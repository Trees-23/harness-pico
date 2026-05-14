"""模型后端适配层。

runtime 只关心一件事：给我一个 prompt，我拿回一段文本。
不同 provider 在 HTTP 接口、响应结构、是否支持 prompt cache 上都有差异，
这些差异都在这里被抹平成统一的 complete() 接口。
"""

import json
import os
import time
from http.client import RemoteDisconnected
import urllib.error
import urllib.request

from .providers.base import ModelResponse, ToolCallRequest


def _messages_to_prompt(messages):
    rendered = []
    for message in messages or []:
        role = str(message.get("role", "")).upper() or "UNKNOWN"
        content = message.get("content", "")
        if isinstance(content, list):
            content = "\n".join(
                str(block.get("text") or block.get("image_url") or block)
                if isinstance(block, dict)
                else str(block)
                for block in content
            )
        rendered.append(f"{role}:\n{content}")
    return "\n\n".join(rendered)


def _openai_content_blocks(content):
    if isinstance(content, list):
        blocks = []
        for block in content:
            if not isinstance(block, dict):
                blocks.append({"type": "input_text", "text": str(block)})
                continue
            if block.get("type") == "image_url":
                blocks.append({"type": "input_image", "image_url": block.get("image_url", {}).get("url", "")})
                continue
            text = block.get("text")
            blocks.append({"type": "input_text", "text": str(text if text is not None else block)})
        return blocks
    return [{"type": "input_text", "text": str(content or "")}]


def _openai_input_from_messages(messages):
    instructions = []
    input_items = []
    for message in messages or []:
        role = str(message.get("role") or "").strip()
        content = message.get("content", "")
        if role == "system":
            instructions.append(str(content or ""))
            continue
        if role == "tool":
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": str(message.get("tool_call_id") or ""),
                    "output": str(content or ""),
                }
            )
            continue
        item = {
            "role": "assistant" if role == "assistant" else "user",
            "content": _openai_content_blocks(content),
        }
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            item["tool_calls"] = list(tool_calls)
        input_items.append(item)
    return "\n\n".join(part for part in instructions if part.strip()), input_items


def _openai_responses_payload_from_messages(
    *,
    model,
    messages,
    max_new_tokens,
    tools=None,
    temperature=None,
    prompt_cache_key=None,
    prompt_cache_retention=None,
    supports_prompt_cache=False,
):
    instructions, input_items = _openai_input_from_messages(messages)
    payload = {
        "model": model,
        "input": input_items,
        "max_output_tokens": max_new_tokens,
        "stream": False,
    }
    if instructions:
        payload["instructions"] = instructions
    if temperature is not None:
        payload["temperature"] = temperature
    if tools:
        payload["tools"] = list(tools)
        payload["tool_choice"] = "auto"
    if supports_prompt_cache and prompt_cache_key:
        payload["prompt_cache_key"] = prompt_cache_key
    if supports_prompt_cache and prompt_cache_retention:
        payload["prompt_cache_retention"] = prompt_cache_retention
    return payload


def _chat_content_blocks(content):
    if isinstance(content, list):
        parts = []
        for block in content:
            if not isinstance(block, dict):
                parts.append(str(block))
                continue
            if block.get("type") == "image_url":
                parts.append({"type": "image_url", "image_url": block.get("image_url", {})})
                continue
            text = block.get("text")
            parts.append(str(text if text is not None else block))
        if all(isinstance(part, str) for part in parts):
            return "\n".join(parts)
        return [
            part if isinstance(part, dict) else {"type": "text", "text": str(part)}
            for part in parts
        ]
    return str(content or "")


def _openai_chat_messages(messages):
    chat_messages = []
    for message in messages or []:
        role = str(message.get("role") or "").strip()
        content = message.get("content", "")
        if role == "tool":
            chat_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(message.get("tool_call_id") or ""),
                    "content": str(content or ""),
                }
            )
            continue
        item = {
            "role": role if role in {"system", "user", "assistant"} else "user",
            "content": _chat_content_blocks(content),
        }
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            item["tool_calls"] = list(tool_calls)
        chat_messages.append(item)
    return chat_messages


def _openai_chat_payload_from_messages(
    *,
    model,
    messages,
    max_new_tokens,
    tools=None,
    temperature=None,
):
    payload = {
        "model": model,
        "messages": _openai_chat_messages(messages),
        "max_tokens": max_new_tokens,
        "stream": False,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if tools:
        payload["tools"] = list(tools)
        payload["tool_choice"] = "auto"
    return payload


def _anthropic_content_blocks(content):
    if isinstance(content, list):
        blocks = []
        for block in content:
            if not isinstance(block, dict):
                blocks.append({"type": "text", "text": str(block)})
                continue
            if block.get("type") == "image_url":
                blocks.append({"type": "text", "text": str(block.get("image_url", {}).get("url", ""))})
                continue
            text = block.get("text")
            blocks.append({"type": "text", "text": str(text if text is not None else block)})
        return blocks
    return [{"type": "text", "text": str(content or "")}]


def _anthropic_messages_from_chat(messages):
    system_parts = []
    anthropic_messages = []
    for message in messages or []:
        role = str(message.get("role") or "").strip()
        content = message.get("content", "")
        if role == "system":
            system_parts.append(str(content or ""))
            continue
        if role == "tool":
            anthropic_messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": str(message.get("tool_call_id") or ""),
                            "content": str(content or ""),
                        }
                    ],
                }
            )
            continue
        blocks = _anthropic_content_blocks(content)
        if role == "assistant":
            for tool_call in message.get("tool_calls") or []:
                function = tool_call.get("function") if isinstance(tool_call, dict) else None
                if not isinstance(function, dict):
                    continue
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": str(tool_call.get("id") or ""),
                        "name": str(function.get("name") or ""),
                        "input": _parse_json_arguments(function.get("arguments") or {}),
                    }
                )
        anthropic_messages.append(
            {
                "role": "assistant" if role == "assistant" else "user",
                "content": blocks,
            }
        )
    return "\n\n".join(part for part in system_parts if part.strip()), anthropic_messages


def _anthropic_payload_from_messages(
    *,
    model,
    messages,
    max_new_tokens,
    tools=None,
    temperature=None,
):
    system, anthropic_messages = _anthropic_messages_from_chat(messages)
    payload = {
        "model": model,
        "messages": anthropic_messages,
        "max_tokens": max_new_tokens,
        "stream": False,
    }
    if system:
        payload["system"] = system
    if temperature is not None:
        payload["temperature"] = temperature
    anthropic_tools = _to_anthropic_tools(tools)
    if anthropic_tools:
        payload["tools"] = anthropic_tools
        payload["tool_choice"] = {"type": "auto"}
    return payload


def _response_metadata(client):
    metadata = dict(getattr(client, "last_completion_metadata", {}) or {})
    usage_keys = {"input_tokens", "output_tokens", "total_tokens", "cached_tokens"}
    usage = {key: value for key, value in metadata.items() if key in usage_keys and value is not None}
    return metadata, usage


class FakeModelClient:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.prompts = []
        self.messages = []
        self.supports_prompt_cache = False
        self.last_completion_metadata = {}

    def complete(self, prompt, max_new_tokens, **kwargs):
        self.prompts.append(prompt)
        if not getattr(self, "last_completion_metadata", None):
            self.last_completion_metadata = {}
        if not self.outputs:
            raise RuntimeError("fake model ran out of outputs")
        return self.outputs.pop(0)

    def complete_with_tools(self, prompt, max_new_tokens, tools=None, **kwargs):
        output = self.complete(prompt, max_new_tokens, **kwargs)
        if isinstance(output, ModelResponse):
            return output
        metadata, usage = _response_metadata(self)
        return ModelResponse(content=str(output), raw=output, metadata=metadata, usage=usage)

    def complete_messages(self, messages, max_new_tokens, **kwargs):
        self.messages.append(list(messages or []))
        return self.complete(_messages_to_prompt(messages), max_new_tokens, **kwargs)

    def complete_messages_with_tools(self, messages, max_new_tokens, tools=None, **kwargs):
        self.messages.append(list(messages or []))
        return self.complete_with_tools(
            _messages_to_prompt(messages),
            max_new_tokens,
            tools=tools,
            **kwargs,
        )


class OllamaModelClient:
    def __init__(self, model, host, temperature, top_p, timeout):
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        self.supports_prompt_cache = False
        self.last_completion_metadata = {}

    def complete(self, prompt, max_new_tokens, **kwargs):
        # Ollama 当前不支持我们这里接入的 prompt cache 语义，
        # 所以 runtime 传下来的缓存参数会被忽略。
        self.last_completion_metadata = {}
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "raw": False,
            "think": False,
            "options": {
                "num_predict": max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
            },
        }
        request = urllib.request.Request(
            self.host + "/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama request failed with HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "Could not reach Ollama.\n"
                "Make sure `ollama serve` is running and the model is available.\n"
                f"Host: {self.host}\n"
                f"Model: {self.model}"
            ) from exc

        if data.get("error"):
            raise RuntimeError(f"Ollama error: {data['error']}")
        return data.get("response", "")

    def complete_with_tools(self, prompt, max_new_tokens, tools=None, **kwargs):
        del tools
        content = self.complete(prompt, max_new_tokens, **kwargs)
        metadata, usage = _response_metadata(self)
        return ModelResponse(content=content, metadata=metadata, usage=usage)

    def complete_messages(self, messages, max_new_tokens, **kwargs):
        return self.complete_messages_with_tools(messages, max_new_tokens, tools=None, **kwargs).content

    def complete_messages_with_tools(self, messages, max_new_tokens, tools=None, **kwargs):
        self.last_completion_metadata = {}
        payload = _openai_responses_payload_from_messages(
            model=self.model,
            messages=messages,
            max_new_tokens=max_new_tokens,
            tools=tools,
            temperature=self.temperature,
            prompt_cache_key=kwargs.get("prompt_cache_key"),
            prompt_cache_retention=kwargs.get("prompt_cache_retention"),
            supports_prompt_cache=self.supports_prompt_cache,
        )

        headers = {
            "Content-Type": "application/json",
            "User-Agent": self.user_agent,
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request = urllib.request.Request(
            self.base_url + "/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        attempts = 3
        for attempt in range(attempts):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body_text = response.read().decode("utf-8")
                    headers = getattr(response, "headers", {}) or {}
                    content_type = headers.get("Content-Type", "")
                break
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                if exc.code >= 500 and attempt < attempts - 1:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise RuntimeError(f"OpenAI-compatible request failed with HTTP {exc.code}: {body}") from exc
            except (urllib.error.URLError, RemoteDisconnected) as exc:
                if attempt < attempts - 1:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise RuntimeError(
                    "Could not reach the OpenAI-compatible backend.\n"
                    f"Base URL: {self.base_url}\n"
                    f"Model: {self.model}"
                ) from exc

        if content_type.startswith("text/event-stream") or body_text.lstrip().startswith("data:"):
            text, response_data = _extract_openai_response_from_sse(body_text)
            response_data = response_data if isinstance(response_data, dict) else {}
        else:
            try:
                response_data = json.loads(body_text)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "OpenAI-compatible error: backend returned non-JSON content that could not be parsed"
                ) from exc
            if response_data.get("error"):
                raise RuntimeError(f"OpenAI-compatible error: {response_data['error']}")
            text = _extract_openai_text(response_data)

        if isinstance(response_data, dict) and response_data:
            self.last_completion_metadata = {
                "prompt_cache_supported": self.supports_prompt_cache,
                "prompt_cache_key": kwargs.get("prompt_cache_key"),
                "prompt_cache_retention": kwargs.get("prompt_cache_retention"),
                **_extract_usage_cache_details(response_data),
            }
        metadata, usage = _response_metadata(self)
        return ModelResponse(
            content=text or "",
            tool_calls=_extract_openai_tool_calls(response_data),
            raw=response_data,
            metadata=metadata,
            usage=usage,
        )


def _normalize_versioned_base_url(base_url):
    base = str(base_url).rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    return base


def _extract_openai_text(data):
    if data.get("output_text"):
        return data["output_text"]

    for item in data.get("output", []):
        for content in item.get("content", []):
            if isinstance(content, dict):
                text = content.get("text")
                if text:
                    return text

    choices = data.get("choices", [])
    if choices:
        message = choices[0].get("message", {})
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if text:
                        return text

    return ""


def _parse_json_arguments(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _extract_openai_tool_calls(data):
    calls = []
    for item in data.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type in {"function_call", "tool_call"}:
            name = str(item.get("name") or "").strip()
            if name:
                calls.append(
                    ToolCallRequest(
                        id=str(item.get("call_id") or item.get("id") or f"tool_{len(calls) + 1}"),
                        name=name,
                        arguments=_parse_json_arguments(item.get("arguments") or {}),
                        raw=item,
                    )
                )
        for content in item.get("content", []) or []:
            if isinstance(content, dict) and content.get("type") in {"function_call", "tool_call"}:
                name = str(content.get("name") or "").strip()
                if name:
                    calls.append(
                        ToolCallRequest(
                            id=str(content.get("call_id") or content.get("id") or f"tool_{len(calls) + 1}"),
                            name=name,
                            arguments=_parse_json_arguments(content.get("arguments") or {}),
                            raw=content,
                        )
                    )

    for choice in data.get("choices", []) or []:
        message = choice.get("message", {}) if isinstance(choice, dict) else {}
        for call in message.get("tool_calls", []) or []:
            if not isinstance(call, dict):
                continue
            function = call.get("function", {}) or {}
            name = str(function.get("name") or "").strip()
            if name:
                calls.append(
                    ToolCallRequest(
                        id=str(call.get("id") or f"tool_{len(calls) + 1}"),
                        name=name,
                        arguments=_parse_json_arguments(function.get("arguments") or {}),
                        raw=call,
                    )
                )
    return calls


def _extract_openai_text_from_sse(body_text):
    last_response = None
    deltas = []
    for line in body_text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            continue
        event_type = event.get("type", "")
        if event_type == "response.output_text.delta":
            delta = event.get("delta")
            if isinstance(delta, str):
                deltas.append(delta)
            continue
        if event_type == "response.output_text.done":
            text = event.get("text")
            if isinstance(text, str) and text:
                return text
        part = event.get("part")
        if isinstance(part, dict):
            text = part.get("text")
            if isinstance(text, str) and text:
                return text
        item = event.get("item")
        if isinstance(item, dict):
            text = _extract_openai_text({"output": [item]})
            if text:
                return text
        response = event.get("response")
        if isinstance(response, dict):
            last_response = response
            text = _extract_openai_text(response)
            if text:
                return text
        text = _extract_openai_text(event)
        if text:
            return text
    if deltas:
        return "".join(deltas)
    if isinstance(last_response, dict):
        return _extract_openai_text(last_response)
    return ""


def _extract_openai_response_from_sse(body_text):
    last_response = None
    deltas = []
    for line in body_text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            continue
        response = event.get("response")
        if isinstance(response, dict):
            last_response = response
            if event.get("type") == "response.completed":
                text = _extract_openai_text(response)
                if text:
                    return text, response
        event_type = event.get("type", "")
        if event_type == "response.output_text.delta":
            delta = event.get("delta")
            if isinstance(delta, str):
                deltas.append(delta)
        elif event_type == "response.output_text.done":
            text = event.get("text")
            if isinstance(text, str) and text:
                return text, last_response or {}
        else:
            text = _extract_openai_text(event)
            if text:
                return text, event
    if deltas:
        return "".join(deltas), last_response or {}
    if isinstance(last_response, dict):
        return _extract_openai_text(last_response), last_response
    return "", {}


def _extract_chat_text(data):
    choices = data.get("choices", [])
    if choices:
        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = [
                str(item.get("text"))
                for item in content
                if isinstance(item, dict) and item.get("text") is not None
            ]
            if texts:
                return "".join(texts)
    return _extract_openai_text(data)


def _extract_chat_tool_calls(data):
    calls = []
    for choice in data.get("choices", []) or []:
        message = choice.get("message", {}) if isinstance(choice, dict) else {}
        for call in message.get("tool_calls", []) or []:
            if not isinstance(call, dict):
                continue
            function = call.get("function", {}) or {}
            name = str(function.get("name") or "").strip()
            if not name:
                continue
            calls.append(
                ToolCallRequest(
                    id=str(call.get("id") or f"tool_{len(calls) + 1}"),
                    name=name,
                    arguments=_parse_json_arguments(function.get("arguments") or {}),
                    raw=call,
                )
            )
    return calls


def _extract_usage_cache_details(data):
    # 把不同 OpenAI-compatible 返回里的 usage 字段整理成统一结构，
    # 让 runtime/trace/report 不需要关心 provider 细节。
    usage = data.get("usage") or {}
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
    input_details = usage.get("input_tokens_details") or usage.get("prompt_tokens_details") or {}
    cached_tokens = int(input_details.get("cached_tokens") or 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": usage.get("total_tokens"),
        "cached_tokens": cached_tokens,
        "cache_hit": cached_tokens > 0,
    }


class OpenAICompatibleModelClient:
    DEFAULT_USER_AGENT = "pico/0.1"

    def __init__(self, model, base_url, api_key, temperature, timeout, user_agent=None, api_style=None):
        self.model = model
        self.base_url = _normalize_versioned_base_url(base_url)
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout
        self.user_agent = user_agent or os.environ.get("OPENAI_USER_AGENT") or self.DEFAULT_USER_AGENT
        self.api_style = (api_style or os.environ.get("OPENAI_API_STYLE") or "responses").strip().lower()
        if self.api_style in {"chat", "chat-completions", "chat_completions"}:
            self.api_style = "chat_completions"
        elif self.api_style not in {"responses", "chat_completions"}:
            self.api_style = "responses"
        # 当前只在明确支持 prompt cache 语义的后端上启用这条链路，
        # 避免对不支持的后端传一个“看起来统一、其实没意义”的伪参数。
        self.supports_prompt_cache = any(host in self.base_url for host in ("openai.com", "right.codes"))
        self.last_completion_metadata = {}

    def _headers(self):
        headers = {
            "Content-Type": "application/json",
            "User-Agent": self.user_agent,
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _post_json(self, path, payload):
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        attempts = 3
        for attempt in range(attempts):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body_text = response.read().decode("utf-8")
                    headers = getattr(response, "headers", {}) or {}
                    content_type = headers.get("Content-Type", "")
                return body_text, content_type
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                if exc.code >= 500 and attempt < attempts - 1:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise RuntimeError(f"OpenAI-compatible request failed with HTTP {exc.code}: {body}") from exc
            except (urllib.error.URLError, RemoteDisconnected) as exc:
                if attempt < attempts - 1:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise RuntimeError(
                    "Could not reach the OpenAI-compatible backend.\n"
                    f"Base URL: {self.base_url}\n"
                    f"Model: {self.model}"
                ) from exc

    def _complete_chat_messages_with_tools(self, messages, max_new_tokens, tools=None):
        payload = _openai_chat_payload_from_messages(
            model=self.model,
            messages=messages,
            max_new_tokens=max_new_tokens,
            tools=tools,
            temperature=self.temperature,
        )
        body_text, content_type = self._post_json("/chat/completions", payload)
        if content_type.startswith("text/event-stream") or body_text.lstrip().startswith("data:"):
            text, response_data = _extract_openai_response_from_sse(body_text)
            response_data = response_data if isinstance(response_data, dict) else {}
        else:
            try:
                response_data = json.loads(body_text)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "OpenAI-compatible error: backend returned non-JSON content that could not be parsed"
                ) from exc
            if response_data.get("error"):
                raise RuntimeError(f"OpenAI-compatible error: {response_data['error']}")
            text = _extract_chat_text(response_data)
        if isinstance(response_data, dict) and response_data:
            self.last_completion_metadata = {
                "prompt_cache_supported": False,
                **_extract_usage_cache_details(response_data),
            }
        metadata, usage = _response_metadata(self)
        return ModelResponse(
            content=text or "",
            tool_calls=_extract_chat_tool_calls(response_data),
            raw=response_data,
            metadata=metadata,
            usage=usage,
        )

    def _complete_responses_messages_with_tools(self, messages, max_new_tokens, tools=None, **kwargs):
        payload = _openai_responses_payload_from_messages(
            model=self.model,
            messages=messages,
            max_new_tokens=max_new_tokens,
            tools=tools,
            temperature=self.temperature,
            prompt_cache_key=kwargs.get("prompt_cache_key"),
            prompt_cache_retention=kwargs.get("prompt_cache_retention"),
            supports_prompt_cache=self.supports_prompt_cache,
        )
        body_text, content_type = self._post_json("/responses", payload)
        if content_type.startswith("text/event-stream") or body_text.lstrip().startswith("data:"):
            text, response_data = _extract_openai_response_from_sse(body_text)
            response_data = response_data if isinstance(response_data, dict) else {}
        else:
            try:
                response_data = json.loads(body_text)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "OpenAI-compatible error: backend returned non-JSON content that could not be parsed"
                ) from exc
            if response_data.get("error"):
                raise RuntimeError(f"OpenAI-compatible error: {response_data['error']}")
            text = _extract_openai_text(response_data)
        if isinstance(response_data, dict) and response_data:
            self.last_completion_metadata = {
                "prompt_cache_supported": self.supports_prompt_cache,
                "prompt_cache_key": kwargs.get("prompt_cache_key"),
                "prompt_cache_retention": kwargs.get("prompt_cache_retention"),
                **_extract_usage_cache_details(response_data),
            }
        metadata, usage = _response_metadata(self)
        return ModelResponse(
            content=text or "",
            tool_calls=_extract_openai_tool_calls(response_data),
            raw=response_data,
            metadata=metadata,
            usage=usage,
        )

    def complete(self, prompt, max_new_tokens, prompt_cache_key=None, prompt_cache_retention=None):
        """向 OpenAI-compatible `/responses` 接口发起一次模型调用。

        为什么存在：
        runtime 不应该知道 HTTP 细节、SSE 细节、usage 字段长什么样，
        更不应该自己去判断 prompt cache 参数要不要带。这个函数把这些后端
        细节都包起来，对上层暴露统一的 `complete()` 行为。

        输入 / 输出：
        - 输入：完整 prompt、最大输出 token，以及可选的 prompt cache 参数
        - 输出：模型最终文本；同时把 usage / cached_tokens 等元数据写进
          `self.last_completion_metadata`

        在 agent 链路里的位置：
        它位于 `Pico.ask()` 的模型调用阶段，是稳定前缀缓存复用链路真正
        落到 provider API 的地方。
        """
        self.last_completion_metadata = {}
        payload = { # 创建http请求
            "model": self.model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt,
                        }
                    ],
                }
            ],
            "max_output_tokens": max_new_tokens,
            "stream": False,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        # runtime 传入的是“稳定前缀”的签名，而不是整段 prompt 的签名。
        # 这样缓存复用针对的是稳定段，不会因为动态 history 每轮变化而失效。
        if self.supports_prompt_cache and prompt_cache_key:
            payload["prompt_cache_key"] = prompt_cache_key
        if self.supports_prompt_cache and prompt_cache_retention:
            payload["prompt_cache_retention"] = prompt_cache_retention

        headers = {
            "Content-Type": "application/json",
            "User-Agent": self.user_agent,
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request = urllib.request.Request(
            self.base_url + "/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        attempts = 3
        for attempt in range(attempts):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body_text = response.read().decode("utf-8")
                    headers = getattr(response, "headers", {}) or {}
                    content_type = headers.get("Content-Type", "")
                break
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                if exc.code >= 500 and attempt < attempts - 1:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise RuntimeError(f"OpenAI-compatible request failed with HTTP {exc.code}: {body}") from exc
            except (urllib.error.URLError, RemoteDisconnected) as exc:
                if attempt < attempts - 1:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise RuntimeError(
                    "Could not reach the OpenAI-compatible backend.\n"
                    f"Base URL: {self.base_url}\n"
                    f"Model: {self.model}"
                ) from exc

        # 有些兼容后端返回普通 JSON，有些返回 SSE。
        # 这里两种都接住，并尽量统一抽取文本和 usage/cache 元数据。
        if content_type.startswith("text/event-stream") or body_text.lstrip().startswith("data:"):
            text, response_data = _extract_openai_response_from_sse(body_text)
            if isinstance(response_data, dict) and response_data:
                # 这些元数据会一路传回 runtime，进入 trace 和 report，
                # 用来观察 prompt cache 是否真的命中。
                self.last_completion_metadata = {
                    "prompt_cache_supported": self.supports_prompt_cache,
                    "prompt_cache_key": prompt_cache_key,
                    "prompt_cache_retention": prompt_cache_retention,
                    **_extract_usage_cache_details(response_data),
                }
            if text:
                return text
            raise RuntimeError("OpenAI-compatible error: could not extract text from event stream response")

        try:
            data = json.loads(body_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "OpenAI-compatible error: backend returned non-JSON content that could not be parsed"
            ) from exc
        if data.get("error"):
            raise RuntimeError(f"OpenAI-compatible error: {data['error']}")
        self.last_completion_metadata = {
            "prompt_cache_supported": self.supports_prompt_cache,
            "prompt_cache_key": prompt_cache_key,
            "prompt_cache_retention": prompt_cache_retention,
            **_extract_usage_cache_details(data),
        }
        return _extract_openai_text(data)

    def complete_with_tools(self, prompt, max_new_tokens, tools=None, prompt_cache_key=None, prompt_cache_retention=None):
        self.last_completion_metadata = {}
        payload = {
            "model": self.model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt,
                        }
                    ],
                }
            ],
            "max_output_tokens": max_new_tokens,
            "stream": False,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if tools:
            payload["tools"] = list(tools)
            payload["tool_choice"] = "auto"
        if self.supports_prompt_cache and prompt_cache_key:
            payload["prompt_cache_key"] = prompt_cache_key
        if self.supports_prompt_cache and prompt_cache_retention:
            payload["prompt_cache_retention"] = prompt_cache_retention

        headers = {
            "Content-Type": "application/json",
            "User-Agent": self.user_agent,
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request = urllib.request.Request(
            self.base_url + "/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        attempts = 3
        for attempt in range(attempts):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body_text = response.read().decode("utf-8")
                    headers = getattr(response, "headers", {}) or {}
                    content_type = headers.get("Content-Type", "")
                break
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                if exc.code >= 500 and attempt < attempts - 1:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise RuntimeError(f"OpenAI-compatible request failed with HTTP {exc.code}: {body}") from exc
            except (urllib.error.URLError, RemoteDisconnected) as exc:
                if attempt < attempts - 1:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise RuntimeError(
                    "Could not reach the OpenAI-compatible backend.\n"
                    f"Base URL: {self.base_url}\n"
                    f"Model: {self.model}"
                ) from exc

        if content_type.startswith("text/event-stream") or body_text.lstrip().startswith("data:"):
            text, response_data = _extract_openai_response_from_sse(body_text)
            response_data = response_data if isinstance(response_data, dict) else {}
        else:
            try:
                response_data = json.loads(body_text)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "OpenAI-compatible error: backend returned non-JSON content that could not be parsed"
                ) from exc
            if response_data.get("error"):
                raise RuntimeError(f"OpenAI-compatible error: {response_data['error']}")
            text = _extract_openai_text(response_data)

        if isinstance(response_data, dict) and response_data:
            self.last_completion_metadata = {
                "prompt_cache_supported": self.supports_prompt_cache,
                "prompt_cache_key": prompt_cache_key,
                "prompt_cache_retention": prompt_cache_retention,
                **_extract_usage_cache_details(response_data),
            }
        metadata, usage = _response_metadata(self)
        return ModelResponse(
            content=text or "",
            tool_calls=_extract_openai_tool_calls(response_data),
            raw=response_data,
            metadata=metadata,
            usage=usage,
        )

    def complete_messages(self, messages, max_new_tokens, **kwargs):
        return self.complete_messages_with_tools(messages, max_new_tokens, tools=None, **kwargs).content

    def complete_messages_with_tools(self, messages, max_new_tokens, tools=None, **kwargs):
        self.last_completion_metadata = {}
        if self.api_style == "chat_completions":
            return self._complete_chat_messages_with_tools(messages, max_new_tokens, tools=tools)
        return self._complete_responses_messages_with_tools(messages, max_new_tokens, tools=tools, **kwargs)


def _extract_anthropic_text(data):
    for item in data.get("content", []):
        if isinstance(item, dict) and item.get("type") == "text":
            text = item.get("text")
            if isinstance(text, str) and text:
                return text
    return ""


def _extract_anthropic_tool_calls(data):
    calls = []
    for item in data.get("content", []) or []:
        if not isinstance(item, dict) or item.get("type") != "tool_use":
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        arguments = item.get("input") or {}
        calls.append(
            ToolCallRequest(
                id=str(item.get("id") or f"tool_{len(calls) + 1}"),
                name=name,
                arguments=arguments if isinstance(arguments, dict) else {},
                raw=item,
            )
        )
    return calls


def _to_anthropic_tools(tools):
    converted = []
    for tool in tools or []:
        function = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not name:
            continue
        converted.append(
            {
                "name": name,
                "description": function.get("description", ""),
                "input_schema": function.get("parameters", {"type": "object", "properties": {}}),
            }
        )
    return converted


class AnthropicCompatibleModelClient:
    def __init__(self, model, base_url, api_key, temperature, timeout):
        self.model = model
        self.base_url = _normalize_versioned_base_url(base_url)
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout
        self.supports_prompt_cache = False
        self.last_completion_metadata = {}

    def complete(self, prompt, max_new_tokens, prompt_cache_key=None, prompt_cache_retention=None):
        # 为了保持统一接口，runtime 仍然会传缓存参数进来；
        # 这里只是显式丢弃，因为当前 Anthropic-compatible 路径没有接缓存复用。
        del prompt_cache_key, prompt_cache_retention
        self.last_completion_metadata = {}
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        }
                    ],
                }
            ],
            "max_tokens": max_new_tokens,
            "stream": False,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }

        request = urllib.request.Request(
            self.base_url + "/messages",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        attempts = 3
        for attempt in range(attempts):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body_text = response.read().decode("utf-8")
                break
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                if exc.code >= 500 and attempt < attempts - 1:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise RuntimeError(f"Anthropic-compatible request failed with HTTP {exc.code}: {body}") from exc
            except (urllib.error.URLError, RemoteDisconnected) as exc:
                if attempt < attempts - 1:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise RuntimeError(
                    "Could not reach the Anthropic-compatible backend.\n"
                    f"Base URL: {self.base_url}\n"
                    f"Model: {self.model}"
                ) from exc

        try:
            data = json.loads(body_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Anthropic-compatible error: backend returned non-JSON content that could not be parsed"
            ) from exc
        if data.get("error"):
            raise RuntimeError(f"Anthropic-compatible error: {data['error']}")
        text = _extract_anthropic_text(data)
        if text:
            return text
        raise RuntimeError("Anthropic-compatible error: could not extract text from response")

    def complete_with_tools(self, prompt, max_new_tokens, tools=None, prompt_cache_key=None, prompt_cache_retention=None):
        del prompt_cache_key, prompt_cache_retention
        self.last_completion_metadata = {}
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        }
                    ],
                }
            ],
            "max_tokens": max_new_tokens,
            "stream": False,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        anthropic_tools = _to_anthropic_tools(tools)
        if anthropic_tools:
            payload["tools"] = anthropic_tools
            payload["tool_choice"] = {"type": "auto"}

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }

        request = urllib.request.Request(
            self.base_url + "/messages",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        attempts = 3
        for attempt in range(attempts):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body_text = response.read().decode("utf-8")
                break
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                if exc.code >= 500 and attempt < attempts - 1:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise RuntimeError(f"Anthropic-compatible request failed with HTTP {exc.code}: {body}") from exc
            except (urllib.error.URLError, RemoteDisconnected) as exc:
                if attempt < attempts - 1:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise RuntimeError(
                    "Could not reach the Anthropic-compatible backend.\n"
                    f"Base URL: {self.base_url}\n"
                    f"Model: {self.model}"
                ) from exc

        try:
            data = json.loads(body_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Anthropic-compatible error: backend returned non-JSON content that could not be parsed"
            ) from exc
        if data.get("error"):
            raise RuntimeError(f"Anthropic-compatible error: {data['error']}")
        metadata, usage = _response_metadata(self)
        return ModelResponse(
            content=_extract_anthropic_text(data),
            tool_calls=_extract_anthropic_tool_calls(data),
            raw=data,
            metadata=metadata,
            usage=usage,
        )

    def complete_messages(self, messages, max_new_tokens, **kwargs):
        return self.complete(_messages_to_prompt(messages), max_new_tokens, **kwargs)

    def complete_messages_with_tools(self, messages, max_new_tokens, tools=None, **kwargs):
        return self.complete_with_tools(
            _messages_to_prompt(messages),
            max_new_tokens,
            tools=tools,
            **kwargs,
        )


__all__ = [
    "AnthropicCompatibleModelClient",
    "FakeModelClient",
    "ModelResponse",
    "OllamaModelClient",
    "OpenAICompatibleModelClient",
    "ToolCallRequest",
]
