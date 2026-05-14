"""Pure Pico tool capabilities used by standard Tool adapters."""

from __future__ import annotations

import shutil
import subprocess
import textwrap
import base64
import mimetypes
import difflib
from pathlib import Path
from typing import Any

from ..skills import BUILTIN_SKILLS_DIR
from ..workspace import IGNORED_PATH_NAMES, clip


_BLOCKED_DEVICE_PATHS = frozenset({
    "/dev/zero", "/dev/random", "/dev/urandom", "/dev/full",
    "/dev/stdin", "/dev/stdout", "/dev/stderr",
    "/dev/tty", "/dev/console",
    "/dev/fd/0", "/dev/fd/1", "/dev/fd/2",
})


def _is_blocked_device(path: str | Path) -> bool:
    raw = str(path)
    try:
        resolved = str(Path(raw).resolve())
    except (OSError, ValueError):
        resolved = raw
    return raw in _BLOCKED_DEVICE_PATHS or resolved in _BLOCKED_DEVICE_PATHS or resolved.startswith("/dev/")


def _detect_image_mime(raw: bytes, path: Path) -> str | None:
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if raw.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if raw.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    mime = mimetypes.guess_type(str(path))[0]
    return mime if mime and mime.startswith("image/") else None


_QUOTE_TABLE = str.maketrans({
    "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"',
})


def _normalize_quotes(text: str) -> str:
    return text.translate(_QUOTE_TABLE)


def tool_list_files(agent: Any, args: dict[str, Any]) -> str:
    path = agent.path(args.get("path", "."))
    if not path.is_dir():
        raise ValueError("path is not a directory")
    entries = [
        item for item in sorted(path.iterdir(), key=lambda item: (item.is_file(), item.name.lower()))
        if item.name not in IGNORED_PATH_NAMES
    ]
    lines = []
    for entry in entries[:200]:
        kind = "[D]" if entry.is_dir() else "[F]"
        lines.append(f"{kind} {entry.relative_to(agent.root)}")
    return "\n".join(lines) or "(empty)"


def tool_read_file(agent: Any, args: dict[str, Any]) -> str:
    if _is_blocked_device(args["path"]):
        raise ValueError(f"reading {args['path']} is blocked")
    raw_path = Path(str(args["path"]))
    if raw_path.is_absolute():
        resolved = raw_path.resolve()
        builtin_root = BUILTIN_SKILLS_DIR.resolve()
        try:
            resolved.relative_to(builtin_root)
        except ValueError:
            path = agent.path(args["path"])
        else:
            path = resolved
    else:
        path = agent.path(args["path"])
    if not path.is_file():
        raise ValueError("path is not a file")
    start = int(args.get("start", 1))
    end = int(args.get("end", 200))
    if start < 1 or end < start:
        raise ValueError("invalid line range")
    raw = path.read_bytes()
    image_mime = _detect_image_mime(raw, path)
    if image_mime:
        return [
            {
                "type": "image_url",
                "image_url": {"url": f"data:{image_mime};base64,{base64.b64encode(raw).decode('ascii')}"},
                "_meta": {"path": str(path)},
            },
            {"type": "text", "text": f"(Image file: {path})"},
        ]
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError("cannot read binary file; only UTF-8 text and images are supported")
    lines = text.replace("\r\n", "\n").splitlines()
    body = "\n".join(f"{number:>4}: {line}" for number, line in enumerate(lines[start - 1:end], start=start))
    try:
        display_path = path.relative_to(agent.root)
    except ValueError:
        display_path = path
    return f"# {display_path}\n{body}"


def tool_search(agent: Any, args: dict[str, Any]) -> str:
    pattern = str(args.get("pattern", "")).strip()
    if not pattern:
        raise ValueError("pattern must not be empty")
    path = agent.path(args.get("path", "."))

    if shutil.which("rg"):
        output_mode = str(args.get("output_mode", "")).strip()
        extra = ["--count"] if output_mode == "count" else []
        result = subprocess.run(
            ["rg", "-n", "--smart-case", "--max-count", "200", *extra, pattern, str(path)],
            cwd=agent.root,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or result.stderr.strip() or "(no matches)"

    matches = []
    files = [path] if path.is_file() else [
        item for item in path.rglob("*")
        if item.is_file() and not any(part in IGNORED_PATH_NAMES for part in item.relative_to(agent.root).parts)
    ]
    for file_path in files:
        for number, line in enumerate(file_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
            if pattern.lower() in line.lower():
                matches.append(f"{file_path.relative_to(agent.root)}:{number}:{line}")
                if len(matches) >= 200:
                    return "\n".join(matches)
    return "\n".join(matches) or "(no matches)"


def tool_glob(agent: Any, args: dict[str, Any]) -> str:
    base = agent.path(args.get("path", "."))
    if not base.is_dir():
        raise ValueError("path is not a directory")
    pattern = str(args.get("pattern", "**/*") or "**/*")
    matches = [
        item
        for item in sorted(base.glob(pattern))
        if item.is_file() and not any(part in IGNORED_PATH_NAMES for part in item.relative_to(agent.root).parts)
    ]
    return "\n".join(str(item.relative_to(agent.root)) for item in matches[:200]) or "(no matches)"


def tool_run_shell(agent: Any, args: dict[str, Any]) -> str:
    command = str(args.get("command", "")).strip()
    if not command:
        raise ValueError("command must not be empty")
    deny_patterns = ("rm -rf /", "mkfs", ":(){", "dd if=", "> /dev/")
    if any(pattern in command for pattern in deny_patterns):
        raise ValueError("command rejected by safety policy")
    timeout = int(args.get("timeout", 20))
    if timeout < 1 or timeout > 600:
        raise ValueError("timeout must be in [1, 600]")
    result = subprocess.run(
        command,
        cwd=agent.root,
        shell=True,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=agent.shell_env(),
    )
    return clip(textwrap.dedent(
        f"""\
        exit_code: {result.returncode}
        stdout:
        {result.stdout.strip() or "(empty)"}
        stderr:
        {result.stderr.strip() or "(empty)"}
        """
    ).strip(), 16_000)


def tool_write_file(agent: Any, args: dict[str, Any]) -> str:
    path = agent.path(args["path"])
    content = str(args["content"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"wrote {path.relative_to(agent.root)} ({len(content)} chars)"


def tool_patch_file(agent: Any, args: dict[str, Any]) -> str:
    path = agent.path(args["path"])
    if not path.is_file():
        raise ValueError("path is not a file")
    old_text = str(args.get("old_text", ""))
    if not old_text:
        raise ValueError("old_text must not be empty")
    if "new_text" not in args:
        raise ValueError("missing new_text")
    text = path.read_text(encoding="utf-8")
    count = text.count(old_text)
    if count == 1:
        path.write_text(text.replace(old_text, str(args["new_text"]), 1), encoding="utf-8")
        return f"patched {path.relative_to(agent.root)}"
    stripped_old = old_text.strip()
    if stripped_old and text.count(stripped_old) == 1:
        path.write_text(text.replace(stripped_old, str(args["new_text"]).strip(), 1), encoding="utf-8")
        return f"patched {path.relative_to(agent.root)}"
    normalized_text = _normalize_quotes(text)
    normalized_old = _normalize_quotes(old_text)
    if normalized_old and normalized_text.count(normalized_old) == 1:
        start = normalized_text.index(normalized_old)
        end = start + len(normalized_old)
        path.write_text(text[:start] + str(args["new_text"]) + text[end:], encoding="utf-8")
        return f"patched {path.relative_to(agent.root)}"
    candidates = difflib.get_close_matches(stripped_old or old_text, text.splitlines(), n=3, cutoff=0.45)
    hint = "; closest lines: " + " | ".join(candidates) if candidates else ""
    raise ValueError(f"old_text must occur exactly once, found {count}{hint}")


def tool_delegate(agent: Any, args: dict[str, Any]) -> str:
    if agent.depth >= agent.max_depth:
        raise ValueError("delegate depth exceeded")
    task = str(args.get("task", "")).strip()
    if not task:
        raise ValueError("task must not be empty")

    from ..loop import PicoLoop

    child = PicoLoop(
        model_client=agent.model_client,
        workspace=agent.workspace,
        session_store=agent.session_store,
        run_store=agent.run_store,
        approval_policy="never",
        max_steps=int(args.get("max_steps", 3)),
        max_new_tokens=agent.max_new_tokens,
        depth=agent.depth + 1,
        max_depth=agent.max_depth,
        read_only=True,
        secret_env_names=agent.secret_env_names,
        shell_env_allowlist=agent.shell_env_allowlist,
    )
    child.session["memory"]["task"] = task
    child.session["memory"]["notes"] = [clip(agent.history_text(), 300)]
    return "delegate_result:\n" + child.ask(task)
