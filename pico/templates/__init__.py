"""Markdown template loading for pico prompts and bootstrap files."""

from functools import lru_cache
from importlib.resources import files as pkg_files
import re
from typing import Any

try:
    from jinja2 import Environment, PackageLoader
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal venvs
    Environment = PackageLoader = None


@lru_cache(maxsize=None)
def read_template(*parts: str) -> str:
    resource = pkg_files(__name__)
    for part in parts:
        resource = resource.joinpath(part)
    return resource.read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _environment():
    if Environment is None or PackageLoader is None:
        return None
    return Environment(
        loader=PackageLoader("pico", "templates"),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_template(name: str, *, strip: bool = False, **kwargs: Any) -> str:
    environment = _environment()
    if environment is not None:
        text = environment.get_template(name).render(**kwargs)
    else:
        text = _render_without_jinja(name, **kwargs)
    return text.rstrip() if strip else text


def _render_without_jinja(name: str, **kwargs: Any) -> str:
    text = read_template(*name.split("/"))

    def include(match: re.Match[str]) -> str:
        include_name = match.group(1)
        try:
            return read_template(*include_name.split("/"))
        except Exception:
            return ""

    text = re.sub(r"\{%\s*include\s+'([^']+)'\s*%\}", include, text)
    text = re.sub(r"\{%\s*raw\s*%\}(.*?)\{%\s*endraw\s*%\}", r"\1", text, flags=re.S)
    text = re.sub(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}", lambda m: str(kwargs.get(m.group(1), "")), text)
    text = re.sub(r"\{#[\s\S]*?#\}", "", text)
    text = re.sub(r"\{%\s*(if|elif|else|endif)[^%]*%\}", "", text)
    return text
