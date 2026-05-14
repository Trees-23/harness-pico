"""Base contracts for Pico tools."""

from __future__ import annotations

from abc import ABC, abstractmethod, update_abstractmethods
from collections.abc import Callable
from copy import deepcopy
from typing import Any, TypeVar

_ToolT = TypeVar("_ToolT", bound="Tool")

_JSON_TYPE_MAP: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


class Schema(ABC):
    """JSON Schema fragment contract used by tools."""

    @staticmethod
    def resolve_json_schema_type(value: Any) -> str | None:
        if isinstance(value, list):
            return next((item for item in value if item != "null"), None)
        return value

    @staticmethod
    def subpath(path: str, key: str) -> str:
        return f"{path}.{key}" if path else key

    @staticmethod
    def validate_json_schema_value(value: Any, schema: dict[str, Any], path: str = "") -> list[str]:
        raw_type = schema.get("type")
        nullable = (isinstance(raw_type, list) and "null" in raw_type) or schema.get("nullable", False)
        schema_type = Schema.resolve_json_schema_type(raw_type)
        label = path or "parameter"

        if nullable and value is None:
            return []
        if schema_type == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
            return [f"{label} should be integer"]
        if schema_type == "number" and (not isinstance(value, _JSON_TYPE_MAP["number"]) or isinstance(value, bool)):
            return [f"{label} should be number"]
        if (
            schema_type in _JSON_TYPE_MAP
            and schema_type not in ("integer", "number")
            and not isinstance(value, _JSON_TYPE_MAP[schema_type])
        ):
            return [f"{label} should be {schema_type}"]

        errors: list[str] = []
        if "enum" in schema and value not in schema["enum"]:
            errors.append(f"{label} must be one of {schema['enum']}")
        if schema_type in ("integer", "number"):
            if "minimum" in schema and value < schema["minimum"]:
                errors.append(f"{label} must be >= {schema['minimum']}")
            if "maximum" in schema and value > schema["maximum"]:
                errors.append(f"{label} must be <= {schema['maximum']}")
        if schema_type == "string":
            if "minLength" in schema and len(value) < schema["minLength"]:
                errors.append(f"{label} must be at least {schema['minLength']} chars")
            if "maxLength" in schema and len(value) > schema["maxLength"]:
                errors.append(f"{label} must be at most {schema['maxLength']} chars")
        if schema_type == "object":
            props = schema.get("properties", {})
            for key in schema.get("required", []):
                if key not in value:
                    errors.append(f"missing required {Schema.subpath(path, key)}")
            for key, item in value.items():
                if key in props:
                    errors.extend(Schema.validate_json_schema_value(item, props[key], Schema.subpath(path, key)))
        if schema_type == "array":
            if "minItems" in schema and len(value) < schema["minItems"]:
                errors.append(f"{label} must have at least {schema['minItems']} items")
            if "maxItems" in schema and len(value) > schema["maxItems"]:
                errors.append(f"{label} must be at most {schema['maxItems']} items")
            if "items" in schema:
                prefix = f"{path}[{{}}]" if path else "[{}]"
                for index, item in enumerate(value):
                    errors.extend(Schema.validate_json_schema_value(item, schema["items"], prefix.format(index)))
        return errors

    @staticmethod
    def fragment(value: Any) -> dict[str, Any]:
        to_json_schema = getattr(value, "to_json_schema", None)
        if callable(to_json_schema):
            return to_json_schema()
        if isinstance(value, dict):
            return value
        raise TypeError(f"Expected schema object or dict, got {type(value).__name__}")

    @abstractmethod
    def to_json_schema(self) -> dict[str, Any]:
        ...

    def validate_value(self, value: Any, path: str = "") -> list[str]:
        return Schema.validate_json_schema_value(value, self.to_json_schema(), path)


class Tool(ABC):
    """Pure capability contract for agent tools."""

    _TYPE_MAP = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    _BOOL_TRUE = frozenset(("true", "1", "yes"))
    _BOOL_FALSE = frozenset(("false", "0", "no"))

    @staticmethod
    def _resolve_type(value: Any) -> str | None:
        return Schema.resolve_json_schema_type(value)

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        ...

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        ...

    @property
    def read_only(self) -> bool:
        return False

    @property
    def exclusive(self) -> bool:
        return False

    @property
    def concurrency_safe(self) -> bool:
        return self.read_only and not self.exclusive

    @abstractmethod
    async def execute(self, **kwargs: Any) -> Any:
        ...

    def _cast_object(self, value: Any, schema: dict[str, Any]) -> Any:
        if not isinstance(value, dict):
            return value
        props = schema.get("properties", {})
        return {key: self._cast_value(item, props[key]) if key in props else item for key, item in value.items()}

    def cast_params(self, params: dict[str, Any]) -> dict[str, Any]:
        schema = self.parameters or {}
        if schema.get("type", "object") != "object":
            return params
        cast = self._cast_object(params, schema)
        return cast if isinstance(cast, dict) else params

    def _cast_value(self, value: Any, schema: dict[str, Any]) -> Any:
        schema_type = self._resolve_type(schema.get("type"))
        if schema_type == "boolean" and isinstance(value, bool):
            return value
        if schema_type == "integer" and isinstance(value, int) and not isinstance(value, bool):
            return value
        if schema_type in self._TYPE_MAP and schema_type not in ("boolean", "integer", "array", "object"):
            expected = self._TYPE_MAP[schema_type]
            if isinstance(value, expected):
                return value
        if isinstance(value, str) and schema_type in ("integer", "number"):
            try:
                return int(value) if schema_type == "integer" else float(value)
            except ValueError:
                return value
        if schema_type == "string":
            return value if value is None else str(value)
        if schema_type == "boolean" and isinstance(value, str):
            lowered = value.lower()
            if lowered in self._BOOL_TRUE:
                return True
            if lowered in self._BOOL_FALSE:
                return False
            return value
        if schema_type == "array" and isinstance(value, list):
            items = schema.get("items")
            return [self._cast_value(item, items) for item in value] if isinstance(items, dict) else value
        if schema_type == "object" and isinstance(value, dict):
            return self._cast_object(value, schema)
        return value

    def validate_params(self, params: dict[str, Any]) -> list[str]:
        if not isinstance(params, dict):
            return [f"parameters must be an object, got {type(params).__name__}"]
        schema = self.parameters or {}
        if schema.get("type", "object") != "object":
            raise ValueError(f"Schema must be object type, got {schema.get('type')!r}")
        return Schema.validate_json_schema_value(params, {**schema, "type": "object"}, "")

    def to_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def tool_parameters(schema: dict[str, Any]) -> Callable[[type[_ToolT]], type[_ToolT]]:
    def decorator(cls: type[_ToolT]) -> type[_ToolT]:
        stored_schema = deepcopy(schema)

        @property
        def parameters(self: Tool) -> dict[str, Any]:
            return deepcopy(stored_schema)

        cls.parameters = parameters  # type: ignore[assignment]
        return update_abstractmethods(cls)

    return decorator
