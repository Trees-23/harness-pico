"""JSON Schema builders for Pico tools."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .base import Schema


class StringSchema(Schema):
    def __init__(
        self,
        description: str = "",
        *,
        min_length: int | None = None,
        max_length: int | None = None,
        enum: tuple[Any, ...] | list[Any] | None = None,
        nullable: bool = False,
    ) -> None:
        self._description = description
        self._min_length = min_length
        self._max_length = max_length
        self._enum = tuple(enum) if enum is not None else None
        self._nullable = nullable

    def to_json_schema(self) -> dict[str, Any]:
        schema_type: Any = ["string", "null"] if self._nullable else "string"
        out: dict[str, Any] = {"type": schema_type}
        if self._description:
            out["description"] = self._description
        if self._min_length is not None:
            out["minLength"] = self._min_length
        if self._max_length is not None:
            out["maxLength"] = self._max_length
        if self._enum is not None:
            out["enum"] = list(self._enum)
        return out


class IntegerSchema(Schema):
    def __init__(
        self,
        value: int = 0,
        *,
        description: str = "",
        minimum: int | None = None,
        maximum: int | None = None,
        enum: tuple[int, ...] | list[int] | None = None,
        nullable: bool = False,
    ) -> None:
        self._value = value
        self._description = description
        self._minimum = minimum
        self._maximum = maximum
        self._enum = tuple(enum) if enum is not None else None
        self._nullable = nullable

    def to_json_schema(self) -> dict[str, Any]:
        schema_type: Any = ["integer", "null"] if self._nullable else "integer"
        out: dict[str, Any] = {"type": schema_type}
        if self._description:
            out["description"] = self._description
        if self._minimum is not None:
            out["minimum"] = self._minimum
        if self._maximum is not None:
            out["maximum"] = self._maximum
        if self._enum is not None:
            out["enum"] = list(self._enum)
        return out


class NumberSchema(Schema):
    def __init__(
        self,
        value: float = 0.0,
        *,
        description: str = "",
        minimum: float | None = None,
        maximum: float | None = None,
        enum: tuple[float, ...] | list[float] | None = None,
        nullable: bool = False,
    ) -> None:
        self._value = value
        self._description = description
        self._minimum = minimum
        self._maximum = maximum
        self._enum = tuple(enum) if enum is not None else None
        self._nullable = nullable

    def to_json_schema(self) -> dict[str, Any]:
        schema_type: Any = ["number", "null"] if self._nullable else "number"
        out: dict[str, Any] = {"type": schema_type}
        if self._description:
            out["description"] = self._description
        if self._minimum is not None:
            out["minimum"] = self._minimum
        if self._maximum is not None:
            out["maximum"] = self._maximum
        if self._enum is not None:
            out["enum"] = list(self._enum)
        return out


class BooleanSchema(Schema):
    def __init__(self, *, description: str = "", default: bool | None = None, nullable: bool = False) -> None:
        self._description = description
        self._default = default
        self._nullable = nullable

    def to_json_schema(self) -> dict[str, Any]:
        schema_type: Any = ["boolean", "null"] if self._nullable else "boolean"
        out: dict[str, Any] = {"type": schema_type}
        if self._description:
            out["description"] = self._description
        if self._default is not None:
            out["default"] = self._default
        return out


class ArraySchema(Schema):
    def __init__(
        self,
        items: Any | None = None,
        *,
        description: str = "",
        min_items: int | None = None,
        max_items: int | None = None,
        nullable: bool = False,
    ) -> None:
        self._items_schema = items if items is not None else StringSchema("")
        self._description = description
        self._min_items = min_items
        self._max_items = max_items
        self._nullable = nullable

    def to_json_schema(self) -> dict[str, Any]:
        schema_type: Any = ["array", "null"] if self._nullable else "array"
        out: dict[str, Any] = {"type": schema_type, "items": Schema.fragment(self._items_schema)}
        if self._description:
            out["description"] = self._description
        if self._min_items is not None:
            out["minItems"] = self._min_items
        if self._max_items is not None:
            out["maxItems"] = self._max_items
        return out


class ObjectSchema(Schema):
    def __init__(
        self,
        properties: Mapping[str, Any] | None = None,
        *,
        required: list[str] | None = None,
        description: str = "",
        additional_properties: bool | dict[str, Any] | None = None,
        nullable: bool = False,
        **kwargs: Any,
    ) -> None:
        self._properties = dict(properties or {}, **kwargs)
        self._required = list(required or [])
        self._root_description = description
        self._additional_properties = additional_properties
        self._nullable = nullable

    def to_json_schema(self) -> dict[str, Any]:
        schema_type: Any = ["object", "null"] if self._nullable else "object"
        out: dict[str, Any] = {
            "type": schema_type,
            "properties": {key: Schema.fragment(value) for key, value in self._properties.items()},
        }
        if self._required:
            out["required"] = self._required
        if self._root_description:
            out["description"] = self._root_description
        if self._additional_properties is not None:
            out["additionalProperties"] = self._additional_properties
        return out


def tool_parameters_schema(
    *,
    required: list[str] | None = None,
    description: str = "",
    **properties: Any,
) -> dict[str, Any]:
    return ObjectSchema(required=required, description=description, **properties).to_json_schema()
