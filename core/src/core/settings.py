"""Helpers for configuring globals in one place."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import MISSING, dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, TypeVar, Union, get_args, get_origin, get_type_hints
from types import UnionType

import yaml

from core.logging import LoggingSettings


@dataclass(slots=True, frozen=True)
class CoreSettings:
    """Global settings."""

    logging: LoggingSettings = LoggingSettings()

    # LLM settings
    llm: str = ""
    embed_model: str = ""
    openai_api_key: str = ""

    # common app settings
    title = ("",)
    description = ("",)
    version = ("",)


T = TypeVar("T")


def load_dataclass_from_yaml(dataclass_type: type[T], yaml_path: str | Path) -> T:
    """Load a dataclass instance from a YAML file with environment fallbacks.

    Args:
        dataclass_type: Dataclass type that defines the desired configuration schema.
        yaml_path: Path to the YAML file that contains hierarchical configuration data.

    Returns:
        An instance of ``dataclass_type`` populated with YAML values. Missing values or
        empty string values for string fields are filled from environment variables
        derived from the hierarchical field path (uppercase, underscore-separated).

    Raises:
        FileNotFoundError: If ``yaml_path`` does not exist.
        TypeError: If ``dataclass_type`` is not a dataclass or YAML content is invalid.
        ValueError: If a required value cannot be resolved from YAML defaults or the
            environment.
    """

    if not is_dataclass(dataclass_type):
        msg = f"Expected a dataclass type, got {dataclass_type!r}"
        raise TypeError(msg)

    path = Path(yaml_path)
    raw_data: Mapping[str, Any] = _load_yaml_mapping(path)
    type_hints = get_type_hints(dataclass_type, include_extras=True)

    return _build_dataclass(dataclass_type, raw_data, type_hints, [])


def _load_yaml_mapping(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    if not isinstance(loaded, Mapping):
        msg = f"Top-level YAML content must be a mapping, got {type(loaded).__name__}"
        raise TypeError(msg)

    return loaded


def _build_dataclass(
    dataclass_type: type[T],
    data: Mapping[str, Any],
    type_hints: dict[str, Any],
    path_segments: list[str],
) -> T:
    field_values: dict[str, Any] = {}

    for field in fields(dataclass_type):
        field_type = type_hints.get(field.name, field.type)
        current_path = path_segments + [field.name]
        field_data = data.get(field.name, MISSING)

        resolved = _resolve_field_value(
            field_type=field_type,
            field=field,
            value=field_data,
            current_path=current_path,
        )
        field_values[field.name] = resolved

    return dataclass_type(**field_values)


def _resolve_field_value(  # noqa: PLR0913 - keep signature explicit for readability
    field_type: Any,
    field,
    value: Any,
    current_path: list[str],
) -> Any:
    base_type, allows_none = _strip_optional(field_type)
    base_type = _strip_annotated(base_type)

    if is_dataclass(base_type):
        mapping_value = _ensure_mapping(value, current_path)
        nested_type_hints = get_type_hints(base_type, include_extras=True)
        return _build_dataclass(
            base_type, mapping_value, nested_type_hints, current_path
        )

    resolved_value = value
    env_var_name = _to_env_var(current_path)

    if resolved_value is MISSING or (
        _is_string_type(base_type) and resolved_value == ""
    ):
        env_value = os.getenv(env_var_name)
        if env_value is not None:
            return _convert_value(env_value, base_type)

        if resolved_value is MISSING:
            return _fallback_default(field, allows_none, current_path, env_var_name)

        return _handle_empty_string_fallback(
            field, allows_none, current_path, env_var_name
        )

    if resolved_value is None:
        if allows_none:
            return None
        joined_path = ".".join(current_path)
        msg = f"Field '{joined_path}' does not allow null values"
        raise ValueError(msg)

    return _convert_value(resolved_value, base_type)


def _fallback_default(
    field, allows_none: bool, path: list[str], env_var_name: str
) -> Any:
    if field.default is not MISSING:
        return field.default
    if field.default_factory is not MISSING:  # type: ignore[attr-defined]
        return field.default_factory()  # type: ignore[misc]
    if allows_none:
        return None

    joined_path = ".".join(path)
    msg = (
        "Missing configuration for field "
        f"'{joined_path}'. Provide it in YAML or set {env_var_name}."
    )
    raise ValueError(msg)


def _handle_empty_string_fallback(
    field,
    allows_none: bool,
    path: list[str],
    env_var_name: str,
) -> Any:
    if field.default is not MISSING:
        return field.default
    if field.default_factory is not MISSING:  # type: ignore[attr-defined]
        return field.default_factory()  # type: ignore[misc]
    if allows_none:
        return None

    joined_path = ".".join(path)
    msg = (
        "Empty string provided for field "
        f"'{joined_path}'. Set {env_var_name} to override the value."
    )
    raise ValueError(msg)


def _ensure_mapping(value: Any, path: list[str]) -> Mapping[str, Any]:
    if value is MISSING or value is None:
        return {}
    if isinstance(value, Mapping):
        return value

    joined_path = ".".join(path)
    msg = (
        f"Expected mapping for nested dataclass field '{joined_path}', "
        f"got {type(value).__name__}."
    )
    raise TypeError(msg)


def _convert_value(value: Any, target_type: Any) -> Any:
    if value is None:
        return None

    if target_type is Any:
        return value

    origin = get_origin(target_type)
    if origin is not None:
        return value

    if isinstance(target_type, type):
        if issubclass(target_type, Enum):
            return _convert_enum(value, target_type)
        if issubclass(target_type, Path):
            return Path(value)

    if isinstance(value, target_type):  # type: ignore[arg-type]
        return value

    if target_type is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lower = value.strip().lower()
            truthy = {"1", "true", "t", "yes", "y", "on"}
            falsy = {"0", "false", "f", "no", "n", "off"}
            if lower in truthy:
                return True
            if lower in falsy:
                return False
            msg = f"Cannot convert value '{value}' to bool"
            raise ValueError(msg)
        if isinstance(value, (int, float)):
            return bool(value)
        msg = f"Cannot convert value of type {type(value).__name__} to bool"
        raise ValueError(msg)

    if target_type is int:
        return int(value)
    if target_type is float:
        return float(value)
    if target_type is str:
        return str(value)

    if isinstance(target_type, type):
        try:
            return target_type(value)
        except Exception as exc:  # pragma: no cover - defensive path
            joined_path = ", ".join(map(str, target_type.__mro__))
            raise ValueError(f"Cannot convert value to {joined_path}") from exc

    return value


def _convert_enum(value: Any, enum_type: type[Enum]) -> Enum:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type[value]  # type: ignore[index]
    except KeyError:
        try:
            return enum_type(value)
        except ValueError as exc:
            options = ", ".join(member.name for member in enum_type)
            msg = f"Cannot convert value '{value}' to Enum {enum_type.__name__}: {options}"
            raise ValueError(msg) from exc


def _strip_optional(field_type: Any) -> tuple[Any, bool]:
    origin = get_origin(field_type)
    if origin not in {UnionType, Union}:
        return field_type, False

    args = get_args(field_type)
    non_none_args = [arg for arg in args if arg is not type(None)]  # noqa: E721
    allows_none = len(non_none_args) != len(args)

    if allows_none and len(non_none_args) == 1:
        return non_none_args[0], True

    return field_type, allows_none


def _strip_annotated(field_type: Any) -> Any:
    if Annotated is not None and get_origin(field_type) is Annotated:
        return get_args(field_type)[0]
    return field_type


def _is_string_type(field_type: Any) -> bool:
    return field_type is str


def _to_env_var(path: list[str]) -> str:
    def normalize(segment: str) -> str:
        return segment.replace("-", "_").upper()

    return "__".join(normalize(segment) for segment in path)
