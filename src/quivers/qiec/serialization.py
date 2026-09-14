"""Canonical wire codec for stable QIEC graphs.

The codec is an explicit allowlist, not a generic Python object serializer.
Only frozen kernel records, concrete identifier classes, kernel enums, tuples,
bytes, and JSON scalars cross the boundary.  Runtime attachments, handler
callbacks, modules, mappings, and unknown future node tags are rejected.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING, fields, is_dataclass
from enum import Enum
from functools import cache
import json
import math
import sys
from typing import (
    Any,
    Literal,
    TypeAliasType,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from quivers.qiec.coverage import (
    BranchPattern,
    BranchRefinement,
    CoverageResult,
    CoverageStatus,
    Reachability,
)
from quivers.qiec.declarations import ConstructorDecl, FamilyDecl, FieldDef
from quivers.qiec.effects import (
    ArgumentDef,
    ComputationType,
    EffectDef,
    EffectRequest,
    EffectRow,
    HandlerClauseDef,
    HandlerReturnClauseDef,
    HandlerDef,
    OperationDef,
    ResumptionGrade,
    RowEntry,
    RowSubstitution,
    RowUnification,
    RowVariable,
)
from quivers.qiec.evidence import BranchGiven, Reflexivity
from quivers.qiec.identifiers import (
    AttachmentId,
    ConstructorId,
    DynamicAddressFrame,
    EffectId,
    EffectInstanceId,
    EqualityId,
    FamilyId,
    ComputationId,
    HandlerId,
    OperationId,
    QIEC_ABI,
    RowVariableId,
    SiteId,
    SiteProvenance,
    SourceOrigin,
    StableId,
    StaticScopeId,
    StaticVariableId,
    TypeId,
)
from quivers.qiec.kinds import (
    ArrowKind,
    ContextSort,
    EffectBinder,
    EffectKind,
    IndexBinder,
    NatSort,
    RowKind,
    ShapeSort,
    TypeBinder,
    TypeKind,
    UserIndexSort,
)
from quivers.qiec.module import NamedComputation, NamedEffectInstance, QiecModule
from quivers.qiec.substitution import StaticSubstitution
from quivers.qiec.terms import (
    NewInstance,
    Resume,
    AttachmentRef,
    Bind,
    Call,
    Case,
    CaseBranch,
    CaseMotive,
    ConstructorValue,
    EvidenceValue,
    Handle,
    LiteralValue,
    Local,
    Perform,
    Return,
    TransportValue,
    Computation,
    LiteralData,
    Value,
    Var,
)
from quivers.qiec.types import (
    EffectRef,
    EffectVariable,
    EqualityType,
    FunctionType,
    IndexConstructor,
    IndexLiteral,
    IndexVariable,
    ShapeIndex,
    TypeApplication,
    TypeConstructorRef,
    TypeVariable,
)


QIEC_WIRE_FORMAT = "qiec-json/v1"

type JsonValue = (
    None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
)


class SerializationError(ValueError):
    """A value is not in the stable QIEC wire language."""


def _class_tag(class_: type[object]) -> str:
    module = class_.__module__.removeprefix("quivers.qiec.")
    return f"{module}.{class_.__qualname__}"


_ID_CLASSES = (
    TypeId,
    ComputationId,
    FamilyId,
    ConstructorId,
    EffectId,
    OperationId,
    EffectInstanceId,
    HandlerId,
    AttachmentId,
    SiteId,
    EqualityId,
    StaticScopeId,
    StaticVariableId,
    RowVariableId,
)

_ENUM_CLASSES = (
    ResumptionGrade,
    CoverageStatus,
    Reachability,
)

_NODE_CLASSES = (
    # source identity and kinds
    SourceOrigin,
    DynamicAddressFrame,
    SiteProvenance,
    TypeKind,
    EffectKind,
    RowKind,
    ArrowKind,
    NatSort,
    ShapeSort,
    ContextSort,
    UserIndexSort,
    TypeBinder,
    IndexBinder,
    EffectBinder,
    # static terms
    IndexVariable,
    IndexLiteral,
    IndexConstructor,
    ShapeIndex,
    TypeVariable,
    TypeConstructorRef,
    TypeApplication,
    FunctionType,
    EqualityType,
    EffectVariable,
    EffectRef,
    # declarations, rows, and signatures
    FieldDef,
    FamilyDecl,
    ConstructorDecl,
    RowVariable,
    RowEntry,
    EffectRow,
    RowSubstitution,
    RowUnification,
    ComputationType,
    ArgumentDef,
    OperationDef,
    EffectDef,
    HandlerClauseDef,
    HandlerReturnClauseDef,
    HandlerDef,
    EffectRequest,
    NamedEffectInstance,
    NamedComputation,
    QiecModule,
    StaticSubstitution,
    # equality and coverage results
    Reflexivity,
    BranchGiven,
    BranchPattern,
    BranchRefinement,
    CoverageResult,
    # stable value/computation graph
    Local,
    Var,
    LiteralValue,
    ConstructorValue,
    EvidenceValue,
    AttachmentRef,
    TransportValue,
    Return,
    Bind,
    Perform,
    Handle,
    CaseMotive,
    CaseBranch,
    Case,
    Call,
    Resume,
    NewInstance,
)

_IDS = {_class_tag(class_): class_ for class_ in _ID_CLASSES}
_ENUMS = {_class_tag(class_): class_ for class_ in _ENUM_CLASSES}
_NODES: dict[str, type[object]] = {
    _class_tag(class_): class_ for class_ in _NODE_CLASSES
}


def _encode(value: object) -> JsonValue:
    if isinstance(value, StableId):
        tag = _class_tag(type(value))
        if tag not in _IDS:
            raise SerializationError(f"unregistered QIEC identifier class {tag!r}")
        return {"$id": tag, "value": value.to_data()}
    if isinstance(value, Enum):
        tag = _class_tag(type(value))
        if tag not in _ENUMS:
            raise SerializationError(f"unregistered QIEC enum class {tag!r}")
        return {"$enum": tag, "value": cast(str, value.value)}
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SerializationError("non-finite floats have no canonical JSON form")
        return value
    if isinstance(value, bytes):
        return {"$bytes": value.hex()}
    if isinstance(value, tuple):
        return {"$tuple": [_encode(item) for item in value]}
    if callable(value):
        raise SerializationError("host callables cannot enter stable QIEC")
    if is_dataclass(value) and not isinstance(value, type):
        tag = _class_tag(type(value))
        if tag not in _NODES:
            raise SerializationError(
                f"runtime or unregistered dataclass cannot enter stable QIEC: {tag!r}"
            )
        return {
            "$type": tag,
            "fields": [
                {"name": field.name, "value": _encode(getattr(value, field.name))}
                for field in fields(value)
            ],
        }
    raise SerializationError(
        f"runtime value of type {type(value).__module__}.{type(value).__qualname__} "
        "cannot enter stable QIEC"
    )


def _exact_keys(value: dict[str, JsonValue], expected: set[str], subject: str) -> None:
    if set(value) != expected:
        raise SerializationError(
            f"malformed {subject}: expected keys {sorted(expected)!r}, "
            f"got {sorted(value)!r}"
        )


@cache
def _resolved_field_types(class_: type[object]) -> dict[str, object]:
    """Resolve one allowlisted record's annotations, including cyclic aliases."""

    namespace = dict(vars(sys.modules[class_.__module__]))
    # ``effects`` imports ``Value`` only while type checking to break its
    # runtime import cycle with ``terms``.  Supply every recursive wire alias
    # explicitly so runtime validation sees the same schema as the checker.
    namespace.update(
        {
            "Computation": Computation,
            "LiteralData": LiteralData,
            "Value": Value,
        }
    )
    try:
        return cast(
            dict[str, object],
            get_type_hints(class_, globalns=namespace, localns=namespace),
        )
    except Exception as error:  # pragma: no cover - an internal schema defect
        raise SerializationError(
            f"cannot resolve wire schema for {_class_tag(class_)!r}: {error}"
        ) from error


def _shape_name(annotation: object) -> str:
    if isinstance(annotation, TypeAliasType):
        return annotation.__name__
    return str(annotation).removeprefix("typing.")


def _validate_runtime_shape(
    value: object,
    annotation: object,
    *,
    path: str,
) -> None:
    """Reject decoded values that do not inhabit their declared wire type."""

    if annotation is Any:
        return
    if isinstance(annotation, TypeAliasType):
        _validate_runtime_shape(value, annotation.__value__, path=path)
        return

    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin is Literal:
        if not any(
            type(value) is type(candidate) and value == candidate
            for candidate in arguments
        ):
            raise SerializationError(
                f"invalid field {path}: expected {_shape_name(annotation)}, "
                f"got {type(value).__name__}"
            )
        return
    if origin is Union:
        for candidate in arguments:
            try:
                _validate_runtime_shape(value, candidate, path=path)
            except SerializationError:
                continue
            return
        raise SerializationError(
            f"invalid field {path}: expected {_shape_name(annotation)}, "
            f"got {type(value).__name__}"
        )
    if origin is tuple:
        if not isinstance(value, tuple):
            raise SerializationError(
                f"invalid field {path}: expected {_shape_name(annotation)}, "
                f"got {type(value).__name__}"
            )
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            for position, item in enumerate(value):
                _validate_runtime_shape(
                    item,
                    arguments[0],
                    path=f"{path}[{position}]",
                )
            return
        if len(value) != len(arguments):
            raise SerializationError(
                f"invalid field {path}: expected a {len(arguments)}-item tuple, "
                f"got {len(value)} items"
            )
        for position, (item, item_type) in enumerate(
            zip(value, arguments, strict=True)
        ):
            _validate_runtime_shape(
                item,
                item_type,
                path=f"{path}[{position}]",
            )
        return

    if annotation is None:
        annotation = type(None)
    if isinstance(annotation, type):
        primitive = annotation in {bool, bytes, float, int, str, type(None)}
        matches = (
            type(value) is annotation if primitive else isinstance(value, annotation)
        )
        if not matches:
            raise SerializationError(
                f"invalid field {path}: expected {_shape_name(annotation)}, "
                f"got {type(value).__name__}"
            )
        if is_dataclass(annotation):
            annotations = _resolved_field_types(annotation)
            for field in fields(annotation):
                _validate_runtime_shape(
                    getattr(value, field.name),
                    annotations[field.name],
                    path=f"{path}.{field.name}",
                )
        return

    raise SerializationError(
        f"unsupported wire-schema annotation {_shape_name(annotation)} at {path}"
    )


def _decode(value: JsonValue) -> object:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SerializationError("non-finite floats have no canonical JSON form")
        return value
    if isinstance(value, list):
        raise SerializationError("bare JSON arrays are not QIEC tuples")
    if not isinstance(value, dict):  # pragma: no cover - closed JsonValue union
        raise SerializationError("invalid QIEC JSON value")

    if "$bytes" in value:
        _exact_keys(value, {"$bytes"}, "bytes value")
        encoded = value["$bytes"]
        if not isinstance(encoded, str):
            raise SerializationError("encoded bytes must be a hexadecimal string")
        try:
            return bytes.fromhex(encoded)
        except ValueError as error:
            raise SerializationError("encoded bytes are not hexadecimal") from error

    if "$tuple" in value:
        _exact_keys(value, {"$tuple"}, "tuple")
        items = value["$tuple"]
        if not isinstance(items, list):
            raise SerializationError("$tuple must contain an array")
        return tuple(_decode(item) for item in items)

    if "$id" in value:
        _exact_keys(value, {"$id", "value"}, "identifier")
        tag = value["$id"]
        encoded = value["value"]
        if not isinstance(tag, str) or tag not in _IDS:
            raise SerializationError(f"unknown QIEC identifier tag {tag!r}")
        if not isinstance(encoded, str):
            raise SerializationError("identifier value must be a string")
        try:
            return _IDS[tag].parse(encoded)
        except ValueError as error:
            raise SerializationError(str(error)) from error

    if "$enum" in value:
        _exact_keys(value, {"$enum", "value"}, "enum")
        tag = value["$enum"]
        encoded = value["value"]
        if not isinstance(tag, str) or tag not in _ENUMS:
            raise SerializationError(f"unknown QIEC enum tag {tag!r}")
        try:
            return _ENUMS[tag](encoded)
        except (TypeError, ValueError) as error:
            raise SerializationError(
                f"invalid value {encoded!r} for QIEC enum {tag!r}"
            ) from error

    if "$type" in value:
        _exact_keys(value, {"$type", "fields"}, "kernel node")
        tag = value["$type"]
        encoded_fields = value["fields"]
        if not isinstance(tag, str) or tag not in _NODES:
            raise SerializationError(f"unknown QIEC node tag {tag!r}")
        if not isinstance(encoded_fields, list):
            raise SerializationError("kernel node fields must be an array")
        class_ = _NODES[tag]
        class_fields = fields(cast(Any, class_))
        expected_names = [field.name for field in class_fields]
        actual_names: list[str] = []
        arguments: dict[str, object] = {}
        for encoded_field in encoded_fields:
            if not isinstance(encoded_field, dict):
                raise SerializationError("kernel field entry must be an object")
            _exact_keys(encoded_field, {"name", "value"}, "kernel field")
            name = encoded_field["name"]
            if not isinstance(name, str):
                raise SerializationError("kernel field name must be a string")
            if name in arguments:
                raise SerializationError(f"duplicate kernel field {name!r}")
            actual_names.append(name)
            arguments[name] = _decode(encoded_field["value"])
        if actual_names != expected_names:
            raise SerializationError(
                f"fields for {tag!r} must be {expected_names!r}, got {actual_names!r}"
            )
        tag_field = next((field for field in class_fields if field.name == "tag"), None)
        if tag_field is not None and tag_field.default is not MISSING:
            actual_tag = arguments["tag"]
            if actual_tag != tag_field.default:
                raise SerializationError(
                    f"invalid literal tag for {tag!r}: expected "
                    f"{tag_field.default!r}, got {actual_tag!r}"
                )
        annotations = _resolved_field_types(class_)
        for name, argument in arguments.items():
            _validate_runtime_shape(
                argument,
                annotations[name],
                path=f"{tag}.{name}",
            )
        try:
            factory = cast(Callable[..., object], class_)
            return factory(**arguments)
        except Exception as error:
            raise SerializationError(f"invalid {tag!r} node: {error}") from error

    raise SerializationError("unknown object in QIEC wire data")


def to_data(value: object) -> dict[str, JsonValue]:
    """Encode a stable QIEC graph in its versioned wire envelope."""
    return {
        "$schema": QIEC_WIRE_FORMAT,
        "abi": QIEC_ABI,
        "root": _encode(value),
    }


def from_data(data: object) -> object:
    """Decode a QIEC graph, rejecting other ABIs and unknown node tags."""
    if not isinstance(data, dict) or any(not isinstance(key, str) for key in data):
        raise SerializationError("QIEC wire envelope must be an object")
    typed = cast(dict[str, JsonValue], data)
    _exact_keys(typed, {"$schema", "abi", "root"}, "QIEC wire envelope")
    if typed["$schema"] != QIEC_WIRE_FORMAT:
        raise SerializationError(f"unsupported QIEC wire format {typed['$schema']!r}")
    if typed["abi"] != QIEC_ABI:
        raise SerializationError(
            f"QIEC ABI mismatch: expected {QIEC_ABI!r}, got {typed['abi']!r}"
        )
    return _decode(typed["root"])


def dumps(value: object) -> str:
    """Serialize a QIEC graph as deterministic UTF-8-compatible JSON text."""
    return json.dumps(
        to_data(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def loads(text: str | bytes | bytearray) -> object:
    """Deserialize one canonical QIEC JSON document."""
    try:
        data = json.loads(text, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise SerializationError(f"invalid QIEC JSON: {error}") from error
    return from_data(data)


def _unique_object(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    """Build a JSON object while rejecting duplicate keys at every depth."""

    result: dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in result:
            raise SerializationError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


__all__ = [
    "JsonValue",
    "QIEC_WIRE_FORMAT",
    "SerializationError",
    "dumps",
    "from_data",
    "loads",
    "to_data",
]
