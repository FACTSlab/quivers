"""Stable identifiers and source provenance for the QIEC kernel.

Identifiers in the kernel are content-derived references, not process-local
object identities.  This keeps the serialized core independent of Python
objects and gives migrations a stable target when source locations move.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from hashlib import sha256
import json
from typing import ClassVar, Literal, Self


QIEC_ABI = "qiec-core/v1alpha1"


def _canonical_value(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return {"$bytes": value.hex()}
    if isinstance(value, StableId):
        return value.to_data()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "$type": f"{type(value).__module__}.{type(value).__qualname__}",
            **{
                field.name: _canonical_value(getattr(value, field.name))
                for field in fields(value)
            },
        }
    raise TypeError(f"value has no canonical QIEC encoding: {value!r}")


def _canonical_bytes(parts: tuple[object, ...]) -> bytes:
    """Encode identifier inputs in the canonical QIEC representation."""
    return json.dumps(
        _canonical_value(parts),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


@dataclass(frozen=True, slots=True, order=True)
class StableId:
    """A namespaced, SHA-256-derived kernel identifier.

    The human-readable declaration name is intentionally not stored in the
    identifier.  Renaming display labels must not silently alter equality;
    callers choose the declaration path supplied to :meth:`derive` and record
    renames explicitly in a migration map.
    """

    digest: str
    namespace: ClassVar[str] = "id"

    def __post_init__(self) -> None:
        if len(self.digest) != 64 or any(
            c not in "0123456789abcdef" for c in self.digest
        ):
            raise ValueError(f"invalid {self.namespace} digest: {self.digest!r}")

    @classmethod
    def derive(cls, *parts: object) -> Self:
        payload = (QIEC_ABI, cls.namespace, *parts)
        return cls(sha256(_canonical_bytes(payload)).hexdigest())

    @classmethod
    def parse(cls, text: str) -> Self:
        prefix = f"qiec:{cls.namespace}:"
        if not text.startswith(prefix):
            raise ValueError(f"expected {prefix!r} identifier, got {text!r}")
        return cls(text.removeprefix(prefix))

    def __str__(self) -> str:
        return f"qiec:{self.namespace}:{self.digest}"

    def to_data(self) -> str:
        """Return the stable wire representation."""
        return str(self)


class TypeId(StableId):
    namespace = "type"


class FamilyId(StableId):
    namespace = "family"


class ConstructorId(StableId):
    namespace = "constructor"


class EffectId(StableId):
    namespace = "effect"


class OperationId(StableId):
    namespace = "operation"


class EffectInstanceId(StableId):
    namespace = "effect-instance"


class HandlerId(StableId):
    namespace = "handler"


class AttachmentId(StableId):
    namespace = "attachment"


class SiteId(StableId):
    namespace = "site"


class EqualityId(StableId):
    namespace = "equality"


class StaticScopeId(StableId):
    namespace = "static-scope"


class StaticVariableId(StableId):
    namespace = "static-variable"


class RowVariableId(StableId):
    namespace = "row-variable"


@dataclass(frozen=True, slots=True)
class SourceOrigin:
    """Stable structural source origin with optional diagnostic coordinates.

    ``structural_path`` is the protocol path after parsing and name
    resolution.  Line and column are diagnostic metadata and therefore do not
    participate in :meth:`site_id`.
    """

    module: str
    structural_path: tuple[str | int, ...]
    role: str
    source_protocol: str
    file: str | None = None
    line: int | None = None
    column: int | None = None

    def site_id(self) -> SiteId:
        return SiteId.derive(
            self.source_protocol,
            self.module,
            self.structural_path,
            self.role,
        )

    def to_data(self) -> dict[str, object]:
        return {
            "module": self.module,
            "structural_path": list(self.structural_path),
            "role": self.role,
            "source_protocol": self.source_protocol,
            "file": self.file,
            "line": self.line,
            "column": self.column,
        }


type AddressRelation = Literal[
    "preserve",
    "split",
    "duplicate",
    "eliminate",
]


@dataclass(frozen=True, slots=True)
class DynamicAddressFrame:
    """One serializable component of a dynamic effect address."""

    scope: str
    key: str | int


@dataclass(frozen=True, slots=True)
class SiteProvenance:
    """Static and dynamic provenance of an effect request."""

    origin: SourceOrigin
    dynamic_path: tuple[DynamicAddressFrame, ...] = ()
    resumption_path: tuple[int, ...] = ()
    relation: AddressRelation = "preserve"
    parents: tuple[SiteId, ...] = ()

    @property
    def static_site(self) -> SiteId:
        return self.origin.site_id()

    def dynamic_key(
        self,
    ) -> tuple[str, tuple[tuple[str, str | int], ...], tuple[int, ...]]:
        return (
            str(self.static_site),
            tuple((frame.scope, frame.key) for frame in self.dynamic_path),
            self.resumption_path,
        )

    def to_data(self) -> dict[str, object]:
        return {
            "origin": self.origin.to_data(),
            "static_site": self.static_site.to_data(),
            "dynamic_path": [
                {"scope": frame.scope, "key": frame.key} for frame in self.dynamic_path
            ],
            "resumption_path": list(self.resumption_path),
            "relation": self.relation,
            "parents": [parent.to_data() for parent in self.parents],
        }


__all__ = [
    "AddressRelation",
    "AttachmentId",
    "ConstructorId",
    "DynamicAddressFrame",
    "EffectId",
    "EffectInstanceId",
    "EqualityId",
    "FamilyId",
    "HandlerId",
    "OperationId",
    "QIEC_ABI",
    "SiteId",
    "SiteProvenance",
    "SourceOrigin",
    "StableId",
    "StaticScopeId",
    "StaticVariableId",
    "TypeId",
    "RowVariableId",
]
