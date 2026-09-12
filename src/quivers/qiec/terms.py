"""Stable value and computation terms for QIEC.

Terms contain no Python callables.  Runtime values and handler bodies cross
the boundary only through stable attachment and handler identifiers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from quivers.qiec.effects import EffectRequest
from quivers.qiec.evidence import EqualityEvidence
from quivers.qiec.identifiers import (
    AttachmentId,
    ConstructorId,
    EffectInstanceId,
    HandlerId,
    StaticScopeId,
)
from quivers.qiec.kinds import Telescope
from quivers.qiec.types import StaticArgument, TypeExpr


@dataclass(frozen=True, slots=True)
class Local:
    name: str
    type: TypeExpr


@dataclass(frozen=True, slots=True)
class Var:
    local: Local
    tag: Literal["var"] = "var"


type LiteralData = None | bool | int | float | str | bytes | tuple[LiteralData, ...]


@dataclass(frozen=True, slots=True)
class LiteralValue:
    value: LiteralData
    type: TypeExpr
    tag: Literal["literal"] = "literal"


@dataclass(frozen=True, slots=True)
class ConstructorValue:
    constructor: ConstructorId
    static_arguments: tuple[StaticArgument, ...]
    fields: tuple[Value, ...]
    result_type: TypeExpr
    tag: Literal["constructor"] = "constructor"


@dataclass(frozen=True, slots=True)
class EvidenceValue:
    evidence: EqualityEvidence
    tag: Literal["evidence"] = "evidence"


@dataclass(frozen=True, slots=True)
class AttachmentRef:
    """A stable reference to a runtime-owned host value."""

    attachment: AttachmentId
    type: TypeExpr
    tag: Literal["attachment"] = "attachment"


@dataclass(frozen=True, slots=True)
class TransportValue:
    """Transport a value along kernel-checked equality evidence."""

    evidence: EqualityEvidence
    value: Value
    target_type: TypeExpr
    tag: Literal["transport"] = "transport"


type Value = (
    Var
    | LiteralValue
    | ConstructorValue
    | EvidenceValue
    | AttachmentRef
    | TransportValue
)


@dataclass(frozen=True, slots=True)
class Return:
    value: Value
    tag: Literal["return"] = "return"


@dataclass(frozen=True, slots=True)
class Bind:
    binder: Local
    first: Computation
    then: Computation
    tag: Literal["bind"] = "bind"


@dataclass(frozen=True, slots=True)
class Perform:
    request: EffectRequest
    tag: Literal["perform"] = "perform"


@dataclass(frozen=True, slots=True)
class Handle:
    instance: EffectInstanceId
    handler: HandlerId
    computation: Computation
    static_arguments: tuple[StaticArgument, ...] = ()
    tag: Literal["handle"] = "handle"


@dataclass(frozen=True, slots=True)
class CaseMotive:
    """The result family of an indexed case expression."""

    indices: Telescope
    result_type: TypeExpr


@dataclass(frozen=True, slots=True)
class CaseBranch:
    """One stable GADT branch with no host-language closure."""

    constructor: ConstructorId
    static_arguments: tuple[StaticArgument, ...]
    fields: tuple[Local, ...]
    body: Computation
    scope: StaticScopeId


@dataclass(frozen=True, slots=True)
class Case:
    scrutinee: Value
    motive: CaseMotive
    branches: tuple[CaseBranch, ...]
    tag: Literal["case"] = "case"


type Computation = Return | Bind | Perform | Handle | Case


__all__ = [
    "AttachmentRef",
    "Bind",
    "Case",
    "CaseBranch",
    "CaseMotive",
    "Computation",
    "ConstructorValue",
    "EvidenceValue",
    "Handle",
    "LiteralData",
    "LiteralValue",
    "Local",
    "Perform",
    "Return",
    "TransportValue",
    "Value",
    "Var",
]
