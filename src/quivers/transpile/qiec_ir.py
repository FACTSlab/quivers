"""Typed, target-independent IR for the Quivers Indexed Effect Core.

The QIEC kernel deliberately uses frozen dataclasses: they are convenient for
type checking and evaluation, but are not part of the structural transpiler
IR.  This module is the lossless boundary between the two.  Every stable QIEC
construct has a corresponding :mod:`didactic` model, so a backend can inspect
effects, values, evidence, resumptions, and source provenance without parsing
an opaque JSON side channel.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Literal, cast

import didactic.api as dx

from quivers.qiec import QiecModule
from quivers.qiec.identifiers import StableId


type IRScalar = None | bool | int | float | str
type IRPathPart = str | int


class IRQiecId(dx.Model):
    """A stable QIEC identifier, split into namespace and digest."""

    namespace: str
    digest: str

    @property
    def text(self) -> str:
        return f"qiec:{self.namespace}:{self.digest}"


class IRQiecSourceOrigin(dx.Model):
    module: str
    structural_path: tuple[IRPathPart, ...]
    role: str
    source_protocol: str
    file: str | None = None
    line: int | None = None
    column: int | None = None


class IRQiecDynamicAddressFrame(dx.Model):
    scope: str
    key: str | int


class IRQiecSiteProvenance(dx.Model):
    origin: IRQiecSourceOrigin
    dynamic_path: tuple[IRQiecDynamicAddressFrame, ...] = ()
    resumption_path: tuple[int, ...] = ()
    relation: Literal["preserve", "split", "duplicate", "eliminate"] = "preserve"
    parents: tuple[IRQiecId, ...] = ()


class IRQiecKind(dx.TaggedUnion, discriminator="kind"):
    """Kind in the QIEC static language."""


class IRQiecTypeKind(IRQiecKind):
    kind: Literal["type"] = "type"


class IRQiecEffectKind(IRQiecKind):
    kind: Literal["effect"] = "effect"


class IRQiecRowKind(IRQiecKind):
    kind: Literal["row"] = "row"


class IRQiecArrowKind(IRQiecKind):
    domain: IRQiecKind
    codomain: IRQiecKind
    kind: Literal["arrow"] = "arrow"


class IRQiecIndexSort(dx.TaggedUnion, discriminator="kind"):
    """Closed index sort."""


class IRQiecNatSort(IRQiecIndexSort):
    kind: Literal["nat"] = "nat"


class IRQiecShapeSort(IRQiecIndexSort):
    rank: int | None = None
    kind: Literal["shape"] = "shape"


class IRQiecContextSort(IRQiecIndexSort):
    signature: str
    kind: Literal["context"] = "context"


class IRQiecUserIndexSort(IRQiecIndexSort):
    name: str
    constructors: tuple[str, ...]
    arities: tuple[int, ...]
    kind: Literal["user"] = "user"


class IRQiecBinder(dx.TaggedUnion, discriminator="kind"):
    name: str
    refinable: bool = False


class IRQiecTypeBinder(IRQiecBinder):
    type_kind: IRQiecKind
    kind: Literal["type"] = "type"


class IRQiecIndexBinder(IRQiecBinder):
    sort: IRQiecIndexSort
    kind: Literal["index"] = "index"


class IRQiecEffectBinder(IRQiecBinder):
    kind: Literal["effect"] = "effect"


class IRQiecStatic(dx.TaggedUnion, discriminator="kind"):
    """Type, index, or effect argument in the QIEC static language."""


class IRQiecIndexVariable(IRQiecStatic):
    name: str
    sort: IRQiecIndexSort
    identity: IRQiecId | None = None
    kind: Literal["index-variable"] = "index-variable"


class IRQiecIndexLiteral(IRQiecStatic):
    value: int | str
    sort: IRQiecIndexSort
    kind: Literal["index-literal"] = "index-literal"


class IRQiecIndexConstructor(IRQiecStatic):
    name: str
    arguments: tuple[IRQiecStatic, ...]
    sort: IRQiecIndexSort
    kind: Literal["index-constructor"] = "index-constructor"


class IRQiecShapeIndex(IRQiecStatic):
    dimensions: tuple[IRQiecStatic, ...]
    kind: Literal["shape-index"] = "shape-index"


class IRQiecTypeVariable(IRQiecStatic):
    name: str
    type_kind: IRQiecKind
    identity: IRQiecId | None = None
    kind: Literal["type-variable"] = "type-variable"


class IRQiecTypeConstructor(dx.Model):
    id: IRQiecId
    name: str
    telescope: tuple[IRQiecBinder, ...] = ()


class IRQiecTypeApplication(IRQiecStatic):
    constructor: IRQiecTypeConstructor
    arguments: tuple[IRQiecStatic, ...] = ()
    kind: Literal["type-application"] = "type-application"


class IRQiecFunctionType(IRQiecStatic):
    parameter: IRQiecStatic
    result: IRQiecStatic
    kind: Literal["function-type"] = "function-type"


class IRQiecEqualityType(IRQiecStatic):
    classifier: IRQiecKind | IRQiecIndexSort
    left: IRQiecStatic
    right: IRQiecStatic
    kind: Literal["equality-type"] = "equality-type"


class IRQiecEffectVariable(IRQiecStatic):
    name: str
    identity: IRQiecId | None = None
    kind: Literal["effect-variable"] = "effect-variable"


class IRQiecEffectRef(IRQiecStatic):
    id: IRQiecId
    name: str
    arguments: tuple[IRQiecStatic, ...] = ()
    kind: Literal["effect-ref"] = "effect-ref"


class IRQiecLocal(dx.Model):
    name: str
    type: IRQiecStatic


class IRQiecRowVariable(dx.Model):
    name: str
    identity: IRQiecId
    lacks: tuple[IRQiecId, ...] = ()


class IRQiecRowEntry(dx.Model):
    instance: IRQiecId
    effect: IRQiecEffectRef


class IRQiecEffectRow(dx.Model):
    entries: tuple[IRQiecRowEntry, ...] = ()
    tail: IRQiecRowVariable | None = None


class IRQiecComputationType(dx.Model):
    effects: IRQiecEffectRow
    result: IRQiecStatic


class IRQiecEvidence(dx.TaggedUnion, discriminator="kind"):
    """Kernel-issued equality evidence."""


class IRQiecReflexivity(IRQiecEvidence):
    equality: IRQiecEqualityType
    kind: Literal["reflexivity"] = "reflexivity"


class IRQiecBranchGiven(IRQiecEvidence):
    id: IRQiecId
    equality: IRQiecEqualityType
    kind: Literal["branch-given"] = "branch-given"


class IRQiecLiteral(dx.TaggedUnion, discriminator="kind"):
    """JSON-safe literal data, including bytes and nested tuples."""


class IRQiecNullLiteral(IRQiecLiteral):
    kind: Literal["null"] = "null"


class IRQiecBoolLiteral(IRQiecLiteral):
    value: bool
    kind: Literal["bool"] = "bool"


class IRQiecIntLiteral(IRQiecLiteral):
    value: int
    kind: Literal["int"] = "int"


class IRQiecFloatLiteral(IRQiecLiteral):
    value: float
    kind: Literal["float"] = "float"


class IRQiecStringLiteral(IRQiecLiteral):
    value: str
    kind: Literal["string"] = "string"


class IRQiecBytesLiteral(IRQiecLiteral):
    value: str
    kind: Literal["bytes"] = "bytes"


class IRQiecTupleLiteral(IRQiecLiteral):
    items: tuple[IRQiecLiteral, ...]
    kind: Literal["tuple"] = "tuple"


class IRQiecValue(dx.TaggedUnion, discriminator="kind"):
    """Stable value node consumed by QIEC-capable renderers."""


class IRQiecVar(IRQiecValue):
    local: IRQiecLocal
    kind: Literal["var"] = "var"


class IRQiecLiteralValue(IRQiecValue):
    value: IRQiecLiteral
    type: IRQiecStatic
    kind: Literal["literal"] = "literal"


class IRQiecConstructorValue(IRQiecValue):
    constructor: IRQiecId
    static_arguments: tuple[IRQiecStatic, ...]
    fields: tuple[IRQiecValue, ...]
    result_type: IRQiecStatic
    kind: Literal["constructor"] = "constructor"


class IRQiecEvidenceValue(IRQiecValue):
    evidence: IRQiecEvidence
    kind: Literal["evidence"] = "evidence"


class IRQiecAttachmentRef(IRQiecValue):
    attachment: IRQiecId
    type: IRQiecStatic
    kind: Literal["attachment"] = "attachment"


class IRQiecTransportValue(IRQiecValue):
    evidence: IRQiecEvidence
    value: IRQiecValue
    target_type: IRQiecStatic
    kind: Literal["transport"] = "transport"


class IRQiecEffectRequest(dx.Model):
    instance: IRQiecId
    effect: IRQiecEffectRef
    operation: IRQiecId
    static_arguments: tuple[IRQiecStatic, ...]
    arguments: tuple[IRQiecValue, ...]
    result_type: IRQiecStatic
    origin: IRQiecSiteProvenance


class IRQiecComputation(dx.TaggedUnion, discriminator="kind"):
    """Stable computation node consumed by QIEC-capable renderers."""


class IRQiecReturn(IRQiecComputation):
    value: IRQiecValue
    kind: Literal["return"] = "return"


class IRQiecBind(IRQiecComputation):
    binder: IRQiecLocal
    first: IRQiecComputation
    then: IRQiecComputation
    kind: Literal["bind"] = "bind"


class IRQiecPerform(IRQiecComputation):
    request: IRQiecEffectRequest
    kind: Literal["perform"] = "perform"


class IRQiecHandle(IRQiecComputation):
    instance: IRQiecId
    handler: IRQiecId
    computation: IRQiecComputation
    static_arguments: tuple[IRQiecStatic, ...] = ()
    kind: Literal["handle"] = "handle"


class IRQiecCaseMotive(dx.Model):
    indices: tuple[IRQiecBinder, ...]
    result_type: IRQiecStatic


class IRQiecCaseBranch(dx.Model):
    constructor: IRQiecId
    static_arguments: tuple[IRQiecStatic, ...]
    fields: tuple[IRQiecLocal, ...]
    body: IRQiecComputation
    scope: IRQiecId


class IRQiecCase(IRQiecComputation):
    scrutinee: IRQiecValue
    motive: IRQiecCaseMotive
    branches: tuple[IRQiecCaseBranch, ...]
    kind: Literal["case"] = "case"


class IRQiecFieldDef(dx.Model):
    name: str
    type: IRQiecStatic


class IRQiecFamilyDecl(dx.Model):
    id: IRQiecId
    name: str
    parameters: tuple[IRQiecBinder, ...]
    indices: tuple[IRQiecBinder, ...]
    constructors: tuple[IRQiecId, ...]
    closed: bool = True


class IRQiecConstructorDecl(dx.Model):
    id: IRQiecId
    family: IRQiecId
    name: str
    telescope: tuple[IRQiecBinder, ...]
    fields: tuple[IRQiecFieldDef, ...]
    result_indices: tuple[IRQiecStatic, ...]


class IRQiecArgumentDef(dx.Model):
    name: str
    type: IRQiecStatic


class IRQiecOperationDef(dx.Model):
    id: IRQiecId
    name: str
    telescope: tuple[IRQiecBinder, ...]
    arguments: tuple[IRQiecArgumentDef, ...]
    result_type: IRQiecStatic


class IRQiecEffectDef(dx.Model):
    ref: IRQiecEffectRef
    telescope: tuple[IRQiecBinder, ...]
    operations: tuple[IRQiecOperationDef, ...]


class IRQiecHandlerClauseDef(dx.Model):
    operation: IRQiecId
    grade: Literal["0", "aff", "1", "omega"]


class IRQiecHandlerDef(dx.Model):
    id: IRQiecId
    name: str
    effect: IRQiecEffectRef
    clauses: tuple[IRQiecHandlerClauseDef, ...]
    input_type: IRQiecStatic
    output_type: IRQiecStatic
    introduced: IRQiecEffectRow
    total: bool
    forwards_unknown: bool
    telescope: tuple[IRQiecBinder, ...]


class IRQiecNamedEffectInstance(dx.Model):
    name: str
    entry: IRQiecRowEntry
    origin: IRQiecSourceOrigin


class IRQiecNamedComputation(dx.Model):
    name: str
    telescope: tuple[IRQiecBinder, ...]
    parameters: tuple[IRQiecLocal, ...]
    body: IRQiecComputation
    type: IRQiecComputationType
    origin: IRQiecSourceOrigin


class IRQiecModule(dx.Model):
    """Lossless structural projection of one checked :class:`QiecModule`."""

    module: str
    source_protocol: str
    index_sorts: tuple[IRQiecUserIndexSort, ...] = ()
    families: tuple[IRQiecFamilyDecl, ...] = ()
    constructors: tuple[IRQiecConstructorDecl, ...] = ()
    effects: tuple[IRQiecEffectDef, ...] = ()
    instances: tuple[IRQiecNamedEffectInstance, ...] = ()
    handlers: tuple[IRQiecHandlerDef, ...] = ()
    computations: tuple[IRQiecNamedComputation, ...] = ()
    abi: str


def _id(value: StableId) -> IRQiecId:
    return IRQiecId(namespace=value.namespace, digest=value.digest)


def _literal(value: object) -> IRQiecLiteral:
    if value is None:
        return IRQiecNullLiteral()
    if isinstance(value, bool):
        return IRQiecBoolLiteral(value=value)
    if isinstance(value, int):
        return IRQiecIntLiteral(value=value)
    if isinstance(value, float):
        return IRQiecFloatLiteral(value=value)
    if isinstance(value, str):
        return IRQiecStringLiteral(value=value)
    if isinstance(value, bytes):
        return IRQiecBytesLiteral(value=value.hex())
    if isinstance(value, tuple):
        return IRQiecTupleLiteral(items=tuple(_literal(item) for item in value))
    raise TypeError(f"unsupported QIEC literal {value!r}")


def _convert(value: object) -> object:  # noqa: C901, PLR0911, PLR0912
    """Recursively project a checked kernel record into structural IR."""
    from quivers.qiec import declarations as d
    from quivers.qiec import effects as e
    from quivers.qiec import evidence as ev
    from quivers.qiec import identifiers as ids
    from quivers.qiec import kinds as k
    from quivers.qiec import module as m
    from quivers.qiec import terms as tm
    from quivers.qiec import types as ty

    if isinstance(value, StableId):
        return _id(value)
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, tuple):
        return tuple(_convert(item) for item in value)
    if isinstance(value, ids.SourceOrigin):
        return IRQiecSourceOrigin(
            module=value.module,
            structural_path=value.structural_path,
            role=value.role,
            source_protocol=value.source_protocol,
            file=value.file,
            line=value.line,
            column=value.column,
        )
    if isinstance(value, ids.DynamicAddressFrame):
        return IRQiecDynamicAddressFrame(scope=value.scope, key=value.key)
    if isinstance(value, ids.SiteProvenance):
        return IRQiecSiteProvenance(
            origin=cast(IRQiecSourceOrigin, _convert(value.origin)),
            dynamic_path=cast(
                tuple[IRQiecDynamicAddressFrame, ...], _convert(value.dynamic_path)
            ),
            resumption_path=value.resumption_path,
            relation=value.relation,
            parents=cast(tuple[IRQiecId, ...], _convert(value.parents)),
        )
    kind_map = {
        k.TypeKind: IRQiecTypeKind,
        k.EffectKind: IRQiecEffectKind,
        k.RowKind: IRQiecRowKind,
    }
    if type(value) in kind_map:
        return kind_map[type(value)]()
    if isinstance(value, k.ArrowKind):
        return IRQiecArrowKind(
            domain=cast(IRQiecKind, _convert(value.domain)),
            codomain=cast(IRQiecKind, _convert(value.codomain)),
        )
    sort_map = {k.NatSort: IRQiecNatSort}
    if type(value) in sort_map:
        return sort_map[type(value)]()
    if isinstance(value, k.ShapeSort):
        return IRQiecShapeSort(rank=value.rank)
    if isinstance(value, k.ContextSort):
        return IRQiecContextSort(signature=value.signature)
    if isinstance(value, k.UserIndexSort):
        return IRQiecUserIndexSort(
            name=value.name, constructors=value.constructors, arities=value.arities
        )
    if isinstance(value, k.TypeBinder):
        return IRQiecTypeBinder(
            name=value.name,
            type_kind=cast(IRQiecKind, _convert(value.kind)),
            refinable=value.refinable,
        )
    if isinstance(value, k.IndexBinder):
        return IRQiecIndexBinder(
            name=value.name,
            sort=cast(IRQiecIndexSort, _convert(value.sort)),
            refinable=value.refinable,
        )
    if isinstance(value, k.EffectBinder):
        return IRQiecEffectBinder(name=value.name, refinable=value.refinable)
    if isinstance(value, ty.IndexVariable):
        return IRQiecIndexVariable(
            name=value.name,
            sort=cast(IRQiecIndexSort, _convert(value.sort)),
            identity=cast(IRQiecId | None, _convert(value.identity)),
        )
    if isinstance(value, ty.IndexLiteral):
        return IRQiecIndexLiteral(
            value=value.value, sort=cast(IRQiecIndexSort, _convert(value.sort))
        )
    if isinstance(value, ty.IndexConstructor):
        return IRQiecIndexConstructor(
            name=value.name,
            arguments=cast(tuple[IRQiecStatic, ...], _convert(value.arguments)),
            sort=cast(IRQiecIndexSort, _convert(value.sort)),
        )
    if isinstance(value, ty.ShapeIndex):
        return IRQiecShapeIndex(
            dimensions=cast(tuple[IRQiecStatic, ...], _convert(value.dimensions))
        )
    if isinstance(value, ty.TypeVariable):
        return IRQiecTypeVariable(
            name=value.name,
            type_kind=cast(IRQiecKind, _convert(value.kind)),
            identity=cast(IRQiecId | None, _convert(value.identity)),
        )
    if isinstance(value, ty.TypeConstructorRef):
        return IRQiecTypeConstructor(
            id=_id(value.id),
            name=value.name,
            telescope=cast(tuple[IRQiecBinder, ...], _convert(value.telescope)),
        )
    if isinstance(value, ty.TypeApplication):
        return IRQiecTypeApplication(
            constructor=cast(IRQiecTypeConstructor, _convert(value.constructor)),
            arguments=cast(tuple[IRQiecStatic, ...], _convert(value.arguments)),
        )
    if isinstance(value, ty.FunctionType):
        return IRQiecFunctionType(
            parameter=cast(IRQiecStatic, _convert(value.parameter)),
            result=cast(IRQiecStatic, _convert(value.result)),
        )
    if isinstance(value, ty.EqualityType):
        return IRQiecEqualityType(
            classifier=cast(IRQiecKind | IRQiecIndexSort, _convert(value.kind)),
            left=cast(IRQiecStatic, _convert(value.left)),
            right=cast(IRQiecStatic, _convert(value.right)),
        )
    if isinstance(value, ty.EffectVariable):
        return IRQiecEffectVariable(
            name=value.name, identity=cast(IRQiecId | None, _convert(value.identity))
        )
    if isinstance(value, ty.EffectRef):
        return IRQiecEffectRef(
            id=_id(value.id),
            name=value.name,
            arguments=cast(tuple[IRQiecStatic, ...], _convert(value.arguments)),
        )
    if isinstance(value, tm.Local):
        return IRQiecLocal(
            name=value.name, type=cast(IRQiecStatic, _convert(value.type))
        )
    if isinstance(value, e.RowVariable):
        return IRQiecRowVariable(
            name=value.name,
            identity=_id(value.identity),
            lacks=cast(tuple[IRQiecId, ...], _convert(value.lacks)),
        )
    if isinstance(value, e.RowEntry):
        return IRQiecRowEntry(
            instance=_id(value.instance),
            effect=cast(IRQiecEffectRef, _convert(value.effect)),
        )
    if isinstance(value, e.EffectRow):
        return IRQiecEffectRow(
            entries=cast(tuple[IRQiecRowEntry, ...], _convert(value.entries)),
            tail=cast(IRQiecRowVariable | None, _convert(value.tail)),
        )
    if isinstance(value, e.ComputationType):
        return IRQiecComputationType(
            effects=cast(IRQiecEffectRow, _convert(value.effects)),
            result=cast(IRQiecStatic, _convert(value.result)),
        )
    if isinstance(value, ev.Reflexivity):
        return IRQiecReflexivity(
            equality=cast(IRQiecEqualityType, _convert(value.equality))
        )
    if isinstance(value, ev.BranchGiven):
        return IRQiecBranchGiven(
            id=_id(value.id),
            equality=cast(IRQiecEqualityType, _convert(value.equality)),
        )
    if isinstance(value, tm.Var):
        return IRQiecVar(local=cast(IRQiecLocal, _convert(value.local)))
    if isinstance(value, tm.LiteralValue):
        return IRQiecLiteralValue(
            value=_literal(value.value), type=cast(IRQiecStatic, _convert(value.type))
        )
    if isinstance(value, tm.ConstructorValue):
        return IRQiecConstructorValue(
            constructor=_id(value.constructor),
            static_arguments=cast(
                tuple[IRQiecStatic, ...], _convert(value.static_arguments)
            ),
            fields=cast(tuple[IRQiecValue, ...], _convert(value.fields)),
            result_type=cast(IRQiecStatic, _convert(value.result_type)),
        )
    if isinstance(value, tm.EvidenceValue):
        return IRQiecEvidenceValue(
            evidence=cast(IRQiecEvidence, _convert(value.evidence))
        )
    if isinstance(value, tm.AttachmentRef):
        return IRQiecAttachmentRef(
            attachment=_id(value.attachment),
            type=cast(IRQiecStatic, _convert(value.type)),
        )
    if isinstance(value, tm.TransportValue):
        return IRQiecTransportValue(
            evidence=cast(IRQiecEvidence, _convert(value.evidence)),
            value=cast(IRQiecValue, _convert(value.value)),
            target_type=cast(IRQiecStatic, _convert(value.target_type)),
        )
    if isinstance(value, e.EffectRequest):
        return IRQiecEffectRequest(
            instance=_id(value.instance),
            effect=cast(IRQiecEffectRef, _convert(value.effect)),
            operation=_id(value.operation),
            static_arguments=cast(
                tuple[IRQiecStatic, ...], _convert(value.static_arguments)
            ),
            arguments=cast(tuple[IRQiecValue, ...], _convert(value.arguments)),
            result_type=cast(IRQiecStatic, _convert(value.result_type)),
            origin=cast(IRQiecSiteProvenance, _convert(value.origin)),
        )
    if isinstance(value, tm.Return):
        return IRQiecReturn(value=cast(IRQiecValue, _convert(value.value)))
    if isinstance(value, tm.Bind):
        return IRQiecBind(
            binder=cast(IRQiecLocal, _convert(value.binder)),
            first=cast(IRQiecComputation, _convert(value.first)),
            then=cast(IRQiecComputation, _convert(value.then)),
        )
    if isinstance(value, tm.Perform):
        return IRQiecPerform(request=cast(IRQiecEffectRequest, _convert(value.request)))
    if isinstance(value, tm.Handle):
        return IRQiecHandle(
            instance=_id(value.instance),
            handler=_id(value.handler),
            computation=cast(IRQiecComputation, _convert(value.computation)),
            static_arguments=cast(
                tuple[IRQiecStatic, ...], _convert(value.static_arguments)
            ),
        )
    if isinstance(value, tm.CaseMotive):
        return IRQiecCaseMotive(
            indices=cast(tuple[IRQiecBinder, ...], _convert(value.indices)),
            result_type=cast(IRQiecStatic, _convert(value.result_type)),
        )
    if isinstance(value, tm.CaseBranch):
        return IRQiecCaseBranch(
            constructor=_id(value.constructor),
            static_arguments=cast(
                tuple[IRQiecStatic, ...], _convert(value.static_arguments)
            ),
            fields=cast(tuple[IRQiecLocal, ...], _convert(value.fields)),
            body=cast(IRQiecComputation, _convert(value.body)),
            scope=_id(value.scope),
        )
    if isinstance(value, tm.Case):
        return IRQiecCase(
            scrutinee=cast(IRQiecValue, _convert(value.scrutinee)),
            motive=cast(IRQiecCaseMotive, _convert(value.motive)),
            branches=cast(tuple[IRQiecCaseBranch, ...], _convert(value.branches)),
        )
    if not is_dataclass(value) or isinstance(value, type):
        raise TypeError(f"unsupported QIEC kernel value {value!r}")
    model_map: dict[type[object], type[dx.Model]] = {
        d.FieldDef: IRQiecFieldDef,
        d.FamilyDecl: IRQiecFamilyDecl,
        d.ConstructorDecl: IRQiecConstructorDecl,
        e.ArgumentDef: IRQiecArgumentDef,
        e.OperationDef: IRQiecOperationDef,
        e.EffectDef: IRQiecEffectDef,
        e.HandlerClauseDef: IRQiecHandlerClauseDef,
        e.HandlerDef: IRQiecHandlerDef,
        m.NamedEffectInstance: IRQiecNamedEffectInstance,
        m.NamedComputation: IRQiecNamedComputation,
        m.QiecModule: IRQiecModule,
    }
    model = model_map.get(type(value))
    if model is None:
        raise TypeError(f"unsupported QIEC kernel record {type(value).__qualname__}")
    values = {
        field.name: _convert(getattr(value, field.name)) for field in fields(value)
    }
    return model(**values)


def lower_qiec_ir(module: QiecModule) -> IRQiecModule:
    """Project an already checked kernel module into structural IR."""
    return cast(IRQiecModule, _convert(module))


type QiecFeature = Literal[
    "declarations",
    "return",
    "bind",
    "perform",
    "handle",
    "case",
    "evidence",
    "transport",
    "attachment",
    "structured-value",
    "effectful-row",
    "open-effect-row",
    "type-polymorphism",
    "index-polymorphism",
    "effect-polymorphism",
    "named-parameter",
    "non-scalar-result",
    "non-scalar-parameter",
    "zero-resumption",
    "affine-resumption",
    "linear-resumption",
    "unrestricted-resumption",
]


class QiecTargetCapabilities(dx.Model):
    """Feature set one target can preserve without semantic erasure."""

    features: frozenset[QiecFeature] = frozenset({"declarations"})

    def supports(self, feature: QiecFeature) -> bool:
        return feature in self.features


class QiecCapabilityDiagnostic(dx.Model):
    """One precise target mismatch discovered before renderer dispatch."""

    code: Literal["qiec-capability"] = "qiec-capability"
    target: str
    feature: QiecFeature
    computation: str | None = None
    origin: IRQiecSourceOrigin | None = None
    detail: str

    @property
    def kind(self) -> str:
        subject = self.computation or "module"
        return f"qiec:capability:{self.feature}:{subject}"

    @property
    def message(self) -> str:
        subject = (
            f"QIEC computation `{self.computation}`"
            if self.computation is not None
            else "the QIEC module"
        )
        return f"{subject} requires `{self.feature}`, but {self.target} {self.detail}"


_DECLARATION_ONLY = QiecTargetCapabilities()
_GENERIC_RUNTIME = QiecTargetCapabilities(
    features=frozenset(
        {
            "declarations",
            "return",
            "bind",
            "perform",
            "handle",
            "case",
            "evidence",
            "transport",
            "attachment",
            "structured-value",
            "effectful-row",
            "open-effect-row",
            "type-polymorphism",
            "index-polymorphism",
            "effect-polymorphism",
            "named-parameter",
            "non-scalar-result",
            "non-scalar-parameter",
            "zero-resumption",
            "affine-resumption",
            "linear-resumption",
            "unrestricted-resumption",
        }
    )
)
_STAN_STATIC = QiecTargetCapabilities(
    features=frozenset({"declarations", "return", "bind", "named-parameter"})
)
_GRAPH_STATIC = QiecTargetCapabilities(
    features=frozenset({"declarations", "return", "bind"})
)


def capabilities_for_target(target: str) -> QiecTargetCapabilities:
    """Return the registered QIEC capability set for ``target``.

    The eight host-language backends lower the full stable core through the
    QIEC runtime ABI. Stan, BUGS, and JAGS admit the useful first-order subset:
    closed, monomorphic, effect-free scalar ``return``/``bind`` computations.
    Unknown targets safely retain declaration metadata only.
    """
    normalized = target.removeprefix("qvr-").lower()
    if normalized in {
        "pyro",
        "numpyro",
        "pymc",
        "edward2",
        "turing",
        "gen",
        "webppl",
        "church",
    }:
        return _GENERIC_RUNTIME
    if normalized == "stan":
        return _STAN_STATIC
    if normalized in {"bugs", "jags"}:
        return _GRAPH_STATIC
    return _DECLARATION_ONLY


def _required_features(computation: IRQiecNamedComputation) -> set[QiecFeature]:
    required: set[QiecFeature] = {"return"}
    if computation.parameters:
        required.add("named-parameter")
    if computation.type.effects.entries:
        required.add("effectful-row")
    if computation.type.effects.tail is not None:
        required.add("open-effect-row")
    for binder in computation.telescope:
        if isinstance(binder, IRQiecTypeBinder):
            required.add("type-polymorphism")
        elif isinstance(binder, IRQiecIndexBinder):
            required.add("index-polymorphism")
        else:
            required.add("effect-polymorphism")
    if not _is_scalar_type(computation.type.result):
        required.add("non-scalar-result")
    if any(not _is_scalar_type(parameter.type) for parameter in computation.parameters):
        required.add("non-scalar-parameter")

    def value(node: IRQiecValue) -> None:
        if isinstance(node, IRQiecLiteralValue) and isinstance(
            node.value, IRQiecTupleLiteral
        ):
            required.add("structured-value")
        elif isinstance(node, IRQiecEvidenceValue):
            required.add("evidence")
        elif isinstance(node, IRQiecTransportValue):
            required.update(("evidence", "transport"))
            value(node.value)
        elif isinstance(node, IRQiecAttachmentRef):
            required.add("attachment")
        elif isinstance(node, IRQiecConstructorValue):
            required.add("structured-value")
            for field in node.fields:
                value(field)

    def visit(node: IRQiecComputation) -> None:
        if isinstance(node, IRQiecReturn):
            value(node.value)
        elif isinstance(node, IRQiecBind):
            required.add("bind")
            if not _is_scalar_type(node.binder.type):
                required.add("non-scalar-parameter")
            visit(node.first)
            visit(node.then)
        elif isinstance(node, IRQiecPerform):
            required.add("perform")
            for argument in node.request.arguments:
                value(argument)
        elif isinstance(node, IRQiecHandle):
            required.add("handle")
            visit(node.computation)
        elif isinstance(node, IRQiecCase):
            required.add("case")
            value(node.scrutinee)
            for branch in node.branches:
                visit(branch.body)

    visit(computation.body)
    return required


def _is_scalar_type(type_: IRQiecStatic) -> bool:
    """Whether the static PPL subset can bind ``type_`` as one scalar."""
    return (
        isinstance(type_, IRQiecTypeApplication)
        and not type_.arguments
        and type_.constructor.name in {"Bool", "Int", "Real"}
    )


def analyze_qiec_capabilities(
    module_or_ir: QiecModule | IRQiecModule | None,
    target: str,
    *,
    capabilities: QiecTargetCapabilities | None = None,
) -> tuple[QiecCapabilityDiagnostic, ...]:
    """Report every QIEC feature ``target`` cannot preserve."""
    if module_or_ir is None:
        return ()
    module = (
        module_or_ir
        if isinstance(module_or_ir, IRQiecModule)
        else lower_qiec_ir(module_or_ir)
    )
    supported = capabilities or capabilities_for_target(target)
    diagnostics: list[QiecCapabilityDiagnostic] = []
    if not supported.supports("declarations") and (
        module.index_sorts or module.families or module.effects or module.handlers
    ):
        diagnostics.append(
            QiecCapabilityDiagnostic(
                target=target,
                feature="declarations",
                detail="cannot preserve its typed declarations",
            )
        )
    grade_features = {
        "0": "zero-resumption",
        "aff": "affine-resumption",
        "1": "linear-resumption",
        "omega": "unrestricted-resumption",
    }
    for computation in module.computations:
        required = _required_features(computation)
        handler_ids = _handled_ids(computation.body)
        for handler in module.handlers:
            if handler.id.text not in handler_ids:
                continue
            required.update(
                cast(QiecFeature, grade_features[clause.grade])
                for clause in handler.clauses
            )
        for feature in sorted(required):
            if supported.supports(feature):
                continue
            diagnostics.append(
                QiecCapabilityDiagnostic(
                    target=target,
                    feature=feature,
                    computation=computation.name,
                    origin=computation.origin,
                    detail="has no semantics-preserving lowering for that feature",
                )
            )
    return tuple(diagnostics)


def _handled_ids(node: IRQiecComputation) -> set[str]:
    out: set[str] = set()
    if isinstance(node, IRQiecHandle):
        out.add(node.handler.text)
        out.update(_handled_ids(node.computation))
    elif isinstance(node, IRQiecBind):
        out.update(_handled_ids(node.first))
        out.update(_handled_ids(node.then))
    elif isinstance(node, IRQiecCase):
        for branch in node.branches:
            out.update(_handled_ids(branch.body))
    return out


__all__ = [
    name for name in globals() if name.startswith("IRQiec") or name.startswith("Qiec")
]
__all__ += ["analyze_qiec_capabilities", "capabilities_for_target", "lower_qiec_ir"]
