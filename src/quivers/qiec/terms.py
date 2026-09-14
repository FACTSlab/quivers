"""Stable value and computation terms for QIEC.

Terms contain no Python callables.  Runtime values and handler bodies cross
the boundary only through stable attachment and handler identifiers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from quivers.qiec.effects import EffectRequest, EffectRow
from quivers.qiec.evidence import EqualityEvidence
from quivers.qiec.identifiers import (
    AttachmentId,
    ComputationId,
    ConstructorId,
    DistributionId,
    EffectInstanceId,
    HandlerId,
    PrimitiveId,
    SourceOrigin,
    StaticScopeId,
)
from quivers.qiec.kinds import Telescope
from quivers.qiec.types import EffectRef, StaticArgument, TypeExpr


@dataclass(frozen=True, slots=True)
class Local:
    """A typed binder for a runtime value.

    Locals are compared by identity of name and type together, so two
    binders of the same name at different types are distinct.

    Parameters
    ----------
    name
        The binder's display name.
    type
        The type of every value bound to it.
    """

    name: str
    type: TypeExpr


@dataclass(frozen=True, slots=True)
class Var:
    """Reference to a bound local.

    Parameters
    ----------
    local
        The binder being referenced.
    tag
        The serialization discriminator; always ``"var"``.
    """

    local: Local
    tag: Literal["var"] = "var"


type LiteralData = None | bool | int | float | str | bytes | tuple[LiteralData, ...]


@dataclass(frozen=True, slots=True)
class LiteralValue:
    """A literal host value at a primitive or tuple type.

    Parameters
    ----------
    value
        The literal's data: ``None``, a bool, int, float, str, bytes, or a
        tuple of literal data.
    type
        The literal's type.
    tag
        The serialization discriminator; always ``"literal"``.
    """

    value: LiteralData
    type: TypeExpr
    tag: Literal["literal"] = "literal"


@dataclass(frozen=True, slots=True)
class ConstructorValue:
    """Application of a declared constructor to static and value arguments.

    Parameters
    ----------
    constructor
        The stable identity of the constructor applied.
    static_arguments
        The constructor's telescope instantiation, in binder order.
    fields
        The field values, in declaration order.
    result_type
        The indexed family type the application inhabits.
    tag
        The serialization discriminator; always ``"constructor"``.
    """

    constructor: ConstructorId
    static_arguments: tuple[StaticArgument, ...]
    fields: tuple[Value, ...]
    result_type: TypeExpr
    tag: Literal["constructor"] = "constructor"


@dataclass(frozen=True, slots=True)
class EvidenceValue:
    """Kernel-checked equality evidence used as a value.

    Parameters
    ----------
    evidence
        The evidence term.
    tag
        The serialization discriminator; always ``"evidence"``.
    """

    evidence: EqualityEvidence
    tag: Literal["evidence"] = "evidence"


@dataclass(frozen=True, slots=True)
class AttachmentRef:
    """A stable reference to a runtime-owned host value.

    Parameters
    ----------
    attachment
        The identity under which a runtime provider binds the host value.
    type
        The type the bound value must inhabit.
    tag
        The serialization discriminator; always ``"attachment"``.
    """

    attachment: AttachmentId
    type: TypeExpr
    tag: Literal["attachment"] = "attachment"


@dataclass(frozen=True, slots=True)
class TransportValue:
    """Transport a value along kernel-checked equality evidence.

    Parameters
    ----------
    evidence
        The equality the value is carried across.
    value
        The value at the equality's left type.
    target_type
        The type the transported value inhabits, which the equality's right
        side determines.
    tag
        The serialization discriminator; always ``"transport"``.
    """

    evidence: EqualityEvidence
    value: Value
    target_type: TypeExpr
    tag: Literal["transport"] = "transport"


@dataclass(frozen=True, slots=True)
class PrimitiveApplication:
    """Application of a pure primitive from the closed registry.

    The primitive is named nominally and its signature is fixed, so the
    checker validates the arguments against the registry and a backend
    renders the name it knows or refuses the capability; nothing is
    resolved by inspecting a runtime value.

    Parameters
    ----------
    primitive
        The stable identity of the primitive applied.
    name
        The primitive's nominal name, as the registry spells it.
    arguments
        The value arguments, one per parameter.
    result_type
        The primitive's result type.
    origin
        The application's source location.
    tag
        The serialization discriminator; always ``"primitive"``.
    """

    primitive: PrimitiveId
    name: str
    arguments: tuple[Value, ...]
    result_type: TypeExpr
    origin: SourceOrigin
    tag: Literal["primitive"] = "primitive"


@dataclass(frozen=True, slots=True)
class TupleValue:
    """Construction of a finite product from its components.

    Parameters
    ----------
    items
        The component values, in order.
    result_type
        The product type of the components.
    tag
        The serialization discriminator; always ``"tuple"``.
    """

    items: tuple[Value, ...]
    result_type: TypeExpr
    tag: Literal["tuple"] = "tuple"


@dataclass(frozen=True, slots=True)
class TensorValue:
    """Construction of a tensor from its entries along the outermost axis.

    A rank-one tensor lists its elements; a tensor of higher rank lists
    the tensors one rank lower that are its slices, all of one shape, so
    a matrix literal is a tensor of row tensors.

    Parameters
    ----------
    items
        The entries along the outermost axis, in order.
    result_type
        The ``Tensor`` type constructed, whose leading dimension is the
        number of entries.
    tag
        The serialization discriminator; always ``"tensor"``.
    """

    items: tuple[Value, ...]
    result_type: TypeExpr
    tag: Literal["tensor"] = "tensor"


@dataclass(frozen=True, slots=True)
class Projection:
    """Selection of one component of a finite product.

    Parameters
    ----------
    value
        The product value projected from.
    position
        The zero-based component selected.
    result_type
        The selected component's type.
    tag
        The serialization discriminator; always ``"projection"``.
    """

    value: Value
    position: int
    result_type: TypeExpr
    tag: Literal["projection"] = "projection"


@dataclass(frozen=True, slots=True)
class DistributionValue:
    """Construction of a distribution from a family of the semantic registry.

    The family is named nominally and its parameters are supplied by name,
    so the term records which parameterization the source used and a
    backend renders the spelling it knows without inspecting values.

    Parameters
    ----------
    family
        The stable identity of the family applied.
    name
        The family's source name, as the registry spells it.
    arguments
        The parameters supplied, each named, in source order.
    result_type
        The ``Sampleable`` type of the constructed distribution.
    origin
        The construction's source location.
    tag
        The serialization discriminator; always ``"distribution"``.
    """

    family: DistributionId
    name: str
    arguments: tuple[tuple[str, Value], ...]
    result_type: TypeExpr
    origin: SourceOrigin
    tag: Literal["distribution"] = "distribution"


@dataclass(frozen=True, slots=True)
class SiteValue:
    """A named sample site.

    A site is the address a probabilistic step is observed and traced
    under. Its label is the source name of the step, which is stable data;
    the dynamic address a run adds comes from the request that carries it.

    Parameters
    ----------
    label
        The site's source name.
    result_type
        The ``Site`` type of the value the site produces.
    tag
        The serialization discriminator; always ``"site"``.
    """

    label: str
    result_type: TypeExpr
    tag: Literal["site"] = "site"


@dataclass(frozen=True, slots=True)
class LogDensity:
    """Evaluation of a distribution's log density at a value.

    Parameters
    ----------
    sampleable
        The distribution, of type ``Sampleable[A]``.
    value
        The point evaluated, of type ``A``.
    origin
        The evaluation's source location.
    tag
        The serialization discriminator; always ``"log_density"``.
    """

    sampleable: Value
    value: Value
    origin: SourceOrigin
    tag: Literal["log_density"] = "log_density"


type Value = (
    Var
    | LiteralValue
    | ConstructorValue
    | EvidenceValue
    | AttachmentRef
    | TransportValue
    | PrimitiveApplication
    | TupleValue
    | TensorValue
    | Projection
    | DistributionValue
    | LogDensity
    | SiteValue
)


@dataclass(frozen=True, slots=True)
class Return:
    """Return a value with no effects.

    Parameters
    ----------
    value
        The value produced.
    tag
        The serialization discriminator; always ``"return"``.
    """

    value: Value
    tag: Literal["return"] = "return"


@dataclass(frozen=True, slots=True)
class Bind:
    """Sequence two computations, binding the first's result in the second.

    Parameters
    ----------
    binder
        The local the first computation's result is bound to.
    first
        The computation run first.
    then
        The computation run second, with ``binder`` in scope.
    tag
        The serialization discriminator; always ``"bind"``.
    """

    binder: Local
    first: Computation
    then: Computation
    tag: Literal["bind"] = "bind"


@dataclass(frozen=True, slots=True)
class Perform:
    """Perform one effect operation.

    Parameters
    ----------
    request
        The operation, its instance, static arguments, value arguments, and
        result type.
    tag
        The serialization discriminator; always ``"perform"``.
    """

    request: EffectRequest
    tag: Literal["perform"] = "perform"


@dataclass(frozen=True, slots=True)
class Handle:
    """Run a computation under a handler installed for one effect instance.

    Parameters
    ----------
    instance
        The effect instance whose operations the handler intercepts.
    handler
        The stable identity of the handler installed.
    computation
        The handled computation.
    static_arguments
        The handler's telescope instantiation, in binder order.
    tag
        The serialization discriminator; always ``"handle"``.
    """

    instance: EffectInstanceId
    handler: HandlerId
    computation: Computation
    static_arguments: tuple[StaticArgument, ...] = ()
    tag: Literal["handle"] = "handle"


@dataclass(frozen=True, slots=True)
class CaseMotive:
    """The result family of an indexed case expression.

    Parameters
    ----------
    indices
        The binders abstracting the scrutinee's indices in ``result_type``.
    result_type
        The type every branch must produce, over ``indices``.
    """

    indices: Telescope
    result_type: TypeExpr


@dataclass(frozen=True, slots=True)
class CaseBranch:
    """One stable GADT branch with no host-language closure.

    Parameters
    ----------
    constructor
        The constructor this branch matches.
    static_arguments
        The constructor's static arguments as bound by the pattern, in
        binder order.
    fields
        The locals binding the constructor's fields, in declaration order.
    body
        The computation run when the branch is selected.
    scope
        The static scope identity under which the branch's skolems and
        equality evidence are minted.
    """

    constructor: ConstructorId
    static_arguments: tuple[StaticArgument, ...]
    fields: tuple[Local, ...]
    body: Computation
    scope: StaticScopeId


@dataclass(frozen=True, slots=True)
class Case:
    """Case analysis over an indexed family value.

    Parameters
    ----------
    scrutinee
        The value analyzed.
    motive
        The result family, over the scrutinee's indices.
    branches
        One branch per constructor considered; coverage is checked against
        the exact applied family.
    tag
        The serialization discriminator; always ``"case"``.
    """

    scrutinee: Value
    motive: CaseMotive
    branches: tuple[CaseBranch, ...]
    tag: Literal["case"] = "case"


@dataclass(frozen=True, slots=True)
class Call:
    """Application of a named computation.

    The callee is a stable identifier rather than an inlined body, which
    is what lets a recursive or mutually recursive call graph serialize
    as a finite tree.

    Parameters
    ----------
    callee
        The stable identity of the computation called.
    name
        The callee's display name, for diagnostics and traces.
    static_arguments
        The callee's telescope instantiation, in binder order.
    arguments
        The value arguments, one per callee parameter.
    result_type
        The callee's result type under the instantiation.
    effects
        The callee's effect row under the instantiation.
    origin
        The call site's source location.
    tag
        The serialization discriminator; always ``"call"``.
    """

    callee: ComputationId
    name: str
    static_arguments: tuple[StaticArgument, ...]
    arguments: tuple[Value, ...]
    result_type: TypeExpr
    effects: EffectRow
    origin: SourceOrigin
    tag: Literal["call"] = "call"


@dataclass(frozen=True, slots=True)
class Resume:
    """Invocation of the enclosing handler clause's continuation.

    The resumption is not a value and cannot be stored, so it is a term
    of its own rather than a call to a bound name. That is what makes a
    clause's grade checkable by counting invocations along the paths
    through its body.

    Parameters
    ----------
    value
        The value returned to the suspended computation as the operation's
        result.
    origin
        The resumption's source location.
    tag
        The serialization discriminator; always ``"resume"``.
    """

    value: Value
    origin: SourceOrigin
    tag: Literal["resume"] = "resume"


@dataclass(frozen=True, slots=True)
class NewInstance:
    """Lexically scoped allocation of an effect instance.

    The identity is derived from the module, the enclosing computation,
    the lexical path, and the applied interface, so the same allocation
    site yields the same instance on every run while two sites of the
    same interface stay distinct.

    Parameters
    ----------
    instance
        The scope-derived stable identity of the allocated instance.
    effect
        The applied interface the instance provides.
    body
        The computation within which the instance is in scope.
    origin
        The allocation's source location.
    tag
        The serialization discriminator; always ``"new_instance"``.
    """

    instance: EffectInstanceId
    effect: EffectRef
    body: Computation
    origin: SourceOrigin
    tag: Literal["new_instance"] = "new_instance"


@dataclass(frozen=True, slots=True)
class If:
    """Branch on a Boolean value.

    Both branches must produce the same type; the row of the whole is
    the union of the branches' rows. This is what lets a recursive
    computation stop, since a value-level select would evaluate both
    branches and never terminate.

    Parameters
    ----------
    condition
        The Boolean value branched on.
    then
        The computation run when the condition holds.
    otherwise
        The computation run when it does not.
    tag
        The serialization discriminator; always ``"if"``.
    """

    condition: Value
    then: Computation
    otherwise: Computation
    tag: Literal["if"] = "if"


type Computation = (
    Return | Bind | Perform | Handle | Case | If | Call | Resume | NewInstance
)


__all__ = [
    "AttachmentRef",
    "Bind",
    "Case",
    "Resume",
    "NewInstance",
    "Call",
    "CaseBranch",
    "CaseMotive",
    "Computation",
    "ConstructorValue",
    "DistributionValue",
    "EvidenceValue",
    "Handle",
    "If",
    "LiteralData",
    "LiteralValue",
    "Local",
    "LogDensity",
    "Perform",
    "PrimitiveApplication",
    "Projection",
    "Return",
    "SiteValue",
    "TransportValue",
    "TupleValue",
    "TensorValue",
    "Value",
    "Var",
]
