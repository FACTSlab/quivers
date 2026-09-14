"""Typed source AST for the Quivers Indexed Effect Core surface.

These nodes describe authored QVR syntax, not stable QIEC identities.  Stable
nominal identifiers, lexical instance identifiers, static scopes, and source
provenance are derived by the QVR-to-QIEC lowerer from names and the source
locations carried by every node here.

The QIEC surface uses distinct nodes alongside the categorical-object and
structural-signature ASTs.  In particular, :class:`QiecEffectRef` is not
an ``ObjectEffectApply`` and :class:`QiecFamilyConstructor` is not the
structural-signature ``ConstructorDecl``.
"""

from __future__ import annotations

from typing import Literal

import didactic.api as dx

from quivers.dsl.ast_nodes.declarations import Statement


# ---------------------------------------------------------------------------
# Kinds, index terms, and heterogeneous telescopes
# ---------------------------------------------------------------------------


class QiecKindExpr(dx.TaggedUnion, discriminator="kind"):
    """A QIEC kind appearing in source."""


class QiecTypeKind(QiecKindExpr):
    line: int = 0
    col: int = 0
    kind: Literal["qiec_type_kind"] = "qiec_type_kind"


class QiecEffectKind(QiecKindExpr):
    line: int = 0
    col: int = 0
    kind: Literal["qiec_effect_kind"] = "qiec_effect_kind"


class QiecIndexExpr(dx.TaggedUnion, discriminator="kind"):
    """A source-level term in a closed index sort."""


class QiecIndexName(QiecIndexExpr):
    name: str
    line: int = 0
    col: int = 0
    kind: Literal["qiec_index_name"] = "qiec_index_name"


class QiecIndexLiteral(QiecIndexExpr):
    value: int
    line: int = 0
    col: int = 0
    kind: Literal["qiec_index_literal"] = "qiec_index_literal"


class QiecIndexApplication(QiecIndexExpr):
    constructor: str
    arguments: tuple[QiecIndexExpr, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["qiec_index_application"] = "qiec_index_application"


class QiecShapeIndex(QiecIndexExpr):
    dimensions: tuple[QiecIndexExpr, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["qiec_shape_index"] = "qiec_shape_index"


class QiecIndexSort(dx.TaggedUnion, discriminator="kind"):
    """A closed index sort annotation."""


class QiecNatSort(QiecIndexSort):
    line: int = 0
    col: int = 0
    kind: Literal["qiec_nat_sort"] = "qiec_nat_sort"


class QiecShapeSort(QiecIndexSort):
    rank: int | None = None
    line: int = 0
    col: int = 0
    kind: Literal["qiec_shape_sort"] = "qiec_shape_sort"


class QiecContextSort(QiecIndexSort):
    signature: str
    line: int = 0
    col: int = 0
    kind: Literal["qiec_context_sort"] = "qiec_context_sort"


class QiecUserIndexSort(QiecIndexSort):
    name: str
    line: int = 0
    col: int = 0
    kind: Literal["qiec_user_index_sort"] = "qiec_user_index_sort"


class QiecBinder(dx.TaggedUnion, discriminator="kind"):
    """One entry in a heterogeneous static telescope."""


class QiecTypeBinder(QiecBinder):
    name: str
    binder_kind: QiecKindExpr
    line: int = 0
    col: int = 0
    kind: Literal["qiec_type_binder"] = "qiec_type_binder"


class QiecIndexBinder(QiecBinder):
    name: str
    sort: QiecIndexSort
    line: int = 0
    col: int = 0
    kind: Literal["qiec_index_binder"] = "qiec_index_binder"


class QiecEffectBinder(QiecBinder):
    name: str
    line: int = 0
    col: int = 0
    kind: Literal["qiec_effect_binder"] = "qiec_effect_binder"


class QiecIndexConstructor(dx.Model):
    """One constructor of a closed user-defined index sort."""

    name: str
    arguments: tuple[QiecIndexSort, ...] = ()
    line: int = 0
    col: int = 0


# ---------------------------------------------------------------------------
# Static type and effect terms
# ---------------------------------------------------------------------------


class QiecTypeExpr(dx.TaggedUnion, discriminator="kind"):
    """A value type in the QIEC source stratum."""


class QiecTypeName(QiecTypeExpr):
    name: str
    line: int = 0
    col: int = 0
    kind: Literal["qiec_type_name"] = "qiec_type_name"


class QiecTypeApplication(QiecTypeExpr):
    constructor: str
    static_arguments: tuple[QiecTypeExpr, ...] = ()
    indices: tuple[QiecIndexExpr, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["qiec_type_application"] = "qiec_type_application"


class QiecProductType(QiecTypeExpr):
    components: tuple[QiecTypeExpr, ...]
    line: int = 0
    col: int = 0
    kind: Literal["qiec_product_type"] = "qiec_product_type"


class QiecFunctionType(QiecTypeExpr):
    parameter: QiecTypeExpr
    result: QiecTypeExpr
    line: int = 0
    col: int = 0
    kind: Literal["qiec_function_type"] = "qiec_function_type"


class QiecEffectRef(dx.Model):
    """An applied effect interface such as ``State[Int]``."""

    name: str
    arguments: tuple[QiecTypeExpr, ...] = ()
    line: int = 0
    col: int = 0


type QiecStaticArgument = QiecTypeExpr


# ---------------------------------------------------------------------------
# Effect rows and signature members
# ---------------------------------------------------------------------------


class QiecRowEntry(dx.Model):
    """A named lexical effect instance in an effect row."""

    instance: str
    line: int = 0
    col: int = 0


class QiecEffectRow(dx.Model):
    """Explicit row entries plus an optional constrained open tail."""

    entries: tuple[QiecRowEntry, ...] = ()
    tail: str | None = None
    lacks: tuple[str, ...] = ()
    line: int = 0
    col: int = 0


class QiecValueParameter(dx.Model):
    name: str
    type_expr: QiecTypeExpr
    line: int = 0
    col: int = 0


class QiecFamilyConstructor(dx.Model):
    """A GADT constructor with a local telescope and refined result."""

    name: str
    binders: tuple[QiecBinder, ...] = ()
    arguments: tuple[QiecTypeExpr, ...] = ()
    result: QiecTypeExpr
    line: int = 0
    col: int = 0


class QiecOperationDecl(dx.Model):
    name: str
    binders: tuple[QiecBinder, ...] = ()
    arguments: tuple[QiecTypeExpr, ...] = ()
    result: QiecTypeExpr
    line: int = 0
    col: int = 0


type QiecResumptionGrade = Literal["0", "aff", "1", "omega"]


class QiecHandlerClause(dx.TaggedUnion, discriminator="kind"):
    """One clause of a handler declaration.

    A handler answers a returning computation and each operation it
    covers, and those are different shapes: the return clause binds a
    value, an operation clause binds the operation's arguments and may
    resume.
    """


# ---------------------------------------------------------------------------
# Values and computations
# ---------------------------------------------------------------------------


class QiecValue(dx.TaggedUnion, discriminator="kind"):
    """A stable, serializable value expression accepted by QIEC lowering."""


class QiecVariableValue(QiecValue):
    name: str
    line: int = 0
    col: int = 0
    kind: Literal["qiec_variable_value"] = "qiec_variable_value"


class QiecLiteralValue(QiecValue):
    value: None | bool | int | float | str
    line: int = 0
    col: int = 0
    kind: Literal["qiec_literal_value"] = "qiec_literal_value"


class QiecConstructorValue(QiecValue):
    constructor: str
    static_arguments: tuple[QiecTypeExpr, ...] = ()
    fields: tuple[QiecValue, ...] = ()
    result_type: QiecTypeExpr
    line: int = 0
    col: int = 0
    kind: Literal["qiec_constructor_value"] = "qiec_constructor_value"


class QiecLocalBinding(dx.Model):
    name: str
    type_expr: QiecTypeExpr | None = None
    line: int = 0
    col: int = 0


class QiecEffectRequest(dx.Model):
    instance: str
    operation: str
    static_arguments: tuple[QiecTypeExpr, ...] = ()
    arguments: tuple[QiecValue, ...] = ()
    line: int = 0
    col: int = 0


class QiecHandlerApplication(dx.Model):
    name: str
    static_arguments: tuple[QiecTypeExpr, ...] = ()
    line: int = 0
    col: int = 0


class QiecComputation(dx.TaggedUnion, discriminator="kind"):
    """A computation in the QIEC fine-grain call-by-value surface."""


class QiecCaseMotive(dx.Model):
    indices: tuple[QiecIndexBinder, ...]
    result_type: QiecTypeExpr
    line: int = 0
    col: int = 0


class QiecCaseBranch(dx.Model):
    constructor: str
    static_arguments: tuple[QiecTypeExpr, ...] = ()
    fields: tuple[QiecLocalBinding, ...] = ()
    body: QiecComputation
    line: int = 0
    col: int = 0


class QiecReturnComputation(QiecComputation):
    value: QiecValue
    line: int = 0
    col: int = 0
    kind: Literal["qiec_return_computation"] = "qiec_return_computation"


class QiecBindComputation(QiecComputation):
    binder: QiecLocalBinding
    first: QiecComputation
    then: QiecComputation
    line: int = 0
    col: int = 0
    kind: Literal["qiec_bind_computation"] = "qiec_bind_computation"


class QiecSequenceComputation(QiecComputation):
    first: QiecComputation
    then: QiecComputation
    line: int = 0
    col: int = 0
    kind: Literal["qiec_sequence_computation"] = "qiec_sequence_computation"


class QiecPerformComputation(QiecComputation):
    request: QiecEffectRequest
    line: int = 0
    col: int = 0
    kind: Literal["qiec_perform_computation"] = "qiec_perform_computation"


class QiecHandleComputation(QiecComputation):
    instance: str
    handler: QiecHandlerApplication
    body: QiecComputation
    line: int = 0
    col: int = 0
    kind: Literal["qiec_handle_computation"] = "qiec_handle_computation"


class QiecCaseComputation(QiecComputation):
    scrutinee: QiecValue
    motive: QiecCaseMotive
    branches: tuple[QiecCaseBranch, ...]
    line: int = 0
    col: int = 0
    kind: Literal["qiec_case_computation"] = "qiec_case_computation"


class QiecPureBinding(QiecComputation):
    """`let x = VALUE`, binding a pure expression.

    Separate from [`QiecBindComputation`][.], which binds a computation's
    result. The two have different typing rules and neither is sugar for
    the other.
    """

    binder: QiecLocalBinding
    value: QiecValue
    then: QiecComputation
    line: int = 0
    col: int = 0
    kind: Literal["qiec_pure_binding"] = "qiec_pure_binding"


class QiecCallComputation(QiecComputation):
    """Application of a named computation.

    The callee is resolved against the module's signature table rather
    than against a local binding, so a call may precede the declaration
    it names and mutual recursion needs no forward declaration.
    """

    callee: str
    static_arguments: tuple[QiecTypeExpr, ...] = ()
    arguments: tuple[QiecValue, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["qiec_call_computation"] = "qiec_call_computation"


class QiecResumeComputation(QiecComputation):
    """Invocation of the enclosing handler clause's continuation.

    The resumption is not a named local, so there is exactly one way to
    invoke it. A `value` of None resumes with unit.
    """

    value: QiecValue | None = None
    line: int = 0
    col: int = 0
    kind: Literal["qiec_resume_computation"] = "qiec_resume_computation"


class QiecInstanceComputation(QiecComputation):
    """`with instance x : E in`, allocating a lexically scoped instance.

    The binder is live only in `body`. Its identity is derived rather
    than generated, so the same allocation site yields the same instance
    on every run.
    """

    name: str
    effect: QiecEffectRef
    body: QiecComputation
    line: int = 0
    col: int = 0
    kind: Literal["qiec_instance_computation"] = "qiec_instance_computation"


class QiecHandlerReturnClause(QiecHandlerClause):
    """What a handler does with the value its computation returns."""

    binder: QiecLocalBinding
    body: QiecComputation
    line: int = 0
    col: int = 0
    kind: Literal["qiec_handler_return_clause"] = "qiec_handler_return_clause"


class QiecHandlerOperationClause(QiecHandlerClause):
    """One operation a handler covers.

    A `body` of None is a signature rather than an implementation, and is
    legal only where the declaration says `implementation=foreign`.
    """

    operation: str
    grade: QiecResumptionGrade
    binders: tuple[QiecBinder, ...] = ()
    parameters: tuple[QiecLocalBinding, ...] = ()
    body: QiecComputation | None = None
    line: int = 0
    col: int = 0
    kind: Literal["qiec_handler_operation_clause"] = "qiec_handler_operation_clause"


# ---------------------------------------------------------------------------
# Top-level declarations
# ---------------------------------------------------------------------------


class QiecIndexDecl(Statement):
    name: str
    constructors: tuple[QiecIndexConstructor, ...]
    docs: tuple[str, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["index_decl"] = "index_decl"


class QiecFamilyDecl(Statement):
    name: str
    parameters: tuple[QiecBinder, ...] = ()
    indices: tuple[QiecIndexBinder, ...] = ()
    result_kind: QiecKindExpr
    constructors: tuple[QiecFamilyConstructor, ...] = ()
    docs: tuple[str, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["indexed_family_decl"] = "indexed_family_decl"


class QiecEffectDecl(Statement):
    name: str
    binders: tuple[QiecBinder, ...] = ()
    operations: tuple[QiecOperationDecl, ...] = ()
    docs: tuple[str, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["effect_decl"] = "effect_decl"


class QiecEffectInstanceDecl(Statement):
    name: str
    effect: QiecEffectRef
    docs: tuple[str, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["effect_instance_decl"] = "effect_instance_decl"


type QiecHandlerImplementation = Literal["authored", "foreign"]


class QiecHandlerDecl(Statement):
    name: str
    binders: tuple[QiecBinder, ...] = ()
    effect: QiecEffectRef
    input_type: QiecTypeExpr
    output_type: QiecTypeExpr
    introduced: QiecEffectRow = QiecEffectRow()
    coverage: Literal["total", "partial"] = "total"
    forwards_unknown: bool = False
    implementation: QiecHandlerImplementation = "authored"
    clauses: tuple[QiecHandlerClause, ...] = ()
    duplicate_options: tuple[str, ...] = ()
    docs: tuple[str, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["handler_decl"] = "handler_decl"


class QiecComputationDecl(Statement):
    name: str
    binders: tuple[QiecBinder, ...] = ()
    parameters: tuple[QiecValueParameter, ...] = ()
    result_type: QiecTypeExpr
    effects: QiecEffectRow
    body: QiecComputation
    docs: tuple[str, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["computation_decl"] = "computation_decl"


__all__ = [name for name in globals() if name.startswith("Qiec")]
