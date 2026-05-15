"""AST node definitions for the quivers DSL.

The AST is a direct representation of the parsed `.qvr` source. Each
node carries source-location info for error reporting. Recursive sums
use ``dx.TaggedUnion`` with a ``kind`` discriminator: variants subclass
their sum's root and pin ``kind`` to a ``Literal[...]`` value.
"""

from typing import Literal

import didactic.api as dx


# ---------------------------------------------------------------------------
# axis-role surface: per-distribution event / batch axis specification
# ---------------------------------------------------------------------------


class AxisSpec(dx.Model):
    """Axis-role specification on a distribution clause.

    Surface form: ``over <axes> [iid over <axes>]``.

    ``over`` names the event axes — the axes on which the family's
    joint structure (an MVN covariance, a MatrixNormal Kronecker pair,
    a GP kernel) lives.  The axis count must match the family's
    declared event rank; the positional ordering corresponds
    positionally to the family's event-axis ordering.

    ``iid_over`` is an optional readability assertion naming the batch
    axes (the complement of ``over`` in the surrounding morphism's
    type signature).  Inconsistency with the type signature or with
    ``over`` is a compile-time error.

    Axis names resolve against the named factors of the surrounding
    morphism's dom/cod.  The reserved tokens ``dom`` and ``cod`` are
    legal shortcuts only when the corresponding side is a single
    unfactored object.
    """

    over: tuple[str, ...]
    iid_over: tuple[str, ...] = ()
    line: int = 0
    col: int = 0


class MorphismPrior(dx.Model):
    """Parameter prior on a ``latent`` morphism's representing tensor.

    Surface form: ``~ Family(args) [options] [axis_role_clause]``.

    Promotes the declared morphism from a free-parameter point
    estimate to a random variable whose representing tensor is drawn
    from the named family at the requested axis-role configuration.
    Categorically: the morphism becomes the deterministic wrap of a
    sample from ``family(args)``, with the family's event/batch
    structure controlled by ``axes``.
    """

    family: str
    args: tuple[str | float, ...] = ()
    options: dict[str, str] = dx.field(default_factory=dict)
    axes: AxisSpec | None = None
    line: int = 0
    col: int = 0


# ---------------------------------------------------------------------------
# type expressions (categorical objects: products and coproducts of finsets)
# ---------------------------------------------------------------------------


class TypeExpr(dx.TaggedUnion, discriminator="kind"):
    """Sum of type-expression node kinds."""


class TypeName(TypeExpr):
    """A named type reference (identifier or integer literal)."""

    name: str
    line: int = 0
    col: int = 0
    kind: Literal["type_name"] = "type_name"


class TypeProduct(TypeExpr):
    """Product type: ``A * B``."""

    components: tuple[TypeExpr, ...]
    line: int = 0
    col: int = 0
    kind: Literal["type_product"] = "type_product"


class TypeCoproduct(TypeExpr):
    """Coproduct type: ``A + B``."""

    components: tuple[TypeExpr, ...]
    line: int = 0
    col: int = 0
    kind: Literal["type_coproduct"] = "type_coproduct"


class TypeSlash(TypeExpr):
    """Residuated slash type: ``result / argument`` or ``result \\ argument``.

    Legal only when both operands inhabit a residuated universe (typically
    a ``FreeResiduated`` object). The compiler enforces this at use-site.
    """

    result: TypeExpr
    argument: TypeExpr
    direction: Literal["/", "\\"]
    line: int = 0
    col: int = 0
    kind: Literal["type_slash"] = "type_slash"


class TypeEffectApply(TypeExpr):
    """Effect-typed type-application: ``T(X)``, ``Continuation[ρ](NP)``.

    The ``effect`` field names the effect (a previously-declared
    ``EffectDecl`` or stdlib effect); ``args`` are its applied arguments.
    Legal only inside a ``FreeResiduated`` whose ``effects`` list mentions
    the named effect.
    """

    effect: str
    args: tuple[TypeExpr, ...]
    line: int = 0
    col: int = 0
    kind: Literal["type_effect_apply"] = "type_effect_apply"


# ---------------------------------------------------------------------------
# space expressions (continuous spaces)
# ---------------------------------------------------------------------------


class SpaceExpr(dx.TaggedUnion, discriminator="kind"):
    """Sum of space-expression node kinds."""


class SpaceName(SpaceExpr):
    """A bare identifier referencing a previously declared space."""

    name: str
    line: int = 0
    col: int = 0
    kind: Literal["space_name"] = "space_name"


class SpaceConstructor(SpaceExpr):
    """Space constructor call: ``Euclidean(3)`` or ``Euclidean(2, low=0.0, high=1.0)``."""

    constructor: str
    args: tuple[str, ...] = ()
    kwargs: dict[str, str] = dx.field(default_factory=dict)
    line: int = 0
    col: int = 0
    kind: Literal["space_constructor"] = "space_constructor"


class SpaceProduct(SpaceExpr):
    """Product space: ``A * B``."""

    components: tuple[SpaceExpr, ...]
    line: int = 0
    col: int = 0
    kind: Literal["space_product"] = "space_product"


# ---------------------------------------------------------------------------
# value (morphism) expressions
# ---------------------------------------------------------------------------


class Expr(dx.TaggedUnion, discriminator="kind"):
    """Sum of value-expression node kinds (morphism computations)."""


class ExprIdent(Expr):
    """Reference to a named morphism or binding."""

    name: str
    line: int = 0
    col: int = 0
    kind: Literal["expr_ident"] = "expr_ident"


class ExprIdentity(Expr):
    """Identity morphism: ``identity(A)``."""

    object_name: str
    line: int = 0
    col: int = 0
    kind: Literal["expr_identity"] = "expr_identity"


class ExprFromData(Expr):
    """Data-derived initializer ``from_data("KEY")``.

    The string key is resolved against the runtime data dictionary
    at fit time; the morphism's tensor is the looked-up value.
    The resulting morphism is :class:`ObservedMorphism` — its
    entries are frozen / structural inputs, not learnable
    parameters.
    """

    key: str
    line: int = 0
    col: int = 0
    kind: Literal["expr_from_data"] = "expr_from_data"


class ExprFreeze(Expr):
    """Detach gradients: ``inner.freeze`` materialises ``inner``'s
    tensor with ``detach()`` and wraps the result as a frozen
    :class:`ObservedMorphism`. Used to pin a learned composition
    as a structural input that the downstream model treats as
    constant."""

    inner: Expr
    line: int = 0
    col: int = 0
    kind: Literal["expr_freeze"] = "expr_freeze"


class ExprDagger(Expr):
    """Compact-closed dagger / transpose of an expression."""

    inner: Expr
    line: int = 0
    col: int = 0
    kind: Literal["expr_dagger"] = "expr_dagger"


class ExprTrace(Expr):
    """Compact-closed trace of an expression along a named object."""

    inner: Expr
    object_name: str
    line: int = 0
    col: int = 0
    kind: Literal["expr_trace"] = "expr_trace"


class ExprChangeBase(Expr):
    """Change-of-base: apply a transformation (an algebra
    homomorphism or :class:`MorphismTransformation`) to a
    morphism.

    The transformation is a first-class value: ``phi`` is any
    expression whose compile-time value is a
    :class:`MorphismTransformation` or
    :class:`AlgebraHomomorphism`.  Concretely:

    * A bare identifier resolving a registered singleton
      (``f.change_base(expectation)``) or a let-bound trans value
      (``f.change_base(t)``).
    * A constructor call (``f.change_base(softmax(B))``).
    * A composition (``f.change_base(t1 >>> t2)``).
    """

    inner: Expr
    phi: Expr
    line: int = 0
    col: int = 0
    kind: Literal["expr_change_base"] = "expr_change_base"


class ExprTransCompose(Expr):
    """Composition of two transformations: ``t1 >>> t2`` denotes
    sequential application — first apply ``t1``, then ``t2``.

    Required: ``t1.target == t2.source`` (typed at compile time;
    a mismatch raises :class:`CompileError`).  The result behaves
    as a transformation with ``source = t1.source`` and
    ``target = t2.target``.
    """

    left: Expr
    right: Expr
    line: int = 0
    col: int = 0
    kind: Literal["expr_trans_compose"] = "expr_trans_compose"


class ExprCup(Expr):
    """Compact-closed unit ``η_A : I → A ⊗ A`` for a named object."""

    object_name: str
    line: int = 0
    col: int = 0
    kind: Literal["expr_cup"] = "expr_cup"


class ExprCap(Expr):
    """Compact-closed counit ``ε_A : A ⊗ A → I`` for a named object."""

    object_name: str
    line: int = 0
    col: int = 0
    kind: Literal["expr_cap"] = "expr_cap"


class ExprCompose(Expr):
    """Algebra-typed sequential composition.

    The ``op`` field selects which enrichment algebra's monoidal
    structure to use for the V-Cat composition:

    * ``">>"`` — ProductFuzzyAlgebra noisy-OR (the default).
    * ``"<<"`` — reverse ProductFuzzyAlgebra.
    * ``">=>"`` — Kleisli composition (operands' shared algebra).
    * ``"*>"`` — Markov sum-product.
    * ``"~>"`` — LogProb (log-space sum-product).
    * ``"||>"`` — Gödel (lattice min/max with Heyting implication).
    * ``"?>"`` — Viterbi (max-plus tropical, best path).
    * ``"&&>"`` — Boolean (∧/∨).
    * ``"+>"`` — Łukasiewicz (probabilistic sum bounded by 1).

    Each operator carries its own algebra; cross-operator
    composition in one chain requires explicit ``.change_base(φ)``
    between segments.
    """

    left: Expr
    right: Expr
    op: str = ">>"
    line: int = 0
    col: int = 0
    kind: Literal["expr_compose"] = "expr_compose"


class ExprTensorProduct(Expr):
    """Parallel product: ``left @ right``."""

    left: Expr
    right: Expr
    line: int = 0
    col: int = 0
    kind: Literal["expr_tensor_product"] = "expr_tensor_product"


class ExprMarginalize(Expr):
    """Marginalization: ``expr.marginalize(A, B, ...)``."""

    inner: Expr
    names: tuple[str, ...]
    line: int = 0
    col: int = 0
    kind: Literal["expr_marginalize"] = "expr_marginalize"


class ExprFan(Expr):
    """Fan-out (diagonal) composition: ``fan(f, g, ...)``."""

    exprs: tuple[Expr, ...]
    line: int = 0
    col: int = 0
    kind: Literal["expr_fan"] = "expr_fan"


class ExprRepeat(Expr):
    """Iterated composition: ``repeat(f, n)`` or ``repeat(f)``."""

    expr: Expr
    count: int | None
    line: int = 0
    col: int = 0
    kind: Literal["expr_repeat"] = "expr_repeat"


class ExprStack(Expr):
    """Independent multi-layer composition: ``stack(f, n)``."""

    expr: Expr
    count: int
    line: int = 0
    col: int = 0
    kind: Literal["expr_stack"] = "expr_stack"


class ExprScan(Expr):
    """Temporal scan: ``scan(cell)`` or ``scan(cell, init=learned)``."""

    expr: Expr
    init: str = "zeros"
    line: int = 0
    col: int = 0
    kind: Literal["expr_scan"] = "expr_scan"


class ExprParser(Expr):
    """Deductive parser assembled from rules."""

    rules: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    terminal: str | None = None
    start: str | int = "S"
    depth: int = 1
    constructors: tuple[str, ...] | None = None
    line: int = 0
    col: int = 0
    kind: Literal["expr_parser"] = "expr_parser"


class ExprChartFold(Expr):
    """Desugared parser-construction primitive.

    Surface form: ``chart_fold(lex=, binary=, unary=, start=, depth=,
    effect_depth=)``.

    Constructs a chart parser from morphism-valued arguments rather
    than from a list of named rule schemas. ``lex`` is a morphism
    Token -> Cat; ``binary`` is a Cat * Cat -> Cat morphism (the
    union of all binary rule schemas); ``unary`` is an optional
    Cat -> Cat morphism (the union of all unary rule schemas);
    ``start`` is the goal category name (or integer index);
    ``depth`` is the maximum category nesting depth; ``effect_depth``
    bounds effect-stack nesting (defaults to 0).
    """

    lex: Expr
    binary: Expr | None = None
    unary: Expr | None = None
    start: str | int = "S"
    depth: int = 1
    effect_depth: int = 0
    line: int = 0
    col: int = 0
    kind: Literal["expr_chart_fold"] = "expr_chart_fold"


class ExprCurry(Expr):
    """Residuation-witness curry combinator.

    For an inner morphism ``f : X * Y -> Z`` whose codomain ``Z``
    inhabits a residuated universe, ``f.curry_right`` denotes the
    morphism ``X -> Z/Y`` and ``f.curry_left`` denotes ``Y -> X\\Z``.

    The categorical interpretation is the right (resp. left) component
    of the residuation-adjunction unit/counit triangle. Validity of the
    construction is checked at compile time: domain must factor as a
    non-commutative product and codomain must inhabit a residuated
    universe (a :class:`FreeResiduated` object in scope).
    """

    inner: Expr
    direction: Literal["right", "left"]
    line: int = 0
    col: int = 0
    kind: Literal["expr_curry"] = "expr_curry"


# ---------------------------------------------------------------------------
# let-step arithmetic expressions
# ---------------------------------------------------------------------------


class LetExprNode(dx.TaggedUnion, discriminator="kind"):
    """Sum of let-step arithmetic expression nodes."""


class LetExprBinOp(LetExprNode):
    """Binary arithmetic operation in a let expression."""

    op: Literal["+", "-", "*", "/"]
    left: LetExprNode
    right: LetExprNode
    kind: Literal["let_expr_binop"] = "let_expr_binop"


class LetExprUnaryOp(LetExprNode):
    """Unary negation in a let expression."""

    operand: LetExprNode
    kind: Literal["let_expr_unary"] = "let_expr_unary"


class LetExprCall(LetExprNode):
    """Built-in function call in a let expression."""

    func: str
    args: tuple[LetExprNode, ...]
    kind: Literal["let_expr_call"] = "let_expr_call"


class LetExprLiteral(LetExprNode):
    """Numeric literal in a let expression."""

    value: float
    kind: Literal["let_expr_literal"] = "let_expr_literal"


class LetExprVar(LetExprNode):
    """Variable reference in a let expression."""

    name: str
    kind: Literal["let_expr_var"] = "let_expr_var"


class LetExprIndex(LetExprNode):
    """Indexed access into a finite-domain-indexed family ``a[i]``.

    Categorically the *pullback* morphism: given a finite-fibration
    ``index : N → A`` and a per-A morphism ``arr : A → B``, the
    indexed expression ``arr[index[n]]`` denotes
    ``arr ∘ index : N → B`` — the natural Kleisli pullback of
    ``arr`` along ``index``.

    Attributes
    ----------
    array : LetExprNode
        The indexed-family expression (typically a :class:`LetExprVar`
        naming a previously-drawn plate variable).
    indices : tuple of LetExprNode
        The index expressions; supports multi-dim indexing for
        nested plates (``coefs[subj[n], k]``).
    """

    array: LetExprNode
    indices: tuple[LetExprNode, ...]
    kind: Literal["let_expr_index"] = "let_expr_index"


class LetExprString(LetExprNode):
    """String literal in a let expression.

    Used for tokenisation, lexicon keys, and as ground-atom names
    in LF constructors like ``pred("dog")`` and
    ``forall("x", body)``. The runtime represents these as plain
    Python strings.
    """

    value: str
    kind: Literal["let_expr_string"] = "let_expr_string"


class LetExprList(LetExprNode):
    """List literal in a let expression — ``[a, b, c]``.

    Categorically a free-monoid element over the value sublanguage;
    the runtime represents it as a Python list (with autograd
    flowing through tensor-valued items).
    """

    items: tuple[LetExprNode, ...]
    kind: Literal["let_expr_list"] = "let_expr_list"


class LetExprLambda(LetExprNode):
    """Lambda expression ``param -> body`` in a let expression.

    Closes over the surrounding let-environment at instantiation
    time. Categorically a curried function in the Kleisli
    setting; used as the argument to fold / map / filter / reduce
    combinators.
    """

    param: str
    body: LetExprNode
    kind: Literal["let_expr_lambda"] = "let_expr_lambda"


class LetFactorBinder(dx.Model):
    """One ``<var> : <Index>`` binder in a multi-axis factor expression.

    The variable name binds to integer index values 0, 1, ...,
    |Index|-1 in the surrounding factor body.  The index type
    expression resolves to a finite-set object whose cardinality
    is the corresponding axis size of the constructed tensor.
    """

    var: str
    index: TypeExpr
    line: int = 0
    col: int = 0


class LetFactorCase(dx.Model):
    """One ``<integer> -> <body>`` case in a factor pattern-match.

    The label is the integer index this case populates; the body
    is the value at that index.  The compiler verifies that the
    union of labels across all cases covers ``{0, ..., |Index|-1}``
    exactly.
    """

    label: int
    value: LetExprNode
    line: int = 0
    col: int = 0


class LetExprFactor(LetExprNode):
    """Multi-axis factor expression: assemble an indexed tensor.

    Surface forms:

    ``factor v1 : I1, v2 : I2, ..., vn : In in <body>`` denotes the
    tensor of shape ``(|I1|, ..., |In|, *body_shape)`` whose value
    at position ``(i1, ..., in)`` is ``body[v_k := i_k]``.

    ``factor v : I in { 0 -> e0, 1 -> e1, ... }`` denotes the
    single-axis case-structured form: the body at index `k` is the
    expression labelled `k`, and the labels must cover
    ``{0, ..., |I|-1}`` exactly.  Multi-axis case form is not
    accepted; the uniform body form (which can itself contain
    conditionals on the binders) is the general construction.

    Categorically the left adjoint of indexing.  Single-axis is a
    section of the trivial bundle ``I -> body_type``; multi-axis is
    a section over the product ``I1 x ... x In``.  The dual
    operation is the index pullback ``arr[i1, ..., in]``
    (:class:`LetExprIndex`); together they realize the indexed-family
    colim / lim pair in the slice category over ``FinSet``.
    """

    binders: tuple[LetFactorBinder, ...]
    body: LetExprNode | None = None
    cases: tuple[LetFactorCase, ...] = ()
    kind: Literal["let_expr_factor"] = "let_expr_factor"


class LetExprMethodCall(LetExprNode):
    """Method call ``receiver.method(args)`` in a let expression.

    The receiver is itself a let-expression (typically a variable
    reference to a let-bound chart-valued, list-valued, or other
    object-valued value); the method is dispatched at runtime
    against the receiver's type. Used primarily for chart-view
    queries (``chart.weight(item)``, ``chart.enumerate(pattern)``,
    ``chart.goal_weight()``).
    """

    receiver: LetExprNode
    method: str
    args: tuple[LetExprNode, ...]
    kind: Literal["let_expr_method_call"] = "let_expr_method_call"


# ---------------------------------------------------------------------------
# program-block steps
# ---------------------------------------------------------------------------


class ProgramStep(dx.TaggedUnion, discriminator="kind"):
    """Sum of program-block step node kinds."""


class BindStep(ProgramStep):
    """A Kleisli bind inside a program block — the unified step shape.

    Surface forms:

    .. code-block:: qvr

        v        <- F(args)                              # mode=sample, scalar
        v : A    <- F(args)                              # mode=sample, A-indexed plate
        (a, b)   <- F(args)                              # destructuring tuple bind
        observe v        <- F(args)                      # mode=score, scalar
        observe r : N    <- F(theta[N])                  # mode=score, N-indexed
        marginalize c    <- F(args) in { steps }         # mode=marginal, scoped
        marginalize c : A <- F(args) in { steps }        # mode=marginal, A-indexed

    Categorical denotation:

    * ``mode="sample"`` extends the trace by a fresh Kleisli arrow
      :math:`\\Phi \\to \\mathcal{G}(\\Phi \\times K)`. When ``index``
      is non-``None`` the iso
      :math:`\\mathbf{Kern}(\\mathbf{1}, K^A) \\cong \\mathbf{Kern}(A, K)`
      lifts the per-fiber family to an indexed family.
    * ``mode="score"`` is a sub-probabilistic Kleisli arrow
      :math:`\\Phi \\to \\mathcal{G}_{\\le 1}(\\Phi)` clamping the
      bound coordinate to a runtime-supplied observation; the
      indexed form denotes the batched-likelihood kernel
      :math:`\\prod_{n} p_F(r_{\\mathrm{obs}}(n); \\theta(n, \\phi))`.
    * ``mode="marginal"`` introduces a coordinate, executes the
      scope's steps with that coordinate in trace context, and at
      the end of the scope pushes forward through the projection
      :math:`\\pi_{\\Phi} : \\Phi \\times C \\to \\Phi` (logsumexp for
      discrete, fibrewise integration for continuous). The
      coordinate is local to ``scope``.

    Attributes
    ----------
    vars : tuple[str, ...]
        Bound names. For sample mode, may be a tuple for
        destructuring; score and marginal modes always carry a
        single name.
    index : TypeExpr | None
        Optional index-set annotation; non-``None`` for plate /
        vectorized / indexed-marginalize forms.
    morphism : str
        Family / morphism name on the kernel-expression RHS.
    args : tuple
        Family arguments. Strings of the form ``"name[Index]"`` are
        bracket-indexed family sections — categorically sections of
        an ``Index``-indexed family.
    mode : Literal["sample", "score", "marginal"]
        Kleisli-bind mode.
    scope : tuple[ProgramStep, ...] | None
        Integration scope; non-``None`` iff ``mode == "marginal"``.
    """

    vars: tuple[str, ...]
    morphism: str
    args: tuple[str | float, ...] | None = None
    index: TypeExpr | None = None
    mode: Literal["sample", "score", "marginal"] = "sample"
    scope: tuple[ProgramStep, ...] | None = None
    # Axis-role clause: ``over <axes> [iid over <axes>]`` on the
    # family invocation.  Configures the event/batch decomposition
    # of the family over the named axes of the step's type
    # annotation (``: T``).  Required when ``family.event_rank > 0``
    # and ambiguous from the annotation alone; rejected at compile
    # time on mismatch with the family's event rank.
    axes: AxisSpec | None = None
    # ``over G`` on the marginalize-mode bind declares the grouping
    # plate.  ``over_obj`` is the single plate name; ``over_objs``
    # is the tuple of plate names when the user wrote a type
    # product (e.g. ``over G * H``).  Unused on score / sample
    # binds.
    over: str | None = None
    over_objs: tuple[str, ...] | None = None
    # ``via idx`` (single fibration) or ``via product(idx_a, idx_b)``
    # (product fibration) on a score-mode bind inside a grouped
    # marginalize body.  Every observe inside the body carries its
    # own ``via`` clause naming the fibration into the shared
    # grouping plate.  Unused on sample / marginal binds.
    via: str | None = None
    via_axes: tuple[str, ...] | None = None
    # `reduction = logsumexp | sum | mean`.
    reduction: str | None = None
    line: int = 0
    col: int = 0
    kind: Literal["bind_step"] = "bind_step"


class LetStep(ProgramStep):
    """A deterministic ``let`` binding inside a program block.

    The ``value`` field always holds a :class:`LetExprNode` — bare floats
    and bare identifier aliases are wrapped in :class:`LetExprLiteral` and
    :class:`LetExprVar` respectively at parse time.
    """

    name: str
    value: LetExprNode
    line: int = 0
    col: int = 0
    kind: Literal["let_step"] = "let_step"


# ---------------------------------------------------------------------------
# Internal compiler-only step shapes.
#
# The parser emits exclusively :class:`BindStep` and :class:`LetStep` for
# program bodies under the v0.5 unified surface. The compiler expands a
# BindStep into one of the four specialized forms below at the entry to
# `_compile_program`, based on the bind's `mode` and `index` fields:
#
#   - sample, no index  -> DrawStep
#   - sample, with idx  -> PlateDrawStep
#   - score, no index   -> DrawStep with is_observed=True
#   - score, with idx   -> VectorisedObserveStep
#   - marginal          -> MarginalizeStep (the scope steps are expanded
#                          inline; the variable is registered for that scope)
#
# These types are not part of the public surface — they are an internal
# IR consumed by the rest of the compiler / template-expansion / runtime
# step-builder machinery. Keeping them lets the compiler's existing deep
# code paths continue unchanged while the surface presents a single
# Kleisli-bind form.
# ---------------------------------------------------------------------------


class DrawStep(ProgramStep):
    """Internal compiler IR: a scalar sample or score step.

    Synthesised from a :class:`BindStep` with no index annotation;
    ``is_observed`` distinguishes sample (``False``) from score
    (``True``).
    """

    vars: tuple[str, ...]
    morphism: str
    args: tuple[str | float, ...] | None = None
    is_observed: bool = False
    line: int = 0
    col: int = 0
    kind: Literal["draw_step"] = "draw_step"


class PlateDrawStep(ProgramStep):
    """Internal compiler IR: an A-indexed sample step.

    Synthesised from a :class:`BindStep` with ``mode='sample'`` and
    an index annotation. Categorically a Kern-morphism
    :math:`A \\to \\mathcal{G}(K)` realised as a single tensor of
    shape ``(|A|, *K.shape)``.
    """

    name: str
    index: TypeExpr
    codomain: TypeExpr
    morphism: str
    args: tuple[str | float, ...] | None = None
    line: int = 0
    col: int = 0
    kind: Literal["plate_draw_step"] = "plate_draw_step"


class VectorisedObserveStep(ProgramStep):
    """Internal compiler IR: an A-indexed score step.

    Synthesised from a :class:`BindStep` with ``mode='score'`` and
    an index annotation. Denotes the sub-probabilistic kernel
    :math:`\\Phi \\to \\mathcal{G}_{\\le 1}(\\Phi)` with score
    :math:`\\prod_{n} p_F(r_{\\mathrm{obs}}(n); \\theta(n, \\phi))`.
    """

    index_var: str
    index_set: TypeExpr
    morphism: str
    args: tuple[str | float, ...] | None = None
    response_var: str = ""
    # ``via <idx>`` clause on the originating observe surface step.
    # Inside a grouped marginalize body this names the per-observe
    # fibration into the shared grouping plate; the product form
    # ``via product(...)`` populates ``fibration_axes`` instead.
    # Outside a grouped body both fields are unused.
    fibration_var: str | None = None
    fibration_axes: tuple[str, ...] | None = None
    line: int = 0
    col: int = 0
    kind: Literal["vectorized_observe_step"] = "vectorized_observe_step"


class GroupedLatentInitStep(ProgramStep):
    """Internal compiler IR: initialize the latent's environment
    slot to ``torch.arange(class_size)`` at the start of a grouped
    marginalize block's body.

    The body's downstream ``let`` and ``observe`` steps then see the
    latent as a length-``K`` index tensor; any arithmetic involving
    the latent broadcasts across the class axis. The terminal
    captured observe (see :class:`GroupedBodyObserveStep`) overwrites
    this slot with the per-(N, K) log-likelihood tensor the
    marginalize step consumes.
    """

    latent_name: str
    class_size: int
    line: int = 0
    col: int = 0
    kind: Literal["grouped_latent_init_step"] = "grouped_latent_init_step"


class GroupedBodyObserveStep(ProgramStep):
    """Internal compiler IR: a captured observe inside a grouped
    marginalize block.

    The body of a grouped marginalize block ends with an observe
    step whose per-row log-likelihood depends on the latent. Rather
    than accumulating the scalar log-density into the program-level
    joint (the normal observe path), this captured form:

    1. Computes ``family.log_prob(theta, response)`` per row,
       broadcasting ``theta`` across the class axis if it carries
       one (because upstream ``let`` steps referenced the latent).
    2. Stores the resulting ``(N, K)`` tensor at the marginalize
       block's latent slot, where the
       :class:`MarginalizeStep`'s runtime callable picks it up,
       applies the prior, and reduces.

    Categorically: the captured observe is the body's
    contribution to the right Kan extension along the fibration in
    :math:`\\mathbf{Kern}` — the per-(row, class) log-likelihood
    tensor that the per-group accumulator scatter-adds.
    """

    response_var: str
    morphism: str
    args: tuple[str | float, ...] | None = None
    index_set: TypeExpr | None = None
    index_var: str = ""
    latent_name: str = ""
    # Per-observe fibration into the shared grouping plate.
    # ``fibration_var`` carries a single-axis fibration's name;
    # ``fibration_axes`` carries a product-fibration's tuple of
    # axis names.  Exactly one is set inside a grouped body; the
    # other is ``None``.
    fibration_var: str | None = None
    fibration_axes: tuple[str, ...] | None = None
    # The env slot the captured-observe's (N_m, K) per-row
    # per-class log-likelihood is written to.  Unique per observe
    # inside a single grouped body so the surrounding
    # MarginalizeStep can collect each axis's contribution
    # separately and pair it with the right fibration.
    ll_slot: str = ""
    line: int = 0
    col: int = 0
    kind: Literal["grouped_body_observe_step"] = "grouped_body_observe_step"


class GroupedObserveEntry(dx.Model):
    """One entry in a grouped :class:`MarginalizeStep`'s
    ``body_observes`` list: the pairing of an env slot (where
    the captured observe writes its ``(N_m, K)`` per-row
    per-class log-likelihood) with the fibration that carries
    those rows into the shared grouping plate.

    Exactly one of ``fibration_var`` and ``fibration_axes`` is
    non-``None``: a single-axis fibration uses
    ``fibration_var``; a product fibration uses
    ``fibration_axes`` (whose arity must match the marginalize
    header's product-plate arity).

    Both ``None`` flags a nested-marginalize entry: the inner
    block has already performed its own scatter, so the outer
    block consumes the ``(|G|, K)`` tensor at ``ll_slot``
    directly with no further fibration.
    """

    ll_slot: str
    fibration_var: str | None = None
    fibration_axes: tuple[str, ...] | None = None


class MarginalizeStep(ProgramStep):
    """Internal compiler IR: a marginalisation reduction.

    The :class:`BindStep` for marginalize is expanded by the
    compiler into: (1) a sample step that introduces the
    coordinate, (2) the scope's steps, (3) this MarginalizeStep
    that pushes forward through the projection
    :math:`\\pi_{\\Phi} : \\Phi \\times C \\to \\Phi`.

    When the surface block carries ``over G via idx``, the
    reduction is fibred: the body's per-row log-density tensor
    of shape ``(N, K)`` is scatter-added along ``via_var`` to
    shape ``(|G|, K)``, the categorical prior ``probs_var``
    contributes ``log probs[g, k]`` per (group, class), and the
    final log-sum-exp over the class axis is summed over groups.
    This denotes the right Kan extension along the fibration
    :math:`r : \\text{Resp} \\to G` in :math:`\\mathbf{Kern}`,
    followed by integration of the class axis under the
    categorical prior.
    """

    var_name: str
    class_size: int = 0
    probs_var: str | None = None
    over_obj: str | None = None
    # Product grouping plate: a tuple of plate names whose
    # cardinalities multiply to give the flat group cardinality.
    # ``None`` for a single grouping plate; in that case
    # ``over_obj`` carries the singleton name.
    over_objs: tuple[str, ...] | None = None
    body_ll_var: str | None = None
    # Grouped form: ordered tuple of per-observe entries.  Each
    # entry pairs an env slot (where the observe writes its
    # ``(N_m, K)`` log-likelihood) with the fibration into the
    # shared grouping plate.  See :class:`GroupedObserveEntry`
    # for the field semantics.  ``None`` outside a grouped body.
    body_observes: tuple[GroupedObserveEntry, ...] | None = None
    # Per-group reduction over the class axis. ``None`` defaults
    # to ``"logsumexp"`` at the runtime call site.
    reduction: str | None = None
    line: int = 0
    col: int = 0
    kind: Literal["marginalize_step"] = "marginalize_step"


# ---------------------------------------------------------------------------
# top-level statements
# ---------------------------------------------------------------------------


class Statement(dx.TaggedUnion, discriminator="kind"):
    """Sum of top-level statement kinds."""


type CompositionLevel = Literal[
    "algebra", "semigroupoid", "bilinear_form", "composition_rule"
]
"""Algebraic level the file declares for its composition rule.

The four levels correspond to the
:class:`~quivers.core.algebras.CompositionRule`-hierarchy:

* ``"algebra"`` requires a full :class:`Algebra` (unit, zero,
  meet, negate, identity, dagger, cup/cap).
* ``"semigroupoid"`` requires a :class:`Semigroupoid`
  (associative `tensor_op`, no identity required).
* ``"bilinear_form"`` requires a :class:`BilinearForm`
  (no associativity promise).
* ``"composition_rule"`` is permissive: any
  :class:`CompositionRule` is accepted.
"""


class CompositionRuleEntry(dx.Model):
    """One entry of a composition-rule body block.

    Function-valued entries (``tensor_op``, ``join``, ``negation``,
    ``meet``) declare a lambda; value-valued entries (``unit``,
    ``zero``) declare a numeric literal. The ``params`` tuple is
    empty for value-valued entries.
    """

    key: str
    params: tuple[str, ...] = ()
    body: "LetExprNode"
    line: int = 0
    col: int = 0


class AlgebraDecl(Statement):
    """Composition-rule selection: ``algebra <name>``,
    ``semigroupoid <name>``, ``bilinear_form <name>``, or
    ``composition_rule <name>``, with an optional inline body.

    Without a body the declaration looks up ``name`` in the
    compiler's :data:`_ALGEBRA_REGISTRY` and verifies the
    registered rule matches the keyword's algebraic level. With a
    body, the declaration *defines* a fresh composition rule
    named ``name`` whose operations come from the supplied
    expressions; the keyword fixes the rule's level.
    """

    name: str
    declared_level: CompositionLevel = "algebra"
    body: tuple[CompositionRuleEntry, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["algebra_decl"] = "algebra_decl"


class CategoryDecl(Statement):
    """Category atom declaration: ``category <name>``."""

    name: str
    line: int = 0
    col: int = 0
    kind: Literal["category_decl"] = "category_decl"


class RuleDecl(Statement):
    """Rule-of-inference declaration.

    Premises and conclusion are :class:`TypeExpr` patterns drawn from
    the unified type-expression family: ``TypeName``, ``TypeProduct``,
    ``TypeSlash`` (residuated), and ``TypeEffectApply`` (effect-typed).
    """

    name: str
    variables: tuple[str, ...]
    premises: tuple[TypeExpr, ...]
    conclusion: TypeExpr
    line: int = 0
    col: int = 0
    kind: Literal["rule_decl"] = "rule_decl"


class SchemaDecl(Statement):
    """Pattern-polymorphic morphism schema declaration.

    Surface form: ``schema r[X, Y : Cat] : (X/Y) * Y -> X``.

    Parameters are encoded as two parallel tuples — :attr:`parameter_names`
    holds, for each parameter group, the tuple of variable names (e.g.
    ``("X", "Y")`` for ``X, Y : Cat``); :attr:`parameter_types` holds the
    corresponding type expressions. The arity invariant
    ``len(parameter_names) == len(parameter_types)`` is enforced via a
    dx.axiom.

    Arity (binary vs. unary) is derived from the domain shape: a
    top-level :class:`TypeProduct` with two components produces a
    binary schema; any other domain shape produces a unary schema.
    """

    name: str
    parameter_names: tuple[tuple[str, ...], ...]
    parameter_types: tuple[TypeExpr, ...]
    domain: TypeExpr
    codomain: TypeExpr
    docs: tuple[str, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["schema_decl"] = "schema_decl"

    __axioms__ = (
        dx.axiom(
            "length parameter_names == length parameter_types",
            message="schema parameter_names and parameter_types must align",
        ),
    )


class ObjectInitializer(dx.TaggedUnion, discriminator="kind"):
    """Sum of object-initializer kinds for the ``=`` form of ObjectDecl."""


class EnumSetLiteral(ObjectInitializer):
    """A ``{NP, S, VP}``-shaped enum-set initializer."""

    elements: tuple[str, ...]
    line: int = 0
    col: int = 0
    kind: Literal["enum_set_literal"] = "enum_set_literal"


class FreeResiduatedExpr(ObjectInitializer):
    """A ``FreeResiduated(generators, depth=, ops=[...])`` initializer."""

    generators: str
    depth: int = 1
    ops: tuple[str, ...] = ("slash",)
    line: int = 0
    col: int = 0
    kind: Literal["free_residuated_expr"] = "free_residuated_expr"


class FreeMonoidExpr(ObjectInitializer):
    """A ``FreeMonoid(generators, max_length=)`` initializer."""

    generators: str
    max_length: int
    line: int = 0
    col: int = 0
    kind: Literal["free_monoid_expr"] = "free_monoid_expr"


class ObjectDecl(Statement):
    """Object declaration.

    Three surface forms:

    - ``object X : 3`` — anonymous-element FinSet of cardinality 3.
      ``type_expr`` carries the TypeExpr; ``init`` is None.
    - ``object Atoms = {NP, S, VP}`` — EnumSet of named atoms.
      ``init`` carries an :class:`EnumSetLiteral`; ``type_expr`` is None.
    - ``object Cat = FreeResiduated(Atoms, depth=4)`` — residuated
      category universe. ``init`` carries a :class:`FreeResiduatedExpr`.

    Doc comments (``##``-prefixed lines) immediately preceding the
    declaration are accumulated into :attr:`docs`.
    """

    name: str
    type_expr: TypeExpr | None = None
    init: ObjectInitializer | None = None
    docs: tuple[str, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["object_decl"] = "object_decl"


class MorphismDecl(Statement):
    """Morphism declaration: ``latent|observed <name> : <dom> -> <cod>``.

    ``morphism_kind`` distinguishes ``"latent"`` from ``"observed"``;
    ``kind`` is the dx.TaggedUnion discriminator and is fixed for this
    variant.
    """

    morphism_kind: Literal["latent", "observed"]
    name: str
    domain: TypeExpr
    codomain: TypeExpr
    init_expr: Expr | None = None
    options: dict[str, str] = dx.field(default_factory=dict)
    # Parameter prior on the morphism's representing tensor.  Legal
    # only on ``latent`` declarations; promotes the morphism from
    # a free-parameter point estimate to a random morphism whose
    # tensor is drawn from the named family.
    prior: MorphismPrior | None = None
    docs: tuple[str, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["morphism_decl"] = "morphism_decl"


class SpaceDecl(Statement):
    """Space declaration: ``space <name> : <space_expr>``."""

    name: str
    space_expr: SpaceExpr
    line: int = 0
    col: int = 0
    kind: Literal["space_decl"] = "space_decl"


class TypeAliasDecl(Statement):
    """Space-level alias declaration: ``type Latent = Euclidean(16)``."""

    name: str
    space_expr: SpaceExpr
    line: int = 0
    col: int = 0
    kind: Literal["type_alias_decl"] = "type_alias_decl"


class AliasDecl(Statement):
    """Object-level type alias: ``alias Sentence = Cat / NP``.

    Binds ``name`` to the resolved :class:`SetObject` of ``type_expr``
    in the compiler's object environment. The alias is transparent:
    every later occurrence of ``name`` resolves to the underlying
    object, with no reference-counting or recursion bound.
    """

    name: str
    type_expr: TypeExpr
    docs: tuple[str, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["alias_decl"] = "alias_decl"


class BundleDecl(Statement):
    """First-class schema-bundle binding.

    Surface form: ``bundle CCG = [forward_app, backward_app,
    harmonic_composition]``. Binds ``name`` to a tuple of schema
    references; ``parser(rules=CCG, ...)`` and
    ``chart_fold(binary=CCG, ...)`` resolve the bundle by name and
    splice its members into the rule list.
    """

    name: str
    rules: tuple[str, ...]
    docs: tuple[str, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["bundle_decl"] = "bundle_decl"


class KernelDecl(Statement):
    """Markov-kernel declaration: ``kernel f : A -> B [~ Family ...]``.

    Without a ``~`` clause, declares a lookup-table kernel on finite
    sets — a categorical kernel :math:`A \\to D(B)` realised as a
    learnable matrix of conditional probabilities.

    With a ``~ Family [options] [axes]`` clause, declares a parametric
    kernel :math:`A \\to G(B)` whose family's parameters are produced
    from the input by a parameter network at sample time.  The
    optional ``axes`` clause configures the family's event/batch
    decomposition over codomain factors.
    """

    name: str
    domain: TypeExpr
    codomain: TypeExpr
    family: str | None = None
    options: dict[str, str] = dx.field(default_factory=dict)
    axes: AxisSpec | None = None
    replicate: int | None = None
    line: int = 0
    col: int = 0
    kind: Literal["kernel_decl"] = "kernel_decl"


class DiscretizeDecl(Statement):
    """Discretize boundary: ``discretize <name> : <space> -> <n_bins>``."""

    name: str
    space_name: str
    n_bins: int
    options: dict[str, str] = dx.field(default_factory=dict)
    line: int = 0
    col: int = 0
    kind: Literal["discretize_decl"] = "discretize_decl"


class EmbedDecl(Statement):
    """Embed boundary: ``embed <name> : <finset> -> <space>``."""

    name: str
    domain_name: str
    codomain_name: str
    replicate: int | None = None
    line: int = 0
    col: int = 0
    kind: Literal["embed_decl"] = "embed_decl"


class ProgramParam(dx.TaggedUnion, discriminator="kind"):
    """Sum of typed-program-parameter variants.

    A *parametric* program declaration takes a list of typed
    parameters (objects, scalars, or morphisms) and denotes a
    dependent kernel

    .. math::

        \\Pi (p_1 : P_1) \\ldots \\Pi (p_n : P_n).\\ \\mathbf{Kern}(\\mathrm{dom}(p),\\, \\mathrm{cod}(p))

    in the indexed family of Kleisli arrows over the parameter
    category. Each call site substitutes specific arguments, yielding
    a concrete Kern-morphism with fresh latent factors inlined into
    the caller's trace; the freshness corresponds to the fact that
    distinct call sites contribute distinct factors to the parent's
    joint kernel.
    """


class ObjectParam(ProgramParam):
    """Object-typed program parameter: ``G : FinSet`` / ``Space`` / ``Object``.

    Denotes a dependent quantification over an object of the
    relevant subcategory: ``FinSet`` ranges over finite-set objects,
    ``Space`` over continuous spaces, ``Object`` over either.
    """

    name: str
    universe: Literal["FinSet", "Space", "Object"]
    line: int = 0
    col: int = 0
    kind: Literal["object_param"] = "object_param"


class ScalarParam(ProgramParam):
    """Scalar-valued program parameter: ``s : Real`` / ``Nat``.

    Denotes a dependent quantification over a hom-object of scalar
    type (real or nonnegative-integer values, used as
    hyperparameters and cardinalities respectively).
    """

    name: str
    scalar_kind: Literal["Real", "Nat"]
    line: int = 0
    col: int = 0
    kind: Literal["scalar_param"] = "scalar_param"


class MorphismParam(ProgramParam):
    """Morphism-typed program parameter: ``f : Mor[A, B]``.

    Denotes a dependent quantification over the hom-set
    :math:`\\mathbf{Kern}(A, B)`; the body may reference ``f`` as a
    family in any plate-draw or draw step whose codomain matches
    ``B``.
    """

    name: str
    domain: TypeExpr
    codomain: TypeExpr
    line: int = 0
    col: int = 0
    kind: Literal["morphism_param"] = "morphism_param"


class ExprMorphismCall(Expr):
    """Call expression ``callee(arg1, arg2, …)`` resolving to a
    morphism-level operation.

    Used by :class:`ContractionDecl` declarations: when the user
    writes ``let out = op_apply(arg1, arg2, kernel)``, the
    ``op_apply`` identifier resolves to a registered contraction
    and the arguments are looked up in the morphism scope.
    """

    callee: str
    args: tuple[str, ...]
    line: int = 0
    col: int = 0
    kind: Literal["expr_morphism_call"] = "expr_morphism_call"


class ContractionInput(dx.Model):
    """One input wire of a :class:`ContractionDecl` declaration."""

    name: str
    input_domain: "TypeExpr"
    input_codomain: "TypeExpr"
    line: int = 0
    col: int = 0


class ContractionDecl(Statement):
    """Operadic n-ary contraction declaration.

    Surface form (type-driven, inferred wiring)::

        contraction op_apply (
            arg1 : A -> B,
            arg2 : A -> C,
            kernel : (B * C) -> D
        ) : A -> D
            rule product_fuzzy

    The typed signature determines the einsum implicitly: each axis
    in the output (here ``A`` and ``D``) propagates; each axis that
    appears in ≥ 2 inputs but not in the output (``B``, ``C``) is
    contracted via the rule's join.

    Two opt-in disambiguators handle cases the inference cannot
    derive from the signature alone:

    * ``share T1, T2, ...`` keeps the listed axes element-wise
      (broadcast / propagated) even when they appear in multiple
      inputs.  Stored on :attr:`shared_axes`.
    * ``wiring "<einsum>"`` is the explicit escape hatch.  Stored
      verbatim on :attr:`wiring_spec`.

    Exactly zero or one of those clauses appears in source.  The
    compiler dispatches: ``wiring_spec`` non-empty → use it
    verbatim; otherwise build the einsum from the typed signature
    plus ``shared_axes``.

    Compiles to a callable that wraps
    :class:`~quivers.core.wiring.EinsumWiring`.
    """

    name: str
    inputs: tuple[ContractionInput, ...]
    domain: "TypeExpr"
    codomain: "TypeExpr"
    rule_name: str
    wiring_spec: str = ""
    shared_axes: tuple[str, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["contraction_decl"] = "contraction_decl"


class ProgramDecl(Statement):
    """Monadic program block — the unique program-form in QVR.

    A program is either *concrete* (no ``type_params``) — denoting a
    single Kern-morphism ``dom → cod`` — or *parametric* (with
    ``type_params``) — denoting a dependent family of Kern-morphisms
    indexed by the parameters. Parametric programs are not compiled
    into a runtime ``MonadicProgram`` directly; the compiler stores
    them as templates and inlines a freshly-renamed copy of the body
    at each call site.

    Effects and posterior modifier:

    * ``effects`` carries the capability set declared after ``!``:
      ``frozenset({"Sample", "Score"})``, ``frozenset({"Marginal"})``,
      ``frozenset({"Pure"})``, etc. ``None`` means unannotated (the
      compiler infers the set from the body but does not enforce a
      restriction).
    * ``over_model`` declares the program is a *posterior block*
      over another program's latents — replacing the standalone
      ``posterior`` keyword. A program with ``over_model`` set is
      routed by the compiler to the posterior registry; its body
      must be deterministic (``Pure`` or at most ``Pure | Marginal``).
    """

    name: str
    params: tuple[str, ...] | None
    domain: TypeExpr
    codomain: TypeExpr
    draws: tuple[ProgramStep, ...]
    return_vars: tuple[str, ...]
    return_labels: tuple[str, ...] | None = None
    effects: frozenset[str] | None = None
    over_model: str | None = None
    type_params: tuple[ProgramParam, ...] | None = None
    docs: tuple[str, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["program_decl"] = "program_decl"


class LetDecl(Statement):
    """Let binding: ``let <name> = <expr> [where let_decl+]``.

    The optional ``where`` clause carries nested :class:`LetDecl`
    statements; the field is typed as ``tuple[Statement, ...]`` (the
    union root) because didactic does not yet accept self-referential
    forward refs in field annotations. The parser only ever writes
    :class:`LetDecl` instances into the tuple.
    """

    name: str
    expr: Expr
    where: tuple[Statement, ...] | None = None
    line: int = 0
    col: int = 0
    kind: Literal["let_decl"] = "let_decl"


class ExportDecl(Statement):
    """Module-level export: ``export <expr>``.

    Any number per module; each selects a top-level morphism /
    posterior / deduction for the compiled output. Replaces v0.4's
    ``output`` keyword (which permitted exactly one per module);
    semantically a public binding in the module namespace.
    """

    expr: Expr
    line: int = 0
    col: int = 0
    kind: Literal["export_decl"] = "export_decl"


class SequentRule(dx.Model):
    """A named sequent-style inference rule inside a deduction block.

    ``rule name : premises |- conclusion``. Patterns may contain
    single-uppercase-identifier wildcards (e.g. ``X``, ``Y``) that
    bind to actual category subexpressions when the rule fires.
    """

    name: str
    premises: tuple[TypeExpr, ...]
    conclusion: TypeExpr
    line: int = 0
    col: int = 0


class LexiconEntry(dx.Model):
    """A single entry in a deduction's lexicon block.

    Maps a literal word string to a (category, logical-form)
    pair, with an optional learnable log-weight. Multiple
    entries with the same ``word`` give the model a latent
    disjunction over (cat, lf) options for that word — at
    chart-construction time, each becomes an independent span
    axiom and the chart's semiring aggregates them (the
    standard semiring-parsing realisation of latent lexical
    categories / logical forms).

    Either or both of ``category`` and ``lf`` may carry
    structural metavariables; under Curry-Howard these are
    types and proof witnesses, and the lexicon is a
    *family of typed terms* in the deduction's term algebra.
    Both slots are syntactically open enough to express
    type-logical grammars, dependent-type systems,
    Datalog facts, edit-distance alignment moves, or any other
    item algebra the deduction operates over.

    Attributes
    ----------
    word : str
        The literal lexical token (typically a surface word in
        an NLP setting, but may be any string-keyed item label).
    category : TypeExpr
        The entry's syntactic / type-theoretic category. Used
        as the leftmost field of the span axiom emitted at chart
        time.
    lf : LetExprNode
        The entry's logical form / proof witness, as a
        let-sublanguage expression evaluated at axiom-injection
        time.
    learnable : bool
        Whether to allocate a per-entry :class:`nn.Parameter`
        log-weight (default scale ``0.0``) that the optimizer
        can adjust during training.
    """

    word: str
    category: TypeExpr
    lf: LetExprNode
    learnable: bool = False
    line: int = 0
    col: int = 0


class DeductionDecl(Statement):
    """A weighted-deduction-system declaration.

    Surface form::

        deduction NAME : Domain -> Codomain {
            atoms { A, B, ... }
            rule r1 : premises |- conclusion
            rule r2 : ...
            semiring  SemiringName
            start     StartSymbol
            depth     N

            # One of three optional axiom-source forms:
            lexicon {                          # inline lexicon block
                "every" : Cat = lf @ learnable
                ...
            }
            lexicon from "path/to/lexicon.tsv" with learnable
                                               # file-loaded lexicon
            axioms = some_kernel_morphism      # general axiom source
        }

    Categorical denotation: the system denotes a
    :math:`\\mathcal{V}`-presheaf-valued morphism
    :math:`\\mathrm{Domain} \\to \\mathbf{Set}^{I^{\\mathrm{op}}}_{K}`,
    computed as the least pre-fixed point of the rule-system
    functor in the :math:`K`-enriched lattice of charts. The
    axiom-source field declares the kernel
    :math:`\\mathrm{Input} \\to \\mathrm{List}(I \\times K)` that
    produces the chart's initial items from an input value;
    ``lexicon`` is a sugar specialization for the
    label-indexed-lookup case.
    """

    name: str
    domain: TypeExpr
    codomain: TypeExpr
    atoms: tuple[str, ...]
    rules: tuple[SequentRule, ...]
    semiring: str | None = None
    start: str | None = None
    depth: int | None = None
    lexicon: tuple[LexiconEntry, ...] = ()
    lexicon_from_file: str | None = None
    lexicon_from_file_learnable: bool = False
    axioms_source: str | None = None
    item_signature: str | None = None
    item_encoder: str | None = None
    line: int = 0
    col: int = 0
    kind: Literal["deduction_decl"] = "deduction_decl"


# ---------------------------------------------------------------------------
# structural-compression: signatures, encoders, decoders, losses
# ---------------------------------------------------------------------------


class SortVocabLiteral(dx.Model):
    """One entry of a data sort's closed vocabulary.

    The literal carries its surface text plus a tag so the compiler
    can decode each entry into the canonical Python value the
    runtime stores in :attr:`Sort.vocab` (``str``, ``int``, or
    ``float``).
    """

    kind: Literal["string", "integer", "float"]
    text: str


class SortDecl(dx.Model):
    """One sort within a signature.

    `kind` is one of ``"object"``, ``"index"``, ``"data"``. The dim
    is optional at the signature level; if absent, every encoder /
    decoder over this signature must supply it. ``vocab`` is the
    closed-vocabulary literal sequence for data sorts (empty for
    object / index sorts).
    """

    name: str
    kind: Literal["object", "index", "data"]
    dim: int | None = None
    vocab: tuple[SortVocabLiteral, ...] = ()
    line: int = 0
    col: int = 0


class ConstructorDecl(dx.Model):
    """A typed operation `name : s_1, ..., s_n -> s` in a signature."""

    name: str
    domain: tuple[str, ...]
    codomain: str
    line: int = 0
    col: int = 0


class BinderVar(dx.Model):
    """A variable introduced by a binder.

    ``var`` and ``annot`` are names used for diagnostics only —
    references are by de-Bruijn index inside the scope. ``sort`` is
    the sort of the variable itself; ``annot_sort`` is the sort of
    the variable's type annotation, if one is supplied. When
    ``annot_sort`` is set, the binder constructor takes one
    additional positional argument (immediately preceding the
    bound variable's role in the scope) of that sort, which the
    encoder / decoder thread into Γ alongside the variable's
    embedding.
    """

    var: str
    sort: str
    annot: str | None = None
    annot_sort: str | None = None


class BinderArg(dx.Model):
    """An argument of a binder constructor; ``scoped`` arguments live
    in the extended context."""

    arg: str
    sort: str


class BinderDecl(dx.Model):
    """A binder constructor introducing new scoped variables."""

    name: str
    binds: tuple[BinderVar, ...]
    scoped: tuple[BinderArg, ...]
    codomain: str
    line: int = 0
    col: int = 0


class VertexKindDecl(dx.Model):
    """A vertex kind in a graph-shaped signature."""

    name: str
    kind: Literal["object", "index", "data"]
    dim: int | None = None
    line: int = 0
    col: int = 0


class EdgeKindDecl(dx.Model):
    """An edge kind in a graph-shaped signature.

    ``directed`` is True for ``src -> tgt``, False for ``src -- tgt``.
    """

    name: str
    src: str
    tgt: str
    directed: bool = True
    line: int = 0
    col: int = 0


class SignatureDecl(Statement):
    """A signature block declaring an algebra over which encoders,
    decoders, and rules are defined.

    A signature may be **inductive** (sorts + constructors + binders)
    or **graph-shaped** (vertex_kinds + edge_kinds), or both.
    """

    name: str
    params: tuple[str, ...] = ()
    sorts: tuple[SortDecl, ...] = ()
    constructors: tuple[ConstructorDecl, ...] = ()
    binders: tuple[BinderDecl, ...] = ()
    vertex_kinds: tuple[VertexKindDecl, ...] = ()
    edge_kinds: tuple[EdgeKindDecl, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["signature_decl"] = "signature_decl"


class SortDim(dx.Model):
    """A `(sort, dim)` association declared in a encoder/decoder."""

    sort: str
    dim: int


class EncoderVarInit(dx.Model):
    """One `var_init <var_sort> [from <annot_sort> [as <name>]]` rule.

    ``annot_sort=None`` is the unannotated-binder case (no type
    annotation; the body sees no extra arg).  When ``annot_sort`` is
    set, ``ty`` is the body's parameter name bound to the annotation
    embedding.
    """

    var_sort: str
    annot_sort: str | None = None
    ty: str | None = None
    body: "LetExprNode"
    line: int = 0
    col: int = 0


class EncoderRule(dx.Model):
    """A per-operation encoder function.

    The body is a let-expression evaluated in an environment where the
    constructor arguments (as named in ``args``) are bound to child
    vectors, plus framework-supplied helpers (``ctx`` for binder
    contexts, ``state``/``prefix`` for recurrent / attention shapes).

    ``mode`` selects sequence sugar:

    * ``"plain"`` — direct algebra-hom rule (default).
    * ``"recurrent"`` — left-fold, ``state`` carries the accumulator.
    * ``"attention"`` — ``prefix`` carries the running list of prior
      compressed children.
    """

    op: str
    args: tuple[str, ...]
    body: "LetExprNode"
    mode: Literal["plain", "recurrent", "attention"] = "plain"
    state_var: str | None = None
    prefix_var: str | None = None
    line: int = 0
    col: int = 0


class EncoderInitRule(dx.Model):
    """Graph-signature initializer: maps vertex `data` payloads to
    initial vertex embeddings before message passing."""

    kind: str
    arg: str
    body: "LetExprNode"
    line: int = 0
    col: int = 0


class EncoderMessageRule(dx.Model):
    """Graph-signature message: maps a `(src, tgt)` pair on an edge
    kind to a message vector."""

    edge_kind: str
    src: str
    tgt: str
    body: "LetExprNode"
    line: int = 0
    col: int = 0


class EncoderUpdateRule(dx.Model):
    """Graph-signature update: maps `(self_embed, aggregated_msgs)`
    to the next vertex embedding, per vertex kind."""

    vertex_kind: str
    self_var: str
    msgs_var: str
    body: "LetExprNode"
    line: int = 0
    col: int = 0


class EncoderDecl(Statement):
    """An algebra homomorphism from an inductive or graph signature
    to a fixed-dimension vector carrier.

    Surface form::

        encoder C over Sig {
            dim Term = 64
            App(f, x)        |-> mlp_app([f, x])
            Lam(ty, body)    |-> mlp_lam([ty, body])
            var_init(ty)     |-> mlp_var_init(ty)
            iterations 4                       # graph-only
            init Atom(a)     |-> atom_embed[a] # graph-only
            message[e](s, t) |-> mlp_msg([s, t])
            update[V](s, m)  |-> gru_update(s, m)
            readout          |-> mean_pool
        }
    """

    name: str
    signature: str
    sig_args: tuple[str, ...] = ()
    dims: tuple[SortDim, ...] = ()
    op_rules: tuple[EncoderRule, ...] = ()
    init_rules: tuple[EncoderInitRule, ...] = ()
    message_rules: tuple[EncoderMessageRule, ...] = ()
    update_rules: tuple[EncoderUpdateRule, ...] = ()
    iterations: int | None = None
    readout: "LetExprNode | None" = None
    var_inits: tuple[EncoderVarInit, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["encoder_decl"] = "encoder_decl"


class DecoderDecl(Statement):
    """A Kleisli coalgebraic decoder over an inductive or graph
    signature.

    Surface form::

        decoder D over Sig depth 8 {
            dim Term = 64
            structure(v)      |-> structure_logits(v)
            primitive(v)      |-> primitive_logits(v)
            factor(v)         |-> factor_split(v)
            binder_select(v)  |-> binder_logits(v)
            body              |-> recursive
        }
    """

    name: str
    signature: str
    sig_args: tuple[str, ...] = ()
    depth: int = 8
    dims: tuple[SortDim, ...] = ()
    structure: "LetExprNode | None" = None
    structure_arg: str | None = None
    primitive: "LetExprNode | None" = None
    primitive_arg: str | None = None
    factor: "LetExprNode | None" = None
    factor_arg: str | None = None
    binder_select: "LetExprNode | None" = None
    binder_select_arg: str | None = None
    recursive_default: bool = True
    line: int = 0
    col: int = 0
    kind: Literal["decoder_decl"] = "decoder_decl"


class LossAttachment(dx.Model):
    """Where a loss fires.

    ``attachment_kind`` is one of:

    * ``"global"`` — fires once per training step.
    * ``"program"`` — fires after a named program invocation.
    * ``"deduction"`` — fires after a named deduction's chart build.
    * ``"encoder"`` / ``"decoder"`` — fires after the named call.
    * ``"rule"`` — fires on each application of a named rule.
    * ``"chart"`` — fires once on the chart of a named deduction.
    """

    attachment_kind: Literal[
        "global", "program", "deduction", "encoder", "decoder", "rule", "chart"
    ] = "global"
    target: str | None = None
    rule_deduction: str | None = None


class LossDecl(Statement):
    """A weighted scalar loss, attachable to any training site."""

    name: str
    weight: "LetExprNode | None" = None
    attachment: LossAttachment = dx.field(default_factory=LossAttachment)
    body: "LetExprNode"
    line: int = 0
    col: int = 0
    kind: Literal["loss_decl"] = "loss_decl"


# ---------------------------------------------------------------------------
# module
# ---------------------------------------------------------------------------


class Module(dx.Model):
    """A complete .qvr program (sequence of statements)."""

    statements: tuple[Statement, ...] = dx.field(default_factory=tuple)
