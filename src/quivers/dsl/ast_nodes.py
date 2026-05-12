"""AST node definitions for the quivers DSL.

The AST is a direct representation of the parsed `.qvr` source. Each
node carries source-location info for error reporting. Recursive sums
use ``dx.TaggedUnion`` with a ``kind`` discriminator: variants subclass
their sum's root and pin ``kind`` to a ``Literal[...]`` value.
"""

from typing import Literal

import didactic.api as dx


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


class ExprCompose(Expr):
    """Sequential composition: ``left >> right``."""

    left: Expr
    right: Expr
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
    bounds effect-stack nesting (Phase 7; defaults to 0).
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
        vectorised / indexed-marginalize forms.
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
# BindStep into one of the four specialised forms below at the entry to
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
    line: int = 0
    col: int = 0
    kind: Literal["vectorised_observe_step"] = "vectorised_observe_step"


class MarginalizeStep(ProgramStep):
    """Internal compiler IR: a marginalisation reduction.

    The :class:`BindStep` for marginalize is expanded by the
    compiler into: (1) a sample step that introduces the
    coordinate, (2) the scope's steps, (3) this MarginalizeStep
    that pushes forward through the projection
    :math:`\\pi_{\\Phi} : \\Phi \\times C \\to \\Phi`.
    """

    var_name: str
    line: int = 0
    col: int = 0
    kind: Literal["marginalize_step"] = "marginalize_step"


# ---------------------------------------------------------------------------
# top-level statements
# ---------------------------------------------------------------------------


class Statement(dx.TaggedUnion, discriminator="kind"):
    """Sum of top-level statement kinds."""


class QuantaleDecl(Statement):
    """Quantale selection: ``quantale <name>``."""

    name: str
    line: int = 0
    col: int = 0
    kind: Literal["quantale_decl"] = "quantale_decl"


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


class ContinuousMorphismDecl(Statement):
    """Continuous morphism declaration."""

    name: str
    domain: TypeExpr
    codomain: TypeExpr
    family: str
    options: dict[str, str] = dx.field(default_factory=dict)
    replicate: int | None = None
    line: int = 0
    col: int = 0
    kind: Literal["continuous_morphism_decl"] = "continuous_morphism_decl"


class StochasticMorphismDecl(Statement):
    """Stochastic morphism declaration."""

    name: str
    domain: TypeExpr
    codomain: TypeExpr
    replicate: int | None = None
    line: int = 0
    col: int = 0
    kind: Literal["stochastic_morphism_decl"] = "stochastic_morphism_decl"


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


# ---------------------------------------------------------------------------
# module
# ---------------------------------------------------------------------------


class Module(dx.Model):
    """A complete .qvr program (sequence of statements)."""

    statements: tuple[Statement, ...] = dx.field(default_factory=tuple)
