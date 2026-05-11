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


class DrawStep(ProgramStep):
    """A single ``draw`` or ``observe`` step inside a program block."""

    vars: tuple[str, ...]
    morphism: str
    args: tuple[str | float, ...] | None = None
    is_observed: bool = False
    line: int = 0
    col: int = 0
    kind: Literal["draw_step"] = "draw_step"


class PlateDrawStep(ProgramStep):
    """A finite-domain-indexed draw: ``draw v : A -> B ~ F(args)``.

    Denotes the indexed family of independent draws

        v(a) ~ F(args(a))   for each a in [[A]]

    realised in the program trace as a single random variable
    ``v : A → B`` of function-space type. In the Giry-monad
    Kleisli semantics, this is

        S[draw v : A → B ~ F](φ) = δ_φ ⊗ Π_{a∈A} F(args(a))

    i.e. the joint distribution over the function-space ``B^A``
    factorises as the independent product across the index set.

    Compiled to a tensor of shape ``(|A|, *B.shape)`` whose
    distribution under the variational posterior is one independent
    copy of ``F`` per row.

    Attributes
    ----------
    name : str
        Bound name of the indexed random variable.
    index : TypeExpr
        The index set ``A`` (a previously-declared object).
    codomain : TypeExpr
        The per-index codomain ``B`` (typically ``Euclidean(K)``).
    morphism : str
        Distribution family name (``Normal``, ``MultivariateNormal``,
        etc.).
    args : tuple
        Family arguments. May contain :class:`GatherExpr` so the
        prior's hyperparameters can depend on the index.
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
    """A batched observation: ``observe r[n] ~ F(θ[n]) for n in N``.

    Denotes the product likelihood

        Π_{n ∈ [[N]]} p_F(r_obs(n); θ(n))

    realised in the sub-probabilistic Giry monad as

        S[obs r[n] ~ F(θ[n]) for n in N] : Φ → G_{≤1}(Φ)
        S[..](φ, B) = 1_B(φ) · Π_{n ∈ N} p_F(r_obs(n); θ(n, φ))

    The trace context is preserved; the total mass of the resulting
    measure is the joint likelihood of the dataset.

    Attributes
    ----------
    index_var : str
        Loop variable bound across the observation index set.
    index_set : TypeExpr
        The observation index set ``N`` (an object declared at
        module level, typically of FinSet kind).
    morphism : str
        Distribution family name.
    args : tuple
        Family arguments, which may include :class:`GatherExpr` and
        :class:`LetExprNode` sub-expressions referencing the loop
        variable.
    response_var : str
        The data column whose entry at index ``n`` provides the
        observed value of the ``n``-th observation.
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
    """A discrete-latent marginalisation: ``marginalize v``.

    Given a previously-drawn discrete latent ``v : Φ → G(C)``, the
    marginalisation pushes forward through the projection
    ``π_{Φ\\C} : Φ × C → Φ``:

        marg(v) : Φ → G(Φ)
        marg(v) = G(π_{Φ\\C}) ∘ S[draw v]

    Numerically realised as ``log_sum_exp`` over the ``C`` axis in
    the trace's accumulated log-likelihood.

    Attributes
    ----------
    var_name : str
        Name of the previously-drawn discrete latent variable.
    """

    var_name: str
    line: int = 0
    col: int = 0
    kind: Literal["marginalize_step"] = "marginalize_step"


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


class ProgramDecl(Statement):
    """Monadic program block with optional named params and tuple returns."""

    name: str
    params: tuple[str, ...] | None
    domain: TypeExpr
    codomain: TypeExpr
    draws: tuple[ProgramStep, ...]
    return_vars: tuple[str, ...]
    return_labels: tuple[str, ...] | None = None
    docs: tuple[str, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["program_decl"] = "program_decl"


class PosteriorDecl(Statement):
    """A posterior / generated-quantities block.

    Runs *after* the model program has been conditioned on data; its
    body is a deterministic function of the posterior over latents.
    Categorically: given the conditioned model's posterior kernel
    ``q(θ | data) : Data → G(Latents)``, the posterior block denotes a
    morphism ``Latents → τ_out`` in **Kern** which lifts to
    ``Data → G(τ_out)`` by post-composition.

    Operationally: each posterior sample (variational draw, or one
    MCMC iterate) is run through the body; the runner aggregates
    the per-sample outputs.

    Surface form mirrors :class:`ProgramDecl`:

    .. code-block:: qvr

        posterior class_probs (model) : Item -> Simplex(4)
            let logprob = item_loglik + log(class_prior)
            let probs = softmax(logprob)
            return probs

    Attributes
    ----------
    name : str
        The posterior-quantity's name.
    model : str
        Name of the model program whose posterior is consumed.
    params : tuple of str
        Optional parameters supplied at evaluation time.
    domain : TypeExpr
        Domain of the resulting kernel.
    codomain : TypeExpr
        Codomain of the resulting kernel.
    steps : tuple of ProgramStep
        Body steps. ``draw`` is disallowed (posterior is deterministic
        post-conditioning); ``observe`` is disallowed; ``let`` and
        ``marginalize`` are permitted.
    return_vars : tuple of str
        Tuple of variables to return as the posterior quantity.
    """

    name: str
    model: str
    params: tuple[str, ...] | None
    domain: TypeExpr
    codomain: TypeExpr
    steps: tuple[ProgramStep, ...]
    return_vars: tuple[str, ...]
    return_labels: tuple[str, ...] | None = None
    docs: tuple[str, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["posterior_decl"] = "posterior_decl"


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


class OutputDecl(Statement):
    """Output declaration: ``output <expr>``."""

    expr: Expr
    line: int = 0
    col: int = 0
    kind: Literal["output_decl"] = "output_decl"


# ---------------------------------------------------------------------------
# module
# ---------------------------------------------------------------------------


class Module(dx.Model):
    """A complete .qvr program (sequence of statements)."""

    statements: tuple[Statement, ...] = dx.field(default_factory=tuple)
