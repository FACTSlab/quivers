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
# category patterns (for rule declarations)
#
# Phase 1 of the categorial-effects integration adds TypeSlash and
# TypeEffectApply above and will eventually fold the CatPattern family
# into TypeExpr; until the tree-sitter grammar is regenerated to emit
# unified type-expression nodes for residuated patterns, the CatPattern
# variants below remain in place as the surface for `rule` declarations.
# ---------------------------------------------------------------------------


class CatPattern(dx.TaggedUnion, discriminator="kind"):
    """Sum of category-pattern node kinds."""


class CatPatternName(CatPattern):
    """A named category pattern element (variable or atom)."""

    name: str
    line: int = 0
    col: int = 0
    kind: Literal["cat_pattern_name"] = "cat_pattern_name"


class CatPatternSlash(CatPattern):
    """A slash category pattern: ``result/argument`` or ``result\\argument``."""

    result: CatPattern
    argument: CatPattern
    direction: Literal["/", "\\"]
    line: int = 0
    col: int = 0
    kind: Literal["cat_pattern_slash"] = "cat_pattern_slash"


class CatPatternProduct(CatPattern):
    """A product category pattern: ``left * right``."""

    left: CatPattern
    right: CatPattern
    line: int = 0
    col: int = 0
    kind: Literal["cat_pattern_product"] = "cat_pattern_product"


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
    """Rule-of-inference declaration."""

    name: str
    variables: tuple[str, ...]
    premises: tuple[CatPattern, ...]
    conclusion: CatPattern
    line: int = 0
    col: int = 0
    kind: Literal["rule_decl"] = "rule_decl"


class ObjectDecl(Statement):
    """Object declaration: ``object <name> : <type_expr>``."""

    name: str
    type_expr: TypeExpr
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
