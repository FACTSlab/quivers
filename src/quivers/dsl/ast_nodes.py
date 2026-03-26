"""AST node definitions for the quivers DSL.

The AST is a direct representation of the parsed .qvr source.
Each node carries source location info for error reporting.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# type expressions (describe categorical objects)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TypeName:
    """A named type reference (identifier or integer literal).

    Parameters
    ----------
    name : str
        The identifier or integer string.
    line : int
        Source line.
    col : int
        Source column.
    """

    name: str
    line: int = 0
    col: int = 0


@dataclass(frozen=True)
class TypeProduct:
    """Product type: A * B.

    Parameters
    ----------
    components : tuple of type expression nodes
        The component types.
    line : int
        Source line.
    col : int
        Source column.
    """

    components: tuple[TypeName | TypeProduct | TypeCoproduct, ...]
    line: int = 0
    col: int = 0


@dataclass(frozen=True)
class TypeCoproduct:
    """Coproduct type: A + B.

    Parameters
    ----------
    components : tuple of type expression nodes
        The component types.
    line : int
        Source line.
    col : int
        Source column.
    """

    components: tuple[TypeName | TypeProduct | TypeCoproduct, ...]
    line: int = 0
    col: int = 0


# union of all type expression nodes
TypeExpr = TypeName | TypeProduct | TypeCoproduct


# ---------------------------------------------------------------------------
# category patterns (for rule declarations)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CatPatternName:
    """A named category pattern element (variable or atom).

    In a rule declaration ``rule fwd(X, Y) : X/Y, Y => X``,
    ``X`` and ``Y`` are pattern variables (bound in the parameter
    list) and any other name would be a category atom constant.

    Parameters
    ----------
    name : str
        The identifier.
    line : int
        Source line.
    col : int
        Source column.
    """

    name: str
    line: int = 0
    col: int = 0


@dataclass(frozen=True)
class CatPatternSlash:
    """A slash category pattern: result/argument or result\\argument.

    Parameters
    ----------
    result : CatPattern
        The result category pattern.
    argument : CatPattern
        The argument category pattern.
    direction : str
        ``"/"`` (forward) or ``"\\\\"`` (backward).
    line : int
        Source line.
    col : int
        Source column.
    """

    result: CatPatternName | CatPatternSlash | CatPatternProduct
    argument: CatPatternName | CatPatternSlash | CatPatternProduct
    direction: str
    line: int = 0
    col: int = 0


@dataclass(frozen=True)
class CatPatternProduct:
    """A product category pattern: left * right.

    Parameters
    ----------
    left : CatPattern
        Left component.
    right : CatPattern
        Right component.
    line : int
        Source line.
    col : int
        Source column.
    """

    left: CatPatternName | CatPatternSlash | CatPatternProduct
    right: CatPatternName | CatPatternSlash | CatPatternProduct
    line: int = 0
    col: int = 0


# union of all category pattern nodes
CatPattern = CatPatternName | CatPatternSlash | CatPatternProduct


# ---------------------------------------------------------------------------
# space expressions (describe continuous spaces)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpaceConstructor:
    """Space constructor call: Euclidean(3) or Euclidean(2, low=0.0, high=1.0).

    Parameters
    ----------
    constructor : str
        Constructor name (e.g. "Euclidean", "Simplex").
    args : tuple[str, ...]
        Positional arguments as strings.
    kwargs : dict[str, str]
        Keyword arguments as string pairs.
    line : int
        Source line.
    col : int
        Source column.
    """

    constructor: str
    args: tuple[str, ...] = ()
    kwargs: dict[str, str] = field(default_factory=dict)
    line: int = 0
    col: int = 0


@dataclass(frozen=True)
class SpaceProduct:
    """Product space: A * B.

    Parameters
    ----------
    components : tuple of space expression nodes
        The component spaces.
    line : int
        Source line.
    col : int
        Source column.
    """

    components: tuple[SpaceConstructor | TypeName | SpaceProduct, ...]
    line: int = 0
    col: int = 0


# union of all space expression nodes
SpaceExpr = SpaceConstructor | TypeName | SpaceProduct


# ---------------------------------------------------------------------------
# value expressions (describe morphism computations)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExprIdent:
    """Reference to a named morphism or binding.

    Parameters
    ----------
    name : str
        The identifier.
    line : int
        Source line.
    col : int
        Source column.
    """

    name: str
    line: int = 0
    col: int = 0


@dataclass(frozen=True)
class ExprIdentity:
    """Identity morphism: identity(A).

    Parameters
    ----------
    object_name : str
        The object to form the identity on.
    line : int
        Source line.
    col : int
        Source column.
    """

    object_name: str
    line: int = 0
    col: int = 0


@dataclass(frozen=True)
class ExprCompose:
    """Sequential composition: left >> right.

    Parameters
    ----------
    left : Expr
        Left morphism (applied first).
    right : Expr
        Right morphism (applied second).
    line : int
        Source line.
    col : int
        Source column.
    """

    left: Expr
    right: Expr
    line: int = 0
    col: int = 0


@dataclass(frozen=True)
class ExprTensorProduct:
    """Parallel product: left @ right.

    Parameters
    ----------
    left : Expr
        Left morphism.
    right : Expr
        Right morphism.
    line : int
        Source line.
    col : int
        Source column.
    """

    left: Expr
    right: Expr
    line: int = 0
    col: int = 0


@dataclass(frozen=True)
class ExprMarginalize:
    """Marginalization: expr.marginalize(A, B, ...).

    Parameters
    ----------
    inner : Expr
        The morphism to marginalize.
    names : tuple[str, ...]
        Object names to marginalize over.
    line : int
        Source line.
    col : int
        Source column.
    """

    inner: Expr
    names: tuple[str, ...]
    line: int = 0
    col: int = 0


@dataclass(frozen=True)
class ExprFan:
    """Fan-out (diagonal) composition: fan(f, g, ...).

    Copies a single input to N morphisms and concatenates their
    outputs into a product space. If f: A -> B and g: A -> C,
    then fan(f, g): A -> B * C.

    Parameters
    ----------
    exprs : tuple[Expr, ...]
        The morphisms to fan out to.
    line : int
        Source line.
    col : int
        Source column.
    """

    exprs: tuple[Expr, ...]
    line: int = 0
    col: int = 0


@dataclass(frozen=True)
class ExprRepeat:
    """Iterated composition: repeat(f, n) or repeat(f).

    With a count: compile-time unrolling, ``repeat(f, 3) = f >> f >> f``.
    Without a count: runtime-variable ``RepeatMorphism`` whose step
    count can be set via ``n_steps`` before each forward pass.

    Parameters
    ----------
    expr : Expr
        The morphism to repeat.
    count : int or None
        Number of repetitions. ``None`` means runtime-variable.
    line : int
        Source line.
    col : int
        Source column.
    """

    expr: Expr
    count: int | None
    line: int = 0
    col: int = 0


@dataclass(frozen=True)
class ExprStack:
    """Independent multi-layer composition: stack(f, n).

    Create n deep copies of morphism f with independent parameters
    and compose them sequentially: f_1 >> f_2 >> ... >> f_n.

    Parameters
    ----------
    expr : Expr
        The morphism to replicate.
    count : int
        Number of independent copies (must be >= 1).
    line : int
        Source line.
    col : int
        Source column.
    """

    expr: Expr
    count: int
    line: int = 0
    col: int = 0


@dataclass(frozen=True)
class ExprScan:
    """Temporal scan: scan(cell) or scan(cell, init=learned).

    Apply a recurrent cell across a sequence, threading hidden
    state. The cell must have product domain A * H and codomain H.
    The scan produces a morphism A -> H that iterates over the
    time dimension of its input.

    Parameters
    ----------
    expr : Expr
        The cell morphism expression.
    init : str
        Initialization strategy: ``"zeros"`` or ``"learned"``.
    line : int
        Source line.
    col : int
        Source column.
    """

    expr: Expr
    init: str = "zeros"
    line: int = 0
    col: int = 0


@dataclass(frozen=True)
class ExprParser:
    """Deductive parser assembled from rules.

    Rules are resolved at compile time: each name in ``rules`` is
    looked up first in ``SCHEMA_REGISTRY`` (yielding a categorical
    rule schema functor) and then in the morphism environment
    (yielding a user-declared morphism whose type determines its
    role in the deductive system).

    When all rules are schemas, the compiler builds a chart parser
    over a generated category system.  When all rules are morphisms,
    their types are inspected — a morphism ``N → N ⊗ N`` contributes
    binary deductions, and ``N → T`` contributes lexical axioms —
    and the compiler assembles the appropriate deductive system.

    Parameters
    ----------
    rules : tuple of str
        Names of rule schema primitives or declared morphisms.
    categories : tuple of str
        Atomic category names supplied inline. If empty and rules
        are schemas, the compiler uses ``category`` declarations.
    terminal : str or None
        Name of the declared ``object`` serving as the terminal
        vocabulary. Required for schema-based parsers so that the
        lexicon (category → terminal distribution) is well-typed.
        If None, the compiler raises an error for schema rules.
    start : str or int
        Start category name or nonterminal index. Default "S".
    depth : int
        Maximum nesting depth for type constructors (default 1).
    constructors : tuple of str or None
        Type constructor names. If None, defaults to ["slash"].
    line : int
        Source line.
    col : int
        Source column.
    """

    rules: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    terminal: str | None = None
    start: str | int = "S"
    depth: int = 1
    constructors: tuple[str, ...] | None = None
    line: int = 0
    col: int = 0


# union of all expression nodes
Expr = (
    ExprIdent
    | ExprIdentity
    | ExprCompose
    | ExprTensorProduct
    | ExprMarginalize
    | ExprFan
    | ExprRepeat
    | ExprStack
    | ExprScan
    | ExprParser
)


# ---------------------------------------------------------------------------
# top-level statements
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuantaleDecl:
    """Quantale selection: quantale <name>.

    Parameters
    ----------
    name : str
        The quantale name (e.g. product_fuzzy, boolean, godel).
    line : int
        Source line.
    col : int
        Source column.
    """

    name: str
    line: int = 0
    col: int = 0


@dataclass(frozen=True)
class CategoryDecl:
    """Category atom declaration: category <name>.

    Declares an atomic category label for use in deductive parsers.
    Unlike ``object`` declarations (which define finite sets with
    cardinality), category atoms are generators for a free categorical
    structure — the ``CategorySystem`` built by the compiler from
    atoms, type constructors, and a depth bound.

    Parameters
    ----------
    name : str
        The category atom name.
    line : int
        Source line.
    col : int
        Source column.
    """

    name: str
    line: int = 0
    col: int = 0


@dataclass(frozen=True)
class RuleDecl:
    """Rule-of-inference declaration.

    Syntax::

        rule <name>(<var>, ...) : <premise>, ... => <conclusion>

    Declares a schematic inference rule. Variables in the parameter
    list are universally quantified over category atoms; the compiler
    generates all concrete instantiations when the rule is applied
    to a ``CategorySystem``.

    Binary rules have two premises (e.g., ``X/Y, Y => X``).
    Unary rules have one premise (e.g., ``A => B/(A\\B)``).

    Parameters
    ----------
    name : str
        The rule's binding name.
    variables : tuple of str
        Universally quantified pattern variables.
    premises : tuple of CatPattern
        Antecedent category patterns (1 for unary, 2 for binary).
    conclusion : CatPattern
        Consequent category pattern.
    line : int
        Source line.
    col : int
        Source column.
    """

    name: str
    variables: tuple[str, ...]
    premises: tuple[CatPattern, ...]
    conclusion: CatPattern
    line: int = 0
    col: int = 0


@dataclass(frozen=True)
class ObjectDecl:
    """Object declaration: object <name> : <type_expr>.

    Parameters
    ----------
    name : str
        The object's binding name.
    type_expr : TypeExpr
        The type expression describing the object.
    line : int
        Source line.
    col : int
        Source column.
    """

    name: str
    type_expr: TypeExpr
    line: int = 0
    col: int = 0


@dataclass(frozen=True)
class MorphismDecl:
    """Morphism declaration: latent|observed <name> : <dom> -> <cod> [= <expr>].

    Parameters
    ----------
    kind : str
        Either "latent" or "observed".
    name : str
        The morphism's binding name.
    domain : TypeExpr
        The domain type expression.
    codomain : TypeExpr
        The codomain type expression.
    init_expr : Expr or None
        Optional initialization expression (e.g. identity).
    options : dict[str, str]
        Optional key=value options (e.g. scale=0.3).
    line : int
        Source line.
    col : int
        Source column.
    """

    kind: str
    name: str
    domain: TypeExpr
    codomain: TypeExpr
    init_expr: Expr | None = None
    options: dict[str, str] = field(default_factory=dict)
    line: int = 0
    col: int = 0


@dataclass(frozen=True)
class SpaceDecl:
    """Space declaration: space <name> : <space_expr>.

    Parameters
    ----------
    name : str
        The space binding name.
    space_expr : SpaceExpr
        The space constructor or reference.
    line : int
        Source line.
    col : int
        Source column.
    """

    name: str
    space_expr: SpaceExpr
    line: int = 0
    col: int = 0


@dataclass(frozen=True)
class ContinuousMorphismDecl:
    """Continuous morphism: continuous <name>[N] : <dom> -> <cod> ~ <family> [opts].

    When ``replicate`` is set, N independent morphisms are created
    with names ``name_0`` through ``name_{N-1}``, and the base name
    is registered as a group reference.

    Parameters
    ----------
    name : str
        The morphism binding name (or base name if replicated).
    domain : TypeExpr
        Domain type expression (may be a product type like A * B).
    codomain : TypeExpr
        Codomain type expression.
    family : str
        Distribution family name (e.g. "Normal", "Dirichlet", "Flow").
    options : dict[str, str]
        Optional key=value options (e.g. n_layers=4, hidden_dim=32).
    replicate : int or None
        If set, create this many independent copies.
    line : int
        Source line.
    col : int
        Source column.
    """

    name: str
    domain: TypeExpr
    codomain: TypeExpr
    family: str
    options: dict[str, str] = field(default_factory=dict)
    replicate: int | None = None
    line: int = 0
    col: int = 0


@dataclass(frozen=True)
class StochasticMorphismDecl:
    """Stochastic morphism: stochastic <name>[N] : <dom> -> <cod>.

    Parameters
    ----------
    name : str
        The morphism binding name (or base name if replicated).
    domain : TypeExpr
        Domain type expression.
    codomain : TypeExpr
        Codomain type expression.
    replicate : int or None
        If set, create this many independent copies.
    line : int
        Source line.
    col : int
        Source column.
    """

    name: str
    domain: TypeExpr
    codomain: TypeExpr
    replicate: int | None = None
    line: int = 0
    col: int = 0


@dataclass(frozen=True)
class DiscretizeDecl:
    """Discretize boundary: discretize <name> : <space> -> <n_bins>.

    Parameters
    ----------
    name : str
        The morphism binding name.
    space_name : str
        The source continuous space name.
    n_bins : int
        Number of discrete bins.
    options : dict[str, str]
        Optional key=value options.
    line : int
        Source line.
    col : int
        Source column.
    """

    name: str
    space_name: str
    n_bins: int
    options: dict[str, str] = field(default_factory=dict)
    line: int = 0
    col: int = 0


@dataclass(frozen=True)
class EmbedDecl:
    """Embed boundary: embed <name>[N] : <finset> -> <space>.

    Parameters
    ----------
    name : str
        The morphism binding name (or base name if replicated).
    domain_name : str
        The source FinSet name.
    codomain_name : str
        The target continuous space name.
    replicate : int or None
        If set, create this many independent copies.
    line : int
        Source line.
    col : int
        Source column.
    """

    name: str
    domain_name: str
    codomain_name: str
    replicate: int | None = None
    line: int = 0
    col: int = 0


@dataclass(frozen=True)
class DrawStep:
    """A single draw step inside a program block.

    Supports single bindings (``draw x ~ f``) and destructuring
    (``draw (x, y) ~ f``), as well as single or multiple arguments
    (``f(x)`` or ``f(x, y)``). Also supports the ``observe`` keyword
    to mark sites as observed for inference.

    Arguments may be variable names (str) or float literals for
    inline distribution specifications like ``draw x ~ Normal(0.0, 1.0)``.

    Parameters
    ----------
    vars : tuple[str, ...]
        Bound variable name(s). Single-element tuple for simple
        binding, multi-element for destructuring.
    morphism : str
        Name of the morphism or distribution family to sample from.
    args : tuple[str | float, ...] or None
        Arguments: bound variable names (str) and/or literal values
        (float) to apply morphism to, or None for program input.
    is_observed : bool
        Whether this site is observed (from ``observe`` keyword).
    line : int
        Source line.
    col : int
        Source column.
    """

    vars: tuple[str, ...]
    morphism: str
    args: tuple[str | float, ...] | None = None
    is_observed: bool = False
    line: int = 0
    col: int = 0


@dataclass(frozen=True)
class LetExprBinOp:
    """Binary arithmetic operation in a let expression.

    Parameters
    ----------
    op : str
        Operator: '+', '-', '*', '/'.
    left : LetExprNode
        Left operand.
    right : LetExprNode
        Right operand.
    """

    op: str
    left: LetExprNode
    right: LetExprNode


@dataclass(frozen=True)
class LetExprUnaryOp:
    """Unary negation in a let expression.

    Parameters
    ----------
    operand : LetExprNode
        The operand to negate.
    """

    operand: LetExprNode


@dataclass(frozen=True)
class LetExprCall:
    """Built-in function call in a let expression.

    Parameters
    ----------
    func : str
        Function name (sigmoid, exp, log, abs, softplus).
    args : tuple[LetExprNode, ...]
        Function arguments.
    """

    func: str
    args: tuple[LetExprNode, ...]


@dataclass(frozen=True)
class LetExprLiteral:
    """Numeric literal in a let expression.

    Parameters
    ----------
    value : float
        The literal value.
    """

    value: float


@dataclass(frozen=True)
class LetExprVar:
    """Variable reference in a let expression.

    Parameters
    ----------
    name : str
        The variable name.
    """

    name: str


# union type for let expression nodes
LetExprNode = LetExprBinOp | LetExprUnaryOp | LetExprCall | LetExprLiteral | LetExprVar


@dataclass(frozen=True)
class LetStep:
    """A deterministic let binding inside a program block.

    Binds a variable to a constant literal, a variable alias, or
    an arithmetic expression over bound variables and built-in
    functions. Does not involve any distribution and contributes
    zero to the joint log-density.

    Examples::

        let cg_complement = 1
        let alias = theta_know
        let threshold = 0.5
        let logit = log(p) - log(1.0 - p)
        let combined = x * 0.5 + y * 0.5

    Parameters
    ----------
    name : str
        Variable name to bind.
    value : float or str or LetExprNode
        Constant literal (float), name of a previously bound
        variable (str), or an expression tree (LetExprNode).
    line : int
        Source line.
    col : int
        Source column.
    """

    name: str
    value: float | str | LetExprNode
    line: int = 0
    col: int = 0


@dataclass(frozen=True)
class ProgramDecl:
    """Monadic program block with optional named params and tuple returns.

    A program block defines a ContinuousMorphism via monadic sequencing
    of draw and let steps. Supports product type domains/codomains,
    named input parameters for sub-programs, tuple returns, and
    deterministic let bindings.

    Examples::

        program p : A -> B
            draw x ~ f
            return x

        program sub(y, z) : Belief * Belief -> Truth * Truth
            draw c ~ bern_c(y)
            draw d ~ bern_d(z)
            return (c, d)

        program model : Entity -> Truth * Truth * Truth
            draw x ~ LogitNormal(0.0, 1.0)
            let cg = 1
            draw b ~ Bernoulli(x)
            return (state: b, cg: cg, prob: x)

    Parameters
    ----------
    name : str
        The program binding name.
    params : tuple[str, ...] or None
        Named input parameters (for product-domain sub-programs).
    domain : TypeExpr
        Domain type expression (may be a product type).
    codomain : TypeExpr
        Codomain type expression (may be a product type).
    draws : tuple[DrawStep | LetStep, ...]
        The draw and let steps in order.
    return_vars : tuple[str, ...]
        Bound variable name(s) to return. Single-element for simple
        return, multi-element for tuple return.
    return_labels : tuple[str, ...] or None
        Optional labels for tuple return fields. When set, the
        output dict uses these labels as keys instead of the
        variable names.
    line : int
        Source line.
    col : int
        Source column.
    """

    name: str
    params: tuple[str, ...] | None
    domain: TypeExpr
    codomain: TypeExpr
    draws: tuple[DrawStep | LetStep, ...]
    return_vars: tuple[str, ...]
    return_labels: tuple[str, ...] | None = None
    line: int = 0
    col: int = 0


@dataclass(frozen=True)
class LetDecl:
    """Let binding: let <name> = <expr> [where let_decl+].

    Parameters
    ----------
    name : str
        The binding name.
    expr : Expr
        The bound expression.
    where : tuple[LetDecl, ...] | None
        Optional where clause bindings.
    line : int
        Source line.
    col : int
        Source column.
    """

    name: str
    expr: Expr
    where: tuple[LetDecl, ...] | None = None
    line: int = 0
    col: int = 0


@dataclass(frozen=True)
class OutputDecl:
    """Output declaration: output <expr>.

    Parameters
    ----------
    expr : Expr
        The expression to compile as the program output.
    line : int
        Source line.
    col : int
        Source column.
    """

    expr: Expr
    line: int = 0
    col: int = 0


# union of all statement types
Statement = (
    QuantaleDecl
    | CategoryDecl
    | RuleDecl
    | ObjectDecl
    | MorphismDecl
    | SpaceDecl
    | ContinuousMorphismDecl
    | StochasticMorphismDecl
    | DiscretizeDecl
    | EmbedDecl
    | ProgramDecl
    | LetDecl
    | OutputDecl
)


@dataclass
class Module:
    """A complete .qvr program (sequence of statements).

    Parameters
    ----------
    statements : list[Statement]
        The top-level statements in order.
    """

    statements: list[Statement] = field(default_factory=list)
