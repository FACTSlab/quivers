"""JSON round-trip tests for every migrated didactic Model.

Each test case is a representative instance of a Model variant; the
roundtrip fixture dumps it to JSON, validates back from JSON, and asserts
equality. This pins didactic's serialization contract for the migrated
classes and surfaces any encoding/decoding asymmetries.

didactic 0.5.0 uses the union root's :py:meth:`model_validate_json` for
TaggedUnion dispatch on the ``kind`` discriminator; flat dx.Model
subclasses use their own.
"""

import pytest

# AST nodes
from quivers.dsl.ast_nodes import (
    CatPattern,
    CatPatternName,
    CatPatternProduct,
    CatPatternSlash,
    CategoryDecl,
    ContinuousMorphismDecl,
    DiscretizeDecl,
    DrawStep,
    EmbedDecl,
    Expr,
    ExprCompose,
    ExprFan,
    ExprIdent,
    ExprIdentity,
    ExprMarginalize,
    ExprParser,
    ExprRepeat,
    ExprScan,
    ExprStack,
    ExprTensorProduct,
    LetDecl,
    LetExprBinOp,
    LetExprCall,
    LetExprLiteral,
    LetExprNode,
    LetExprUnaryOp,
    LetExprVar,
    LetStep,
    Module,
    MorphismDecl,
    ObjectDecl,
    OutputDecl,
    ProgramDecl,
    ProgramStep,
    QuantaleDecl,
    RuleDecl,
    SpaceConstructor,
    SpaceDecl,
    SpaceExpr,
    SpaceName,
    SpaceProduct,
    Statement,
    StochasticMorphismDecl,
    TypeCoproduct,
    TypeExpr,
    TypeName,
    TypeProduct,
)

# stochastic Category union
from quivers.stochastic.categories import (
    AtomicCategory,
    Category,
    ModalCategory,
    ProductCategory,
    SlashCategory,
    UnitCategory,
)

# RuleSystem
from quivers.stochastic._rule_system import RuleSystem

# core/objects
from quivers.core.objects import (
    CoproductSet,
    FinSet,
    FreeMonoid,
    ProductSet,
    SetObject,
)

# categorical/monoidal
from quivers.categorical.monoidal import EmptySet

# continuous/spaces
from quivers.continuous.spaces import (
    ContinuousSpace,
    Euclidean,
    PositiveReals,
    ProductSpace,
    Simplex,
)


# ---------------------------------------------------------------------------
# instances
# ---------------------------------------------------------------------------


# AST: type expressions
_TYPE_NAME = TypeName(name="State")
_TYPE_PRODUCT = TypeProduct(components=(TypeName(name="A"), TypeName(name="B")))
_TYPE_COPRODUCT = TypeCoproduct(components=(_TYPE_NAME, TypeName(name="X")))

# AST: cat patterns
_CAT_NAME = CatPatternName(name="X")
_CAT_SLASH = CatPatternSlash(
    result=_CAT_NAME, argument=CatPatternName(name="Y"), direction="/"
)
_CAT_PRODUCT = CatPatternProduct(left=_CAT_NAME, right=CatPatternName(name="Z"))

# AST: space expressions
_SPACE_NAME = SpaceName(name="R3")
_SPACE_CTOR = SpaceConstructor(
    constructor="Euclidean", args=("3",), kwargs={"low": "0.0", "high": "1.0"}
)
_SPACE_PRODUCT = SpaceProduct(components=(_SPACE_NAME, _SPACE_CTOR))

# AST: value expressions
_E_IDENT = ExprIdent(name="f")
_E_IDENTITY = ExprIdentity(object_name="A")
_E_COMPOSE = ExprCompose(left=_E_IDENT, right=ExprIdent(name="g"))
_E_TENSOR = ExprTensorProduct(left=_E_IDENT, right=ExprIdent(name="g"))
_E_FAN = ExprFan(exprs=(_E_IDENT, ExprIdent(name="g")))
_E_REPEAT = ExprRepeat(expr=_E_IDENT, count=3)
_E_STACK = ExprStack(expr=_E_IDENT, count=4)
_E_SCAN = ExprScan(expr=_E_IDENT, init="zeros")
_E_MARG = ExprMarginalize(inner=_E_IDENT, names=("A",))
_E_PARSER = ExprParser(
    rules=("evaluation",),
    categories=("S", "NP"),
    terminal="Token",
    start="S",
    depth=1,
)

# AST: let-expr nodes
_LE_LIT = LetExprLiteral(value=0.5)
_LE_VAR = LetExprVar(name="theta")
_LE_BINOP = LetExprBinOp(op="+", left=_LE_VAR, right=_LE_LIT)
_LE_UNARY = LetExprUnaryOp(operand=_LE_VAR)
_LE_CALL = LetExprCall(func="log", args=(_LE_VAR,))

# AST: program steps
_DRAW = DrawStep(vars=("x",), morphism="f", args=None)
_LET_STEP = LetStep(name="y", value=_LE_BINOP)

# AST: top-level statements
_QDECL = QuantaleDecl(name="product_fuzzy")
_CDECL = CategoryDecl(name="S")
_RDECL = RuleDecl(
    name="app",
    variables=("X", "Y"),
    premises=(_CAT_SLASH, CatPatternName(name="Y")),
    conclusion=CatPatternName(name="X"),
)
_ODECL = ObjectDecl(name="State", type_expr=_TYPE_NAME)
_MDECL = MorphismDecl(
    morphism_kind="latent",
    name="f",
    domain=_TYPE_NAME,
    codomain=_TYPE_NAME,
    options={"scale": "0.3"},
)
_SDECL = SpaceDecl(name="R3", space_expr=_SPACE_CTOR)
_CMDECL = ContinuousMorphismDecl(
    name="g",
    domain=_TYPE_NAME,
    codomain=_TYPE_NAME,
    family="Normal",
    options={},
)
_SMDECL = StochasticMorphismDecl(
    name="t", domain=_TYPE_NAME, codomain=_TYPE_NAME
)
_DDECL = DiscretizeDecl(name="d", space_name="R3", n_bins=10, options={})
_EDECL = EmbedDecl(name="e", domain_name="X", codomain_name="R3")
_PDECL = ProgramDecl(
    name="p",
    params=None,
    domain=_TYPE_NAME,
    codomain=_TYPE_NAME,
    draws=(_DRAW,),
    return_vars=("x",),
)
_LDECL = LetDecl(name="h", expr=_E_IDENT)
_OUTDECL = OutputDecl(expr=_E_IDENT)

_MODULE = Module(statements=(_QDECL, _ODECL, _MDECL, _OUTDECL))

# stochastic categories
_CAT_ATOM = AtomicCategory(name="S")
_CAT_SLASH_RT = SlashCategory(
    result=_CAT_ATOM, argument=AtomicCategory(name="NP"), direction="/"
)
_CAT_PROD_RT = ProductCategory(left=_CAT_ATOM, right=AtomicCategory(name="N"))
_CAT_UNIT = UnitCategory()
_CAT_MODAL = ModalCategory(modality="◇", inner=_CAT_ATOM)

# RuleSystem
_RULE_SYS = RuleSystem(
    binary_rules=((0, 1, 2),),
    unary_rules=((0, 1),),
    n_categories=3,
    description="demo",
    binary_weights=(1.0,),
    unary_weights=(0.5,),
)

# core objects
_FINSET = FinSet(name="X", cardinality=4)
_PRODUCT_SET = ProductSet(components=(_FINSET, FinSet(name="Y", cardinality=3)))
_COPRODUCT_SET = CoproductSet(components=(_FINSET, _FINSET))
_FREE_MONOID = FreeMonoid(generators=_FINSET, max_length=2)

# EmptySet (categorical/monoidal)
_EMPTY = EmptySet()

# continuous spaces
_EUCLID = Euclidean(name="x", dim=3)
_EUCLID_BOUNDED = Euclidean(name="u", dim=2, low=0.0, high=1.0)
_SIMPLEX = Simplex(name="probs", dim=4)
_POSREAL = PositiveReals(name="sigma", dim=1)
_PRODUCT_SPACE = ProductSpace(components=(_EUCLID, _SIMPLEX))


# ---------------------------------------------------------------------------
# (root, instance) pairs for parametrization
#
# For TaggedUnion variants, the *root* is the class with model_validate_json;
# the instance must round-trip via the root for discriminator dispatch.
# Flat Models use themselves as the root.
# ---------------------------------------------------------------------------


CASES: list[tuple[type, object]] = [
    # AST: type expressions
    (TypeExpr, _TYPE_NAME),
    (TypeExpr, _TYPE_PRODUCT),
    (TypeExpr, _TYPE_COPRODUCT),
    # AST: cat patterns
    (CatPattern, _CAT_NAME),
    (CatPattern, _CAT_SLASH),
    (CatPattern, _CAT_PRODUCT),
    # AST: space expressions
    (SpaceExpr, _SPACE_NAME),
    (SpaceExpr, _SPACE_CTOR),
    (SpaceExpr, _SPACE_PRODUCT),
    # AST: value expressions
    (Expr, _E_IDENT),
    (Expr, _E_IDENTITY),
    (Expr, _E_COMPOSE),
    (Expr, _E_TENSOR),
    (Expr, _E_FAN),
    (Expr, _E_REPEAT),
    (Expr, _E_STACK),
    (Expr, _E_SCAN),
    (Expr, _E_MARG),
    (Expr, _E_PARSER),
    # AST: let-expr nodes
    (LetExprNode, _LE_LIT),
    (LetExprNode, _LE_VAR),
    (LetExprNode, _LE_BINOP),
    (LetExprNode, _LE_UNARY),
    (LetExprNode, _LE_CALL),
    # AST: program steps
    (ProgramStep, _DRAW),
    (ProgramStep, _LET_STEP),
    # AST: top-level statements
    (Statement, _QDECL),
    (Statement, _CDECL),
    (Statement, _RDECL),
    (Statement, _ODECL),
    (Statement, _MDECL),
    (Statement, _SDECL),
    (Statement, _CMDECL),
    (Statement, _SMDECL),
    (Statement, _DDECL),
    (Statement, _EDECL),
    (Statement, _PDECL),
    (Statement, _LDECL),
    (Statement, _OUTDECL),
    # AST: module
    (Module, _MODULE),
    # stochastic categories
    (Category, _CAT_ATOM),
    (Category, _CAT_SLASH_RT),
    (Category, _CAT_PROD_RT),
    (Category, _CAT_UNIT),
    (Category, _CAT_MODAL),
    # RuleSystem
    (RuleSystem, _RULE_SYS),
    # core objects
    (SetObject, _FINSET),
    (SetObject, _PRODUCT_SET),
    (SetObject, _COPRODUCT_SET),
    (SetObject, _FREE_MONOID),
    (SetObject, _EMPTY),
    # continuous spaces
    (ContinuousSpace, _EUCLID),
    (ContinuousSpace, _EUCLID_BOUNDED),
    (ContinuousSpace, _SIMPLEX),
    (ContinuousSpace, _POSREAL),
    (ContinuousSpace, _PRODUCT_SPACE),
]


def _id(case: tuple[type, object]) -> str:
    root, instance = case
    return f"{root.__name__}-{type(instance).__name__}"


@pytest.mark.parametrize("root,instance", CASES, ids=[_id(c) for c in CASES])
def test_json_roundtrip(root: type, instance: object) -> None:
    """Each migrated dx.Model survives a JSON round-trip via its root."""
    raw = instance.model_dump_json()  # type: ignore[attr-defined]
    parsed = root.model_validate_json(raw)  # type: ignore[attr-defined]
    assert parsed == instance, (
        f"round-trip mismatch for {type(instance).__name__} via {root.__name__}\n"
        f"  json   = {raw}\n"
        f"  parsed = {parsed}\n"
        f"  expect = {instance}"
    )
