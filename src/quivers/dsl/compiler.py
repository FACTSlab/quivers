"""Compiler: transform a quivers DSL AST into a trainable Program.

The compiler walks the AST in declaration order, building up an
environment of objects, spaces, and morphisms, then compiles the
output expression into a quivers.Program (nn.Module).

Supports both discrete (FinSet-based) and continuous (ContinuousSpace-
based) morphisms, including stochastic (Markov kernels), boundary
(Discretize/Embed), and parameterized family distributions.
"""

from __future__ import annotations
from collections.abc import Callable
from typing import Any  # pre-existing usages; new code should use precise types

import torch
import torch.nn as nn
from quivers.continuous.spaces import (
    ContinuousSpace,
    Euclidean,
    PositiveReals,
    Simplex,
    UnitInterval,
)
from quivers.continuous.morphisms import AnySpace
from quivers.continuous.plate import marginalize_grouped
from quivers.core.objects import SetObject, FinSet, ProductSet
from quivers.core.quantales import (
    BOOLEAN,
    BilinearForm,
    CompositionRule,
    CustomBilinearForm,
    CustomQuantale,
    CustomSemigroupoid,
    PRODUCT_FUZZY,
    Quantale,
    Semigroupoid,
    material_implication,
)
from quivers.core.wiring import EinsumWiring
from quivers.core.trans import TransSeq
from quivers.core.morphism_transformations import (
    bayes_invert as _bayes_invert,
    l1_normalize as _l1_normalize,
    l2_normalize as _l2_normalize,
    softmax as _softmax,
)
from quivers.core.morphisms import morphism as make_latent, identity as make_identity
from quivers.core.morphisms import cap as _make_cap, cup as _make_cup
from quivers.core.quantale_morphisms import (
    COUNTING_FROM_REAL,
    COUNTING_TO_REAL,
    EXPECTATION,
    LOG_PROB as _LOG_PROB_HOM,
    MATERIAL_IMPLICATION,
    MAX_PLUS as _MAX_PLUS_HOM,
    PROBABILITY_CLAMP,
    PROBABILITY_TO_REAL,
    embedding as _make_embedding,
    threshold as _make_threshold,
)


from quivers.program import Program
from quivers.structural.encoder import (
    Encoder,
    _PerOpFn,
    make_default_op_fn,
    make_default_var_init,
)
from quivers.structural.decoder import Decoder
from quivers.stochastic import StochasticMorphism
from quivers.stochastic.agenda import (
    DeductionSystem,
    InferenceRule,
    Wildcard,
    cky_agenda,
    depth_first_agenda,
    semi_naive_agenda,
)
from quivers.stochastic.semiring import (
    BOOLEAN as SEMIRING_BOOLEAN,
    COUNTING as SEMIRING_COUNTING,
    LOG_PROB as SEMIRING_LOG_PROB,
    VITERBI as SEMIRING_VITERBI,
)
from quivers.structural.losses import LossEntry, LossRegistry
from quivers.structural.signature import (
    Binder,
    BinderArgSpec,
    BinderVarSpec,
    Constructor,
    EdgeKind,
    Signature,
    Sort,
    SortVocabEntry,
    VertexKind,
)
from quivers.dsl.ast_nodes import SortVocabLiteral
from quivers.dsl.ast_nodes import (
    AxisSpec,
    Module,
    MorphismPrior,
    Statement,
    CompositionRuleEntry,
    ContractionDecl,
    QuantaleDecl,
    CategoryDecl,
    RuleDecl,
    ObjectDecl,
    MorphismDecl,
    SpaceDecl,
    KernelDecl,
    AliasDecl,
    BundleDecl,
    DiscretizeDecl,
    EmbedDecl,
    EnumSetLiteral,
    FreeMonoidExpr,
    FreeResiduatedExpr,
    SchemaDecl,
    BindStep,
    EncoderDecl,
    DecoderDecl,
    DeductionDecl,
    LossDecl,
    SignatureDecl,
    DrawStep,
    ExportDecl,
    TypeEffectApply,
    TypeSlash,
    LetStep,
    LetExprBinOp,
    LetExprIndex,
    LetExprLambda,
    LetExprList,
    LetExprMethodCall,
    LetExprString,
    LetExprUnaryOp,
    LetExprCall,
    LetExprLiteral,
    LetExprVar,
    LetExprNode,
    GroupedBodyObserveStep,
    GroupedObserveEntry,
    GroupedLatentInitStep,
    MarginalizeStep,
    MorphismParam,
    ObjectParam,
    PlateDrawStep,
    ProgramDecl,
    ProgramStep,
    ScalarParam,
    VectorisedObserveStep,
    LetDecl,
    TypeExpr,
    TypeName,
    TypeProduct,
    SpaceExpr,
    SpaceConstructor,
    Expr,
    ExprIdent,
    ExprIdentity,
    ExprCompose,
    ExprTensorProduct,
    ExprChartFold,
    ExprCurry,
    ExprCap,
    ExprChangeBase,
    ExprTransCompose,
    ExprCup,
    ExprDagger,
    ExprFreeze,
    ExprFromData,
    ExprMarginalize,
    ExprMorphismCall,
    ExprTrace,
    ExprFan,
    ExprRepeat,
    ExprStack,
    ExprScan,
    ExprParser,
)

# Registry of composition rules the DSL knows about by name.
# Each entry maps a user-facing keyword (the right-hand side of
# ``quantale X``, ``semigroupoid X``, ``bilinear_form X``,
# ``composition_rule X``) to its concrete rule instance.  The
# type is widened to ``CompositionRule`` (the common parent of
# ``Quantale``, ``Semigroupoid``, and ``BilinearForm``) so
# user-defined non-quantale composition rules can be registered
# alongside the shipped ones.
_QUANTALE_REGISTRY: dict[str, "CompositionRule"] = {
    "product_fuzzy": PRODUCT_FUZZY,
    "boolean": BOOLEAN,
}


class _CompiledContraction:
    """Compiled form of a :class:`ContractionDecl`.

    Stores the einsum wiring object, the declared output domain
    and codomain (so the compiler can re-tag the result morphism
    after the contraction), the expected input morphism types
    (for shape validation), and the active composition rule.

    A plain class (not a :class:`didactic.api.Model`) because the
    fields are opaque runtime objects — :class:`EinsumWiring`,
    :class:`SetObject` / :class:`ContinuousSpace`,
    :class:`CompositionRule` instances — that don't translate to a
    panproto sort.
    """

    __slots__ = ("name", "wiring", "domain", "codomain", "input_types", "quantale")

    def __init__(
        self,
        name: str,
        wiring: "EinsumWiring",
        domain: "AnySpace",
        codomain: "AnySpace",
        input_types: tuple[tuple[str, "AnySpace", "AnySpace"], ...],
        quantale: "CompositionRule",
    ) -> None:
        self.name = name
        self.wiring = wiring
        self.domain = domain
        self.codomain = codomain
        self.input_types = input_types
        self.quantale = quantale


def _numel_shape(shape) -> int:
    """Total cardinality of an object's shape — used to compare
    declared vs actual morphism arities at contraction call sites."""
    n = 1
    for s in shape:
        n *= int(s)
    return n


def _wrap_join_dim(user_join):
    """Wrap a user-supplied ``join`` function so it accepts the
    ``(tensor, dim)`` calling convention expected by the
    :class:`CompositionRule` interface.

    User-defined ``join(t) = expr`` bodies see ``t`` as a tensor
    and reduce over its last axes via let-expression builtins
    like ``sum(t)``, ``prod(t)``, ``logsumexp(t)``. The compose
    machinery, however, calls ``join(tensor, dim)`` with an
    explicit reduction axis. We bridge by reshaping the tensor
    so the contracted axes collapse to a single trailing axis,
    invoking ``user_join`` on the reshaped tensor, and reshaping
    the result back.
    """

    def _join(t, dim):
        if isinstance(dim, int):
            dims: tuple[int, ...] = (dim,)
        else:
            dims = tuple(dim)
        # Sort dims, normalize negatives.
        ndim = t.dim()
        dims = tuple(sorted((d % ndim for d in dims), reverse=True))
        # Permute the dims-to-reduce to the trailing axes.
        keep = [i for i in range(ndim) if i not in dims]
        permuted = t.permute(*keep, *reversed(dims))
        # Collapse the trailing reduce-axes into one.
        keep_shape = permuted.shape[: len(keep)]
        flat = permuted.reshape(*keep_shape, -1)
        # The user's reducer takes only one argument; we feed
        # the flattened tensor and accept whatever they reduce.
        reduced = user_join(flat)
        return reduced

    return _join


def _build_default_trans_singletons() -> dict:
    """Built-in transformation singletons available as bare-name
    references in the DSL.

    Each value is a :class:`QuantaleHomomorphism` or
    :class:`MorphismTransformation` (collectively, a ``Trans``
    value).  Bare-name lookup produces the singleton:

        let phi = expectation
        let g = f.change_base(phi)
    """
    return {
        "expectation": EXPECTATION,
        "log_prob": _LOG_PROB_HOM,
        "max_plus": _MAX_PLUS_HOM,
        "material_implication": MATERIAL_IMPLICATION,
        "threshold": _make_threshold(0.5),
        "boolean_embedding": _make_embedding(BOOLEAN, PRODUCT_FUZZY),
        "probability_clamp": PROBABILITY_CLAMP,
        "probability_to_real": PROBABILITY_TO_REAL,
        "counting_from_real": COUNTING_FROM_REAL,
        "counting_to_real": COUNTING_TO_REAL,
    }


def _build_default_trans_constructors() -> dict:
    """Built-in transformation constructors — callables that
    accept compile-time arguments (objects, morphisms) resolved
    from the DSL scope and return a :class:`MorphismTransformation`.

    Surface form ``softmax(B)`` / ``bayes_invert(prior)`` parses
    as :class:`ExprMorphismCall` and dispatches here when the
    callee resolves into this dict.
    """
    return {
        "softmax": _softmax,
        "l1_normalize": _l1_normalize,
        "l2_normalize": _l2_normalize,
        "bayes_invert": _bayes_invert,
    }


def _register_extra_quantales() -> None:
    """Lazily register every shipped quantale into the
    ``quantale <name>`` resolution table the DSL uses at module
    top.

    The registration is idempotent and short-circuits when the
    table is already populated. Catching :class:`ImportError`
    keeps the compiler usable for users who don't have the
    optional dependencies (e.g. the stochastic module pulls in
    ``torch.distributions`` heavily).
    """
    if "lukasiewicz" not in _QUANTALE_REGISTRY:
        try:
            from quivers.core.quantales import (
                COUNTING,
                DUAL_GODEL,
                DUAL_LUKASIEWICZ,
                GODEL,
                LOG_PROB,
                LUKASIEWICZ,
                MAX_PLUS,
                PROBABILITY,
                REAL,
                TROPICAL,
            )
            from quivers.core.quantales import BOOLEAN_DUAL, REICHENBACH

            _QUANTALE_REGISTRY["lukasiewicz"] = LUKASIEWICZ
            _QUANTALE_REGISTRY["godel"] = GODEL
            _QUANTALE_REGISTRY["tropical"] = TROPICAL
            _QUANTALE_REGISTRY["max_plus"] = MAX_PLUS
            _QUANTALE_REGISTRY["log_prob"] = LOG_PROB
            _QUANTALE_REGISTRY["real"] = REAL
            _QUANTALE_REGISTRY["probability"] = PROBABILITY
            _QUANTALE_REGISTRY["counting"] = COUNTING
            # Built-in non-quantale composition rules.
            _QUANTALE_REGISTRY["material_impl"] = material_implication()
            # Named de-Morgan duals — each is the corresponding
            # ``X.dual()`` exposed under a DSL-friendly name so a
            # user can write ``quantale reichenbach``.
            _QUANTALE_REGISTRY["reichenbach"] = REICHENBACH
            _QUANTALE_REGISTRY["boolean_dual"] = BOOLEAN_DUAL
            _QUANTALE_REGISTRY["dual_lukasiewicz"] = DUAL_LUKASIEWICZ
            _QUANTALE_REGISTRY["dual_godel"] = DUAL_GODEL
        except ImportError:
            pass
    if "markov" not in _QUANTALE_REGISTRY:
        try:
            from quivers.stochastic import MARKOV

            _QUANTALE_REGISTRY["markov"] = MARKOV
        except ImportError:
            pass


_FAMILY_REGISTRY: dict[str, type] | None = None


# Event rank declared by each registered family.  Used by the
# compiler to validate axis-role clauses (`over <axes> [iid over
# <axes>]`): the number of names in `over` must equal the family's
# event rank.  Scalar families have rank 0 (every codomain axis is
# iid); vector families have rank 1; matrix families have rank 2.
# Multivariate, matrix, and correlation families that take more
# than one named event axis declare it here; everything else
# defaults to 0 in the lookup helper.
_FAMILY_EVENT_RANK: dict[str, int] = {
    "MultivariateNormal": 1,
    "LowRankMVN": 1,
    "Dirichlet": 1,
    "OneHotCategorical": 1,
    "RelaxedOneHotCategorical": 1,
    "LogisticNormal": 1,
    "Wishart": 2,
    "InverseWishart": 2,
    "MatrixNormal": 2,
    "LKJCholesky": 2,
    "GP": 1,
    "Horseshoe": 1,
}


def _family_event_rank(family_name: str) -> int:
    """Return the declared event rank of a family (0 for scalar)."""
    return _FAMILY_EVENT_RANK.get(family_name, 0)


def _shape_size(obj) -> int:
    """Return the total size of a SetObject or ContinuousSpace.

    For a SetObject this is the cardinality; for a ContinuousSpace
    it is the product of its declared shape dimensions.
    """
    if hasattr(obj, "cardinality"):
        return int(obj.cardinality)
    if hasattr(obj, "dim"):
        return int(obj.dim)
    shape = getattr(obj, "shape", None)
    if shape is not None:
        size = 1
        for d in shape:
            size *= int(d)
        return size
    raise TypeError(
        f"object {obj!r} has no cardinality / dim / shape; cannot "
        f"determine size for axis-role lookup"
    )


def _type_factor_names(texpr) -> tuple[bool, tuple[str, ...]]:
    """Extract the axis names from a TypeExpr.

    Returns ``(is_singleton, names)`` where ``is_singleton`` is True
    when the type is a single unfactored object (the ``dom``/``cod``
    shortcuts are legal) and ``names`` is the tuple of declared
    factor names (for product types each component must be a named
    object).

    The DSL surface treats axes as named components of the
    surrounding morphism's dom/cod.  A product type ``A * B`` has
    factor names ``("A", "B")``; a single unfactored type has zero
    factor names and admits the shortcut.  Parametric type
    constructors like ``Euclidean(D)`` are treated as singletons
    whose axis is the construct's argument name (``D``).
    """
    if isinstance(texpr, TypeName):
        return True, (texpr.name,)
    if isinstance(texpr, TypeEffectApply):
        # e.g. Euclidean(D): axis name is the argument's name when
        # it's itself a TypeName.
        args = getattr(texpr, "args", None)
        if args and len(args) == 1 and isinstance(args[0], TypeName):
            return True, (args[0].name,)
        return True, ()
    if isinstance(texpr, TypeProduct):
        names = []
        for c in texpr.components:
            _, sub = _type_factor_names(c)
            names.extend(sub)
        return False, tuple(names)
    return False, ()


def _available_axes_for(dom, cod) -> set[str]:
    """The legal axis names for an axis-role clause on a morphism
    declaration with the given dom and cod TypeExprs.

    Returns the union of:
      - ``dom`` shortcut if the dom is a single unfactored object,
      - ``cod`` shortcut if the cod is a single unfactored object,
      - every named factor of dom and cod.
    """
    dom_singleton, dom_names = _type_factor_names(dom)
    cod_singleton, cod_names = _type_factor_names(cod)
    out: set[str] = set(dom_names) | set(cod_names)
    if dom_singleton:
        out.add("dom")
    if cod_singleton:
        out.add("cod")
    return out


def _validate_axis_spec(
    axes: AxisSpec,
    family_name: str,
    available_axes: set[str],
    line: int,
    col: int,
) -> None:
    """Reject malformed axis-role clauses at compile time.

    Checks:

    1. Every name in ``over`` and ``iid_over`` is one of the
       ``available_axes`` (the named factors of the surrounding
       morphism's dom/cod plus the ``dom``/``cod`` shortcuts when
       legal).
    2. ``over`` and ``iid_over`` are disjoint.
    3. ``len(over) == family.event_rank``.  Mismatched arity is an
       error rather than a silent reinterpretation: a flat MVN
       over ``dim(A)*dim(B)`` and a MatrixNormal over ``(A, B)``
       are categorically distinct families with different
       covariance structures.
    """
    expected = _family_event_rank(family_name)
    if len(axes.over) != expected:
        raise CompileError(
            f"axis-role clause: family {family_name!r} has event_rank "
            f"{expected}, but `over` lists {len(axes.over)} axis name(s)",
            line, col,
        )
    bad = [a for a in axes.over if a not in available_axes]
    if bad:
        raise CompileError(
            f"axis-role clause: unknown axis name(s) {bad!r} in `over`; "
            f"available: {sorted(available_axes)}",
            line, col,
        )
    bad_iid = [a for a in axes.iid_over if a not in available_axes]
    if bad_iid:
        raise CompileError(
            f"axis-role clause: unknown axis name(s) {bad_iid!r} in "
            f"`iid over`; available: {sorted(available_axes)}",
            line, col,
        )
    overlap = set(axes.over) & set(axes.iid_over)
    if overlap:
        raise CompileError(
            f"axis-role clause: axes {sorted(overlap)!r} appear in both "
            f"`over` and `iid over`; each axis has exactly one role",
            line, col,
        )


def _get_family_registry() -> dict[str, type]:
    """Lazily build the distribution family registry."""
    global _FAMILY_REGISTRY
    if _FAMILY_REGISTRY is not None:
        return _FAMILY_REGISTRY
    from quivers.continuous.families import (
        ConditionalNormal,
        ConditionalLogitNormal,
        ConditionalBeta,
        ConditionalTruncatedNormal,
        ConditionalDirichlet,
        ConditionalCauchy,
        ConditionalLaplace,
        ConditionalGumbel,
        ConditionalLogNormal,
        ConditionalStudentT,
        ConditionalExponential,
        ConditionalGamma,
        ConditionalChi2,
        ConditionalHalfCauchy,
        ConditionalHalfNormal,
        ConditionalInverseGamma,
        ConditionalWeibull,
        ConditionalPareto,
        ConditionalKumaraswamy,
        ConditionalContinuousBernoulli,
        ConditionalFisherSnedecor,
        ConditionalUniform,
        ConditionalMultivariateNormal,
        ConditionalLowRankMVN,
        ConditionalRelaxedBernoulli,
        ConditionalRelaxedOneHotCategorical,
        ConditionalWishart,
        ConditionalInverseWishart,
        ConditionalMatrixNormal,
        ConditionalGaussianProcess,
        ConditionalHorseshoe,
        ConditionalBernoulli,
        ConditionalCategorical,
    )

    _FAMILY_REGISTRY = {
        "Normal": ConditionalNormal,
        "LogitNormal": ConditionalLogitNormal,
        "Beta": ConditionalBeta,
        "TruncatedNormal": ConditionalTruncatedNormal,
        "Dirichlet": ConditionalDirichlet,
        "Cauchy": ConditionalCauchy,
        "Laplace": ConditionalLaplace,
        "Gumbel": ConditionalGumbel,
        "LogNormal": ConditionalLogNormal,
        "StudentT": ConditionalStudentT,
        "Exponential": ConditionalExponential,
        "Gamma": ConditionalGamma,
        "Chi2": ConditionalChi2,
        "HalfCauchy": ConditionalHalfCauchy,
        "HalfNormal": ConditionalHalfNormal,
        "InverseGamma": ConditionalInverseGamma,
        "Weibull": ConditionalWeibull,
        "Pareto": ConditionalPareto,
        "Kumaraswamy": ConditionalKumaraswamy,
        "ContinuousBernoulli": ConditionalContinuousBernoulli,
        "FisherSnedecor": ConditionalFisherSnedecor,
        "Uniform": ConditionalUniform,
        "MultivariateNormal": ConditionalMultivariateNormal,
        "LowRankMVN": ConditionalLowRankMVN,
        "RelaxedBernoulli": ConditionalRelaxedBernoulli,
        "RelaxedOneHotCategorical": ConditionalRelaxedOneHotCategorical,
        "Wishart": ConditionalWishart,
        "InverseWishart": ConditionalInverseWishart,
        "MatrixNormal": ConditionalMatrixNormal,
        "GP": ConditionalGaussianProcess,
        "Horseshoe": ConditionalHorseshoe,
        "Bernoulli": ConditionalBernoulli,
        "Categorical": ConditionalCategorical,
    }
    try:
        from quivers.continuous.families import ConditionalGeneralizedPareto

        _FAMILY_REGISTRY["GeneralizedPareto"] = ConditionalGeneralizedPareto
    except ImportError, AttributeError:
        pass
    return _FAMILY_REGISTRY


_SPACE_CONSTRUCTORS: (
    dict[str, type[ContinuousSpace] | Callable[..., ContinuousSpace]] | None
) = None


def _get_space_constructors() -> dict[
    str, type[ContinuousSpace] | Callable[..., ContinuousSpace]
]:
    """Lazily build the space constructor registry."""
    global _SPACE_CONSTRUCTORS
    if _SPACE_CONSTRUCTORS is not None:
        return _SPACE_CONSTRUCTORS
    from quivers.continuous.spaces import (
        Euclidean,
        Simplex,
        PositiveReals,
        UnitInterval,
        ProductSpace,
    )

    _SPACE_CONSTRUCTORS = {
        "Euclidean": Euclidean,
        "Simplex": Simplex,
        "PositiveReals": PositiveReals,
        "UnitInterval": UnitInterval,
        "ProductSpace": ProductSpace,
    }
    return _SPACE_CONSTRUCTORS


class _ChartHandlerComposite(torch.nn.Module):
    """Post-handler composition over a chart parser's output.

    Wraps a base ``InsideAlgorithm`` (or any callable returning a
    ``(batch, N)``-shaped tensor of log-probabilities over the start
    symbol's enriched category cell) and composes one or more
    handler morphisms on the output. Each handler's tensor is taken
    as a ``N × N'`` log-probability transition that reduces the
    effect stack on the output cell.
    """

    def __init__(self, base, handler) -> None:
        super().__init__()
        self._base = base
        self._handler = handler
        # Register the handler's module so parameters and buffers are
        # tracked through training.
        if hasattr(handler, "module"):
            self._handler_mod = handler.module()
        else:
            self._handler_mod = handler

    def forward(self, tokens):
        base_out = self._base(tokens)
        # base_out shape: (batch,) for the start-symbol log-prob, or
        # (batch, N) for the cell distribution. Handlers reduce the
        # cell distribution along the N axis via log-space matrix
        # multiplication; if base_out is scalar, the handler is a
        # no-op identity on the start-symbol axis.
        if base_out.dim() == 1:
            return base_out
        log_handler = torch.log(self._handler.tensor.clamp(min=1e-30))
        # log[batch, B] = logsumexp_A(base_out[batch, A] + log_handler[A, B])
        return torch.logsumexp(base_out.unsqueeze(2) + log_handler.unsqueeze(0), dim=1)

    def __repr__(self) -> str:
        return f"ChartHandlerComposite({self._base!r} ; {self._handler!r})"


class CompileError(Exception):
    """Raised when the compiler encounters a semantic error.

    Parameters
    ----------
    message : str
        Error description.
    line : int
        Source line number (0 if unknown).
    col : int
        Source column number (0 if unknown).
    """

    def __init__(self, message: str, line: int = 0, col: int = 0) -> None:
        self.line = line
        self.col = col
        loc = f"line {line}, col {col}: " if line else ""
        super().__init__(f"{loc}{message}")


def _decode_vocab_literal(
    sig_name: str,
    sort_name: str,
    lit: "SortVocabLiteral",
) -> str | int | float:
    """Decode a sort-vocabulary literal's surface text into the
    Python value the runtime indexes by.

    String literals are unescaped via the standard Python escape
    rules (so ``"\\n"`` decodes to a newline). Integer and float
    literals decode via the built-in numeric constructors.
    """
    if lit.kind == "string":
        raw = lit.text
        if not (len(raw) >= 2 and raw[0] == '"' and raw[-1] == '"'):
            raise CompileError(
                f"signature {sig_name!r}: sort {sort_name!r} vocab entry "
                f"{raw!r} is not a well-formed string literal"
            )
        inner = raw[1:-1]
        return inner.encode("utf-8").decode("unicode_escape")
    if lit.kind == "integer":
        return int(lit.text)
    if lit.kind == "float":
        return float(lit.text)
    raise CompileError(
        f"signature {sig_name!r}: sort {sort_name!r} unknown vocab literal "
        f"kind {lit.kind!r}"
    )


class Compiler:
    """Compile a quivers DSL AST into a Program.

    The compiler maintains three environments:

    - objects: name -> SetObject (discrete finite sets)
    - spaces: name -> ContinuousSpace
    - morphisms: name -> Morphism or ContinuousMorphism (any morphism-like)

    It processes statements in order and compiles the output
    expression into a Program wrapping the morphism DAG.

    Parameters
    ----------
    module : Module
        The parsed AST.
    """

    def __init__(self, module: Module) -> None:
        self._module = module
        self._quantale: Quantale = PRODUCT_FUZZY
        self._categories: list[str] = []
        self._rules: dict = {}
        self._bundles: dict[str, tuple[str, ...]] = {}
        self._aliases: dict[str, TypeExpr] = {}
        self._alias_names: set[str] = set()
        # Built-in transformation catalog: singletons looked up by
        # bare name (``expectation``, ``log_prob``, …) and
        # constructors invoked with arguments (``softmax(B)``,
        # ``bayes_invert(prior)``).  Disjoint from
        # :attr:`_transformations`, which holds user let-bound
        # transformations defined inside the module.
        self._trans_singletons: dict = _build_default_trans_singletons()
        self._trans_constructors: dict = _build_default_trans_constructors()
        # User-defined transformations bound via ``let t = …``.
        # Disjoint from :attr:`_morphisms`: a ``let`` whose RHS
        # resolves to a transformation lands here; a ``let`` whose
        # RHS resolves to a morphism lands in ``_morphisms``.
        self._transformations: dict = {}
        self._objects: dict[str, SetObject] = {}
        self._spaces: dict = {}
        self._morphisms: dict = {}
        self._groups: dict[str, list[str]] = {}
        self._output_expr: Expr | None = None
        # Parametric-program templates: dependent kernels Π(p:P).Kern(dom(p),cod(p))
        # stored as their unsubstituted AST decl. Instantiated at each call
        # site by parameter substitution + α-renaming of internal latents.
        self._program_templates: dict[str, ProgramDecl] = {}
        # Operadic contraction declarations. Each entry is callable
        # from the DSL at let-binding sites; the value records the
        # compiled :class:`EinsumWiring` plus the declared domain /
        # codomain typing for shape-checking at invocation time.
        self._contractions: dict[str, "_CompiledContraction"] = {}

    @property
    def categories(self) -> list[str]:
        """The declared category atoms."""
        return list(self._categories)

    @property
    def rules(self) -> dict:
        """The compiled rule schemas from ``rule`` declarations."""
        return dict(self._rules)

    @property
    def objects(self) -> dict[str, SetObject]:
        """The compiled object environment."""
        return dict(self._objects)

    @property
    def spaces(self) -> dict:
        """The compiled space environment."""
        return dict(self._spaces)

    @property
    def morphisms(self) -> dict:
        """The compiled morphism environment."""
        return dict(self._morphisms)

    @property
    def quantale(self) -> Quantale:
        """The active quantale."""
        return self._quantale

    def compile(self) -> Program:
        """Compile the module into a trainable Program.

        Returns
        -------
        Program
            The compiled nn.Module wrapping the morphism DAG.

        Raises
        ------
        CompileError
            On semantic errors (undefined names, type mismatches, etc.).
        """
        _register_extra_quantales()
        for stmt in self._module.statements:
            self._compile_statement(stmt)
        if self._output_expr is None:
            # A module may declare only structural artifacts
            # (signatures, encoders, decoders, losses) with no
            # exported morphism; the returned Program is a container
            # carrying those artifacts.
            program = Program(None)
        else:
            root_morphism = self._compile_expr(self._output_expr)
            program = Program(root_morphism)
        # Attach the compiler's deduction and posterior registries to
        # the Program so downstream callers can reach them after
        # `quivers.dsl.load(...)`.
        program.deductions = getattr(self, "_deductions", {})
        program.posteriors = getattr(self, "_posteriors", {})
        program.signatures = getattr(self, "_signatures", {})
        program.encoders = getattr(self, "_encoders", {})
        program.decoders = getattr(self, "_decoders", {})
        program.losses = getattr(self, "_loss_registry", None)
        # Wire the loss registry into every compiled deduction so the
        # agenda's rule-firing and chart-completion paths can fire
        # rule-attached and chart-attached losses automatically.
        if program.losses is not None:
            for name, system in program.deductions.items():
                system._loss_registry = program.losses  # type: ignore[attr-defined]
                system._deduction_name = name  # type: ignore[attr-defined]
        return program

    def compile_env(self) -> dict:
        """Compile all statements and return the full environment.

        Useful for inspection without requiring an output declaration.

        Returns
        -------
        dict
            Combined environment of objects, spaces, morphisms, and the quantale.
        """
        _register_extra_quantales()
        for stmt in self._module.statements:
            self._compile_statement(stmt)
        env: dict = {}
        env["__quantale__"] = self._quantale
        for name, obj in self._objects.items():
            env[name] = obj
        for name, space in self._spaces.items():
            env[name] = space
        for name, morph in self._morphisms.items():
            env[name] = morph
        for name, rule in self._rules.items():
            env[name] = rule
        return env

    def _compile_statement(self, stmt: Statement) -> None:
        """Dispatch to the appropriate statement compiler."""
        if isinstance(stmt, QuantaleDecl):
            self._compile_quantale(stmt)
        elif isinstance(stmt, CategoryDecl):
            self._compile_category(stmt)
        elif isinstance(stmt, RuleDecl):
            self._compile_rule(stmt)
        elif isinstance(stmt, SchemaDecl):
            self._compile_schema(stmt)
        elif isinstance(stmt, AliasDecl):
            self._compile_alias(stmt)
        elif isinstance(stmt, BundleDecl):
            self._compile_bundle(stmt)
        elif isinstance(stmt, ObjectDecl):
            self._compile_object(stmt)
        elif isinstance(stmt, MorphismDecl):
            self._compile_morphism(stmt)
        elif isinstance(stmt, SpaceDecl):
            self._compile_space(stmt)
        elif isinstance(stmt, KernelDecl):
            self._compile_kernel(stmt)
        elif isinstance(stmt, DiscretizeDecl):
            self._compile_discretize(stmt)
        elif isinstance(stmt, EmbedDecl):
            self._compile_embed(stmt)
        elif isinstance(stmt, ProgramDecl):
            self._compile_program(stmt)
        elif isinstance(stmt, ContractionDecl):
            self._compile_contraction(stmt)
        elif isinstance(stmt, LetDecl):
            self._compile_let(stmt)
        elif isinstance(stmt, ExportDecl):
            self._compile_export(stmt)
        elif isinstance(stmt, DeductionDecl):
            self._compile_deduction(stmt)
        elif isinstance(stmt, SignatureDecl):
            self._compile_signature(stmt)
        elif isinstance(stmt, EncoderDecl):
            self._compile_encoder(stmt)
        elif isinstance(stmt, DecoderDecl):
            self._compile_decoder(stmt)
        elif isinstance(stmt, LossDecl):
            self._compile_loss(stmt)
        else:
            raise CompileError(f"unknown statement type: {type(stmt).__name__}")

    def _compile_quantale(self, decl: QuantaleDecl) -> None:
        """Set the active composition rule for this module.

        Four surface forms (distinguished by ``decl.declared_level``):

        * ``quantale X`` — X must be a Quantale.
        * ``semigroupoid X`` — X must be a Semigroupoid.
        * ``bilinear_form X`` — X must be a BilinearForm.
        * ``composition_rule X`` — X must be any CompositionRule.

        Two body forms:

        * **No body** — ``X`` names a registered rule in the
          composition-rule registry; the compiler verifies it
          matches the declared level.
        * **With body** — ``decl.body`` is a list of entries
          defining the rule's operations from scratch. The
          compiler evaluates each entry's expression and builds
          a fresh ``CustomQuantale``, ``CustomSemigroupoid``, or
          ``CustomBilinearForm`` of the declared level, then
          registers it under the supplied name.
        """
        level = decl.declared_level
        if decl.body:
            rule = self._build_custom_composition_rule(decl)
        else:
            name = decl.name.lower()
            if name not in _QUANTALE_REGISTRY:
                raise CompileError(
                    f"unknown {level} {decl.name!r}; available: "
                    f"{', '.join(sorted(_QUANTALE_REGISTRY))}",
                    decl.line,
                    decl.col,
                )
            rule = _QUANTALE_REGISTRY[name]
        self._verify_composition_rule_level(rule, decl, level)
        self._quantale = rule  # type: ignore[assignment]

    def _verify_composition_rule_level(
        self,
        rule: "CompositionRule",
        decl: QuantaleDecl,
        level: str,
    ) -> None:
        """Confirm ``rule`` satisfies the algebraic level declared
        by the keyword."""
        required = {
            "quantale": Quantale,
            "semigroupoid": Semigroupoid,
            "bilinear_form": BilinearForm,
            "composition_rule": CompositionRule,
        }
        required_class = required.get(level, CompositionRule)
        if not isinstance(rule, required_class):
            actual = (
                "Quantale"
                if isinstance(rule, Quantale)
                else "Semigroupoid"
                if isinstance(rule, Semigroupoid)
                else "BilinearForm"
                if isinstance(rule, BilinearForm)
                else "CompositionRule"
            )
            raise CompileError(
                f"{level} {decl.name!r}: registered rule is a "
                f"{actual}, which is not at level {level!r}. "
                f"Either declare it with the matching keyword or "
                f"register a rule at the right level.",
                decl.line,
                decl.col,
            )

    def _build_custom_composition_rule(self, decl: QuantaleDecl) -> "CompositionRule":
        """Build a fresh CompositionRule instance from a
        ``quantale name { … }`` body, dispatching on the declared
        level."""
        entries: dict[str, "CompositionRuleEntry"] = {}
        for entry in decl.body:
            if entry.key in entries:
                raise CompileError(
                    f"{decl.declared_level} {decl.name!r}: duplicate "
                    f"entry {entry.key!r}",
                    entry.line,
                    entry.col,
                )
            entries[entry.key] = entry
        level = decl.declared_level
        required_keys = {
            "quantale": {"tensor_op", "join", "unit", "zero"},
            "semigroupoid": {"tensor_op", "join"},
            "bilinear_form": {"tensor_op", "join"},
            "composition_rule": {"tensor_op", "join"},
        }[level]
        missing = required_keys - set(entries)
        if missing:
            raise CompileError(
                f"{level} {decl.name!r}: missing required entries {sorted(missing)}",
                decl.line,
                decl.col,
            )
        # Compile each entry to a Python callable (function-valued
        # entry like ``tensor_op(a, b) = …``) or a numeric value
        # (literal-valued entry like ``unit = 1.0``). The union of
        # callable and float is the natural type here; we keep
        # them separate downstream so the level builders type-check
        # each slot precisely.
        compiled: dict[str, "Callable[..., torch.Tensor] | float"] = {}
        for key, entry in entries.items():
            compiled[key] = self._compile_composition_rule_entry(entry, decl)
        # Build the concrete rule at the declared level.
        tensor_op = compiled["tensor_op"]
        join = compiled["join"]
        if not callable(tensor_op):
            raise CompileError(
                f"{level} {decl.name!r}: ``tensor_op`` must be a "
                f"function entry like ``tensor_op(a, b) = …``",
                decl.line,
                decl.col,
            )
        if not callable(join):
            raise CompileError(
                f"{level} {decl.name!r}: ``join`` must be a "
                f"function entry like ``join(t) = …``",
                decl.line,
                decl.col,
            )
        join_wrapped = _wrap_join_dim(join)
        if level == "quantale":
            unit = compiled["unit"]
            zero = compiled["zero"]
            if callable(unit) or callable(zero):
                raise CompileError(
                    f"quantale {decl.name!r}: ``unit`` and ``zero`` "
                    f"must be value entries (no parens)",
                    decl.line,
                    decl.col,
                )
            negate_fn = compiled.get("negation")
            if negate_fn is not None and not callable(negate_fn):
                raise CompileError(
                    f"quantale {decl.name!r}: ``negation`` must be "
                    f"a function entry like ``negation(a) = …``",
                    decl.line,
                    decl.col,
                )
            return CustomQuantale(
                name=decl.name,
                tensor_op=tensor_op,
                join=join_wrapped,
                unit=float(unit),
                zero=float(zero),
                negate=negate_fn,
                verify=False,
            )
        if level == "semigroupoid":
            return CustomSemigroupoid(
                name=decl.name,
                tensor_op=tensor_op,
                join=join_wrapped,
                verify_associative=False,
            )
        # ``bilinear_form`` and ``composition_rule`` both build a
        # CustomBilinearForm: ``bilinear_form`` is the weakest
        # named level (no associativity promise), and
        # ``composition_rule`` is the permissive surface that
        # admits any rule, for which BilinearForm is the right
        # default since it makes no claim beyond the
        # CompositionRule interface.
        return CustomBilinearForm(
            name=decl.name,
            tensor_op=tensor_op,
            join=join_wrapped,
        )

    def _compile_composition_rule_entry(
        self,
        entry: "CompositionRuleEntry",
        decl: QuantaleDecl,
    ) -> "Callable[..., torch.Tensor] | float":
        """Compile one ``key(params) = body`` or ``key = body``
        entry to a Python callable (when ``entry.params`` is
        non-empty) or to its evaluated numeric value (when
        ``entry.params`` is empty)."""
        body_fn = Compiler._compile_let_expr(entry.body)
        if not entry.params:
            try:
                return body_fn({})
            except Exception as exc:
                raise CompileError(
                    f"{decl.declared_level} {decl.name!r}: value "
                    f"entry {entry.key!r} could not be evaluated: {exc}",
                    entry.line,
                    entry.col,
                ) from exc
        param_names = entry.params

        def _callable(*args):
            if len(args) != len(param_names):
                raise TypeError(
                    f"{entry.key}(): expected {len(param_names)} "
                    f"arguments, got {len(args)}"
                )
            env = dict(zip(param_names, args))
            return body_fn(env)

        return _callable

    def _compile_category(self, decl: CategoryDecl) -> None:
        """Register a category atom declaration.

        Category atoms are generators for a free categorical structure,
        distinct from finite set objects.  They are used by the parser
        compiler to build a ``CategorySystem``.
        """
        if decl.name in self._categories:
            raise CompileError(
                f"category {decl.name!r} already declared", decl.line, decl.col
            )
        self._categories.append(decl.name)

    def _compile_rule(self, decl: RuleDecl) -> None:
        """Compile a rule-of-inference declaration into a RuleSchema.

        Creates a ``PatternBinarySchema`` (2 premises) or
        ``PatternUnarySchema`` (1 premise) and registers it by name
        so it can be resolved in ``parser(rules=[...])``.
        """
        from quivers.stochastic.schema import (
            PatternBinarySchema,
            PatternUnarySchema,
            SCHEMA_REGISTRY,
        )

        if decl.name in self._rules:
            raise CompileError(
                f"rule {decl.name!r} already declared", decl.line, decl.col
            )
        if decl.name in SCHEMA_REGISTRY:
            raise CompileError(
                f"rule {decl.name!r} shadows a built-in schema; choose a different name",
                decl.line,
                decl.col,
            )
        variables = frozenset(decl.variables)
        n_premises = len(decl.premises)
        if n_premises == 2:
            schema = PatternBinarySchema(
                left_pattern=decl.premises[0],
                right_pattern=decl.premises[1],
                conclusion_pattern=decl.conclusion,
                variables=variables,
                name=decl.name,
            )
        elif n_premises == 1:
            schema = PatternUnarySchema(
                premise_pattern=decl.premises[0],
                conclusion_pattern=decl.conclusion,
                variables=variables,
                name=decl.name,
            )
        else:
            raise CompileError(
                f"rule {decl.name!r} has {n_premises} premises; only unary (1) and binary (2) rules are supported",
                decl.line,
                decl.col,
            )
        self._rules[decl.name] = schema

    def _compile_schema(self, decl: SchemaDecl) -> None:
        """Compile a pattern-polymorphic schema declaration.

        Creates a ``PatternBinarySchema`` when the declared domain is a
        :class:`TypeProduct` with two components, otherwise a
        ``PatternUnarySchema``. Pattern variables are the union of the
        ``names`` lists across all :class:`SchemaParameter` entries; the
        parameter type-expression is consulted only for well-formedness
        (it must reference a residuated universe in scope; the
        type-checker does not yet enforce this — the chart-parser
        catches mismatches at firing time).
        """
        from quivers.stochastic.schema import (
            PatternBinarySchema,
            PatternUnarySchema,
            SCHEMA_REGISTRY,
        )

        if decl.name in self._rules:
            raise CompileError(
                f"schema {decl.name!r} already declared",
                decl.line,
                decl.col,
            )
        if decl.name in SCHEMA_REGISTRY:
            raise CompileError(
                f"schema {decl.name!r} shadows a built-in schema; choose a different name",
                decl.line,
                decl.col,
            )

        variables: frozenset[str] = frozenset(
            n for group in decl.parameter_names for n in group
        )

        # Decide arity from the domain shape:
        #  - top-level TypeProduct with exactly 2 components → binary
        #  - any other shape (TypeName, TypeSlash, TypeEffectApply,
        #    or a non-binary TypeProduct) → unary
        if isinstance(decl.domain, TypeProduct) and len(decl.domain.components) == 2:
            left, right = decl.domain.components
            schema = PatternBinarySchema(
                left_pattern=left,
                right_pattern=right,
                conclusion_pattern=decl.codomain,
                variables=variables,
                name=decl.name,
            )
        else:
            schema = PatternUnarySchema(
                premise_pattern=decl.domain,
                conclusion_pattern=decl.codomain,
                variables=variables,
                name=decl.name,
            )

        self._rules[decl.name] = schema

    def _compile_alias(self, decl) -> None:
        """Compile an ``alias Foo = ...`` type-level alias.

        Two cases:

        - The right-hand side resolves cleanly as a :class:`SetObject`
          (TypeName / TypeProduct / TypeCoproduct over named objects).
          The alias binds to that SetObject in :attr:`self._objects`,
          so ``Foo`` is usable wherever an ordinary object reference
          is — `latent f : Foo -> Bar`, `parser(rules=..., terminal=Foo)`
          etc.
        - The right-hand side is a residuated pattern (TypeSlash /
          TypeEffectApply) or otherwise fails SetObject resolution.
          The alias is recorded in :attr:`self._aliases` for textual
          substitution at use site (inside schema patterns).
        """
        if decl.name in self._alias_names:
            raise CompileError(
                f"alias {decl.name!r} already declared",
                decl.line,
                decl.col,
            )
        if decl.name in self._objects:
            raise CompileError(
                f"alias {decl.name!r} shadows an existing object",
                decl.line,
                decl.col,
            )
        self._alias_names.add(decl.name)
        try:
            resolved = self._resolve_type(decl.type_expr, decl.name)
        except TypeError, KeyError:
            # Residuated / effect-typed RHS: record as a syntactic
            # alias for substitution at schema-pattern use site.
            self._aliases[decl.name] = decl.type_expr
            return
        self._objects[decl.name] = resolved

    def _compile_bundle(self, decl) -> None:
        """Compile a ``bundle CCG = [r1, r2, ...]`` rule bundle.

        Each entry must resolve at compile time as either a previously-
        declared rule / schema or as a built-in entry of
        :data:`SCHEMA_REGISTRY`. The bundle is recorded under its name
        in ``self._bundles`` so ``parser(rules=CCG)`` and
        ``chart_fold(binary=CCG)`` can splice its members.
        """
        from quivers.stochastic.schema import SCHEMA_REGISTRY

        if decl.name in self._bundles:
            raise CompileError(
                f"bundle {decl.name!r} already declared",
                decl.line,
                decl.col,
            )
        if decl.name in self._rules or decl.name in SCHEMA_REGISTRY:
            raise CompileError(
                f"bundle {decl.name!r} shadows a rule / built-in schema",
                decl.line,
                decl.col,
            )
        # Member references are resolved lazily at use-site (in the
        # parser-rules expander) so that bundles can forward-reference
        # other bundles. Cycles surface as ``cycle through ...`` errors
        # at expansion time.
        self._bundles[decl.name] = tuple(decl.rules)

    def _compile_object(self, decl: ObjectDecl) -> None:
        """Compile an object declaration into the environment.

        Three surface forms are recognized:

        - ``object X : <type_expr>`` — resolves via the
          :class:`TypeExprToSetObject` lens.
        - ``object Atoms = {NP, S, VP}`` — constructs an
          :class:`EnumSet`.
        - ``object Cat = FreeResiduated(Atoms, depth=, ops=[...])`` —
          constructs a :class:`FreeResiduated` over a previously-declared
          :class:`EnumSet`.
        """
        from quivers.core.objects import EnumSet, FreeResiduated

        if decl.name in self._objects:
            raise CompileError(
                f"object {decl.name!r} already declared", decl.line, decl.col
            )

        if decl.type_expr is not None:
            obj = self._resolve_type(decl.type_expr, decl.name)
            self._objects[decl.name] = obj
            return

        if decl.init is None:
            raise CompileError(
                f"object {decl.name!r} has no type or initializer",
                decl.line,
                decl.col,
            )

        if isinstance(decl.init, EnumSetLiteral):
            self._objects[decl.name] = EnumSet(
                name=decl.name, elements=decl.init.elements
            )
            return

        if isinstance(decl.init, FreeMonoidExpr):
            from quivers.core.objects import FinSet, FreeMonoid

            gen = self._objects.get(decl.init.generators)
            if not isinstance(gen, FinSet):
                raise CompileError(
                    f"FreeMonoid generators {decl.init.generators!r} must "
                    f"reference a previously-declared FinSet (got "
                    f"{type(gen).__name__ if gen else 'undefined'})",
                    decl.line,
                    decl.col,
                )
            self._objects[decl.name] = FreeMonoid(
                generators=gen, max_length=decl.init.max_length
            )
            return

        if isinstance(decl.init, FreeResiduatedExpr):
            gen = self._objects.get(decl.init.generators)
            if not isinstance(gen, EnumSet):
                raise CompileError(
                    f"FreeResiduated generators {decl.init.generators!r} must "
                    f"reference a previously-declared EnumSet (got "
                    f"{type(gen).__name__ if gen else 'undefined'})",
                    decl.line,
                    decl.col,
                )
            self._objects[decl.name] = FreeResiduated(
                generators=gen,
                depth=decl.init.depth,
                ops=decl.init.ops,
            )
            return

        raise CompileError(
            f"unrecognized object initializer for {decl.name!r}",
            decl.line,
            decl.col,
        )

    def _compile_morphism(self, decl: MorphismDecl) -> None:
        """Compile a morphism declaration into the environment."""
        if decl.name in self._morphisms:
            raise CompileError(
                f"morphism {decl.name!r} already declared", decl.line, decl.col
            )
        if decl.prior is not None:
            if decl.morphism_kind != "latent":
                raise CompileError(
                    f"morphism prior `~ Family(...)` is legal only on "
                    f"`latent` declarations; got {decl.morphism_kind!r}",
                    decl.line, decl.col,
                )
            if decl.prior.axes is not None:
                _validate_axis_spec(
                    decl.prior.axes,
                    decl.prior.family,
                    _available_axes_for(decl.domain, decl.codomain),
                    decl.line, decl.col,
                )
        domain = self._resolve_type(decl.domain)
        codomain = self._resolve_type(decl.codomain)
        if decl.morphism_kind == "latent":
            scale = float(decl.options.get("scale", "0.5"))
            morph = make_latent(
                domain, codomain, init_scale=scale, quantale=self._quantale
            )
        elif decl.morphism_kind == "observed":
            if decl.init_expr is not None:
                morph = self._compile_expr(decl.init_expr)
                # The init expression's domain/codomain may be
                # anonymous (e.g. ``from_data(...)`` synthesizes
                # them from the tensor shape). Accept a shape
                # match and rebind to the user-declared types so
                # downstream code sees the correct named objects.
                #
                # Compatibility is checked at the storage level
                # rather than the type level: a flat init tensor
                # whose total numel matches the declared product
                # codomain's numel is accepted, then reshaped to
                # the declared factored shape. This is the
                # categorical view that ``B = B1 * B2 * ... * Bk``
                # and ``B'`` of cardinality ``|B1| * ... * |Bk|``
                # are isomorphic objects; the tensor storage is
                # the same up to reshape.
                if morph.domain != domain or morph.codomain != codomain:

                    def _numel(shape):
                        n = 1
                        for s in shape:
                            n *= int(s)
                        return n

                    init_d = _numel(morph.domain.shape)
                    init_c = _numel(morph.codomain.shape)
                    decl_d = _numel(domain.shape)
                    decl_c = _numel(codomain.shape)
                    if init_d == decl_d and init_c == decl_c:
                        from quivers.core.morphisms import (
                            ObservedMorphism as _Obs,
                        )

                        # Reshape the tensor to match the declared
                        # factored shape. ``Tensor.reshape`` is a
                        # no-op when the storage already matches.
                        target_shape = tuple(domain.shape) + tuple(codomain.shape)
                        reshaped = morph.tensor.reshape(target_shape)
                        morph = _Obs(
                            domain,
                            codomain,
                            reshaped,
                            quantale=morph.quantale,
                        )
                    else:
                        raise CompileError(
                            f"morphism {decl.name!r} init expression has "
                            f"type {morph.domain!r} -> {morph.codomain!r} "
                            f"(numel {init_d} -> {init_c}), expected "
                            f"{domain!r} -> {codomain!r} "
                            f"(numel {decl_d} -> {decl_c})",
                            decl.line,
                            decl.col,
                        )
            else:
                raise CompileError(
                    f"observed morphism {decl.name!r} requires an initializer (e.g. = identity({decl.domain}))",
                    decl.line,
                    decl.col,
                )
        else:
            raise CompileError(
                f"unknown morphism kind {decl.morphism_kind!r}", decl.line, decl.col
            )
        self._morphisms[decl.name] = morph

    def _compile_space(self, decl: SpaceDecl) -> None:
        """Compile a space declaration into the space environment."""
        if decl.name in self._spaces:
            raise CompileError(
                f"space {decl.name!r} already declared", decl.line, decl.col
            )
        space = self._resolve_space(decl.space_expr, decl.name)
        self._spaces[decl.name] = space

    def _compile_kernel(self, decl: KernelDecl) -> None:
        """Compile a Markov-kernel declaration ``kernel f : A -> B [~ F ...]``.

        Without a ``~`` clause the declaration is a lookup-table
        kernel on finite sets, realised as a
        :class:`quivers.stochastic.StochasticMorphism`.  With a ``~``
        clause it is a parametric kernel whose family parameters are
        produced from the input by a parameter network at sample
        time; the ``decl.axes`` clause configures the family's
        event/batch decomposition over codomain factors.  Replicate
        counts produce N independent copies named ``name_0`` through
        ``name_{N-1}`` with the base name registered as a group.
        """
        if decl.name in self._morphisms:
            raise CompileError(
                f"morphism {decl.name!r} already declared", decl.line, decl.col
            )
        count = decl.replicate if decl.replicate is not None else 1
        names = (
            [f"{decl.name}_{i}" for i in range(count)]
            if decl.replicate is not None
            else [decl.name]
        )
        if decl.axes is not None:
            if decl.family is None:
                raise CompileError(
                    "axis-role clause requires a `~ Family` clause on the "
                    "kernel declaration; lookup-table kernels do not carry "
                    "a parametric family.",
                    decl.line, decl.col,
                )
            _validate_axis_spec(
                decl.axes,
                decl.family,
                _available_axes_for(decl.domain, decl.codomain),
                decl.line, decl.col,
            )
        if decl.family is None:
            domain = self._resolve_type(decl.domain)
            codomain = self._resolve_type(decl.codomain)
            for name in names:
                self._morphisms[name] = StochasticMorphism(domain, codomain)
        else:
            domain = self._resolve_any_space(decl.domain)
            codomain = self._resolve_any_space(decl.codomain)
            for name in names:
                morph = self._make_continuous_morphism(
                    domain, codomain, decl.family, decl.options, decl
                )
                self._morphisms[name] = morph
        if decl.replicate is not None:
            self._groups[decl.name] = names

    def _make_continuous_morphism(
        self, domain, codomain, family_name: str, options: dict[str, str], decl
    ):
        """Create a single continuous morphism from a family name."""
        if family_name == "Flow":
            from quivers.continuous.flows import ConditionalFlow

            n_layers = int(options.get("n_layers", "4"))
            hidden_dim = int(options.get("hidden_dim", "64"))
            return ConditionalFlow(
                domain, codomain, n_layers=n_layers, hidden_dim=hidden_dim
            )
        registry = _get_family_registry()
        if family_name not in registry:
            raise CompileError(
                f"unknown distribution family {family_name!r}; available: {', '.join(sorted(registry))}",
                decl.line,
                decl.col,
            )
        cls = registry[family_name]
        hidden_dim = int(options.get("hidden_dim", "64"))
        kwargs: dict = {"hidden_dim": hidden_dim}
        if "rank" in options:
            kwargs["rank"] = int(options["rank"])
        if "temperature" in options:
            kwargs["temperature"] = float(options["temperature"])
        # For event_rank-2 matrix families (MatrixNormal) the
        # constructor needs explicit row/column dims, which come
        # from the axis-role clause's named factors.  When the
        # user wrote ``~ MatrixNormal over (X, Y)``, X resolves
        # to the rows axis and Y to the cols axis (positional
        # ordering corresponds positionally to the family's
        # declared event-axis ordering: rows first, cols second).
        axes = getattr(decl, "axes", None)
        if family_name == "MatrixNormal" and axes is not None:
            if len(axes.over) != 2:
                raise CompileError(
                    f"MatrixNormal requires `over (rows_axis, cols_axis)`; "
                    f"got over={axes.over!r}",
                    decl.line, decl.col,
                )
            rows_axis, cols_axis = axes.over
            kwargs["rows"] = self._axis_dim(decl, rows_axis)
            kwargs["cols"] = self._axis_dim(decl, cols_axis)
        return cls(domain, codomain, **kwargs)

    def _axis_dim(self, decl, axis_name: str) -> int:
        """Resolve an axis name to its dimension.

        ``axis_name`` is either a declared factor name of the
        morphism's dom or cod, or the reserved shortcut ``dom`` /
        ``cod``.  Returns the cardinality / dim of the resolved
        object or space.
        """
        # The dom/cod shortcuts resolve to the morphism's dom/cod
        # objects directly.
        if axis_name == "dom":
            obj = self._resolve_any_space(decl.domain)
            return _shape_size(obj)
        if axis_name == "cod":
            obj = self._resolve_any_space(decl.codomain)
            return _shape_size(obj)
        # Otherwise the axis is a named factor of dom or cod.  For
        # an unfactored side whose argument carries the same name
        # (``Euclidean(D)`` with object D in scope) the axis name
        # is the object's name; resolve it directly.
        if axis_name in self._objects:
            return int(self._objects[axis_name].cardinality)
        if axis_name in self._spaces:
            return _shape_size(self._spaces[axis_name])
        raise CompileError(
            f"axis-role clause: cannot resolve axis name {axis_name!r} to a "
            f"dimension; not a declared object/space, and not a `dom`/`cod` "
            f"shortcut", decl.line, decl.col,
        )

    def _compile_discretize(self, decl: DiscretizeDecl) -> None:
        """Compile a discretize boundary morphism."""
        if decl.name in self._morphisms:
            raise CompileError(
                f"morphism {decl.name!r} already declared", decl.line, decl.col
            )
        if decl.space_name not in self._spaces:
            raise CompileError(
                f"undefined space {decl.space_name!r}", decl.line, decl.col
            )
        from quivers.continuous.boundaries import Discretize

        space = self._spaces[decl.space_name]
        morph = Discretize(space, n_bins=decl.n_bins)
        self._morphisms[decl.name] = morph

    def _compile_embed(self, decl: EmbedDecl) -> None:
        """Compile an embed boundary morphism."""
        if decl.name in self._morphisms:
            raise CompileError(
                f"morphism {decl.name!r} already declared", decl.line, decl.col
            )
        if decl.domain_name not in self._objects:
            raise CompileError(
                f"undefined object {decl.domain_name!r}", decl.line, decl.col
            )
        if decl.codomain_name not in self._spaces:
            raise CompileError(
                f"undefined space {decl.codomain_name!r}", decl.line, decl.col
            )
        from quivers.continuous.boundaries import Embed

        domain = self._objects[decl.domain_name]
        codomain = self._spaces[decl.codomain_name]
        count = decl.replicate if decl.replicate is not None else 1
        names = (
            [f"{decl.name}_{i}" for i in range(count)]
            if decl.replicate is not None
            else [decl.name]
        )
        for name in names:
            assert isinstance(domain, FinSet)
            morph = Embed(domain, codomain)
            self._morphisms[name] = morph
        if decl.replicate is not None:
            self._groups[decl.name] = names

    def _expand_bind_steps(
        self, steps: tuple[ProgramStep, ...]
    ) -> tuple[ProgramStep, ...]:
        """Translate v0.5 :class:`BindStep` IR into the compiler's
        internal step-IR (:class:`DrawStep`, :class:`PlateDrawStep`,
        :class:`VectorisedObserveStep`, :class:`MarginalizeStep`).

        The expansion is purely a syntactic refinement: each
        BindStep dispatches on its ``mode`` and ``index`` fields
        to one of the four internal step shapes. Marginalize binds
        additionally inline a synthesized sample step for the
        coordinate, followed by the scope's recursively-expanded
        steps, followed by a :class:`MarginalizeStep` reduction.

        ``LetStep`` passes through unchanged. The expansion
        preserves the Kleisli-arrow denotation of the program body
        — it is a reorganisation of the surface IR, not a change
        of semantics.
        """
        out: list[ProgramStep] = []
        for step in steps:
            if isinstance(step, LetStep):
                out.append(step)
                continue
            if not isinstance(step, BindStep):
                # Pass-through for any internal-IR step that has
                # already been expanded (e.g., template-inlined
                # bodies that synthesized internal steps directly).
                out.append(step)
                continue
            if step.mode == "sample":
                if step.index is None:
                    out.append(
                        DrawStep(
                            vars=step.vars,
                            morphism=step.morphism,
                            args=step.args,
                            is_observed=False,
                            line=step.line,
                            col=step.col,
                        )
                    )
                else:
                    if len(step.vars) != 1:
                        raise CompileError(
                            "indexed sample bind must bind a single name",
                            step.line,
                            step.col,
                        )
                    # The per-row codomain for a plate-draw is taken
                    # from the family's natural codomain at compile
                    # time; the IR carries a `codomain` field that
                    # the compiler's PlateDrawStep handler resolves
                    # via the family's domain/codomain dimensions.
                    # For the v0.5 unified surface, the index annotation
                    # `: A` declares the index set; the per-row codomain
                    # is implicit (taken from the family). We supply a
                    # placeholder `TypeName("1")` which the family
                    # resolver interprets as "scalar per-row codomain"
                    # (Euclidean(1)); families that declare richer
                    # codomains override this.
                    out.append(
                        PlateDrawStep(
                            name=step.vars[0],
                            index=step.index,
                            codomain=TypeName(name="1", line=step.line, col=step.col),
                            morphism=step.morphism,
                            args=step.args,
                            line=step.line,
                            col=step.col,
                        )
                    )
            elif step.mode == "score":
                if step.index is None:
                    out.append(
                        DrawStep(
                            vars=step.vars,
                            morphism=step.morphism,
                            args=step.args,
                            is_observed=True,
                            line=step.line,
                            col=step.col,
                        )
                    )
                else:
                    if len(step.vars) != 1:
                        raise CompileError(
                            "indexed observe bind must bind a single name",
                            step.line,
                            step.col,
                        )
                    out.append(
                        VectorisedObserveStep(
                            index_var=step.vars[0],
                            index_set=step.index,
                            morphism=step.morphism,
                            args=step.args,
                            response_var=step.vars[0],
                            fibration_var=step.via,
                            fibration_axes=step.via_axes,
                            line=step.line,
                            col=step.col,
                        )
                    )
            elif step.mode == "marginal":
                if len(step.vars) != 1:
                    raise CompileError(
                        "marginalize bind must bind a single name",
                        step.line,
                        step.col,
                    )
                # Grouped form: `over G` (single plate) or
                # `over G * H` (product grouping plate). The
                # fibration data lives on each observe inside the
                # body via that observe's `via <idx>` (or
                # `via product(...)`) clause; the header carries
                # only the grouping plate.
                has_over = step.over is not None or step.over_objs is not None
                has_grouping = has_over
                # Normalize the grouping plate into tuple form so
                # the downstream code handles single + product
                # uniformly.
                over_names: tuple[str, ...] | None = None
                if step.over_objs is not None:
                    over_names = step.over_objs
                elif step.over is not None:
                    over_names = (step.over,)
                # Resolve the per-class size from the latent's index
                # annotation when present. The grouped form requires
                # an explicit `: K` annotation so the prior's class
                # axis can be sized at compile time.
                class_size = 0
                if step.index is not None:
                    if isinstance(step.index, TypeName):
                        nm = step.index.name
                        if nm.isdigit():
                            class_size = int(nm)
                        elif nm in self._objects:
                            class_size = int(self._objects[nm].size)
                if has_over and class_size == 0:
                    raise CompileError(
                        "grouped marginalize requires an explicit class-set "
                        "annotation (e.g. `marginalize c : K <- ...`) so the "
                        "class axis is sized at compile time",
                        step.line,
                        step.col,
                    )
                if over_names is not None:
                    for nm in over_names:
                        if nm not in self._objects:
                            raise CompileError(
                                f"grouped marginalize: `over` object "
                                f"{nm!r} is not a declared object",
                                step.line,
                                step.col,
                            )
                # Reduction defaults to logsumexp when unspecified.
                reduction = step.reduction or "logsumexp"
                if reduction not in ("logsumexp", "sum", "mean"):
                    raise CompileError(
                        f"grouped marginalize: unknown reduction "
                        f"{reduction!r}; expected one of logsumexp, "
                        f"sum, mean",
                        step.line,
                        step.col,
                    )
                # Extract the categorical prior's `probs` argument.
                probs_var: str | None = None
                if has_over:
                    if not step.args:
                        raise CompileError(
                            "grouped marginalize requires the latent's "
                            "categorical family to carry a probs argument "
                            "(e.g. `Categorical(probs)`)",
                            step.line,
                            step.col,
                        )
                    first = step.args[0]
                    if not isinstance(first, str):
                        raise CompileError(
                            "grouped marginalize: the categorical family's "
                            "first argument must be a named probs tensor "
                            f"(got literal {first!r})",
                            step.line,
                            step.col,
                        )
                    probs_var = first
                # Introduce the coordinate. For an *ungrouped*
                # marginalize block the latent is a real sample
                # from its categorical prior; for a *grouped*
                # block it is the implicit class-index vector
                # ``torch.arange(K)`` so any downstream let-step
                # expression that references the latent broadcasts
                # across the class axis. The terminal observe in
                # the body (rewritten below) overwrites this slot
                # with the per-(N, K) log-likelihood tensor the
                # marginalize step consumes.
                latent_name = step.vars[0]
                if has_grouping:
                    out.append(
                        GroupedLatentInitStep(
                            latent_name=latent_name,
                            class_size=class_size,
                            line=step.line,
                            col=step.col,
                        )
                    )
                elif step.index is None:
                    out.append(
                        DrawStep(
                            vars=step.vars,
                            morphism=step.morphism,
                            args=step.args,
                            is_observed=False,
                            line=step.line,
                            col=step.col,
                        )
                    )
                else:
                    out.append(
                        PlateDrawStep(
                            name=step.vars[0],
                            index=step.index,
                            codomain=TypeName(name="1", line=step.line, col=step.col),
                            morphism=step.morphism,
                            args=step.args,
                            line=step.line,
                            col=step.col,
                        )
                    )
                # Scope's steps. For a grouped block, rewrite every
                # VectorisedObserveStep in the expanded body as a
                # GroupedBodyObserveStep that writes its per-row
                # per-class log-likelihood to a dedicated env slot.
                # The surrounding MarginalizeStep's runtime callable
                # collects each slot, pairs it with that observe's
                # `via <idx>` fibration, and scatter-sums each
                # contribution into the shared `(|G|, K)`
                # accumulator before the reduction.
                scope_steps = step.scope if step.scope is not None else ()
                expanded_scope = list(self._expand_bind_steps(scope_steps))
                body_observes: list[GroupedObserveEntry] = []
                if has_grouping:
                    # Walk the body forward; rewrite each observe
                    # into a GroupedBodyObserveStep with a unique
                    # ll_slot and capture its per-observe fibration.
                    for j, scoped_step in enumerate(expanded_scope):
                        if isinstance(scoped_step, VectorisedObserveStep):
                            if (
                                scoped_step.fibration_var is None
                                and scoped_step.fibration_axes is None
                            ):
                                raise CompileError(
                                    "grouped marginalize: every observe "
                                    "inside the body must carry its own "
                                    "`via <idx>` clause "
                                    "(e.g. `observe r : N via idx <- ...`); "
                                    f"observe {scoped_step.response_var!r} "
                                    "has no `via`",
                                    scoped_step.line,
                                    scoped_step.col,
                                )
                            ll_slot = (
                                f"_grouped_ll_{latent_name}_"
                                f"{len(body_observes)}"
                            )
                            expanded_scope[j] = GroupedBodyObserveStep(
                                response_var=scoped_step.response_var,
                                morphism=scoped_step.morphism,
                                args=scoped_step.args,
                                index_set=scoped_step.index_set,
                                index_var=scoped_step.index_var,
                                latent_name=latent_name,
                                fibration_var=scoped_step.fibration_var,
                                fibration_axes=scoped_step.fibration_axes,
                                ll_slot=ll_slot,
                                line=scoped_step.line,
                                col=scoped_step.col,
                            )
                            body_observes.append(
                                GroupedObserveEntry(
                                    ll_slot=ll_slot,
                                    fibration_var=scoped_step.fibration_var,
                                    fibration_axes=scoped_step.fibration_axes,
                                )
                            )
                    if not body_observes:
                        # Nested case: a grouped marginalize block whose
                        # body's only contribution to the per-group
                        # accumulator is an inner grouped block.  The
                        # inner block's MarginalizeStep produces the
                        # (N_outer, K_outer) tensor the outer block
                        # consumes.  Re-point the inner's
                        # ``body_ll_var`` at the outer latent so the
                        # outer codegen finds the tensor at the
                        # expected slot, and record a single
                        # body_observes entry whose ll_slot is the
                        # outer latent's name (no per-observe
                        # fibration, since the inner block already
                        # performed its own scatter-add).
                        nested_marg_idx: int | None = None
                        for j in range(len(expanded_scope) - 1, -1, -1):
                            if isinstance(expanded_scope[j], MarginalizeStep):
                                nested_marg_idx = j
                                break
                        if nested_marg_idx is None:
                            raise CompileError(
                                "grouped marginalize: the body must "
                                "contain at least one `observe` step "
                                "(or an inner grouped marginalize) "
                                "whose per-row log-likelihood produces "
                                "the per-group accumulator's "
                                "contribution",
                                step.line,
                                step.col,
                            )
                        inner_marg = expanded_scope[nested_marg_idx]
                        assert isinstance(inner_marg, MarginalizeStep)
                        expanded_scope[nested_marg_idx] = MarginalizeStep(
                            var_name=inner_marg.var_name,
                            class_size=inner_marg.class_size,
                            probs_var=inner_marg.probs_var,
                            over_obj=inner_marg.over_obj,
                            over_objs=inner_marg.over_objs,
                            body_ll_var=latent_name,
                            body_observes=inner_marg.body_observes,
                            reduction=inner_marg.reduction,
                            line=inner_marg.line,
                            col=inner_marg.col,
                        )
                        body_observes.append(
                            GroupedObserveEntry(ll_slot=latent_name)
                        )
                out.extend(expanded_scope)
                # Pushforward reduction. When grouped, the
                # MarginalizeStep carries the list of per-observe
                # (ll_slot, fibration) entries the runtime callable
                # consumes; the legacy single-fibration fields are
                # gone.
                single_over = (
                    over_names[0]
                    if over_names is not None and len(over_names) == 1
                    else None
                )
                product_overs = (
                    over_names
                    if over_names is not None and len(over_names) > 1
                    else None
                )
                out.append(
                    MarginalizeStep(
                        var_name=step.vars[0],
                        class_size=class_size,
                        probs_var=probs_var,
                        over_obj=single_over,
                        over_objs=product_overs,
                        body_ll_var=step.vars[0],
                        body_observes=(
                            tuple(body_observes) if has_grouping else None
                        ),
                        reduction=step.reduction,
                        line=step.line,
                        col=step.col,
                    )
                )
            else:
                raise CompileError(
                    f"unknown bind mode {step.mode!r}",
                    step.line,
                    step.col,
                )
        return tuple(out)

    def _verify_effects(
        self, decl: ProgramDecl, steps: tuple[ProgramStep, ...]
    ) -> None:
        """Verify the program body's effect usage matches `! effects`.

        Each program step contributes to the program's *actual*
        effect set:

        * ``DrawStep`` / ``PlateDrawStep`` with ``is_observed=False``
          contribute ``Sample``.
        * ``DrawStep`` / ``VectorisedObserveStep`` with score-bind
          shape contribute ``Score``.
        * ``MarginalizeStep`` contributes ``Marginal``.
        * ``LetStep`` contributes nothing (purely deterministic).

        If the declaration includes ``! effects``, the actual set
        must be a subset of the declared set. A declared
        ``Pure`` rejects any of {Sample, Score, Marginal}.
        """
        if decl.effects is None:
            return  # unannotated → no verification
        declared = decl.effects
        actual: set[str] = set()
        for step in steps:
            if isinstance(step, (DrawStep, PlateDrawStep)):
                if isinstance(step, DrawStep) and step.is_observed:
                    actual.add("Score")
                else:
                    actual.add("Sample")
            elif isinstance(step, VectorisedObserveStep):
                actual.add("Score")
            elif isinstance(step, MarginalizeStep):
                actual.add("Marginal")
            # LetStep contributes nothing.
        if "Pure" in declared and actual:
            raise CompileError(
                f"program {decl.name!r} is declared `! Pure` but body "
                f"uses effects {sorted(actual)}",
                decl.line,
                decl.col,
            )
        unaccounted = actual - declared - {"Pure"}
        if unaccounted:
            raise CompileError(
                f"program {decl.name!r} body uses effects {sorted(unaccounted)} "
                f"not listed in `! {{{', '.join(sorted(declared))}}}`",
                decl.line,
                decl.col,
            )

    def _expand_template_calls(
        self, steps: tuple[ProgramStep, ...]
    ) -> tuple[ProgramStep, ...]:
        """Inline parametric-program-template call sites in a step list.

        A ``draw v ~ T(args)`` step whose morphism name ``T`` is a
        registered parametric program denotes the instantiation of
        the dependent kernel at the supplied arguments. The body of
        ``T`` is substituted (formal parameters → actual arguments)
        and α-renamed (every locally-bound name is prefixed by
        ``v$``, except the return-variable which is renamed to ``v``
        directly so the call's binding receives the template's
        return value). The renamed step list replaces the call site.

        Recursive template calls (a template body that itself calls
        another template) are handled by post-expansion: after a
        template is inlined its expanded steps are themselves
        recursively expanded, with cycle detection.
        """
        expanded: list[ProgramStep] = []
        for step in steps:
            if (
                isinstance(step, DrawStep)
                and not step.is_observed
                and step.morphism in self._program_templates
            ):
                tmpl = self._program_templates[step.morphism]
                if len(step.vars) != 1:
                    raise CompileError(
                        f"template call {step.morphism!r} may bind only one "
                        f"variable, got tuple {step.vars}",
                        step.line,
                        step.col,
                    )
                bind_name = step.vars[0]
                args = step.args or ()
                inst = self._instantiate_template(tmpl, bind_name, args, step)
                # Recursively expand any nested template calls in the
                # inlined body.
                expanded.extend(self._expand_template_calls(inst))
                continue
            if (
                isinstance(step, PlateDrawStep)
                and step.morphism in self._program_templates
            ):
                raise CompileError(
                    f"template {step.morphism!r} cannot be called from a "
                    f"plate-draw step; use a bare 'draw' inside the template "
                    f"body for the plate or wrap the call in a per-index "
                    f"helper",
                    step.line,
                    step.col,
                )
            expanded.append(step)
        return tuple(expanded)

    def _instantiate_template(
        self,
        tmpl: ProgramDecl,
        bind_name: str,
        args: tuple,
        call_site: ProgramStep,
    ) -> tuple[ProgramStep, ...]:
        """Realise one call site of a parametric program template.

        Categorical denotation: given the dependent kernel
        :math:`\\Pi (p_i : P_i).\\ \\mathbf{Kern}(\\mathrm{dom}(p),\\, \\mathrm{cod}(p))`
        carried by ``tmpl``, return the concrete Kern-morphism at
        ``args`` (a section of the family). The morphism is
        represented as the renamed step list whose internal latents
        contribute their own factors to the caller's joint kernel,
        with the return-variable renamed to ``bind_name`` so the
        call's binding receives the template's output value.

        Substitution + α-renaming together realise the categorical
        substitution lemma: substituting actuals for formals
        commutes with denotation up to renaming-equivalence.
        """
        type_params = tmpl.type_params or ()
        if len(args) != len(type_params):
            raise CompileError(
                f"template {tmpl.name!r} expects {len(type_params)} arguments, "
                f"got {len(args)}",
                call_site.line,
                call_site.col,
            )
        # Build the parameter-substitution environment.
        type_subst: dict[str, TypeExpr] = {}
        value_subst: dict[str, str | float] = {}
        for param, arg in zip(type_params, args):
            if isinstance(param, ObjectParam):
                if not isinstance(arg, str):
                    raise CompileError(
                        f"template {tmpl.name!r}: parameter {param.name!r} "
                        f"({param.universe}) requires a type-name argument, "
                        f"got {arg!r}",
                        call_site.line,
                        call_site.col,
                    )
                # Validate the named object/space matches the declared
                # universe.
                if param.universe == "FinSet" and arg not in self._objects:
                    raise CompileError(
                        f"template {tmpl.name!r}: parameter {param.name!r} : "
                        f"FinSet expects a finite-set object, but {arg!r} is "
                        f"not a declared object",
                        call_site.line,
                        call_site.col,
                    )
                if param.universe == "Space" and arg not in self._spaces:
                    raise CompileError(
                        f"template {tmpl.name!r}: parameter {param.name!r} : "
                        f"Space expects a continuous space, but {arg!r} is "
                        f"not a declared space",
                        call_site.line,
                        call_site.col,
                    )
                if (
                    param.universe == "Object"
                    and arg not in self._objects
                    and arg not in self._spaces
                ):
                    raise CompileError(
                        f"template {tmpl.name!r}: parameter {param.name!r} : "
                        f"Object expects a declared object or space, got {arg!r}",
                        call_site.line,
                        call_site.col,
                    )
                type_subst[param.name] = TypeName(
                    name=arg, line=call_site.line, col=call_site.col
                )
            elif isinstance(param, ScalarParam):
                if isinstance(arg, str):
                    # Scalar parameter passed as a name (e.g., a previously
                    # let-bound scalar in the caller). Pass through as a
                    # string reference; the caller's bound_vars will
                    # resolve it at draw-site time.
                    value_subst[param.name] = arg
                else:
                    value_subst[param.name] = float(arg)
            elif isinstance(param, MorphismParam):
                if not isinstance(arg, str):
                    raise CompileError(
                        f"template {tmpl.name!r}: parameter {param.name!r} : "
                        f"Mor[...] expects a morphism name, got {arg!r}",
                        call_site.line,
                        call_site.col,
                    )
                if arg not in self._morphisms and arg not in self._program_templates:
                    raise CompileError(
                        f"template {tmpl.name!r}: parameter {param.name!r}: "
                        f"morphism {arg!r} is not declared",
                        call_site.line,
                        call_site.col,
                    )
                value_subst[param.name] = arg
            else:
                raise CompileError(
                    f"template {tmpl.name!r}: unknown parameter kind for "
                    f"{getattr(param, 'name', '?')!r}",
                    call_site.line,
                    call_site.col,
                )
        # Collect all locally-bound names in the template body (latents
        # drawn, plate-draws, lets, observe loop-vars). These are
        # α-renamed to live in the caller's namespace.
        local_names = self._collect_template_local_names(tmpl)
        # The return variable (if a single identifier) receives the
        # call's binding name directly; other locals are namespaced.
        return_var = tmpl.return_vars[0] if len(tmpl.return_vars) == 1 else None
        rename: dict[str, str] = {}
        for nm in local_names:
            if nm == return_var:
                rename[nm] = bind_name
            else:
                rename[nm] = f"{bind_name}${nm}"
        # Expand the template body's BindStep IR into the
        # compiler's internal step shapes first (so the rename pass
        # operates on a uniform IR).
        expanded_body = self._expand_bind_steps(tmpl.draws)
        # Walk the expanded body, applying parameter substitution +
        # α-renaming step by step.
        return tuple(
            self._rename_step(step, type_subst, value_subst, rename)
            for step in expanded_body
        )

    def _collect_template_local_names(self, tmpl: ProgramDecl) -> set[str]:
        """All names bound inside the template body (latents + lets).

        Walks the *unexpanded* v0.5 BindStep / LetStep surface; the
        BindStep covers sample / score / marginal modes, contributing
        all bound names to the local-name set for α-renaming.
        """
        out: set[str] = set()

        def _walk(steps):
            for step in steps:
                if isinstance(step, BindStep):
                    out.update(step.vars)
                    if step.scope is not None:
                        _walk(step.scope)
                elif isinstance(step, LetStep):
                    out.add(step.name)
                # Internal IR (post-expand) — also covered, for the
                # case where _collect_local_names is invoked on
                # already-expanded steps.
                elif isinstance(step, DrawStep):
                    out.update(step.vars)
                elif isinstance(step, PlateDrawStep):
                    out.add(step.name)
                elif isinstance(step, VectorisedObserveStep):
                    out.add(step.index_var)
                    if step.response_var:
                        out.add(step.response_var)
                elif isinstance(step, MarginalizeStep):
                    out.add(step.var_name)
                    if step.body_ll_var is not None:
                        out.add(step.body_ll_var)
                elif isinstance(step, GroupedLatentInitStep):
                    out.add(step.latent_name)
                elif isinstance(step, GroupedBodyObserveStep):
                    out.add(step.index_var)
                    if step.response_var:
                        out.add(step.response_var)
                    if step.latent_name:
                        out.add(step.latent_name)

        _walk(tmpl.draws)
        return out

    def _rename_type(
        self, texpr: TypeExpr, type_subst: dict[str, TypeExpr]
    ) -> TypeExpr:
        """Substitute object parameters inside a type expression."""
        if isinstance(texpr, TypeName):
            if texpr.name in type_subst:
                return type_subst[texpr.name]
            return texpr
        if isinstance(texpr, TypeProduct):
            return TypeProduct(
                components=tuple(
                    self._rename_type(c, type_subst) for c in texpr.components
                ),
                line=texpr.line,
                col=texpr.col,
            )
        return texpr

    def _rename_args(
        self,
        args: tuple | None,
        value_subst: dict[str, str | float],
        rename: dict[str, str],
    ) -> tuple | None:
        """Apply parameter substitution and α-renaming inside a draw-arg list."""
        if args is None:
            return None
        out: list = []
        for a in args:
            if isinstance(a, str):
                if a in value_subst:
                    out.append(value_subst[a])
                elif a in rename:
                    out.append(rename[a])
                else:
                    out.append(a)
            else:
                out.append(a)
        return tuple(out)

    def _rename_step(
        self,
        step: ProgramStep,
        type_subst: dict[str, TypeExpr],
        value_subst: dict[str, str | float],
        rename: dict[str, str],
    ) -> ProgramStep:
        """Apply parameter substitution + α-renaming to a single step."""
        if isinstance(step, DrawStep):
            new_vars = tuple(rename.get(v, v) for v in step.vars)
            new_morph = value_subst.get(step.morphism, step.morphism)
            if not isinstance(new_morph, str):
                raise CompileError(
                    f"draw step morphism {step.morphism!r} substituted to a "
                    f"non-string value {new_morph!r}",
                    step.line,
                    step.col,
                )
            return DrawStep(
                vars=new_vars,
                morphism=new_morph,
                args=self._rename_args(step.args, value_subst, rename),
                is_observed=step.is_observed,
                line=step.line,
                col=step.col,
            )
        if isinstance(step, PlateDrawStep):
            new_morph = value_subst.get(step.morphism, step.morphism)
            if not isinstance(new_morph, str):
                raise CompileError(
                    f"plate-draw step morphism {step.morphism!r} substituted "
                    f"to a non-string value {new_morph!r}",
                    step.line,
                    step.col,
                )
            return PlateDrawStep(
                name=rename.get(step.name, step.name),
                index=self._rename_type(step.index, type_subst),
                codomain=self._rename_type(step.codomain, type_subst),
                morphism=new_morph,
                args=self._rename_args(step.args, value_subst, rename),
                line=step.line,
                col=step.col,
            )
        if isinstance(step, LetStep):
            return LetStep(
                name=rename.get(step.name, step.name),
                expr=self._rename_let_expr(step.expr, value_subst, rename),
                line=step.line,
                col=step.col,
            )
        if isinstance(step, VectorisedObserveStep):
            new_morph = value_subst.get(step.morphism, step.morphism)
            if not isinstance(new_morph, str):
                raise CompileError(
                    f"observe step morphism {step.morphism!r} substituted to "
                    f"a non-string value {new_morph!r}",
                    step.line,
                    step.col,
                )
            return VectorisedObserveStep(
                index_var=rename.get(step.index_var, step.index_var),
                index_set=self._rename_type(step.index_set, type_subst),
                morphism=new_morph,
                args=self._rename_args(step.args, value_subst, rename),
                response_var=rename.get(step.response_var, step.response_var),
                line=step.line,
                col=step.col,
            )
        if isinstance(step, MarginalizeStep):
            renamed_probs = (
                rename.get(step.probs_var, step.probs_var)
                if step.probs_var is not None
                else None
            )
            renamed_body_ll = (
                rename.get(step.body_ll_var, step.body_ll_var)
                if step.body_ll_var is not None
                else None
            )
            renamed_body_observes: tuple[GroupedObserveEntry, ...] | None = None
            if step.body_observes is not None:
                renamed_body_observes = tuple(
                    GroupedObserveEntry(
                        ll_slot=rename.get(entry.ll_slot, entry.ll_slot),
                        fibration_var=(
                            rename.get(entry.fibration_var, entry.fibration_var)
                            if entry.fibration_var is not None
                            else None
                        ),
                        fibration_axes=(
                            tuple(rename.get(v, v) for v in entry.fibration_axes)
                            if entry.fibration_axes is not None
                            else None
                        ),
                    )
                    for entry in step.body_observes
                )
            return MarginalizeStep(
                var_name=rename.get(step.var_name, step.var_name),
                class_size=step.class_size,
                probs_var=renamed_probs,
                over_obj=step.over_obj,
                over_objs=step.over_objs,
                body_ll_var=renamed_body_ll,
                body_observes=renamed_body_observes,
                reduction=step.reduction,
                line=step.line,
                col=step.col,
            )
        if isinstance(step, GroupedLatentInitStep):
            return GroupedLatentInitStep(
                latent_name=rename.get(step.latent_name, step.latent_name),
                class_size=step.class_size,
                line=step.line,
                col=step.col,
            )
        if isinstance(step, GroupedBodyObserveStep):
            return GroupedBodyObserveStep(
                response_var=rename.get(step.response_var, step.response_var),
                morphism=step.morphism,
                args=self._rename_args(step.args, value_subst, rename),
                index_set=self._rename_type(step.index_set, type_subst)
                if step.index_set is not None
                else None,
                index_var=rename.get(step.index_var, step.index_var),
                latent_name=rename.get(step.latent_name, step.latent_name),
                line=step.line,
                col=step.col,
            )
        raise CompileError(
            f"unsupported step kind in template body: {type(step).__name__}",
            getattr(step, "line", 0),
            getattr(step, "col", 0),
        )

    def _rename_let_expr(
        self,
        expr: LetExprNode,
        value_subst: dict[str, str | float],
        rename: dict[str, str],
    ) -> LetExprNode:
        """Apply parameter substitution + α-renaming inside a let RHS."""
        if isinstance(expr, LetExprVar):
            if expr.name in value_subst:
                val = value_subst[expr.name]
                if isinstance(val, str):
                    return LetExprVar(name=val, line=expr.line, col=expr.col)
                return LetExprLiteral(value=float(val), line=expr.line, col=expr.col)
            if expr.name in rename:
                return LetExprVar(name=rename[expr.name], line=expr.line, col=expr.col)
            return expr
        if isinstance(expr, LetExprLiteral):
            return expr
        if isinstance(expr, LetExprBinOp):
            return LetExprBinOp(
                op=expr.op,
                lhs=self._rename_let_expr(expr.lhs, value_subst, rename),
                rhs=self._rename_let_expr(expr.rhs, value_subst, rename),
                line=expr.line,
                col=expr.col,
            )
        if isinstance(expr, LetExprUnaryOp):
            return LetExprUnaryOp(
                op=expr.op,
                operand=self._rename_let_expr(expr.operand, value_subst, rename),
                line=expr.line,
                col=expr.col,
            )
        if isinstance(expr, LetExprCall):
            new_callee = value_subst.get(expr.callee, expr.callee)
            if not isinstance(new_callee, str):
                raise CompileError(
                    f"let-expression callee {expr.callee!r} substituted to "
                    f"non-string value {new_callee!r}",
                    expr.line,
                    expr.col,
                )
            return LetExprCall(
                callee=new_callee,
                args=tuple(
                    self._rename_let_expr(a, value_subst, rename) for a in expr.args
                ),
                line=expr.line,
                col=expr.col,
            )
        if isinstance(expr, LetExprIndex):
            new_arr = value_subst.get(expr.array, expr.array)
            if not isinstance(new_arr, str):
                raise CompileError(
                    f"let-expression array {expr.array!r} substituted to "
                    f"non-string value {new_arr!r}",
                    expr.line,
                    expr.col,
                )
            arr_name = (
                rename.get(new_arr, new_arr) if isinstance(new_arr, str) else new_arr
            )
            return LetExprIndex(
                array=arr_name,
                indices=tuple(
                    self._rename_let_expr(i, value_subst, rename) for i in expr.indices
                ),
                line=expr.line,
                col=expr.col,
            )
        return expr

    def _compile_morphism_call(self, expr: ExprMorphismCall):
        """Compile ``callee(arg1, arg2, …)`` — currently used to
        invoke a :class:`ContractionDecl` at a let-binding site.

        Resolves ``expr.callee`` against the registered
        contractions, validates that the argument count matches
        the contraction's expected arity, resolves each argument
        against the morphism scope, and runs the einsum-style
        contraction. Returns the result as an
        :class:`ObservedMorphism` with the contraction's declared
        domain and codomain.
        """
        from quivers.core.morphisms import ObservedMorphism

        contraction = self._contractions.get(expr.callee)
        if contraction is None:
            # Fall through to parametric-program template
            # invocation: ``let applied = p(f)`` is the existing
            # surface and should keep working when ``p`` is a
            # parametric program rather than a contraction.
            if expr.callee in self._program_templates:
                return self._compile_program_template_call(expr)
            raise CompileError(
                f"morphism-call: undefined {expr.callee!r}; "
                f"contractions: {sorted(self._contractions)}; "
                f"program templates: {sorted(self._program_templates)}",
                expr.line,
                expr.col,
            )
        if len(expr.args) != len(contraction.input_types):
            raise CompileError(
                f"contraction {expr.callee!r}: expected "
                f"{len(contraction.input_types)} arguments "
                f"({', '.join(n for n, _, _ in contraction.input_types)}), "
                f"got {len(expr.args)}",
                expr.line,
                expr.col,
            )
        resolved_morphs: list = []
        for arg_name, (param_name, exp_dom, exp_cod) in zip(
            expr.args, contraction.input_types
        ):
            if arg_name not in self._morphisms:
                raise CompileError(
                    f"contraction {expr.callee!r}: argument "
                    f"{arg_name!r} (for parameter {param_name!r}) "
                    f"is not a declared morphism",
                    expr.line,
                    expr.col,
                )
            morph = self._morphisms[arg_name]
            # Light shape check: matching numel for domain / codomain.
            if _numel_shape(morph.domain.shape) != _numel_shape(
                exp_dom.shape
            ) or _numel_shape(morph.codomain.shape) != _numel_shape(exp_cod.shape):
                raise CompileError(
                    f"contraction {expr.callee!r}: argument {arg_name!r} "
                    f"has shape {tuple(morph.domain.shape)} -> "
                    f"{tuple(morph.codomain.shape)} but parameter "
                    f"{param_name!r} declares {tuple(exp_dom.shape)} -> "
                    f"{tuple(exp_cod.shape)}",
                    expr.line,
                    expr.col,
                )
            resolved_morphs.append(morph)
        # Run the wiring.
        tensors = [m.tensor for m in resolved_morphs]
        result_tensor = contraction.wiring.apply(*tensors)
        return ObservedMorphism(
            contraction.domain,
            contraction.codomain,
            result_tensor,
            quantale=contraction.quantale,
        )

    def _compile_program_template_call(self, expr: ExprMorphismCall):
        """Instantiate a parametric program template at a let-binding
        site: ``let applied = p(f, …)`` substitutes the actual
        morphism / object / scalar arguments for the template's
        formal parameters, builds a synthetic non-parametric
        :class:`ProgramDecl`, and compiles it into a runtime
        :class:`MonadicProgram` morphism.

        Realises the dependent-kernel application
        :math:`\\Pi (p:P).\\ \\mathbf{Kern}(\\mathrm{dom}(p),\\, \\mathrm{cod}(p))`
        at the supplied section: the result is the concrete
        Kern-morphism the template denotes at those parameters,
        materialised as a standalone morphism rather than inlined
        into an enclosing program.
        """
        tmpl = self._program_templates[expr.callee]
        type_params = tmpl.type_params or ()
        args = expr.args
        if len(args) != len(type_params):
            raise CompileError(
                f"template {expr.callee!r} expects {len(type_params)} "
                f"arguments, got {len(args)}",
                expr.line,
                expr.col,
            )
        type_subst: dict[str, TypeExpr] = {}
        value_subst: dict[str, str | float] = {}
        for param, arg in zip(type_params, args):
            if isinstance(param, ObjectParam):
                if param.universe == "FinSet" and arg not in self._objects:
                    raise CompileError(
                        f"template {expr.callee!r}: parameter "
                        f"{param.name!r} : FinSet expects a finite-set "
                        f"object, but {arg!r} is not a declared object",
                        expr.line,
                        expr.col,
                    )
                if param.universe == "Space" and arg not in self._spaces:
                    raise CompileError(
                        f"template {expr.callee!r}: parameter "
                        f"{param.name!r} : Space expects a continuous "
                        f"space, but {arg!r} is not a declared space",
                        expr.line,
                        expr.col,
                    )
                if (
                    param.universe == "Object"
                    and arg not in self._objects
                    and arg not in self._spaces
                ):
                    raise CompileError(
                        f"template {expr.callee!r}: parameter "
                        f"{param.name!r} : Object expects a declared "
                        f"object or space, got {arg!r}",
                        expr.line,
                        expr.col,
                    )
                type_subst[param.name] = TypeName(
                    name=arg, line=expr.line, col=expr.col
                )
            elif isinstance(param, ScalarParam):
                value_subst[param.name] = arg
            elif isinstance(param, MorphismParam):
                if arg not in self._morphisms and arg not in self._program_templates:
                    raise CompileError(
                        f"template {expr.callee!r}: parameter "
                        f"{param.name!r}: morphism {arg!r} is not "
                        f"declared",
                        expr.line,
                        expr.col,
                    )
                value_subst[param.name] = arg
            else:
                raise CompileError(
                    f"template {expr.callee!r}: unknown parameter kind "
                    f"for {getattr(param, 'name', '?')!r}",
                    expr.line,
                    expr.col,
                )
        # Substitute parameters through the template body. We do not
        # α-rename locals here: the synthetic program owns its own
        # scope, so the template's local names (return-var, latents,
        # let-bindings) need no further refresh.
        expanded_body = self._expand_bind_steps(tmpl.draws)
        empty_rename: dict[str, str] = {}
        substituted_body = tuple(
            self._rename_step(step, type_subst, value_subst, empty_rename)
            for step in expanded_body
        )
        synth_domain = self._rename_type(tmpl.domain, type_subst)
        synth_codomain = self._rename_type(tmpl.codomain, type_subst)
        synth_name = f"__tmpl_call${expr.callee}${expr.line}${expr.col}"
        # Defensively dodge collisions if the same call site is
        # compiled twice for any reason.
        suffix = 0
        unique_name = synth_name
        while unique_name in self._morphisms:
            suffix += 1
            unique_name = f"{synth_name}#{suffix}"
        synth = ProgramDecl(
            name=unique_name,
            params=None,
            domain=synth_domain,
            codomain=synth_codomain,
            draws=substituted_body,
            return_vars=tmpl.return_vars,
            return_labels=tmpl.return_labels,
            effects=tmpl.effects,
            over_model=None,
            type_params=None,
            docs=(),
            line=expr.line,
            col=expr.col,
        )
        self._compile_program(synth)
        return self._morphisms.pop(unique_name)

    def _compile_contraction(self, decl: ContractionDecl) -> None:
        """Compile a ``contraction`` declaration into a callable
        registered under ``decl.name``.

        Categorically, a contraction is an n-ary operadic morphism:
        it takes ``len(decl.inputs)`` input morphisms, contracts
        them together under the named composition rule using the
        einsum-style wiring spec, and returns a fresh morphism
        with the declared ``domain -> codomain`` typing.

        The compiled callable accepts that many input morphisms (in
        the order declared in ``decl.inputs``) and returns an
        :class:`ObservedMorphism` whose tensor is the contraction
        result. The callable is registered in the morphism table
        so the user can invoke it like any other morphism:
        ``let out = op_apply(arg1, arg2, kernel)``.
        """
        if decl.name in self._morphisms or decl.name in self._program_templates:
            raise CompileError(
                f"contraction {decl.name!r} already declared as a morphism or program",
                decl.line,
                decl.col,
            )
        rule_name = decl.rule_name.lower()
        if rule_name not in _QUANTALE_REGISTRY:
            raise CompileError(
                f"contraction {decl.name!r}: unknown rule "
                f"{decl.rule_name!r}; available: "
                f"{', '.join(sorted(_QUANTALE_REGISTRY))}",
                decl.line,
                decl.col,
            )
        rule = _QUANTALE_REGISTRY[rule_name]
        try:
            wiring = EinsumWiring(rule, decl.wiring_spec)
        except ValueError as exc:
            raise CompileError(
                f"contraction {decl.name!r}: invalid wiring "
                f"{decl.wiring_spec!r}: {exc}",
                decl.line,
                decl.col,
            ) from exc
        expected_arity = wiring.input_arity
        if expected_arity != len(decl.inputs):
            raise CompileError(
                f"contraction {decl.name!r}: wiring spec declares "
                f"{expected_arity} inputs but the parameter list "
                f"has {len(decl.inputs)}",
                decl.line,
                decl.col,
            )
        # Resolve domain / codomain object refs.
        domain_obj = self._resolve_type(decl.domain)
        codomain_obj = self._resolve_type(decl.codomain)
        # Resolve each input's domain / codomain for validation.
        input_types = [
            (
                inp.name,
                self._resolve_type(inp.input_domain),
                self._resolve_type(inp.input_codomain),
            )
            for inp in decl.inputs
        ]
        self._contractions[decl.name] = _CompiledContraction(
            name=decl.name,
            wiring=wiring,
            domain=domain_obj,
            codomain=codomain_obj,
            input_types=input_types,
            quantale=rule,
        )

    def _compile_program(self, decl: ProgramDecl) -> None:
        """Compile a monadic program block into a MonadicProgram.

        Parametric programs (those carrying ``type_params``) are not
        compiled into a runtime ``MonadicProgram`` directly. They
        denote a dependent kernel

        .. math::

            \\Pi (p_1 : P_1) \\ldots \\Pi (p_n : P_n).\\ \\mathbf{Kern}(\\mathrm{dom}(p),\\, \\mathrm{cod}(p))

        in the indexed family of Kleisli arrows over the parameter
        category, and are stored as templates. Each call site of a
        template (a ``draw v ~ template(args)`` step inside another
        program) is realised by substituting the actual arguments
        for the formal parameters and α-renaming all locally-bound
        latents under the call's binding name, then inlining the
        renamed body into the caller's step list. The freshness of
        latent names per call site is the syntactic shadow of the
        fact that distinct call sites contribute distinct factors
        to the parent's joint kernel.
        """
        if decl.type_params is not None:
            # Parametric program — store as a template; defer body
            # compilation until each call site instantiates it.
            if decl.name in self._morphisms or decl.name in self._program_templates:
                raise CompileError(
                    f"morphism {decl.name!r} already declared",
                    decl.line,
                    decl.col,
                )
            if decl.params is not None:
                raise CompileError(
                    f"parametric program {decl.name!r} cannot also take data parameters",
                    decl.line,
                    decl.col,
                )
            self._program_templates[decl.name] = decl
            return
        if decl.name in self._morphisms:
            raise CompileError(
                f"morphism {decl.name!r} already declared", decl.line, decl.col
            )
        from quivers.continuous.programs import MonadicProgram

        domain = self._resolve_any_space(decl.domain)
        codomain = self._resolve_any_space(decl.codomain)
        from quivers.continuous.spaces import ProductSpace as _PS

        if decl.params is not None:
            if isinstance(domain, (ProductSet, _PS)):
                if len(decl.params) != len(domain.components):
                    raise CompileError(
                        f"program has {len(decl.params)} params but domain has {len(domain.components)} components",
                        decl.line,
                        decl.col,
                    )
            elif len(decl.params) != 1:
                raise CompileError(
                    f"program has {len(decl.params)} params but domain is not a product type",
                    decl.line,
                    decl.col,
                )
        bound_vars: dict[str, AnySpace | None] = {}
        if decl.params is not None:
            if isinstance(domain, (ProductSet, _PS)):
                for pname, factor in zip(decl.params, domain.components):
                    bound_vars[pname] = factor
            else:
                bound_vars[decl.params[0]] = domain
        # First, expand the v0.5 unified surface (BindStep) into the
        # internal IR (DrawStep / PlateDrawStep / VectorisedObserveStep /
        # MarginalizeStep) that the rest of the compiler consumes.
        # The expansion translates each BindStep based on its mode +
        # index annotation, and inlines marginalize scopes.
        ir_draws = self._expand_bind_steps(decl.draws)
        # Effect-set verification: walk the expanded IR and check
        # that the declared `!` capability set is consistent with
        # the body's actual effect usage.
        self._verify_effects(decl, ir_draws)
        # Then expand parametric-template call sites by inlining the
        # substituted + α-renamed template body. This realises the
        # dependent-kernel application: each call site is a section
        # of the family Π(p:P).Kern(dom(p), cod(p)) at the supplied
        # arguments, contributing its own factors to the parent's
        # joint kernel.
        expanded_draws = self._expand_template_calls(ir_draws)
        steps: list[tuple] = []
        for step in expanded_draws:
            if isinstance(step, PlateDrawStep):
                # draw v : A -> B ~ Family(args).  By the natural iso
                # Kern(1, B^A) ≅ Kern(A, B), the plate variable IS a
                # Kern-morphism A → B; we realise it as a PlateDraw
                # whose codomain is the flat product space of
                # |A| copies of the per-row family's codomain.
                from quivers.continuous.plate import PlateDraw as _PlateDraw
                from quivers.continuous.spaces import Euclidean as _Euc

                idx_space = self._resolve_any_space(step.index)
                # The per-row codomain `B` is either a declared object /
                # space or an integer literal interpreted as
                # `Euclidean(N)` — the standard convention for
                # continuous per-row families.
                if (
                    isinstance(step.codomain, TypeName)
                    and step.codomain.name.isdigit()
                    and step.codomain.name not in self._objects
                ):
                    cod_space = _Euc(
                        name=f"_plate_codom_{step.name}",
                        dim=int(step.codomain.name),
                    )
                else:
                    cod_space = self._resolve_any_space(step.codomain)
                # Synthesize a DrawStep so we can reuse the inline /
                # family-registry resolution logic. The synthetic step
                # carries the plate's per-row codomain so the family
                # is built at the right dimensionality.
                _synth = DrawStep(
                    vars=(step.name,),
                    morphism=step.morphism,
                    args=step.args,
                    is_observed=False,
                    line=step.line,
                    col=step.col,
                )
                family, step_args = self._resolve_draw_morphism(
                    _synth, bound_vars, cod_space
                )
                plate = _PlateDraw(idx_space.size, family, domain=family.domain)
                if step.name in bound_vars:
                    raise CompileError(
                        f"variable {step.name!r} already bound in program",
                        step.line,
                        step.col,
                    )
                bound_vars[step.name] = plate.codomain
                steps.append(((step.name,), plate, step_args, False))
                continue
            if isinstance(step, GroupedLatentInitStep):
                # Initialise the latent's environment slot to
                # torch.arange(K) so any downstream let-step that
                # references the latent broadcasts across the class
                # axis.
                def _grouped_latent_init(
                    env: dict, _K: int = step.class_size
                ) -> torch.Tensor:
                    return torch.arange(_K, dtype=torch.long)

                if step.latent_name in bound_vars:
                    raise CompileError(
                        f"grouped marginalize: latent {step.latent_name!r} "
                        f"is already bound in scope",
                        step.line,
                        step.col,
                    )
                bound_vars[step.latent_name] = None
                steps.append(((step.latent_name,), None, _grouped_latent_init))
                continue
            if isinstance(step, GroupedBodyObserveStep):
                # The captured observe inside a grouped marginalize
                # block: compute the family's per-row log-likelihood
                # against the supplied response data, broadcasting
                # any (K,)-shaped parameters across the class axis,
                # and store the resulting (N, K) tensor at this
                # observe's dedicated env slot.  The surrounding
                # MarginalizeStep collects each slot, pairs it with
                # the observe's fibration, and scatter-sums each
                # contribution into the shared (|G|, K) accumulator.
                idx_space = self._resolve_any_space(step.index_set)
                _synth = DrawStep(
                    vars=(step.index_var,),
                    morphism=step.morphism,
                    args=step.args,
                    is_observed=True,
                    line=step.line,
                    col=step.col,
                )
                family, step_args = self._resolve_draw_morphism(
                    _synth, bound_vars, codomain
                )
                # Register the response variable so the runtime can
                # supply data via the observations dict.
                if step.response_var not in bound_vars:
                    bound_vars[step.response_var] = family.codomain
                resp_var = step.response_var
                ll_slot = step.ll_slot or step.latent_name
                num_rows = int(idx_space.size)

                def _captured_observe(
                    env: dict,
                    _family=family,
                    _args=step_args,
                    _resp=resp_var,
                    _slot=ll_slot,
                    _N=num_rows,
                ) -> torch.Tensor:
                    # Resolve theta from env using the same input-
                    # stacking rule as the standard runtime path.
                    if _args is None:
                        # No upstream variables to gather; the
                        # family takes the program input directly.
                        theta = env["_x_input"]
                    elif len(_args) == 1:
                        theta = env[_args[0]]
                    else:
                        parts = [env[a] for a in _args]
                        # Broadcast each part to a common shape so
                        # `stack` doesn't choke on mismatched event
                        # dimensions; this preserves the per-(N, K)
                        # broadcast pattern when one part is (N,)
                        # and another is (K,).
                        common_shape = torch.broadcast_shapes(*(p.shape for p in parts))
                        expanded = [p.expand(common_shape) for p in parts]
                        theta = torch.stack(expanded, dim=-1)
                    response = env[_resp]
                    ll = _family.log_prob(theta, response)
                    # Reshape to (N, K). When theta is shape (N,)
                    # (no K dependence) the family returns (N,) and
                    # we add a trailing axis so the latent slot is
                    # always 2-D. When theta is shape (N, K) the
                    # family returns (N, K) directly.
                    if ll.dim() == 1:
                        ll = ll.unsqueeze(-1)
                    elif ll.dim() == 0:
                        ll = ll.reshape(1, 1)
                    return ll

                bound_vars[ll_slot] = None
                steps.append(((ll_slot,), None, _captured_observe))
                continue
            if isinstance(step, VectorisedObserveStep):
                # observe r[n] ~ Family(args) for n in N — the batched-
                # likelihood kernel Φ → G_{≤1}(Φ) with score
                # ∏_n p_F(r_obs(n); θ(n, φ)). Realised as a
                # VectorisedObserve wrapping the per-row family;
                # threads through the existing _StepSpec(is_observed=True)
                # path. The response buffer is supplied at runtime
                # via the `observations` dict on the program.
                from quivers.continuous.plate import (
                    VectorisedObserve as _VectorisedObserve,
                )

                idx_space = self._resolve_any_space(step.index_set)
                _synth = DrawStep(
                    vars=(step.index_var,),
                    morphism=step.morphism,
                    args=step.args,
                    is_observed=True,
                    line=step.line,
                    col=step.col,
                )
                # Use the program's declared codomain as a fallback for
                # type inference on inline distributions; the family's
                # codomain is what actually matters.
                family, step_args = self._resolve_draw_morphism(
                    _synth, bound_vars, codomain
                )
                # Build a placeholder response of the right shape; the
                # actual values are supplied at fit time via the
                # `observations[response_var]` dict-entry. The buffer
                # only carries shape information here.
                resp_shape: tuple[int, ...]
                if hasattr(family.codomain, "dim"):
                    resp_shape = (idx_space.size, int(family.codomain.dim))
                else:
                    resp_shape = (idx_space.size,) + tuple(family.codomain.shape)
                import torch as _torch

                placeholder = _torch.zeros(*resp_shape)
                vec_obs = _VectorisedObserve(family, placeholder)
                # The step's response_var is the data column supplied
                # at fit time. We expose it as the bound name so the
                # runtime's observations[response_var] = data flow
                # automatically clamps the placeholder.
                if step.response_var not in bound_vars:
                    bound_vars[step.response_var] = family.codomain
                steps.append(((step.response_var,), vec_obs, step_args, True))
                continue
            if isinstance(step, MarginalizeStep):
                # marginalize v — pushforward G(π_{Φ\\C}). Realised as a
                # deterministic let-step that applies log-sum-exp across
                # the class axis of the named variable's per-class score
                # tensor. The runtime convention is that the body
                # populates `env[v]` with a per-class log-likelihood
                # tensor before this step executes.
                if step.var_name not in bound_vars:
                    raise CompileError(
                        f"marginalize: variable {step.var_name!r} not bound",
                        step.line,
                        step.col,
                    )
                target_var = step.var_name
                # Grouped form: `over G` (single plate) or
                # `over G * H` (product plate).  Per-observe
                # fibration data lives in ``body_observes``: each
                # entry is ``(ll_slot, fibration_var,
                # fibration_axes)``.  The codegen iterates the
                # entries, builds a parallel list of (ll, idx)
                # pairs from ``env``, and calls the multi-axis
                # form of ``marginalize_grouped``.
                has_grouping = step.over_obj is not None or step.over_objs is not None
                if has_grouping:
                    over_tuple: tuple[str, ...]
                    if step.over_objs is not None:
                        over_tuple = step.over_objs
                    else:
                        assert step.over_obj is not None
                        over_tuple = (step.over_obj,)
                    for over_name in over_tuple:
                        if over_name not in self._objects:
                            raise CompileError(
                                f"grouped marginalize: `over` object "
                                f"{over_name!r} is not a declared object",
                                step.line,
                                step.col,
                            )
                    if step.probs_var is None or step.probs_var not in bound_vars:
                        raise CompileError(
                            f"grouped marginalize: categorical prior "
                            f"{step.probs_var!r} is not bound in program scope",
                            step.line,
                            step.col,
                        )
                    if not step.body_observes:
                        raise CompileError(
                            "grouped marginalize: the body must contain "
                            "at least one observe step carrying its own "
                            "`via <idx>` clause",
                            step.line,
                            step.col,
                        )
                    is_product = len(over_tuple) > 1
                    # Verify each per-observe fibration's arity and
                    # bound-ness in scope.  Each entry is either
                    # single-fibration (``fibration_var`` set) or
                    # product-fibration (``fibration_axes`` set
                    # with length equal to ``len(over_tuple)``).
                    for entry in step.body_observes:
                        slot = entry.ll_slot
                        fib_var = entry.fibration_var
                        fib_axes = entry.fibration_axes
                        if fib_var is None and fib_axes is None:
                            # Nested-marginalize entry: the inner
                            # block has already performed its own
                            # scatter; the outer block consumes its
                            # already-(|G|, K)-shaped output
                            # directly with no further fibration.
                            continue
                        if is_product:
                            if fib_axes is None or len(fib_axes) != len(over_tuple):
                                raise CompileError(
                                    "grouped marginalize: observe writing "
                                    f"to slot {slot!r} must declare a "
                                    f"`via product(...)` clause of arity "
                                    f"{len(over_tuple)} matching the "
                                    "product grouping plate",
                                    step.line,
                                    step.col,
                                )
                            for axis_name in fib_axes:
                                if axis_name not in bound_vars:
                                    raise CompileError(
                                        f"grouped marginalize: per-observe "
                                        f"`via` axis {axis_name!r} is not "
                                        "bound in program scope",
                                        step.line,
                                        step.col,
                                    )
                        else:
                            if fib_axes is not None:
                                raise CompileError(
                                    "grouped marginalize: `via product(...)` "
                                    "on an observe requires the marginalize "
                                    "header to declare a product grouping "
                                    "plate `over G * H * ...`",
                                    step.line,
                                    step.col,
                                )
                            if fib_var not in bound_vars:
                                raise CompileError(
                                    f"grouped marginalize: per-observe "
                                    f"`via` variable {fib_var!r} is not "
                                    "bound in program scope",
                                    step.line,
                                    step.col,
                                )
                    group_sizes = tuple(
                        int(self._objects[name].size) for name in over_tuple
                    )
                    num_classes = step.class_size
                    probs_var = step.probs_var
                    reduction = step.reduction or "logsumexp"
                    observe_specs = tuple(
                        (entry.ll_slot, entry.fibration_var, entry.fibration_axes)
                        for entry in step.body_observes
                    )

                    def _marginalize_grouped_callable(
                        env: dict,
                        _specs: tuple[
                            tuple[str, str | None, tuple[str, ...] | None], ...
                        ] = observe_specs,
                        _probs: str = probs_var,
                        _sizes: tuple[int, ...] = group_sizes,
                        _k: int = num_classes,
                        _reduction: str = reduction,
                        _product: bool = is_product,
                    ) -> torch.Tensor:
                        # Collect per-observe (ll, idx) pairs from
                        # env, with the shape contract that each
                        # ll's trailing axis is the class axis.
                        ll_list: list[torch.Tensor] = []
                        idx_list: list[
                            torch.Tensor | tuple[torch.Tensor, ...]
                        ] = []
                        for slot, fib_var, fib_axes in _specs:
                            ll = env[slot]
                            if ll.shape[-1] != _k:
                                raise ValueError(
                                    f"grouped marginalize: per-row "
                                    f"per-class log-likelihood at slot "
                                    f"{slot!r} must end with the class "
                                    f"axis of size {_k}; got "
                                    f"{tuple(ll.shape)}"
                                )
                            ll_list.append(ll)
                            if fib_axes is not None:
                                idx_tuple = []
                                for axis_name in fib_axes:
                                    idx = env[axis_name]
                                    if idx.dim() == 2 and idx.shape[-1] == 1:
                                        idx = idx.squeeze(-1)
                                    idx_tuple.append(idx.to(torch.long))
                                idx_list.append(tuple(idx_tuple))
                            elif fib_var is not None:
                                idx = env[fib_var]
                                if idx.dim() == 2 and idx.shape[-1] == 1:
                                    idx = idx.squeeze(-1)
                                idx_list.append(idx.to(torch.long))
                            else:
                                # Nested-marginalize entry: the
                                # inner block produced a
                                # (|G|, K)-shaped tensor; bypass
                                # scatter-add for this entry by
                                # using an identity fibration.
                                idx_list.append(
                                    torch.arange(
                                        int(ll.shape[0]),
                                        dtype=torch.long,
                                        device=ll.device,
                                    )
                                )
                        probs = env[_probs]
                        log_prior = torch.log(probs.clamp_min(1e-38))
                        result = marginalize_grouped(
                            ll_list,
                            idx_list,
                            log_prior,
                            _sizes if _product else _sizes[0],
                            reduction=_reduction,
                        )
                        # Outermost block returns a scalar; we wrap
                        # in length-1 so the surrounding joint
                        # accumulator can broadcast cleanly.
                        if result.dim() == 0:
                            return result.reshape(1)
                        return result

                    # In a nested-marginalize context, the outer
                    # block's _expand_bind_steps re-points
                    # ``body_ll_var`` at the *outer* latent name so
                    # the inner block's reduction output is bound
                    # there (where the outer block's runtime
                    # callable expects to find it). Otherwise we
                    # emit a fresh ``_marg_<var>`` slot.
                    marg_name = (
                        step.body_ll_var
                        if step.body_ll_var is not None
                        and step.body_ll_var != step.var_name
                        else f"_marg_{step.var_name}"
                    )
                    bound_vars[marg_name] = None
                    steps.append(((marg_name,), None, _marginalize_grouped_callable))
                    continue

                def _marginalize_callable(
                    env: dict, _v: str = target_var
                ) -> torch.Tensor:
                    tensor = env[_v]
                    return torch.logsumexp(tensor, dim=-1)

                marg_name = (
                    step.body_ll_var
                    if step.body_ll_var is not None
                    and step.body_ll_var != step.var_name
                    else f"_marg_{step.var_name}"
                )
                bound_vars[marg_name] = None
                steps.append(((marg_name,), None, _marginalize_callable))
                continue
            if isinstance(step, LetStep):
                if step.name in bound_vars:
                    raise CompileError(
                        f"variable {step.name!r} already bound in program",
                        step.line,
                        step.col,
                    )
                if isinstance(step.value, LetExprVar):
                    if step.value.name not in bound_vars:
                        raise CompileError(
                            f"undefined variable {step.value.name!r} in let binding",
                            step.line,
                            step.col,
                        )
                    bound_vars[step.name] = bound_vars[step.value.name]
                    steps.append(((step.name,), None, step.value.name))
                elif isinstance(step.value, LetExprLiteral):
                    bound_vars[step.name] = None
                    steps.append(((step.name,), None, step.value.value))
                else:
                    # Let-expressions inside a program body may
                    # reference compiled deductions by name (for
                    # `parse(D, ...)` calls), so we pass the
                    # compiler's deductions dict as the static
                    # `globals_` environment for variable
                    # resolution.
                    self._validate_let_expr_vars(step.value, bound_vars, step)
                    deductions_globals = dict(getattr(self, "_deductions", {}))
                    compiled_fn = self._compile_let_expr(
                        step.value,
                        globals_=deductions_globals,
                    )
                    bound_vars[step.name] = None
                    steps.append(((step.name,), None, compiled_fn))
                continue
            draw = step
            for v in draw.vars:
                if v in bound_vars:
                    raise CompileError(
                        f"variable {v!r} already bound in program", draw.line, draw.col
                    )
            morph, step_args = self._resolve_draw_morphism(draw, bound_vars, codomain)
            if step_args is not None:
                for arg_name in step_args:
                    if arg_name not in bound_vars:
                        raise CompileError(
                            f"undefined variable {arg_name!r} in draw step",
                            draw.line,
                            draw.col,
                        )
            if len(draw.vars) == 1:
                bound_vars[draw.vars[0]] = morph.codomain
            elif isinstance(morph, MonadicProgram) and (not morph._return_is_single):
                if len(draw.vars) != len(morph._return_vars):
                    raise CompileError(
                        f"destructuring {len(draw.vars)} vars but sub-program returns {len(morph._return_vars)}",
                        draw.line,
                        draw.col,
                    )
                for v in draw.vars:
                    bound_vars[v] = None
            elif isinstance(morph.codomain, ProductSet):
                if len(draw.vars) != len(morph.codomain.components):
                    raise CompileError(
                        f"destructuring {len(draw.vars)} vars but codomain has {len(morph.codomain.components)} components",
                        draw.line,
                        draw.col,
                    )
                for v, factor in zip(draw.vars, morph.codomain.components):
                    bound_vars[v] = factor
            else:
                raise CompileError(
                    f"cannot destructure non-product codomain {morph.codomain!r}",
                    draw.line,
                    draw.col,
                )
            steps.append((draw.vars, morph, step_args, draw.is_observed))
        for rv in decl.return_vars:
            if rv not in bound_vars:
                raise CompileError(
                    f"return variable {rv!r} not bound in program", decl.line, decl.col
                )
        prog = MonadicProgram(
            domain,
            codomain,
            steps,
            decl.return_vars,
            params=decl.params,
            return_labels=decl.return_labels,
            effect_set=decl.effects,
        )
        # Posterior-block routing: `over M` programs go to the
        # posterior registry rather than the morphism registry.
        if decl.over_model is not None:
            if not hasattr(self, "_posteriors"):
                self._posteriors = {}
            if decl.over_model not in self._morphisms:
                raise CompileError(
                    f"posterior block 'over {decl.over_model}' references "
                    f"undefined model {decl.over_model!r}",
                    decl.line,
                    decl.col,
                )
            self._posteriors[decl.name] = prog
        else:
            self._morphisms[decl.name] = prog

    def _resolve_draw_morphism(
        self,
        draw,
        bound_vars: dict[str, AnySpace | None],
        program_codomain: SetObject | ContinuousSpace | None,
    ) -> tuple:
        """Resolve a draw step's morphism, handling both named morphisms
        and inline distribution families.

        Parameters
        ----------
        draw : DrawStep
            The draw step to resolve.
        bound_vars : dict
            Currently bound variable types.
        program_codomain : object
            The program's declared codomain (for type inference).

        Returns
        -------
        tuple of (morphism, step_args)
            The compiled morphism and the variable-only args for
            the step spec (None = use program input).
        """
        if draw.morphism in self._morphisms:
            morph = self._morphisms[draw.morphism]
            if draw.args is not None:
                for a in draw.args:
                    if isinstance(a, (int, float)):
                        raise CompileError(
                            f"literal argument {a} not allowed for named morphism {draw.morphism!r}",
                            draw.line,
                            draw.col,
                        )
            step_args = (
                tuple((str(a) for a in draw.args)) if draw.args is not None else None
            )
            return (morph, step_args)
        from quivers.continuous.inline import (
            get_inline_param_names,
            make_inline_distribution,
        )

        param_names = get_inline_param_names(draw.morphism)
        if param_names is not None:
            if draw.args is None:
                raise CompileError(
                    f"inline distribution {draw.morphism!r} requires arguments (e.g. {draw.morphism}(...))",
                    draw.line,
                    draw.col,
                )
            inline_codomain = self._infer_inline_codomain(
                draw.morphism, draw.args, draw.vars, program_codomain
            )
            morph, var_args = make_inline_distribution(
                draw.morphism,
                draw.args,
                inline_codomain,
                variable_types={k: v for k, v in bound_vars.items() if v is not None},
            )
            return (morph, var_args)
        registry = _get_family_registry()
        if draw.morphism in registry:
            raise CompileError(
                f"distribution family {draw.morphism!r} is not supported as an inline distribution; declare it as a continuous morphism instead",
                draw.line,
                draw.col,
            )
        raise CompileError(
            f"undefined morphism or distribution family {draw.morphism!r}",
            draw.line,
            draw.col,
        )

    def _infer_inline_codomain(
        self,
        family: str,
        args: tuple,
        var_names: tuple[str, ...],
        program_codomain: object,
    ):
        """Infer the codomain for an inline distribution.

        Parameters
        ----------
        family : str
            Distribution family name.
        args : tuple
            Arguments from the draw step.
        var_names : tuple[str, ...]
            Bound variable name(s).
        program_codomain : object
            The program's declared codomain.

        Returns
        -------
        AnySpace
            The inferred codomain.
        """
        if family == "LogitNormal":
            return UnitInterval(f"_{var_names[0]}")
        elif family == "Bernoulli":
            return FinSet(name=f"_{var_names[0]}", cardinality=2)
        elif family == "Uniform":
            float_args = [a for a in args if isinstance(a, (int, float))]
            if len(float_args) >= 2:
                low, high = (float(float_args[0]), float(float_args[1]))
                if low == 0.0 and high == 1.0:
                    return UnitInterval(f"_{var_names[0]}")
                return Euclidean(name=f"_{var_names[0]}", dim=1, low=low, high=high)
            return UnitInterval(f"_{var_names[0]}")
        elif family == "TruncatedNormal":
            float_args = {
                i: a for i, a in enumerate(args) if isinstance(a, (int, float))
            }
            if 2 in float_args and 3 in float_args:
                low, high = (float(float_args[2]), float(float_args[3]))
                return Euclidean(name=f"_{var_names[0]}", dim=1, low=low, high=high)
            return UnitInterval(f"_{var_names[0]}")
        elif family == "Normal":
            return Euclidean(name=f"_{var_names[0]}", dim=1)
        elif family == "Beta":
            return UnitInterval(f"_{var_names[0]}")
        elif family in (
            "Exponential",
            "HalfCauchy",
            "HalfNormal",
            "LogNormal",
            "Gamma",
        ):
            return PositiveReals(name=f"_{var_names[0]}", dim=1)
        elif family == "Dirichlet":
            # Inline Dirichlet's simplex dimension:
            #
            # * ``Dirichlet([a_1, …, a_K])`` (parser flattens the
            #   bracketed numeric sequence into K positional literal
            #   floats) → K-simplex.
            # * ``Dirichlet(alpha)`` with a single scalar literal →
            #   simplex dimension comes from the program's declared
            #   codomain (``dim`` for a ContinuousSpace,
            #   ``cardinality`` for a SetObject), defaulting to 2.
            sim_dim: int | None = None
            n_literals = sum(1 for a in args if isinstance(a, (int, float)))
            if n_literals >= 2:
                sim_dim = n_literals
            if sim_dim is None:
                for a in args:
                    if isinstance(a, (list, tuple)) and len(a) > 0:
                        sim_dim = len(a)
                        break
            if sim_dim is None and isinstance(program_codomain, ContinuousSpace):
                sim_dim = getattr(program_codomain, "dim", None)
            if sim_dim is None and isinstance(program_codomain, SetObject):
                # A discrete codomain of cardinality `k` indexes a
                # k-simplex of class probabilities.
                sim_dim = getattr(program_codomain, "cardinality", None)
            if sim_dim is None or sim_dim < 2:
                sim_dim = 2
            return Simplex(name=f"_{var_names[0]}", dim=sim_dim)
        else:
            return Euclidean(name=f"_{var_names[0]}", dim=1)

    def _validate_let_expr_vars(
        self, node: LetExprNode, bound_vars: dict[str, AnySpace | None], step: LetStep
    ) -> None:
        """Validate that all variables in a let expression are bound.

        The validator tolerates references to compiled deductions
        (via ``self._deductions``) and lambdas that bind their own
        parameter — both of these are resolved at runtime by the
        let-expression evaluator's `globals_` channel and its
        lambda-environment extension.

        Free variables that match none of the above are treated as
        *host-data references*: the value is expected to arrive at
        runtime through :func:`quivers.inference.condition`'s data
        dict (e.g. per-row index arrays for hierarchical regression).
        The runtime raises a clear ``KeyError`` if the data dict
        doesn't supply such a name.
        """
        deductions = getattr(self, "_deductions", {})

        # Inner walker carries a set of locally-bound names from
        # surrounding lambdas, so a lambda's `param` is treated as
        # in-scope inside its body.
        def _walk(node, locals_set: set[str]) -> None:
            if isinstance(node, LetExprVar):
                if node.name in bound_vars or node.name in deductions:
                    return
                if node.name in locals_set:
                    return
                # Otherwise the variable is a host-data reference; the
                # trace runtime resolves it against the conditioning
                # data dict at execution time.
                return
            if isinstance(node, LetExprBinOp):
                _walk(node.left, locals_set)
                _walk(node.right, locals_set)
            elif isinstance(node, LetExprUnaryOp):
                _walk(node.operand, locals_set)
            elif isinstance(node, LetExprCall):
                for arg in node.args:
                    _walk(arg, locals_set)
            elif isinstance(node, LetExprList):
                for item in node.items:
                    _walk(item, locals_set)
            elif isinstance(node, LetExprLambda):
                _walk(node.body, locals_set | {node.param})
            elif isinstance(node, LetExprMethodCall):
                _walk(node.receiver, locals_set)
                for arg in node.args:
                    _walk(arg, locals_set)
            elif isinstance(node, LetExprIndex):
                _walk(node.array, locals_set)
                for idx in node.indices:
                    _walk(idx, locals_set)
            # LetExprLiteral, LetExprString carry no variables.

        _walk(node, set())

    @staticmethod
    def _compile_let_expr(
        node: LetExprNode,
        globals_: "dict[str, Any] | None" = None,
    ) -> Callable[[dict[str, "Any"]], "Any"]:
        """Compile a let expression tree into a callable.

        The returned callable takes an environment dict mapping
        names to Python values (tensors, strings, lists, lambdas,
        chart views, structured tuples, ...) and returns the
        expression's value. The let-sublanguage is a small typed
        lambda calculus over heterogeneous values; the runtime
        evaluator preserves autograd through all tensor
        operations.

        Supported node kinds:

        * :class:`LetExprLiteral` — numeric literal → tensor.
        * :class:`LetExprString` — string literal → Python str.
        * :class:`LetExprVar` — variable reference → env lookup.
        * :class:`LetExprBinOp` — arithmetic over tensors.
        * :class:`LetExprUnaryOp` — negation.
        * :class:`LetExprList` — list literal → Python list.
        * :class:`LetExprLambda` — closure over the let environment.
        * :class:`LetExprMethodCall` — dispatch on receiver type.
        * :class:`LetExprCall` — built-in or constructor mode.
        * :class:`LetExprIndex` — tensor gather.
        """
        if isinstance(node, LetExprLiteral):
            val = node.value

            def _literal(env: dict) -> torch.Tensor:
                for v in env.values():
                    if isinstance(v, torch.Tensor):
                        return torch.tensor(val, device=v.device)
                return torch.tensor(val)

            return _literal
        if isinstance(node, LetExprString):
            val = node.value

            def _string(env: dict) -> str:
                return val

            return _string
        if isinstance(node, LetExprVar):
            name = node.name
            globs = globals_ or {}
            constructors = globs.get("__constructors__", frozenset())

            def _var(env: dict):
                if name in env:
                    return env[name]
                if name in constructors:
                    return (name,)
                if name in globs and name != "__constructors__":
                    return globs[name]
                raise CompileError(f"undefined variable {name!r} in let expression")

            return _var
        if isinstance(node, LetExprList):
            item_fns = [
                Compiler._compile_let_expr(it, globals_=globals_) for it in node.items
            ]

            def _list(env: dict) -> list:
                return [fn(env) for fn in item_fns]

            return _list
        if isinstance(node, LetExprLambda):
            param = node.param
            body_fn = Compiler._compile_let_expr(node.body, globals_=globals_)

            def _lambda(env: dict):
                # Returns a Python callable closed over the let-env.
                def _closure(arg):
                    extended = dict(env)
                    extended[param] = arg
                    return body_fn(extended)

                return _closure

            return _lambda
        if isinstance(node, LetExprBinOp):
            left_fn = Compiler._compile_let_expr(node.left, globals_=globals_)
            right_fn = Compiler._compile_let_expr(node.right, globals_=globals_)
            op = node.op

            def _binop(env: dict) -> torch.Tensor:
                l = left_fn(env)
                r = right_fn(env)
                # Promote scalar / int values to tensors.
                if not isinstance(l, torch.Tensor):
                    l = torch.tensor(float(l))
                if not isinstance(r, torch.Tensor):
                    r = torch.tensor(float(r))
                l, r = torch.broadcast_tensors(l, r)
                if op == "+":
                    return l + r
                elif op == "-":
                    return l - r
                elif op == "*":
                    return l * r
                elif op == "/":
                    return l / r
                raise ValueError(f"unknown operator: {op}")

            return _binop
        if isinstance(node, LetExprUnaryOp):
            inner_fn = Compiler._compile_let_expr(node.operand, globals_=globals_)

            def _neg(env: dict):
                v = inner_fn(env)
                if isinstance(v, torch.Tensor):
                    return -v
                return -v

            return _neg
        if isinstance(node, LetExprMethodCall):
            recv_fn = Compiler._compile_let_expr(node.receiver, globals_=globals_)
            method = node.method
            arg_fns = [
                Compiler._compile_let_expr(a, globals_=globals_) for a in node.args
            ]

            def _method(env: dict):
                receiver = recv_fn(env)
                args = [fn(env) for fn in arg_fns]
                fn = getattr(receiver, method, None)
                if fn is None:
                    raise CompileError(
                        f"object {type(receiver).__name__!r} has no method {method!r}"
                    )
                return fn(*args)

            return _method
        if isinstance(node, LetExprCall):
            func_name = node.func
            arg_fns = [
                Compiler._compile_let_expr(a, globals_=globals_) for a in node.args
            ]

            # Built-in tensor operations.
            _TENSOR_BUILTINS = {
                "sigmoid": lambda a: torch.sigmoid(a),
                "exp": lambda a: torch.exp(a),
                "log": lambda a: torch.log(a),
                "abs": lambda a: torch.abs(a),
                "softplus": lambda a: torch.nn.functional.softplus(a),
                "cumsum": lambda a: torch.cumsum(a, dim=-1),
                "softmax": lambda a: torch.softmax(a, dim=-1),
                "log1p": lambda a: torch.log1p(a),
                "sqrt": lambda a: torch.sqrt(a),
                "neg": lambda a: -a,
            }

            def _call(env: dict):
                # Higher-order combinators come first; they consume
                # raw closure args without eager evaluation of the
                # lambda body.
                if func_name == "length":
                    val = arg_fns[0](env)
                    if isinstance(val, list):
                        return float(len(val))
                    if isinstance(val, torch.Tensor):
                        return float(val.shape[0])
                    if isinstance(val, tuple):
                        return float(len(val))
                    raise CompileError(
                        f"length() does not support {type(val).__name__}"
                    )
                if func_name in ("map", "filter"):
                    coll = arg_fns[0](env)
                    fn = arg_fns[1](env)
                    if func_name == "map":
                        return [fn(x) for x in coll]
                    return [x for x in coll if fn(x)]
                if func_name == "fold":
                    # fold(list, init, accumulator_lambda)
                    coll = arg_fns[0](env)
                    init = arg_fns[1](env)
                    fn = arg_fns[2](env)
                    acc = init
                    for x in coll:
                        # The lambda takes one arg (the current
                        # element); inner closures handle accumulation
                        # via additional lambda nesting:
                        #   fold(xs, 0, x -> acc_so_far -> acc + x)
                        # ... this is awkward. Two-argument folds
                        # work better; we expose `fold` as taking a
                        # lambda whose body is itself a lambda
                        # (curried), invoked here as `fn(x)(acc)`.
                        if callable(fn):
                            step = fn(acc)
                            if callable(step):
                                acc = step(x)
                            else:
                                acc = step
                        else:
                            raise CompileError("fold's accumulator must be a lambda")
                    return acc
                if func_name == "logsumexp_over":
                    # logsumexp_over(list, lambda x -> log_weight_x)
                    coll = arg_fns[0](env)
                    fn = arg_fns[1](env)
                    if not coll:
                        return torch.tensor(-float("inf"))
                    weights = []
                    for x in coll:
                        w = fn(x)
                        if not isinstance(w, torch.Tensor):
                            w = torch.tensor(float(w))
                        weights.append(w)
                    return torch.logsumexp(torch.stack(weights), dim=0)
                if func_name == "logsumexp":
                    # logsumexp(a, b, ...) over an explicit list of args
                    coll = [fn(env) for fn in arg_fns]
                    coll = [
                        torch.tensor(float(c)) if not isinstance(c, torch.Tensor) else c
                        for c in coll
                    ]
                    return torch.logsumexp(torch.stack(coll), dim=0)
                if func_name == "parse":
                    # parse(D, input) — invoke a registered deduction
                    # on an axiom list / input. The first arg is a
                    # let_var naming a compiled DeductionSystem; the
                    # remaining arg is the input to feed it.
                    # The runtime env carries a `__compiler__` key
                    # set by the program runner with the compiled
                    # deductions dict.
                    if len(arg_fns) != 2:
                        raise CompileError(
                            "parse() takes exactly two arguments: "
                            "deduction-name and input"
                        )
                    ded = arg_fns[0](env)
                    inp = arg_fns[1](env)
                    if hasattr(ded, "__call__"):
                        return ded(inp)
                    raise CompileError(
                        f"parse() first arg must be a DeductionSystem, "
                        f"got {type(ded).__name__}"
                    )
                # Standard scalar / tensor builtins.
                if func_name in _TENSOR_BUILTINS:
                    args = [fn(env) for fn in arg_fns]
                    return _TENSOR_BUILTINS[func_name](args[0])
                if func_name == "cholesky_quad_form":
                    args = [fn(env) for fn in arg_fns]
                    L_flat, scale = args[0], args[1]
                    K = scale.shape[-1]
                    L = L_flat.reshape(*L_flat.shape[:-1], K, K)
                    mask = torch.tril(torch.ones(K, K, device=L.device, dtype=L.dtype))
                    L = L * mask
                    R = L @ L.transpose(-1, -2)
                    D = scale.unsqueeze(-1) * torch.eye(
                        K, device=L.device, dtype=L.dtype
                    )
                    cov = D @ R @ D
                    return cov.reshape(*cov.shape[:-2], K * K)
                # Constructor mode: build a tuple `(func_name, *args)`
                # only when `func_name` is in the user-declared
                # constructor set (passed via `globals_["__constructors__"]`).
                # The free term algebra over named constructor symbols
                # is thus fully under the user's control — no
                # identifier is silently treated as a constructor.
                constructors = (globals_ or {}).get("__constructors__", frozenset())
                if func_name in constructors:
                    args = [fn(env) for fn in arg_fns]
                    return (func_name, *args)
                raise CompileError(
                    f"unknown function {func_name!r} in let expression; "
                    f"declare it as a constructor (e.g., in a deduction's "
                    f"`atoms` block) or use a registered builtin"
                )

            return _call
        if isinstance(node, LetExprIndex):
            # Indexed gather along the leading axis of the array.
            # Realises the Kleisli pullback ι^* v = v ∘ ι for a finite
            # fibration ι : N → A and a plate variable v : A → B.
            arr_fn = Compiler._compile_let_expr(node.array, globals_=globals_)
            idx_fns = [
                Compiler._compile_let_expr(ix, globals_=globals_) for ix in node.indices
            ]

            def _index(env: dict) -> torch.Tensor:
                arr = arr_fn(env)
                idx_tensors = [fn(env) for fn in idx_fns]
                # Cast each index to a long-typed tensor; broadcast and
                # use advanced indexing along the leading dims of arr.
                long_idx = tuple(
                    ix.to(torch.long) if ix.dtype != torch.long else ix
                    for ix in idx_tensors
                )
                return arr[long_idx]

            return _index
        raise CompileError(f"unknown let expression node: {type(node).__name__}")

    def _compile_let(self, decl: LetDecl) -> None:
        """Compile a let-binding with optional where clause.

        The RHS is first classified by surface shape:

        * If it's an expression that denotes a transformation
          (bare reference to a registered trans singleton or
          let-bound trans, a constructor call against the
          transformation catalog, or ``t1 >>> t2``), the binding
          lands in :attr:`_transformations`.
        * Otherwise it's compiled as a morphism expression and
          lands in :attr:`_morphisms`.

        The two namespaces are disjoint: a name cannot be used as
        both a morphism and a transformation in the same module.
        """
        if hasattr(decl, "where") and decl.where:
            for where_decl in decl.where:
                self._compile_let(where_decl)
        if decl.name in self._morphisms or decl.name in self._transformations:
            raise CompileError(f"name {decl.name!r} already bound", decl.line, decl.col)
        if self._is_trans_expr(decl.expr):
            self._transformations[decl.name] = self._compile_trans_expr(decl.expr)
            return
        morph = self._compile_expr(decl.expr)
        self._morphisms[decl.name] = morph

    def _is_trans_expr(self, expr) -> bool:
        """Return True iff ``expr`` denotes a transformation value.

        The classification is *purely structural* — based on the
        expression's surface shape — and is the criterion the
        let-binding logic uses to choose between the morphism and
        transformation namespaces.  An :class:`ExprMorphismCall`
        whose callee is in :attr:`_trans_constructors` is a
        transformation; the same shape with a callee in
        :attr:`_contractions` is a morphism.
        """
        if isinstance(expr, ExprTransCompose):
            return True
        if isinstance(expr, ExprIdent):
            return (
                expr.name in self._trans_singletons
                or expr.name in self._transformations
                or expr.name in self._trans_constructors
            )
        if isinstance(expr, ExprMorphismCall):
            return expr.callee in self._trans_constructors
        return False

    # ------------------------------------------------------------------
    # Structural-compression compilation
    # ------------------------------------------------------------------

    def _compile_signature(self, decl: SignatureDecl) -> None:
        """Register a signature declaration.

        Builds a runtime :class:`quivers.structural.Signature` from
        the AST node, stashes it on ``self._signatures`` keyed by
        name. Performs sort coverage, codomain validity, and binder
        sort-consistency checks.
        """
        if not hasattr(self, "_signatures"):
            self._signatures: dict[str, Signature] = {}

        if decl.name in self._signatures:
            raise CompileError(
                f"signature {decl.name!r} already declared",
                decl.line,
                decl.col,
            )

        # Sort table.
        sorts: dict[str, Sort] = {}
        for s in decl.sorts:
            if s.name in sorts:
                raise CompileError(
                    f"signature {decl.name!r}: duplicate sort {s.name!r}",
                    s.line,
                    s.col,
                )
            if s.vocab and s.kind != "data":
                raise CompileError(
                    f"signature {decl.name!r}: vocab clause is only valid "
                    f"on `data` sorts; sort {s.name!r} has kind {s.kind!r}",
                    s.line,
                    s.col,
                )
            vocab_entries: list[SortVocabEntry] = []
            seen_vals: set = set()
            for lit in s.vocab:
                value = _decode_vocab_literal(decl.name, s.name, lit)
                if value in seen_vals:
                    raise CompileError(
                        f"signature {decl.name!r}: sort {s.name!r} vocabulary "
                        f"contains duplicate entry {value!r}",
                        s.line,
                        s.col,
                    )
                seen_vals.add(value)
                vocab_entries.append(SortVocabEntry(kind=lit.kind, value=value))
            sorts[s.name] = Sort(
                name=s.name,
                kind=s.kind,
                dim=s.dim,
                vocab=tuple(vocab_entries),
            )

        # Vertex / edge kinds (graph-shaped signatures).
        vertex_kinds: dict[str, VertexKind] = {}
        for v in decl.vertex_kinds:
            if v.name in vertex_kinds:
                raise CompileError(
                    f"signature {decl.name!r}: duplicate vertex_kind {v.name!r}",
                    v.line,
                    v.col,
                )
            vertex_kinds[v.name] = VertexKind(name=v.name, kind=v.kind, dim=v.dim)
        edge_kinds: dict[str, EdgeKind] = {}
        for e in decl.edge_kinds:
            if e.name in edge_kinds:
                raise CompileError(
                    f"signature {decl.name!r}: duplicate edge_kind {e.name!r}",
                    e.line,
                    e.col,
                )
            if e.src not in vertex_kinds:
                raise CompileError(
                    f"signature {decl.name!r}: edge_kind {e.name!r} has "
                    f"unknown source vertex_kind {e.src!r}",
                    e.line,
                    e.col,
                )
            if e.tgt not in vertex_kinds:
                raise CompileError(
                    f"signature {decl.name!r}: edge_kind {e.name!r} has "
                    f"unknown target vertex_kind {e.tgt!r}",
                    e.line,
                    e.col,
                )
            edge_kinds[e.name] = EdgeKind(
                name=e.name,
                src=e.src,
                tgt=e.tgt,
                directed=e.directed,
            )

        # Constructors. Every sort mentioned in a constructor must
        # be declared in the signature's `sorts { … }` block —
        # auto-registering an undeclared sort would mask a real
        # declaration error and leave its dim unspecified.
        _RESERVED_OP_NAMES = {"BoundVar", "Data"}
        constructors: dict[str, Constructor] = {}
        for c in decl.constructors:
            if c.name in _RESERVED_OP_NAMES:
                raise CompileError(
                    f"signature {decl.name!r}: constructor name {c.name!r} "
                    f"is reserved by the framework",
                    c.line,
                    c.col,
                )
            if c.name in constructors:
                raise CompileError(
                    f"signature {decl.name!r}: duplicate constructor {c.name!r}",
                    c.line,
                    c.col,
                )
            for s in c.domain:
                if s not in sorts:
                    raise CompileError(
                        f"signature {decl.name!r}: constructor {c.name!r} "
                        f"references undeclared sort {s!r}; declare it in "
                        f"the signature's `sorts {{ … }}` block",
                        c.line,
                        c.col,
                    )
            if c.codomain not in sorts:
                raise CompileError(
                    f"signature {decl.name!r}: constructor {c.name!r} has "
                    f"unknown codomain sort {c.codomain!r}",
                    c.line,
                    c.col,
                )
            constructors[c.name] = Constructor(
                name=c.name,
                domain=c.domain,
                codomain=c.codomain,
            )

        # Binders. Every sort a binder mentions (variable sort,
        # annotation sort, scoped argument sort, codomain) must
        # already be declared in the signature's `sorts { … }` block
        # — binders introduce structural recursion, so silently
        # auto-registering an object sort would mask a real
        # declaration error and produce a sort whose dim the user
        # never specified.
        binders: dict[str, Binder] = {}
        for b in decl.binders:
            if b.name in _RESERVED_OP_NAMES:
                raise CompileError(
                    f"signature {decl.name!r}: binder name {b.name!r} is "
                    f"reserved by the framework",
                    b.line,
                    b.col,
                )
            if b.name in binders or b.name in constructors:
                raise CompileError(
                    f"signature {decl.name!r}: duplicate binder {b.name!r}",
                    b.line,
                    b.col,
                )
            for v in b.binds:
                if v.sort not in sorts:
                    raise CompileError(
                        f"signature {decl.name!r}: binder {b.name!r} introduces "
                        f"variable of undeclared sort {v.sort!r}",
                        b.line,
                        b.col,
                    )
                if v.annot_sort is not None and v.annot_sort not in sorts:
                    raise CompileError(
                        f"signature {decl.name!r}: binder {b.name!r} variable "
                        f"{v.var!r} annotated by undeclared sort "
                        f"{v.annot_sort!r}",
                        b.line,
                        b.col,
                    )
            for a in b.scoped:
                if a.sort not in sorts:
                    raise CompileError(
                        f"signature {decl.name!r}: binder {b.name!r} scoped arg "
                        f"{a.arg!r} has undeclared sort {a.sort!r}",
                        b.line,
                        b.col,
                    )
            if b.codomain not in sorts:
                raise CompileError(
                    f"signature {decl.name!r}: binder {b.name!r} has "
                    f"unknown codomain sort {b.codomain!r}",
                    b.line,
                    b.col,
                )
            binders[b.name] = Binder(
                name=b.name,
                binds=tuple(
                    BinderVarSpec(
                        var=v.var,
                        sort=v.sort,
                        annot_sort=v.annot_sort,
                    )
                    for v in b.binds
                ),
                scoped=tuple(BinderArgSpec(arg=a.arg, sort=a.sort) for a in b.scoped),
                codomain=b.codomain,
            )

        sig = Signature(
            name=decl.name,
            params=decl.params,
            sorts_t=tuple(sorts.values()),
            constructors_t=tuple(constructors.values()),
            binders_t=tuple(binders.values()),
            vertex_kinds_t=tuple(vertex_kinds.values()),
            edge_kinds_t=tuple(edge_kinds.values()),
        )
        self._signatures[decl.name] = sig

    def _resolve_dim(
        self,
        sig: "Signature",
        sort: str,
        overrides: dict[str, int],
        diag_owner: str,
    ) -> int:
        """Resolve a sort's embedding dimension.

        Priority: the per-encoder / per-decoder dim override
        from the DSL block (``dim Term = 64``), then the
        signature's sort declaration. Raises if neither supplies a
        dim — the user must specify one somewhere.
        """
        if sort in overrides:
            return overrides[sort]
        d = sig.sort_dim(sort)
        if d is not None:
            return d
        raise CompileError(
            f"{diag_owner}: sort {sort!r} has no dim — declare it on the "
            f"signature's `sorts {{ … }}` block (e.g. `Term : object dim 64`) "
            f"or override it on the encoder / decoder block "
            f"(e.g. `dim Term = 64`)"
        )

    def _compile_encoder(self, decl: EncoderDecl) -> None:
        """Compile a encoder block into a runtime Encoder module."""
        if not hasattr(self, "_encoders"):
            self._encoders: dict[str, Encoder] = {}
        if not hasattr(self, "_signatures"):
            self._signatures = {}

        if decl.signature not in self._signatures:
            raise CompileError(
                f"encoder {decl.name!r}: unknown signature {decl.signature!r}",
                decl.line,
                decl.col,
            )
        sig = self._signatures[decl.signature]

        # Per-sort dim resolution.
        overrides: dict[str, int] = {sd.sort: sd.dim for sd in decl.dims}
        sort_dims: dict[str, int] = {}
        _diag = f"encoder {decl.name!r}"
        for s_name, s in sig.sorts.items():
            sort_dims[s_name] = self._resolve_dim(
                sig,
                s_name,
                overrides,
                _diag,
            )
        for v_name in sig.vertex_kinds:
            sort_dims[v_name] = self._resolve_dim(
                sig,
                v_name,
                overrides,
                _diag,
            )

        # Set the compiler's per-let globals so let-expressions in
        # per-op bodies can reference other module-level morphisms,
        # signatures, encoders, deductions, etc.
        globs = self._lex_globals_for_structural()

        modules_owned: list[nn.Module] = []
        op_fns: dict[str, _PerOpFn] = {}

        for rule in decl.op_rules:
            op = rule.op
            if op in sig.constructors:
                domain = sig.constructors[op].domain
            elif op in sig.binders:
                # `Binder.domain()` already produces the positional
                # sort sequence in the order the per-op function
                # receives children: annotation sorts (one per
                # annotated bound variable, outer-context) followed
                # by scoped argument sorts (extended-context).
                domain = sig.binders[op].domain()
            else:
                raise CompileError(
                    f"encoder {decl.name!r}: op {op!r} is not in signature "
                    f"{sig.name!r}",
                    rule.line,
                    rule.col,
                )
            args = rule.args

            if rule.args and len(rule.args) != len(domain):
                raise CompileError(
                    f"encoder {decl.name!r}: op {op!r} expects "
                    f"{len(domain)} arguments, got {len(rule.args)}",
                    rule.line,
                    rule.col,
                )

            body_fn = self._compile_let_expr(rule.body, globals_=globs)

            def make_call(
                body_fn=body_fn,
                args_=args,
                mode=rule.mode,
                state_var=rule.state_var,
                prefix_var=rule.prefix_var,
            ):
                if mode == "recurrent":
                    # The body sees the named children plus an
                    # alias `state_var` for the recursive child's
                    # already-computed embedding.
                    def call(*children):
                        env = {name: child for name, child in zip(args_, children)}
                        if state_var is not None:
                            # Convention: the recursive child is the
                            # last positional in the surface form
                            # `Cons(head, tail) recurrent state |-> ...`.
                            env[state_var] = children[-1]
                        return body_fn(env)

                    return call
                if mode == "attention":
                    # Children are the non-recursive args followed by
                    # (prefix_list, current_step_state) supplied by
                    # `_compress_attention_chain`.
                    def call(*children_with_extras):
                        non_rec = list(children_with_extras[:-2])
                        prefix_list = children_with_extras[-2]
                        state_arg = children_with_extras[-1]
                        # `args_` names the non-recursive children
                        # plus the recursive arg (as declared in the
                        # source). The recursive arg name is the
                        # last in `args_`; it sees the running step
                        # state, mirroring `recurrent`.
                        env = {name: child for name, child in zip(args_[:-1], non_rec)}
                        if args_:
                            env[args_[-1]] = state_arg
                        if prefix_var is not None:
                            env[prefix_var] = prefix_list
                        return body_fn(env)

                    return call

                def call(*children):
                    env = {name: child for name, child in zip(args_, children)}
                    return body_fn(env)

                return call

            op_fns[op] = _PerOpFn(
                op=op,
                mode=rule.mode,
                args=args,
                fn=make_call(),
                state_var=rule.state_var,
                prefix_var=rule.prefix_var,
            )

        # Scaffold defaults for any constructor / binder not given a
        # rule by the user. We compute the per-argument dim sequence
        # in the exact order the framework passes children to the
        # per-op function.
        for op_name in list(sig.constructors) + list(sig.binders):
            if op_name in op_fns:
                continue
            if op_name in sig.constructors:
                c = sig.constructors[op_name]
                arg_dims = tuple(sort_dims[s] for s in c.domain)
                out_dim = sort_dims[c.codomain]
            else:
                b = sig.binders[op_name]
                arg_dims = tuple(sort_dims[s] for s in b.domain())
                out_dim = sort_dims[b.codomain]
            mod, call = make_default_op_fn(op_name, arg_dims, out_dim)
            modules_owned.append(mod)
            op_fns[op_name] = _PerOpFn(
                op=op_name,
                mode="plain",
                args=(),
                fn=call,
            )

        # var_init functions for binders. We allocate one per
        # (variable_sort, annotation_sort) pair that actually appears
        # in the signature's binders, plus one per unannotated
        # variable sort. Each is a learned 2-layer MLP from the
        # annotation's dim (or zero, for unannotated) to the
        # variable sort's dim.
        var_init_fns: dict = {}
        seen_keys: set = set()
        for b in sig.binders.values():
            for spec in b.binds:
                key: tuple[str, str] | str
                if spec.annot_sort is not None:
                    key = (spec.sort, spec.annot_sort)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    in_dim = sort_dims[spec.annot_sort]
                    out_dim = sort_dims[spec.sort]
                    mod, call = make_default_var_init(in_dim, out_dim)
                    modules_owned.append(mod)
                    var_init_fns[key] = call
                else:
                    key = spec.sort
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    out_dim = sort_dims[spec.sort]
                    init_param = nn.Parameter(torch.randn(out_dim) * 0.1)
                    holder = nn.Module()
                    holder.register_parameter(
                        f"unannot_var_{spec.sort}",
                        init_param,
                    )
                    modules_owned.append(holder)

                    def make_unannot(p=init_param):
                        def call(_annot=None):
                            return p

                        return call

                    var_init_fns[key] = make_unannot()

        # User-supplied per-(var_sort, annot_sort) var_init bodies.
        # Each `var_init <V> from <A> as ty |-> body` declaration
        # overrides the scaffolded default for that exact pair; an
        # omitted `from <A>` clause refers to the unannotated case.
        for vi in decl.var_inits:
            body_fn = self._compile_let_expr(vi.body, globals_=globs)
            if vi.annot_sort is None:
                key: tuple[str, str] | str = vi.var_sort

                def make_call(body_fn=body_fn):
                    def call(_annot=None):
                        return body_fn({})

                    return call

                var_init_fns[key] = make_call()
            else:
                if vi.ty is None:
                    raise CompileError(
                        f"encoder {decl.name!r}: var_init for "
                        f"{vi.var_sort!r} from {vi.annot_sort!r} requires "
                        f"an `as <name>` clause to bind the annotation "
                        f"embedding in the body",
                        vi.line,
                        vi.col,
                    )
                key = (vi.var_sort, vi.annot_sort)

                def make_call(body_fn=body_fn, arg=vi.ty):
                    def call(ty):
                        return body_fn({arg: ty})

                    return call

                var_init_fns[key] = make_call()

        # Data embedders: one learnable table per data sort, keyed by
        # the registered vocabulary (built as encountered).
        data_embedders = self._build_data_embedders(sig, sort_dims, modules_owned)

        # Graph specialization.
        iterations = decl.iterations or 0
        init_fns: dict[str, "Callable"] = {}
        message_fns: dict[str, "Callable"] = {}
        update_fns: dict[str, "Callable"] = {}
        readout = None
        for ir in decl.init_rules:
            ib = self._compile_let_expr(ir.body, globals_=globs)

            def init_call(payload, body_fn=ib, arg=ir.arg):
                return body_fn({arg: payload})

            init_fns[ir.kind] = init_call
        for mr in decl.message_rules:
            mb = self._compile_let_expr(mr.body, globals_=globs)

            def msg_call(s, t, body_fn=mb, sv=mr.src, tv=mr.tgt):
                return body_fn({sv: s, tv: t})

            message_fns[mr.edge_kind] = msg_call
        for ur in decl.update_rules:
            ub = self._compile_let_expr(ur.body, globals_=globs)

            def upd_call(slf, msgs, body_fn=ub, sv=ur.self_var, mv=ur.msgs_var):
                return body_fn({sv: slf, mv: msgs})

            update_fns[ur.vertex_kind] = upd_call
        if decl.readout is not None:
            rb = self._compile_let_expr(decl.readout, globals_=globs)

            def readout_call(embeds, body_fn=rb):
                return body_fn({"embeds": embeds})

            readout = readout_call

        comp = Encoder(
            name=decl.name,
            signature=sig,
            sort_dims=sort_dims,
            op_fns=op_fns,
            var_init_fns=var_init_fns,
            data_embedders=data_embedders,
            modules_owned=modules_owned,
            iterations=iterations,
            init_fns=init_fns,
            message_fns=message_fns,
            update_fns=update_fns,
            readout=readout,
        )
        if decl.name in self._morphisms:
            raise CompileError(
                f"encoder {decl.name!r} name conflicts with existing morphism",
                decl.line,
                decl.col,
            )
        self._encoders[decl.name] = comp
        self._morphisms[decl.name] = comp

    def _compile_decoder(self, decl: DecoderDecl) -> None:
        """Compile a decoder block into a runtime Decoder module.

        Scaffolds, for each missing component, a properly-shaped
        learnable neural network — never a heuristic. The user's
        body overrides take precedence in every slot.
        """
        if not hasattr(self, "_decoders"):
            self._decoders: dict[str, Decoder] = {}
        if not hasattr(self, "_signatures"):
            self._signatures = {}

        if decl.signature not in self._signatures:
            raise CompileError(
                f"decoder {decl.name!r}: unknown signature {decl.signature!r}",
                decl.line,
                decl.col,
            )
        sig: Signature = self._signatures[decl.signature]

        overrides: dict[str, int] = {sd.sort: sd.dim for sd in decl.dims}
        sort_dims: dict[str, int] = {}
        _diag = f"decoder {decl.name!r}"
        for s_name in sig.sorts:
            sort_dims[s_name] = self._resolve_dim(
                sig,
                s_name,
                overrides,
                _diag,
            )

        globs = self._lex_globals_for_structural()
        modules_owned: list[nn.Module] = []

        # ---- structure heads, per sort ----
        # Each object sort needs one structure head emitting logits
        # over its candidate set (constructors + binders +
        # BoundVar). We size each head to the candidate set size.
        structure_fns: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {}
        for s_name, s in sig.sorts.items():
            if s.kind != "object":
                continue
            cands = []
            for c_name, c in sig.constructors.items():
                if c.codomain == s_name:
                    cands.append(c_name)
            for b_name, b in sig.binders.items():
                if b.codomain == s_name:
                    cands.append(b_name)
            # Reserve one extra slot for BoundVar; the runtime
            # always restricts to actually-available candidates.
            n_logits = max(len(cands) + 1, 2)
            head = nn.Linear(sort_dims[s_name], n_logits)
            modules_owned.append(head)

            def _make_struct(head=head):
                def call(v: torch.Tensor) -> torch.Tensor:
                    return head(v.reshape(-1))

                return call

            structure_fns[s_name] = _make_struct()

        # User-supplied structure override.
        if decl.structure is not None and decl.structure_arg is not None:
            sb = self._compile_let_expr(decl.structure, globals_=globs)

            def _struct_override(
                v: torch.Tensor, body_fn=sb, arg=decl.structure_arg
            ) -> torch.Tensor:
                return body_fn({arg: v})

            structure_fns["*"] = _struct_override

        # ---- primitive heads, per data sort ----
        # Each data sort needs a head over its (possibly empty)
        # closed vocabulary. The runtime raises if the vocab is
        # unpopulated; here we only allocate when the vocab is set
        # via the compiler's data_vocab attribute (declared
        # separately if and when needed).
        primitive_fns: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {}
        for s_name, s in sig.sorts.items():
            if s.kind != "data":
                continue
            vocab = self._data_vocab_for(sig).get(s_name, [])
            head = nn.Linear(sort_dims[s_name], max(len(vocab), 1))
            modules_owned.append(head)

            def _make_prim(head=head):
                def call(v: torch.Tensor) -> torch.Tensor:
                    return head(v.reshape(-1))

                return call

            primitive_fns[s_name] = _make_prim()

        if decl.primitive is not None and decl.primitive_arg is not None:
            pb = self._compile_let_expr(decl.primitive, globals_=globs)

            def _prim_override(
                v: torch.Tensor, body_fn=pb, arg=decl.primitive_arg
            ) -> torch.Tensor:
                return body_fn({arg: v})

            primitive_fns["*"] = _prim_override

        # ---- factor functions: per object sort, per arity ----
        # Every arity that occurs in the signature gets a learned
        # linear projection `dim -> n*dim` reshaped to a tuple of
        # n sub-vectors. This is the formally correct child split.
        factor_fns: dict[
            str, dict[int, Callable[[torch.Tensor], tuple[torch.Tensor, ...]]]
        ] = {}
        arities_by_sort: dict[str, set[int]] = {}
        for c in sig.constructors.values():
            if c.arity > 0:
                arities_by_sort.setdefault(c.codomain, set()).add(c.arity)
        for b in sig.binders.values():
            if b.arity > 0:
                arities_by_sort.setdefault(b.codomain, set()).add(b.arity)

        for sort, arities in arities_by_sort.items():
            d = sort_dims[sort]
            per_arity: dict[
                int, Callable[[torch.Tensor], tuple[torch.Tensor, ...]]
            ] = {}
            for n in arities:
                lin = nn.Linear(d, d * n)
                modules_owned.append(lin)

                def _make_factor(lin=lin, n=n, d=d):
                    def call(v: torch.Tensor) -> tuple[torch.Tensor, ...]:
                        out = lin(v.reshape(-1))
                        return tuple(out[i * d : (i + 1) * d] for i in range(n))

                    return call

                per_arity[n] = _make_factor()
            factor_fns[sort] = per_arity

        if decl.factor is not None and decl.factor_arg is not None:
            fb = self._compile_let_expr(decl.factor, globals_=globs)

            # The user-supplied factor body is evaluated with the
            # parent vector bound to `decl.factor_arg` and the arity
            # bound to ``n``. It must return a list / tuple of
            # exactly ``n`` sub-vectors. We close over each arity at
            # install time so the runtime sees a per-(sort, n)
            # function with the canonical (vec) -> tuple shape.
            def _make_factor_at_arity(n: int):
                def call(v: torch.Tensor) -> tuple[torch.Tensor, ...]:
                    result = fb({decl.factor_arg: v, "n": n})
                    if not isinstance(result, (list, tuple)):
                        raise RuntimeError(
                            f"decoder {decl.name!r}: factor body must return "
                            f"a list or tuple of sub-vectors, got "
                            f"{type(result).__name__}"
                        )
                    if len(result) != n:
                        raise RuntimeError(
                            f"decoder {decl.name!r}: factor body at arity "
                            f"{n} returned {len(result)} sub-vectors"
                        )
                    return tuple(result)

                return call

            for sort, per_arity in factor_fns.items():
                for n in list(per_arity):
                    per_arity[n] = _make_factor_at_arity(n)

        # ---- binder_select: scores in-scope variables ----
        # A small bilinear scorer between the parent vector and each
        # in-scope variable's embedding. Required by the runtime
        # whenever a BoundVar choice may fire OR an index-sorted
        # child position is decoded.
        principal_dim = next(iter(sort_dims.values()))
        bs_query = nn.Linear(principal_dim, principal_dim)
        bs_key = nn.Linear(principal_dim, principal_dim)
        modules_owned.extend([bs_query, bs_key])

        def _binder_select_default(
            v: torch.Tensor,
            embeds: list[torch.Tensor],
            q=bs_query,
            k=bs_key,
        ) -> torch.Tensor:
            qv = q(v.reshape(-1))
            keys = torch.stack([k(e.reshape(-1)) for e in embeds], dim=0)
            return keys @ qv

        binder_select_fn: Callable[[torch.Tensor, list[torch.Tensor]], torch.Tensor]
        if decl.binder_select is not None and decl.binder_select_arg is not None:
            bb = self._compile_let_expr(decl.binder_select, globals_=globs)

            def _bs_override(
                v: torch.Tensor,
                embeds: list[torch.Tensor],
                body_fn=bb,
                arg=decl.binder_select_arg,
            ) -> torch.Tensor:
                return body_fn({arg: v, "embeds": embeds})

            binder_select_fn = _bs_override
        else:
            binder_select_fn = _binder_select_default

        dec = Decoder(
            name=decl.name,
            signature=sig,
            sort_dims=sort_dims,
            depth=decl.depth,
            structure_fns=structure_fns,
            primitive_fns=primitive_fns,
            factor_fns=factor_fns,
            binder_select_fn=binder_select_fn,
            data_vocab=self._data_vocab_for(sig),
            modules_owned=modules_owned,
        )
        if decl.name in self._morphisms:
            raise CompileError(
                f"decoder {decl.name!r} name conflicts with existing morphism",
                decl.line,
                decl.col,
            )
        self._decoders[decl.name] = dec
        self._morphisms[decl.name] = dec

    def _compile_loss(self, decl: LossDecl) -> None:
        """Compile a loss declaration into a registry entry."""
        if not hasattr(self, "_loss_registry"):
            self._loss_registry = LossRegistry()

        globs = self._lex_globals_for_structural()
        body_fn = self._compile_let_expr(decl.body, globals_=globs)
        weight_fn = None
        if decl.weight is not None:
            weight_fn = self._compile_let_expr(decl.weight, globals_=globs)
        att = decl.attachment
        self._loss_registry.add(
            LossEntry(
                name=decl.name,
                body=body_fn,
                weight=weight_fn,
                attachment_kind=att.attachment_kind,
                target=att.target,
                rule_deduction=att.rule_deduction,
            )
        )

    def _lex_globals_for_structural(self) -> dict:
        """Build the globals dict visible to encoder/decoder/loss
        let-expression bodies. Includes morphisms, encoders,
        decoders, deductions, signatures."""
        globs: dict = {}
        globs.update(self._morphisms)
        for attr in ("_encoders", "_decoders", "_deductions", "_signatures"):
            d = getattr(self, attr, None)
            if d:
                globs.update(d)
        return globs

    def _build_data_embedders(
        self,
        sig: "Signature",
        sort_dims: dict[str, int],
        modules_owned: list,
    ) -> dict[str, "Callable"]:
        """For each data-sort in the signature, build an open-vocab
        keyed embedding table: each distinct data leaf encountered
        at the sort gets a learnable per-key vector allocated on
        first lookup.

        Dim is sourced strictly from ``sort_dims`` — every data
        sort in the signature must have its dim resolved before this
        runs (which the compiler enforces by calling ``_resolve_dim``
        on every sort up front and raising on missing dims).
        """
        out: dict[str, Callable] = {}
        for s_name, s in sig.sorts.items():
            if s.kind != "data":
                continue
            if s_name not in sort_dims:
                raise CompileError(
                    f"encoder over {sig.name!r}: data sort {s_name!r} "
                    f"has no resolved dim"
                )
            dim = sort_dims[s_name]
            table = nn.ParameterDict()
            modules_owned.append(table)

            def make_embed(table=table, dim=dim):
                def call(key):
                    skey = str(key).replace(".", "_")
                    if skey not in table:
                        p = nn.Parameter(torch.randn(dim) * 0.1)
                        table[skey] = p
                    return table[skey]

                return call

            out[s_name] = make_embed()
        return out

    def _data_vocab_for(self, sig: "Signature") -> dict[str, list]:
        """Return the per-data-sort closed vocabulary for use by the
        decoder's primitive heads.

        Vocabularies are declared inline in the signature's
        ``sorts { … }`` block via the ``vocab { … }`` clause on a
        data sort. The runtime list is the surface declaration's
        Python-decoded values in declaration order; the decoder's
        primitive head and ``log_prob`` use this order to index
        token positions.
        """
        return {
            s.name: list(s.vocab_values) for s in sig.sorts.values() if s.kind == "data"
        }

    def _compile_deduction(self, decl: DeductionDecl) -> None:
        """Compile a ``deduction { … }`` block into an agenda-engine
        :class:`DeductionSystem` and register it under ``decl.name``.

        Translates the declarative sequent-style rules into the
        runtime's :class:`InferenceRule` form (with single-uppercase
        identifiers treated as wildcard variables). Resolves the
        semiring by name. Wires the axiom source — one of:

        * a ``lexicon { ... }`` block, compiled into a learnable
          dispatch table keyed on the input token at each position;
        * a ``lexicon from "path"`` declaration, loaded from a TSV
          file at compile time and treated identically to the
          inline form;
        * an ``axioms = source_name`` declaration, naming a
          previously-defined morphism whose callable returns a
          list of `(item, weight)` pairs given an input;
        * none of the above — the user supplies axioms directly at
          call time (identity axiom-injector).

        The result is callable as ``parse(NAME, input)`` from
        program bodies, producing a :class:`ChartView`.
        """
        if not hasattr(self, "_deductions"):
            self._deductions: dict[str, "DeductionSystem"] = {}

        globals_ = dict(getattr(self, "_deductions", {}))
        # The deduction's declared atomic + complex constructor
        # symbols form the user-controlled free term algebra used
        # by lexicon LF expressions, rule weights, and any other
        # let-expressions evaluated inside this deduction's scope.
        # No constructor symbol is privileged by the compiler — the
        # user states the entire algebra explicitly.
        globals_["__constructors__"] = frozenset(decl.atoms)

        if decl.name in self._deductions or decl.name in self._morphisms:
            raise CompileError(
                f"deduction {decl.name!r} already declared",
                decl.line,
                decl.col,
            )

        # Pattern-conversion: TypeExpr -> agenda-engine Pattern.
        # The conversion is fully general; users may use any
        # type-expression shape and the runtime pattern-matcher
        # walks it structurally. Identifiers that match a declared
        # atom name become ground atoms; identifiers NOT in the
        # atoms set are treated as wildcard variables (the standard
        # Prolog / Datalog convention: declared constants are
        # ground; undeclared identifiers in patterns are variables).
        atoms_set = set(decl.atoms)

        def _convert_pattern(texpr):
            if isinstance(texpr, TypeName):
                name = texpr.name
                if name in atoms_set:
                    return ("atom", name)
                # Variable convention: any identifier not in the
                # atoms list (and not a numeric literal) is a
                # wildcard. This permits arbitrary metavariable
                # names — X, Y, Foo, antecedent — without ad-hoc
                # capitalisation rules.
                if name.isdigit():
                    return ("literal", int(name))
                return Wildcard(name)
            if isinstance(texpr, TypeProduct):
                return (
                    "product",
                    tuple(_convert_pattern(c) for c in texpr.components),
                )
            if isinstance(texpr, TypeSlash):
                # Categorial-grammar slash types: X/Y, X\Y.
                return (
                    texpr.direction,
                    _convert_pattern(texpr.result),
                    _convert_pattern(texpr.argument),
                )
            if isinstance(texpr, TypeEffectApply):
                # T(X) = ("effect_apply", T_name, *X_args). The
                # constructor's args are recursively converted; this
                # encodes any structured term — proof witnesses, LF
                # constructors, dependent-type applications.
                args = tuple(_convert_pattern(a) for a in texpr.args)
                return (texpr.effect, *args)
            # Fallback: a structural-equality probe.
            return ("atom", repr(texpr))

        semiring_registry = {
            "LogProb": SEMIRING_LOG_PROB,
            "Boolean": SEMIRING_BOOLEAN,
            "Viterbi": SEMIRING_VITERBI,
            "Counting": SEMIRING_COUNTING,
            "ProductFuzzy": SEMIRING_LOG_PROB,
        }
        semiring = (
            semiring_registry.get(decl.semiring, SEMIRING_LOG_PROB)
            if decl.semiring is not None
            else SEMIRING_LOG_PROB
        )

        inference_rules: list = []
        # The application rule and other generic combinators
        # carry no learnable weight by default; users may wrap
        # the deduction with a `weight_fn` that consults
        # rule-weight parameters.
        for sr in decl.rules:
            premises = tuple(_convert_pattern(p) for p in sr.premises)
            conclusion = _convert_pattern(sr.conclusion)
            inference_rules.append(
                InferenceRule(
                    name=sr.name,
                    premises=premises,
                    conclusion=conclusion,
                )
            )

        # ---- Axiom source ----
        #
        # Resolve in priority order:
        #   1. `axioms = some_morphism` (most general).
        #   2. `lexicon { ... }` or `lexicon from "..."` (sugar
        #      for the label-indexed-lookup case).
        #   3. Identity — input itself is the axiom list.

        if decl.axioms_source is not None:
            # General axiom source — look up the named morphism and
            # invoke it on the input at call time. The morphism may
            # be any callable.
            src_name = decl.axioms_source
            if src_name not in self._morphisms:
                raise CompileError(
                    f"deduction {decl.name!r}: axioms source "
                    f"{src_name!r} is not a declared morphism",
                    decl.line,
                    decl.col,
                )
            morph = self._morphisms[src_name]

            def _axiom_injector(input_value, _morph=morph):
                # The morphism is expected to be a callable that,
                # given the input, returns a list of (item, weight)
                # pairs.
                return list(_morph(input_value))

            axiom_module = morph if isinstance(morph, nn.Module) else None
        elif decl.lexicon or decl.lexicon_from_file is not None:
            # Lexicon-based axiom source. Build a learnable lookup
            # table keyed on the literal word string; emit one
            # axiom per matching entry per input position.
            entries: list[tuple[str, "Any", "Any", bool]] = []
            for entry in decl.lexicon:
                lf_fn = Compiler._compile_let_expr(entry.lf, globals_=globals_)
                # Evaluate the LF eagerly under an empty environment;
                # LF templates in lexicons must be closed expressions.
                try:
                    lf_value = lf_fn({})
                except CompileError as e:
                    raise CompileError(
                        f"deduction {decl.name!r}: lexicon entry for "
                        f"{entry.word!r} has unresolved variable: {e}",
                        entry.line,
                        entry.col,
                    ) from e
                entries.append(
                    (
                        entry.word,
                        _convert_pattern(entry.category),
                        lf_value,
                        entry.learnable,
                    )
                )
            # File-loaded lexicon: TSV with `word\tcategory\tlf` rows.
            if decl.lexicon_from_file is not None:
                file_entries = self._load_lexicon_tsv(
                    decl.lexicon_from_file,
                    decl.lexicon_from_file_learnable,
                    decl,
                )
                entries.extend(file_entries)
            # Allocate one learnable Parameter per learnable entry.
            # We keep the Parameter list on a small nn.Module so it
            # participates in `.parameters()` of any Program that
            # owns the deduction.
            axiom_module = nn.Module()
            param_list: list = []
            for idx, (_w, _cat, _lf, is_learnable) in enumerate(entries):
                if is_learnable:
                    p = nn.Parameter(torch.zeros(()))
                    axiom_module.register_parameter(f"lex_weight_{idx}", p)
                    param_list.append(p)
                else:
                    param_list.append(None)
            # Capture the axiom-injector as a closure over the
            # entries + parameter list.
            entries_local = tuple(entries)
            params_local = tuple(param_list)

            def _axiom_injector(
                input_value, _entries=entries_local, _params=params_local
            ):
                # `input_value` may be a list/tuple of token strings,
                # OR a list of `(token, position)` pairs. We accept
                # bare-string lists for the common case.
                tokens = list(input_value)
                out: list = []
                for pos, tok in enumerate(tokens):
                    if isinstance(tok, tuple) and len(tok) == 2:
                        tok = tok[0]
                    for idx, (word, cat_pat, lf_val, _learn) in enumerate(_entries):
                        if word != tok:
                            continue
                        weight_param = _params[idx]
                        if weight_param is not None:
                            weight_tensor = weight_param
                        else:
                            weight_tensor = torch.tensor(0.0)
                        # Emit a span axiom carrying the lexical
                        # category and LF; positions cover the
                        # single token at [pos, pos+1).
                        item = ("span", pos, pos + 1, cat_pat, lf_val)
                        out.append((item, weight_tensor))
                return out
        else:
            # Identity injector — input is already a list of axioms.
            def _axiom_injector(input_value):
                if isinstance(input_value, list):
                    return input_value
                return list(input_value)

            axiom_module = None

        # Goal: items matching the start symbol's atom form for
        # top-level spans. Users override by composing the parse
        # result with their own predicate.
        start = decl.start

        def _goal(item) -> bool:
            if start is None:
                return True
            if not (isinstance(item, tuple) and len(item) > 0):
                return False
            # Three goal-item shapes the framework recognizes by
            # default; users can override via a custom goal
            # predicate (the `axioms = source_kernel` escape hatch
            # composes with an arbitrary `goal` field on the
            # underlying DeductionSystem).
            head = item[0]
            # 1. Bare atom: ("atom", "S").
            if head == "atom" and len(item) == 2 and item[1] == start:
                return True
            # 2. Head-keyed (Datalog-style): ("reach", ...).
            if isinstance(head, str) and head == start:
                return True
            # 3. CKY-shaped span: ("span", i, j, ("atom", "S"), lf).
            if head == "span" and len(item) >= 4:
                cat = item[3]
                if (
                    isinstance(cat, tuple)
                    and len(cat) == 2
                    and cat[0] == "atom"
                    and cat[1] == start
                ):
                    return True
            return False

        # Choose an agenda strategy. Default to CKY for
        # context-free-shaped systems; depth-first for proof
        # search (Boolean semiring with rule-arity-2-or-less);
        # semi-naive for Datalog-shaped (no aggregation needed).
        if semiring is BOOLEAN and any(len(r.premises) == 1 for r in inference_rules):
            agenda_factory = depth_first_agenda
        elif semiring is BOOLEAN:
            agenda_factory = semi_naive_agenda
        else:
            agenda_factory = cky_agenda

        system = DeductionSystem(
            rules=tuple(inference_rules),
            semiring=semiring,
            axiom_injector=_axiom_injector,
            goal=_goal,
            agenda_factory=agenda_factory,
            max_iterations=10_000,
        )
        # Stash the axiom-module on the system so Programs that
        # reach it (via `parse(NAME, …)`) can include its
        # parameters in their optimizer.
        if axiom_module is not None:
            system._axiom_module = axiom_module  # type: ignore[attr-defined]
        # Attach a signature / encoder pairing, if declared. The
        # chart-query operations (`chart.embedding(pattern)`) consult
        # this attached encoder to compute on-demand item
        # embeddings.
        if decl.item_signature is not None:
            sigs = getattr(self, "_signatures", {})
            if decl.item_signature not in sigs:
                raise CompileError(
                    f"deduction {decl.name!r}: unknown item signature "
                    f"{decl.item_signature!r}",
                    decl.line,
                    decl.col,
                )
            system._item_signature = sigs[decl.item_signature]  # type: ignore[attr-defined]
        if decl.item_encoder is not None:
            comps = getattr(self, "_encoders", {})
            if decl.item_encoder not in comps:
                raise CompileError(
                    f"deduction {decl.name!r}: unknown item encoder "
                    f"{decl.item_encoder!r}",
                    decl.line,
                    decl.col,
                )
            system._item_encoder = comps[decl.item_encoder]  # type: ignore[attr-defined]
        self._deductions[decl.name] = system

    def _load_lexicon_tsv(
        self,
        path: str,
        learnable: bool,
        decl: "DeductionDecl",
    ) -> list[tuple[str, "Any", "Any", bool]]:
        """Load a lexicon from a TSV file at compile time.

        Format: each row has three tab-separated columns:
        ``word``, ``category``, ``lf_template``. The category is
        parsed as a type expression; the LF template is parsed as
        a let-arithmetic expression. Multiple rows per word are
        allowed (latent disjunction).

        Resolved relative to the working directory; paths starting
        with ``/`` are absolute.
        """
        from pathlib import Path
        # Re-parse the category and LF text by feeding them to the
        # tree-sitter parser inside a synthetic dummy program.
        # This keeps the lexicon-file syntax aligned with the
        # main grammar.

        # For simplicity, we expect categories and LFs in a
        # restricted form: bare identifiers for categories
        # (atom names) and bare identifiers for LFs (let_var refs).
        # Richer TSV formats may be supported by adding a custom
        # parser; this is the minimum viable schema.
        p = Path(path)
        if not p.exists():
            raise CompileError(
                f"deduction {decl.name!r}: lexicon file {path!r} not found",
                decl.line,
                decl.col,
            )
        out: list[tuple[str, "Any", "Any", bool]] = []
        with p.open("r", encoding="utf-8") as fh:
            for lineno, raw_line in enumerate(fh, start=1):
                line = raw_line.rstrip("\n")
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 3:
                    raise CompileError(
                        f"deduction {decl.name!r}: lexicon file "
                        f"{path!r}:{lineno}: expected 3 tab-separated "
                        f"columns (word, category, lf), got {len(parts)}",
                        decl.line,
                        decl.col,
                    )
                word, cat_text, lf_text = parts[0], parts[1], parts[2]
                # Build a TypeName for the category atom. (Richer
                # category-shape parsing happens on the live
                # grammar; here we accept atom identifiers as a
                # safe, broadly-useful starting point.)
                cat_pattern = ("atom", cat_text)
                # LF: treat as a constructor-application or atom.
                # If the text contains '(' it's a let-call shape;
                # otherwise it's a bare identifier. Building the
                # corresponding pattern directly:
                if "(" in lf_text:
                    # Parse the LF text as a let-arith expression
                    # by wrapping it in a tiny synthetic program.
                    from quivers.dsl.parser import parse as _parse

                    syn_src = (
                        "object _DummyObj : 1\n"
                        "program _dummy_prog : _DummyObj -> _DummyObj\n"
                        f"    _x <- _f\n"
                        f"    let _lex_lf = {lf_text}\n"
                        "    return _x\n"
                    )
                    syn_mod = _parse(syn_src)
                    # The third statement is the program; its
                    # second step's value carries the parsed LF.
                    prog = next(
                        s
                        for s in syn_mod.statements
                        if hasattr(s, "draws")
                        and getattr(s, "name", None) == "_dummy_prog"
                    )
                    let_step = prog.draws[1]
                    lex_globals = self._lex_globals_for_structural()
                    lf_value = Compiler._compile_let_expr(
                        let_step.value, globals_=lex_globals
                    )({})
                else:
                    lf_value = lf_text
                out.append((word, cat_pattern, lf_value, learnable))
        return out

    def _compile_export(self, decl: ExportDecl) -> None:
        """Record an exported expression.

        Replaces v0.4's single-output model: a module may declare
        any number of ``export`` statements, each selecting a
        top-level binding for the module's public surface. The
        compiled output runner picks the first export; further
        exports become additional accessible morphisms on the
        compiled object.
        """
        if not hasattr(self, "_exports"):
            self._exports = []
        self._exports.append(decl.expr)
        # Maintain backwards compatibility with internal helpers
        # that consult `_output_expr`: the first export wins.
        if self._output_expr is None:
            self._output_expr = decl.expr

    def _resolve_type(self, texpr: TypeExpr, bind_name: str | None = None) -> SetObject:
        """Resolve a type expression into a SetObject.

        Delegates to :class:`~quivers.dsl.resolution.TypeExprToSetObject`,
        a :class:`didactic.api.Lens` parameterized by the current object
        environment. Integer-literal :class:`TypeName` nodes that aren't
        in the environment use ``bind_name`` (falling back to
        ``"_<value>"``) as the synthesized :class:`FinSet` name; this
        thin wrapper is kept so the literal-naming policy stays in
        compiler control.
        """
        from quivers.dsl.resolution import TypeExprToSetObject

        if (
            isinstance(texpr, TypeName)
            and texpr.name.isdigit()
            and texpr.name not in self._objects
            and bind_name is not None
        ):
            return FinSet(name=bind_name, cardinality=int(texpr.name))

        try:
            resolved, _ = TypeExprToSetObject(self._objects).forward(texpr)
        except KeyError as e:
            line = getattr(texpr, "line", 0)
            col = getattr(texpr, "col", 0)
            raise CompileError(str(e).strip("'\""), line, col) from e
        return resolved

    def _resolve_any_space(self, texpr: TypeExpr):
        """Resolve a type expression to either a SetObject or ContinuousSpace.

        Continuous morphism domains/codomains can be either discrete
        objects, continuous spaces, or product types.

        Parameters
        ----------
        texpr : TypeExpr
            The type expression to resolve (TypeName, TypeProduct, etc.).

        Returns
        -------
        SetObject or ContinuousSpace
            The resolved domain/codomain.
        """
        if isinstance(texpr, TypeProduct):
            from quivers.core.objects import ProductSet
            from quivers.continuous.spaces import ContinuousSpace, ProductSpace

            components = [self._resolve_any_space(c) for c in texpr.components]
            if any((isinstance(c, ContinuousSpace) for c in components)):
                return ProductSpace(components=tuple(components))
            return ProductSet(components=tuple(components))
        if not isinstance(texpr, TypeName):
            raise CompileError(
                f"unsupported type expression in domain/codomain: {type(texpr).__name__}",
                getattr(texpr, "line", 0),
                getattr(texpr, "col", 0),
            )
        name = texpr.name
        if name in self._objects:
            return self._objects[name]
        if name in self._spaces:
            return self._spaces[name]
        raise CompileError(f"undefined object or space {name!r}", texpr.line, texpr.col)

    def _resolve_space(self, sexpr: SpaceExpr, bind_name: str | None = None):
        """Resolve a space expression into a ContinuousSpace.

        Delegates to :class:`~quivers.dsl.resolution.SpaceExprToContinuousSpace`,
        a :class:`didactic.api.Lens` parameterized by both the space and
        object environments (a bare identifier may resolve to either).
        """
        from quivers.dsl.resolution import SpaceExprToContinuousSpace

        if isinstance(sexpr, SpaceConstructor):
            constructors = _get_space_constructors()
            cname = sexpr.constructor
            if cname not in constructors:
                raise CompileError(
                    f"unknown space constructor {cname!r}; available: "
                    f"{', '.join(sorted(constructors))}",
                    sexpr.line,
                    sexpr.col,
                )

        try:
            resolved, _ = SpaceExprToContinuousSpace(
                env_spaces=self._spaces,
                env_objects=self._objects,
                name=bind_name or "_anon",
            ).forward(sexpr)
        except (KeyError, ValueError) as e:
            line = getattr(sexpr, "line", 0)
            col = getattr(sexpr, "col", 0)
            raise CompileError(str(e).strip("'\""), line, col) from e
        return resolved

    def bind_data(self, data: dict) -> None:
        """Bind a runtime data dictionary for ``from_data("KEY")``
        initializers.

        Each key in ``data`` maps to a tensor (or tensor-like
        object) that supplies the morphism tensor for any
        ``from_data("KEY")`` expression that references it. The
        bindings are consulted at compile time; supply them BEFORE
        calling :meth:`compile`.
        """
        self._data_bindings = dict(data)

    def _require_quantale(self, op_name: str, line: int, col: int) -> None:
        """Raise ``CompileError`` if the module's composition rule
        isn't a :class:`Quantale`.

        Operations that need a unit, zero, dagger, or compact-closed
        structure (``identity``, ``cup``, ``cap``, ``.dagger``,
        ``.trace``) call this before constructing their result.
        The error names the operation and the active rule so the
        user can decide whether to switch the file's
        ``quantale`` / ``semigroupoid`` / ``bilinear_form``
        declaration or drop the operation.
        """
        rule = self._quantale
        if isinstance(rule, Quantale):
            return
        kind = (
            "semigroupoid"
            if isinstance(rule, Semigroupoid)
            else (
                "bilinear_form"
                if isinstance(rule, BilinearForm)
                else "composition_rule"
            )
        )
        raise CompileError(
            f"{op_name}: requires a Quantale composition rule (needs "
            f"identity / unit / zero); the module declares {kind} "
            f"{rule.name!r}, which does not have those. Switch the "
            f"file-level declaration to ``quantale X`` for a "
            f"Quantale X, or drop the {op_name} operation.",
            line,
            col,
        )

    def _compile_trans_expr(self, expr):
        """Compile an expression that evaluates to a transformation.

        Recognised expression shapes:

        * :class:`ExprIdent` — a bare name resolved against
          :attr:`_transformations` (user let-bindings) and then
          :attr:`_trans_singletons` (built-in singletons).
        * :class:`ExprMorphismCall` — a constructor call whose
          callee names an entry of :attr:`_trans_constructors`;
          each argument is resolved against the surrounding scope
          (objects, morphisms, quantales).
        * :class:`ExprTransCompose` — ``t1 >>> t2``; recursively
          compiles each side, flattens nested sequences, and
          type-checks each adjacent ``target == source`` boundary,
          returning a :class:`TransSeq` (or, for two single-step
          values, an unboxed TransSeq of length 2).

        Returns a value that the runtime understands: a
        :class:`QuantaleHomomorphism`, a
        :class:`MorphismTransformation`, or a :class:`TransSeq`.
        Raises :class:`CompileError` on any unresolved reference
        or type mismatch.
        """
        if isinstance(expr, ExprIdent):
            if expr.name in self._transformations:
                return self._transformations[expr.name]
            if expr.name in self._trans_singletons:
                return self._trans_singletons[expr.name]
            if expr.name in self._trans_constructors:
                raise CompileError(
                    f"change_base: {expr.name!r} is a transformation "
                    f"constructor and needs arguments; call it like "
                    f"`{expr.name}(axis_object)`",
                    expr.line,
                    expr.col,
                )
            raise CompileError(
                f"change_base: undefined transformation "
                f"{expr.name!r}; available singletons: "
                f"{sorted(self._trans_singletons)}; constructors: "
                f"{sorted(self._trans_constructors)}; let-bound: "
                f"{sorted(self._transformations)}",
                expr.line,
                expr.col,
            )
        if isinstance(expr, ExprMorphismCall):
            factory = self._trans_constructors.get(expr.callee)
            if factory is None:
                raise CompileError(
                    f"change_base: undefined transformation "
                    f"constructor {expr.callee!r}; available: "
                    f"{sorted(self._trans_constructors)}",
                    expr.line,
                    expr.col,
                )
            resolved: list = []
            for arg_name in expr.args:
                value = self._resolve_trans_constructor_argument(
                    arg_name,
                    expr.line,
                    expr.col,
                    constructor_name=expr.callee,
                )
                resolved.append(value)
            try:
                return factory(*resolved)
            except (TypeError, ValueError) as e:
                raise CompileError(
                    f"change_base: constructor {expr.callee!r} "
                    f"rejected arguments {expr.args!r}: {e}",
                    expr.line,
                    expr.col,
                ) from e
        if isinstance(expr, ExprTransCompose):
            left_val = self._compile_trans_expr(expr.left)
            right_val = self._compile_trans_expr(expr.right)
            left_steps = (
                left_val.steps if isinstance(left_val, TransSeq) else (left_val,)
            )
            right_steps = (
                right_val.steps if isinstance(right_val, TransSeq) else (right_val,)
            )
            steps = left_steps + right_steps
            # Type-check the seam between each adjacent step.
            for i in range(len(steps) - 1):
                tgt = steps[i].target
                src = steps[i + 1].source
                if type(tgt) is not type(src):
                    raise CompileError(
                        f">>>: target of step {i} ({tgt.name!r}) "
                        f"does not match source of step {i + 1} "
                        f"({src.name!r})",
                        expr.line,
                        expr.col,
                    )
            return TransSeq(steps)
        raise CompileError(
            f"change_base: expression of kind "
            f"{type(expr).__name__!r} does not denote a "
            f"transformation",
            expr.line,
            expr.col,
        )

    def _resolve_trans_constructor_argument(
        self,
        name: str,
        line: int,
        col: int,
        *,
        constructor_name: str,
    ):
        """Resolve a named constructor argument to its compiled
        value.  Tries the value sorts that show up in
        transformation constructors (objects, morphisms,
        continuous morphisms, quantales) and surfaces a clear
        error if the name doesn't resolve.
        """
        if name in self._objects:
            return self._objects[name]
        if name in self._morphisms:
            return self._morphisms[name]
        if (
            hasattr(self, "_continuous_morphisms")
            and name in self._continuous_morphisms
        ):
            return self._continuous_morphisms[name]
        if name in _QUANTALE_REGISTRY:
            return _QUANTALE_REGISTRY[name]
        raise CompileError(
            f"change_base: constructor {constructor_name!r} "
            f"argument {name!r} unresolved (must name an object, "
            f"morphism, or quantale)",
            line,
            col,
        )

    def _apply_trans(self, inner_morph, phi):
        """Apply a transformation value (a singleton, constructor
        result, or :class:`TransSeq`) to a morphism via the
        morphism's own ``change_base``.

        Sequences are unfolded into iterated change_base calls;
        each intermediate result feeds the next step.
        """
        if isinstance(phi, TransSeq):
            current = inner_morph
            for step in phi.steps:
                current = current.change_base(step)
            return current
        return inner_morph.change_base(phi)

    def _compose_with_op(self, left, right, op: str):
        """Dispatch a composition expression to the quantale
        implied by the surface operator.

        Each composition operator carries an enrichment quantale.
        ``>>``, ``<<`` (already swapped to forward), and ``>=>``
        all use the operands' shared quantale (the existing
        :meth:`Morphism.__rshift__` path, which raises
        ``incompatible quantales`` if they differ).

        The new operators (``*>``, ``~>``, ``||>``, ``?>``,
        ``&&>``, ``+>``) each fix the composition quantale at the
        operator and re-tag the operands accordingly. If the
        operands' declared quantales already match the operator's
        target, no base change is needed; otherwise the user must
        have applied an explicit ``.change_base(φ)`` upstream.
        """
        from quivers.core.quantales import (
            COUNTING,
            GODEL,
            LOG_PROB,
            LUKASIEWICZ,
            MAX_PLUS,
            PROBABILITY,
            REAL,
        )
        from quivers.core.morphisms import ComposedMorphism, Morphism
        from quivers.core.quantales import BOOLEAN
        from quivers.stochastic.quantale import MARKOV

        del COUNTING  # exposed via module-level `quantale counting` only
        op_to_quantale: dict[str, object] = {
            ">>": None,  # use operands' shared quantale
            ">=>": None,
            "*>": MARKOV,
            "~>": LOG_PROB,
            "||>": GODEL,
            "?>": MAX_PLUS,
            "&&>": BOOLEAN,
            "+>": LUKASIEWICZ,
            "$>": REAL,
            "%>": PROBABILITY,
        }
        if op not in op_to_quantale:
            raise CompileError(f"unknown composition operator {op!r}", 0, 0)
        target_quantale = op_to_quantale[op]
        if target_quantale is None:
            # ``>>`` and ``>=>``: fall through to the operands' own
            # composition machinery, which uses the shared quantale
            # (and errors on a mismatch as before).
            return left >> right
        # Validate both operands carry the operator's target
        # quantale. The operator does NOT auto-base-change; the
        # user must have applied ``.change_base(...)`` upstream to
        # bring both operands to the target quantale before
        # composing.
        if not isinstance(left, Morphism) or not isinstance(right, Morphism):
            raise TypeError(
                f"composition operator {op!r}: both operands must be "
                f"Morphism instances; got "
                f"{type(left).__name__} {op} {type(right).__name__}"
            )
        for label, m in (("left", left), ("right", right)):
            if type(m.quantale) is not type(target_quantale):
                raise TypeError(
                    f"composition operator {op!r}: {label} operand's "
                    f"quantale is {m.quantale.name!r}, but the "
                    f"operator dispatches to "
                    f"{target_quantale.name!r}; apply "  # type: ignore[union-attr]
                    f"`.change_base(...)` first to convert "
                    f"{label} into the operator's quantale"
                )
        if left.codomain != right.domain:
            raise TypeError(
                f"composition operator {op!r}: cannot compose "
                f"codomain {left.codomain!r} != domain "
                f"{right.domain!r}"
            )
        return ComposedMorphism(left, right)

    def _compile_expr(self, expr: Expr):
        """Compile a value expression into a morphism.

        Parameters
        ----------
        expr : Expr
            The expression to compile.

        Returns
        -------
        Morphism or ContinuousMorphism
            The compiled morphism (possibly a DAG of compositions).
        """
        if isinstance(expr, ExprIdent):
            if expr.name not in self._morphisms:
                raise CompileError(
                    f"undefined morphism {expr.name!r}", expr.line, expr.col
                )
            return self._morphisms[expr.name]
        elif isinstance(expr, ExprMorphismCall):
            return self._compile_morphism_call(expr)
        elif isinstance(expr, ExprIdentity):
            if expr.object_name not in self._objects:
                raise CompileError(
                    f"undefined object {expr.object_name!r}", expr.line, expr.col
                )
            self._require_quantale("identity", expr.line, expr.col)
            obj = self._objects[expr.object_name]
            return make_identity(obj, quantale=self._quantale)
        elif isinstance(expr, ExprDagger):
            inner = self._compile_expr(expr.inner)
            self._require_quantale("dagger", expr.line, expr.col)
            return inner.dagger
        elif isinstance(expr, ExprTrace):
            inner = self._compile_expr(expr.inner)
            if expr.object_name not in self._objects:
                raise CompileError(
                    f"trace: undefined object {expr.object_name!r}",
                    expr.line,
                    expr.col,
                )
            self._require_quantale("trace", expr.line, expr.col)
            try:
                return inner.trace(self._objects[expr.object_name])
            except TypeError as e:
                raise CompileError(str(e), expr.line, expr.col) from e
        elif isinstance(expr, ExprChangeBase):
            inner = self._compile_expr(expr.inner)
            phi = self._compile_trans_expr(expr.phi)
            try:
                return self._apply_trans(inner, phi)
            except TypeError as e:
                raise CompileError(str(e), expr.line, expr.col) from e
        elif isinstance(expr, ExprCup):
            if expr.object_name not in self._objects:
                raise CompileError(
                    f"cup: undefined object {expr.object_name!r}",
                    expr.line,
                    expr.col,
                )
            self._require_quantale("cup", expr.line, expr.col)
            return _make_cup(self._objects[expr.object_name], quantale=self._quantale)
        elif isinstance(expr, ExprCap):
            if expr.object_name not in self._objects:
                raise CompileError(
                    f"cap: undefined object {expr.object_name!r}",
                    expr.line,
                    expr.col,
                )
            self._require_quantale("cap", expr.line, expr.col)
            return _make_cap(self._objects[expr.object_name], quantale=self._quantale)
        elif isinstance(expr, ExprFreeze):
            inner = self._compile_expr(expr.inner)
            # Materialise the inner morphism's tensor with detach()
            # and wrap as an ObservedMorphism. Gradient flow stops
            # at this boundary; the constituent morphisms' parameters
            # are not part of the result's parameter set.
            from quivers.core.morphisms import (
                Morphism as _CatMorph,
                ObservedMorphism as _Obs,
            )

            if not isinstance(inner, _CatMorph):
                raise CompileError(
                    f"freeze: inner expression compiled to "
                    f"{type(inner).__name__}; expected a Morphism",
                    expr.line,
                    expr.col,
                )
            frozen = _Obs(
                inner.domain,
                inner.codomain,
                inner.tensor.detach().clone(),
                quantale=inner.quantale,
            )
            return frozen
        elif isinstance(expr, ExprFromData):
            # ``from_data("KEY")`` cannot resolve the tensor at
            # compile time because the runtime data dict is only
            # available at fit time. We must therefore emit a
            # late-binding ObservedMorphism whose tensor is filled
            # in via a fit-time hook on the program's data
            # dictionary. For the standalone ``let``-expression
            # case this is materialised as a deferred binding the
            # compiler can resolve once the data dict is known.
            #
            # The data dictionary is stored on the Compiler under
            # ``self._data_bindings``; users supply it via
            # :meth:`Compiler.bind_data` BEFORE calling compile()
            # (or via the high-level ``loads()`` ``data=`` kwarg).
            data_dict = getattr(self, "_data_bindings", None)
            if data_dict is None or expr.key not in data_dict:
                available = sorted(data_dict.keys()) if data_dict else []
                raise CompileError(
                    f"from_data: unknown data key {expr.key!r}; "
                    f"available: {available}. Bind the data dict via "
                    f"``Compiler.bind_data(...)`` or the ``data=`` "
                    f"keyword on ``loads()`` before compiling.",
                    expr.line,
                    expr.col,
                )
            from quivers.core.morphisms import ObservedMorphism as _Obs

            tensor = data_dict[expr.key]
            if not isinstance(tensor, torch.Tensor):
                tensor = torch.as_tensor(tensor)
            # The user-supplied domain/codomain on the morphism
            # declaration is in the parent ``observed`` decl; here
            # we synthesize a one-shot ObservedMorphism whose
            # domain/codomain are inferred from the tensor's shape
            # split halfway. For the common ``observed f : A -> B =
            # from_data("KEY")`` pattern the parent decl supplies
            # the correct domain/codomain post-substitution; this
            # let-expression path is for ``let h = from_data(...)``
            # which the surrounding context constrains.
            from quivers.core.objects import FinSet as _FS

            if tensor.dim() < 1:
                raise CompileError(
                    f"from_data: tensor at key {expr.key!r} has "
                    f"rank 0; cannot infer a morphism",
                    expr.line,
                    expr.col,
                )
            # Synthetic anonymous domain/codomain for the
            # let-expression path. The morphism's downstream
            # usage typically discards this synthetic typing
            # because it is reassigned to a properly typed
            # ``observed`` slot, OR used as a tensor source.
            dom = _FS(name=f"_data_dom_{expr.key}", cardinality=tensor.shape[0])
            if tensor.dim() == 1:
                cod = _FS(name="_data_unit", cardinality=1)
                tensor = tensor.unsqueeze(-1)
            else:
                cod_size = int(torch.tensor(tensor.shape[1:]).prod().item())
                cod = _FS(name=f"_data_cod_{expr.key}", cardinality=cod_size)
                if tensor.dim() > 2:
                    tensor = tensor.reshape(tensor.shape[0], -1)
            return _Obs(dom, cod, tensor, quantale=self._quantale)
        elif isinstance(expr, ExprCompose):
            left = self._compile_expr(expr.left)
            right = self._compile_expr(expr.right)
            try:
                return self._compose_with_op(left, right, expr.op)
            except TypeError as e:
                raise CompileError(str(e), expr.line, expr.col) from e
        elif isinstance(expr, ExprTensorProduct):
            left = self._compile_expr(expr.left)
            right = self._compile_expr(expr.right)
            return left @ right
        elif isinstance(expr, ExprMarginalize):
            inner = self._compile_expr(expr.inner)
            sets = []
            for name in expr.names:
                if name not in self._objects:
                    raise CompileError(
                        f"undefined object {name!r} in marginalize", expr.line, expr.col
                    )
                sets.append(self._objects[name])
            try:
                return inner.marginalize(*sets)
            except (TypeError, ValueError) as e:
                raise CompileError(str(e), expr.line, expr.col) from e
        elif isinstance(expr, ExprCurry):
            from quivers.core.morphisms import CurriedMorphism

            inner = self._compile_expr(expr.inner)
            try:
                return CurriedMorphism(inner, direction=expr.direction)
            except (TypeError, ValueError) as e:
                raise CompileError(str(e), expr.line, expr.col) from e
        elif isinstance(expr, ExprChartFold):
            return self._compile_chart_fold(expr)
        elif isinstance(expr, ExprFan):
            from quivers.continuous.morphisms import FanOutMorphism

            components = []
            for sub_expr in expr.exprs:
                if isinstance(sub_expr, ExprIdent) and sub_expr.name in self._groups:
                    for member_name in self._groups[sub_expr.name]:
                        components.append(self._morphisms[member_name])
                else:
                    morph = self._compile_expr(sub_expr)
                    components.append(morph)
            try:
                return FanOutMorphism(components)
            except (TypeError, ValueError) as e:
                raise CompileError(str(e), expr.line, expr.col) from e
        elif isinstance(expr, ExprRepeat):
            morph = self._compile_expr(expr.expr)
            if expr.count is None:
                from quivers.core.morphisms import RepeatMorphism

                try:
                    return RepeatMorphism(morph, n=1)
                except (TypeError, ValueError) as e:
                    raise CompileError(str(e), expr.line, expr.col) from e
            result = morph
            for _ in range(expr.count - 1):
                try:
                    result = result >> morph
                except TypeError as e:
                    raise CompileError(str(e), expr.line, expr.col) from e
            return result
        elif isinstance(expr, ExprStack):
            import copy

            morph = self._compile_expr(expr.expr)
            result = copy.deepcopy(morph)
            for _ in range(expr.count - 1):
                clone = copy.deepcopy(morph)
                try:
                    result = result >> clone
                except TypeError as e:
                    raise CompileError(str(e), expr.line, expr.col) from e
            return result
        elif isinstance(expr, ExprScan):
            from quivers.continuous.scan import ScanMorphism

            cell = self._compile_expr(expr.expr)
            try:
                return ScanMorphism(cell, init=expr.init)
            except TypeError as e:
                raise CompileError(str(e), expr.line, expr.col) from e
        elif isinstance(expr, ExprParser):
            from quivers.stochastic.schema import SCHEMA_REGISTRY

            schemas: list = []
            morphisms: list = []

            def _expand(name: str, seen: frozenset[str]) -> list[str]:
                """Recursively expand a bundle reference into rule names.

                Cycle detection: if ``name`` already appears in ``seen``,
                raises CompileError.
                """
                if name not in self._bundles:
                    return [name]
                if name in seen:
                    raise CompileError(
                        f"bundle cycle through {name!r}",
                        expr.line,
                        expr.col,
                    )
                expanded: list[str] = []
                for member in self._bundles[name]:
                    expanded.extend(_expand(member, seen | {name}))
                return expanded

            resolved_rules: list[str] = []
            for rule_name in expr.rules:
                resolved_rules.extend(_expand(rule_name, frozenset()))

            for rule_name in resolved_rules:
                if rule_name in self._rules:
                    schemas.append(self._rules[rule_name])
                elif (schema_obj := SCHEMA_REGISTRY.get(rule_name)) is not None:
                    schemas.append(schema_obj)
                elif rule_name in self._morphisms:
                    morphisms.append(self._morphisms[rule_name])
                else:
                    raise CompileError(
                        f"unknown rule {rule_name!r}; not a declared rule, schema primitive ({', '.join(sorted(SCHEMA_REGISTRY))}), or a declared morphism",
                        expr.line,
                        expr.col,
                    )
            if schemas and morphisms:
                raise CompileError(
                    "parser() rules must be all schema primitives or all morphism references, not a mix",
                    expr.line,
                    expr.col,
                )
            if morphisms:
                return self._compile_parser_morphisms(morphisms, expr)
            if not schemas:
                raise CompileError(
                    "parser() requires at least one rule", expr.line, expr.col
                )
            return self._compile_parser_schemas(schemas, expr)
        else:
            raise CompileError(f"unknown expression type: {type(expr).__name__}")

    def _compile_parser_morphisms(self, morphisms: list, expr: ExprParser):
        """Compile parser from user-declared morphisms via type inspection.

        Classifies each morphism by its type signature:

        - ``N → N ⊗ N`` (codomain is a product of the domain with
          itself) contributes binary deductions.
        - ``N → T`` (codomain differs from domain) contributes
          lexical axioms.

        Parameters
        ----------
        morphisms : list
            Compiled morphism objects.
        expr : ExprParser
            The AST node (for error reporting).
        """
        from quivers.core.objects import ProductSet
        from quivers.stochastic.inside import InsideAlgorithm

        binary = None
        lexical = None
        for morph in morphisms:
            cod = morph.codomain
            if (
                isinstance(cod, ProductSet)
                and len(cod.components) == 2
                and all((c == morph.domain for c in cod.components))
            ):
                if binary is not None:
                    raise CompileError(
                        "parser() received multiple binary morphisms (codomain = domain ⊗ domain); expected one",
                        expr.line,
                        expr.col,
                    )
                binary = morph
            else:
                if lexical is not None:
                    raise CompileError(
                        "parser() received multiple lexical morphisms; expected one",
                        expr.line,
                        expr.col,
                    )
                lexical = morph
        if binary is None:
            raise CompileError(
                "parser() requires a binary morphism (type N → N ⊗ N) among its rules",
                expr.line,
                expr.col,
            )
        if lexical is None:
            raise CompileError(
                "parser() requires a lexical morphism (type N → T) among its rules",
                expr.line,
                expr.col,
            )
        try:
            start = expr.start if isinstance(expr.start, int) else 0
            return InsideAlgorithm(binary, lexical, start=start)
        except TypeError as e:
            raise CompileError(str(e), expr.line, expr.col) from e

    def _compile_chart_fold(self, expr):
        """Compile a chart_fold(...) primitive expression.

        chart_fold is the explicit form of which the legacy
        parser(rules=...) is sugar. Given a lexical morphism
        ``lex : Token -> Cat`` plus a binary morphism (and optional
        unary morphism) on Cat, it constructs an InsideAlgorithm-based
        chart parser. The user-visible structure of the parser is
        therefore expressible from primitives — no opaque parser()
        call required.

        Effect-typed chart cells (``effect_depth`` > 0) extend the
        category universe to ``Cat × EffectStack_{≤d}`` via the
        class-driven lifting machinery in
        :mod:`quivers.stochastic.effect_lifts`; the caller is expected
        to have constructed ``binary`` (and any ``unary``) over this
        enlarged universe, typically via
        :func:`quivers.stochastic.effect_lifts.lift_rule_set` over the
        declared :class:`EffectDecl` instances in scope. The
        ``effect_depth`` integer flows through to the parser as the
        depth bound used for any depth-truncating reductions over
        intermediate cells.

        Handler firings (``handlers=`` argument) are applied as a
        post-composition step on the parser's denotation: the final
        chart cell is routed through each handler's :meth:`run`
        morphism in declared order, reducing the effect stack as the
        handlers compose.
        """
        from quivers.stochastic.inside import InsideAlgorithm

        lex = self._compile_expr(expr.lex)
        if expr.binary is None:
            raise CompileError(
                "chart_fold(...) requires a binary= argument (a morphism "
                "Cat * Cat -> Cat representing the union of binary rule "
                "schemas)",
                expr.line,
                expr.col,
            )
        binary = self._compile_expr(expr.binary)

        unary = self._compile_expr(expr.unary) if expr.unary is not None else None

        handlers_morphisms: list = []
        for h_expr in getattr(expr, "handlers", ()) or ():
            handlers_morphisms.append(self._compile_expr(h_expr))

        try:
            start = expr.start if isinstance(expr.start, int) else 0
            parser = InsideAlgorithm(binary, lex, start=start, unary=unary)
        except (TypeError, ValueError) as e:
            raise CompileError(str(e), expr.line, expr.col) from e

        # Compose handlers as post-applications on the parser's output.
        # Each handler is a morphism Cat → Cat (or a more refined effect
        # reduction); composition is right-to-left in declaration order.
        result = parser
        for handler in handlers_morphisms:
            result = _ChartHandlerComposite(result, handler)
        return result

    def _compile_parser_schemas(self, schemas: list, expr: ExprParser):
        """Compile parser from schema functors over a category system.

        Parameters
        ----------
        schemas : list
            Schema objects from ``SCHEMA_REGISTRY``.
        expr : ExprParser
            The AST node.
        """
        from quivers.stochastic.categories import CategorySystem
        from quivers.stochastic.parsers import ChartParser

        from quivers.core.objects import FreeResiduated

        if expr.categories:
            categories = list(expr.categories)
        elif self._categories:
            categories = list(self._categories)
        else:
            # Look for a FreeResiduated object in scope and use its
            # generators' atom names. If exactly one residuated universe
            # is declared, this avoids the user having to spell out
            # `categories=[NP, S, VP, ...]` redundantly.
            residuated = [
                obj for obj in self._objects.values() if isinstance(obj, FreeResiduated)
            ]
            if len(residuated) == 1:
                categories = list(residuated[0].generators.elements)
            elif len(residuated) > 1:
                raise CompileError(
                    "parser() with schema rules: multiple FreeResiduated "
                    "objects in scope; pass categories=[...] explicitly to "
                    "select the atom set",
                    expr.line,
                    expr.col,
                )
            else:
                raise CompileError(
                    "parser() with schema rules requires category atoms — declare them via `object Atoms = {NP, S, VP, ...}` plus `object Cat = FreeResiduated(Atoms, ...)`, or pass categories=[NP, S, VP, ...] inline",
                    expr.line,
                    expr.col,
                )
        if expr.constructors is not None:
            cs = CategorySystem.from_generators(
                atoms=categories,
                constructors=list(expr.constructors),
                max_depth=expr.depth,
            )
        else:
            cs = CategorySystem.from_atoms_and_slash_depth(
                categories, max_depth=expr.depth
            )
        schema = schemas[0]
        for piece in schemas[1:]:
            schema = schema | piece
        if expr.terminal is None:
            raise CompileError(
                "parser() with schema rules requires terminal=<object> — the declared object serving as the terminal vocabulary",
                expr.line,
                expr.col,
            )
        if expr.terminal not in self._objects:
            raise CompileError(
                f"terminal={expr.terminal!r} does not refer to a declared object",
                expr.line,
                expr.col,
            )
        n_term = self._objects[expr.terminal].size
        try:
            start = expr.start if isinstance(expr.start, str) else "S"
            return ChartParser.from_schema(schema, cs, n_terminals=n_term, start=start)
        except (TypeError, ValueError) as e:
            raise CompileError(str(e), expr.line, expr.col) from e
