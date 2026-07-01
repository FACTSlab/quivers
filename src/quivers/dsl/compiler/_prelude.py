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

import torch
from quivers.continuous.spaces import (
    ContinuousSpace,
    Euclidean,
    PositiveReals,
    Simplex,
    UnitInterval,
)
from quivers.continuous.morphisms import AnySpace
from quivers.core.algebras import (
    BOOLEAN,
    CompositionRule,
    PRODUCT_FUZZY,
    material_implication,
)
from quivers.core.wiring import EinsumWiring
from quivers.core.morphism_transformations import (
    bayes_invert as _bayes_invert,
    l1_normalize as _l1_normalize,
    l2_normalize as _l2_normalize,
    softmax as _softmax,
)
from quivers.core.algebra_morphisms import (
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


from quivers.dsl.ast_nodes import SortVocabLiteral
from quivers.dsl.ast_nodes import (
    AxisSpec,
    ObjectEffectApply,
    TypeName,
    ObjectProduct,
)

# Registry of composition rules the DSL knows about by name.
# Each entry maps a user-facing keyword (the right-hand side of
# ``algebra X``, ``semigroupoid X``, ``bilinear_form X``,
# ``composition_rule X``) to its concrete rule instance.  The
# type is widened to ``CompositionRule`` (the common parent of
# ``Algebra``, ``Semigroupoid``, and ``BilinearForm``) so
# user-defined non-algebra composition rules can be registered
# alongside the built-in ones.
_ALGEBRA_REGISTRY: dict[str, "CompositionRule"] = {
    "product_fuzzy": PRODUCT_FUZZY,
    "boolean": BOOLEAN,
}


class _CompiledContraction:
    """Compiled form of a `ContractionDecl`.

    Stores the einsum wiring object, the declared output domain
    and codomain (so the compiler can re-tag the result morphism
    after the contraction), the expected input morphism types
    (for shape validation), and the active composition rule.

    A plain class (not a `didactic.api.Model`) because the
    fields are opaque runtime objects — `EinsumWiring`,
    `SetObject` / `ContinuousSpace`,
    `CompositionRule` instances — that don't translate to a
    panproto sort.
    """

    __slots__ = ("name", "wiring", "domain", "codomain", "input_types", "algebra")

    def __init__(
        self,
        name: str,
        wiring: "EinsumWiring",
        domain: "AnySpace",
        codomain: "AnySpace",
        input_types: tuple[tuple[str, "AnySpace", "AnySpace"], ...],
        algebra: "CompositionRule",
    ) -> None:
        self.name = name
        self.wiring = wiring
        self.domain = domain
        self.codomain = codomain
        self.input_types = input_types
        self.algebra = algebra


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
    `CompositionRule` interface.

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

    Each value is a `AlgebraHomomorphism` or
    `MorphismTransformation` (collectively, a ``Trans``
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
    from the DSL scope and return a `MorphismTransformation`.

    Surface form ``softmax(B)`` / ``bayes_invert(prior)`` parses
    as `ExprMorphismCall` and dispatches here when the
    callee resolves into this dict.
    """
    return {
        "softmax": _softmax,
        "l1_normalize": _l1_normalize,
        "l2_normalize": _l2_normalize,
        "bayes_invert": _bayes_invert,
    }


def _register_extra_algebras() -> None:
    """Lazily register every built-in algebra into the
    ``algebra <name>`` resolution table the DSL uses at module
    top.

    The registration is idempotent and short-circuits when the
    table is already populated. Catching `ImportError`
    keeps the compiler usable for users who don't have the
    optional dependencies (e.g. the stochastic module pulls in
    ``torch.distributions`` heavily).
    """
    if "lukasiewicz" not in _ALGEBRA_REGISTRY:
        try:
            from quivers.core.algebras import (
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
            from quivers.core.algebras import BOOLEAN_DUAL, REICHENBACH

            _ALGEBRA_REGISTRY["lukasiewicz"] = LUKASIEWICZ
            _ALGEBRA_REGISTRY["godel"] = GODEL
            _ALGEBRA_REGISTRY["tropical"] = TROPICAL
            _ALGEBRA_REGISTRY["max_plus"] = MAX_PLUS
            _ALGEBRA_REGISTRY["log_prob"] = LOG_PROB
            _ALGEBRA_REGISTRY["real"] = REAL
            _ALGEBRA_REGISTRY["probability"] = PROBABILITY
            _ALGEBRA_REGISTRY["counting"] = COUNTING
            # Built-in non-algebra composition rules.
            _ALGEBRA_REGISTRY["material_impl"] = material_implication()
            # Named de-Morgan duals — each is the corresponding
            # ``X.dual()`` exposed under a DSL-friendly name so a
            # user can write ``algebra reichenbach``.
            _ALGEBRA_REGISTRY["reichenbach"] = REICHENBACH
            _ALGEBRA_REGISTRY["boolean_dual"] = BOOLEAN_DUAL
            _ALGEBRA_REGISTRY["dual_lukasiewicz"] = DUAL_LUKASIEWICZ
            _ALGEBRA_REGISTRY["dual_godel"] = DUAL_GODEL
        except ImportError:
            pass
    if "markov" not in _ALGEBRA_REGISTRY:
        try:
            from quivers.stochastic import MARKOV

            _ALGEBRA_REGISTRY["markov"] = MARKOV
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
    """Extract the axis names from a ObjectExpr.

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
    if isinstance(texpr, ObjectEffectApply):
        # e.g. Euclidean(D): axis name is the argument's name when
        # it's itself a TypeName.
        args = getattr(texpr, "args", None)
        if args and len(args) == 1 and isinstance(args[0], TypeName):
            return True, (args[0].name,)
        return True, ()
    if isinstance(texpr, ObjectProduct):
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
            line,
            col,
        )
    bad = [a for a in axes.over if a not in available_axes]
    if bad:
        raise CompileError(
            f"axis-role clause: unknown axis name(s) {bad!r} in `over`; "
            f"available: {sorted(available_axes)}",
            line,
            col,
        )
    bad_iid = [a for a in axes.iid_over if a not in available_axes]
    if bad_iid:
        raise CompileError(
            f"axis-role clause: unknown axis name(s) {bad_iid!r} in "
            f"`iid over`; available: {sorted(available_axes)}",
            line,
            col,
        )
    overlap = set(axes.over) & set(axes.iid_over)
    if overlap:
        raise CompileError(
            f"axis-role clause: axes {sorted(overlap)!r} appear in both "
            f"`over` and `iid over`; each axis has exactly one role",
            line,
            col,
        )


def _get_family_registry() -> dict[str, type]:
    """Lazily build the distribution family registry."""
    global _FAMILY_REGISTRY
    if _FAMILY_REGISTRY is not None:
        return _FAMILY_REGISTRY
    from quivers.continuous.families import (
        ConditionalBeta,
        ConditionalBernoulli,
        ConditionalBinomial,
        ConditionalCategorical,
        ConditionalCauchy,
        ConditionalChi2,
        ConditionalContinuousBernoulli,
        ConditionalDirichlet,
        ConditionalExponential,
        ConditionalFisherSnedecor,
        ConditionalGamma,
        ConditionalGaussianProcess,
        ConditionalGumbel,
        ConditionalHalfCauchy,
        ConditionalHalfNormal,
        ConditionalHorseshoe,
        ConditionalHurdlePoisson,
        ConditionalInverseGamma,
        ConditionalInverseWishart,
        ConditionalKumaraswamy,
        ConditionalLaplace,
        ConditionalLogNormal,
        ConditionalLogitNormal,
        ConditionalLowRankMVN,
        ConditionalMatrixNormal,
        ConditionalMixtureNormal,
        ConditionalMultivariateNormal,
        ConditionalNegativeBinomial,
        ConditionalNormal,
        ConditionalOrderedLogistic,
        ConditionalPareto,
        ConditionalPoisson,
        ConditionalRelaxedBernoulli,
        ConditionalRelaxedOneHotCategorical,
        ConditionalStudentT,
        ConditionalTruncatedNormal,
        ConditionalUniform,
        ConditionalWeibull,
        ConditionalWishart,
        ConditionalZeroInflatedPoisson,
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
        "Binomial": ConditionalBinomial,
        "Categorical": ConditionalCategorical,
        "Poisson": ConditionalPoisson,
        "NegativeBinomial": ConditionalNegativeBinomial,
        "OrderedLogistic": ConditionalOrderedLogistic,
        "ZeroInflatedPoisson": ConditionalZeroInflatedPoisson,
        "HurdlePoisson": ConditionalHurdlePoisson,
        "MixtureNormal": ConditionalMixtureNormal,
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
