"""Single source of truth for every parametric distribution family
the quivers stack supports.

A :class:`FamilySpec` carries everything the three call paths need:

* the **conditional** path
  (:class:`quivers.continuous.families._IndependentConditional`)
  consumes ``params`` to know how to transform unbounded MLP outputs
  and how to instantiate a :mod:`torch.distributions` object;
* the **fixed inline** path
  (:class:`quivers.continuous.inline.FixedDistribution`) consumes
  ``params`` and ``support`` to construct a distribution from
  literal compile-time floats;
* the **mixed inline** path
  (:class:`quivers.continuous.inline.MixedInlineDistribution`)
  consumes the same spec to build a distribution whose parameters
  are stacked runtime tensors.

Every family registers itself exactly once via :func:`register`; the
inline and conditional generators read from the same record. This
is the architectural seam that keeps the family catalog from
duplicating itself across three modules.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.distributions as D
import torch.nn.functional as F
from torch.distributions import constraints as _constraints

from quivers.core._util import EPS

# ---------------------------------------------------------------------------
# Parameter transforms (used by the conditional path to convert raw
# MLP outputs into constraint-respecting values).
# ---------------------------------------------------------------------------


def _identity(x: torch.Tensor) -> torch.Tensor:
    return x


def _softplus(x: torch.Tensor) -> torch.Tensor:
    return F.softplus(x) + EPS


def _softplus_shifted(x: torch.Tensor) -> torch.Tensor:
    """Positive with a minimum of 0.1 for concentration / df-style params."""
    return F.softplus(x) + 0.1


def _exp(x: torch.Tensor) -> torch.Tensor:
    return x.exp().clamp(min=EPS)


def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


_RAW_TRANSFORMS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "id": _identity,
    "softplus": _softplus,
    "softplus_shifted": _softplus_shifted,
    "exp": _exp,
    "sigmoid": _sigmoid,
}


# Per-named-transform safety clamp for inline parameters. The user
# supplies these directly as either literal floats (in
# ``FixedDistribution``) or runtime tensors (in
# ``MixedInlineDistribution``); we enforce the matching constraint
# at construction time so a slightly-off tensor (e.g. ``scale = 0``
# from a guide draw against a HalfNormal prior at the boundary)
# doesn't tip torch's distribution validation into raising.
def _clamp_id(t: torch.Tensor) -> torch.Tensor:
    return t


def _clamp_positive(t: torch.Tensor) -> torch.Tensor:
    return t.clamp(min=EPS)


def _clamp_positive_shifted(t: torch.Tensor) -> torch.Tensor:
    return t.clamp(min=0.1)


def _clamp_unit_interval(t: torch.Tensor) -> torch.Tensor:
    return t.clamp(min=EPS, max=1.0 - EPS)


_INLINE_CLAMPS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "id": _clamp_id,
    "softplus": _clamp_positive,
    "softplus_shifted": _clamp_positive_shifted,
    "exp": _clamp_positive,
    "sigmoid": _clamp_unit_interval,
}


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------


ParamKind = Literal["scalar", "vector", "integer"]
"""Per-parameter shape contract on the inline call side.

* ``"scalar"`` — broadcasts across the codomain's last axis
  (one parameter value per per-codomain dimension, as in
  ``Normal(loc, scale)``).
* ``"vector"`` — a single vector parameter for the whole sample
  (used by ``Dirichlet`` whose concentration is the full
  simplex-dimensional vector, not a per-component scalar).
* ``"integer"`` — a count-typed parameter (``total_count`` for
  ``Binomial`` / ``NegativeBinomial`` / ``Multinomial``); the
  fixed-factory accepts a Python ``int``.
"""


@dataclass(frozen=True)
class ParamSpec:
    """Spec for a single parameter of a distribution family.

    Attributes
    ----------
    name : str
        Keyword argument expected by the underlying torch
        ``Distribution`` constructor.
    transform : str
        Name of the raw-output transform applied on the conditional
        path (one of the keys of :data:`_RAW_TRANSFORMS`).  Also
        determines the safety clamp on the inline path.
    kind : ParamKind
        Per-parameter shape contract on the inline call side.
    """

    name: str
    transform: str
    kind: ParamKind = "scalar"

    def __post_init__(self) -> None:
        if self.transform not in _RAW_TRANSFORMS:
            raise ValueError(
                f"ParamSpec {self.name}: unknown transform {self.transform!r}. "
                f"Valid transforms: {sorted(_RAW_TRANSFORMS)}"
            )

    @property
    def raw_transform(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return _RAW_TRANSFORMS[self.transform]

    @property
    def inline_clamp(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return _INLINE_CLAMPS[self.transform]


OutputKind = Literal[
    "independent",  # per-dim independent, e.g. Normal[d]
    "vector",  # a single vector output (Dirichlet, LogisticNormal)
    "mvn",  # multivariate normal-style joint
    "matrix",  # matrix-valued (Wishart, LKJCholesky)
    "categorical",  # discrete index over a FinSet
    "mixture",  # MixtureSameFamily wrapper
]
"""How the distribution's output relates to the codomain.

The conditional path reads ``output_kind`` to decide whether to use
the generic per-dim ``_IndependentConditional`` or a hand-written
class.  The inline path reads it to pick the matching codomain
inference rule.
"""


@dataclass(frozen=True)
class FamilySpec:
    """Single source of truth for a distribution family.

    Used by:

    * :class:`quivers.continuous.families._IndependentConditional`
      and the standalone ``ConditionalX`` classes for the
      learnable-parameter path;
    * :class:`quivers.continuous.inline.FixedDistribution` and
      :class:`quivers.continuous.inline.MixedInlineDistribution`
      for the DSL inline call surface.
    """

    name: str
    """DSL key. Programs reference this name in ``<- F(args)`` lines."""

    dist_class: type
    """The underlying :mod:`torch.distributions` class."""

    params: tuple[ParamSpec, ...]
    """Ordered parameter specs, matching the dist's positional / keyword API."""

    support: _constraints.Constraint
    """Output support, used by variational guides to pick a bijector."""

    discrete: bool = False
    """True for integer / categorical outputs."""

    output_kind: OutputKind = "independent"
    """How the distribution's output relates to the codomain shape."""

    docstring: str = ""
    """Used as the generated ``ConditionalX`` class docstring."""

    # Hand-written overrides — used by families that don't fit the
    # generic generators (TruncatedNormal's bounded inline support,
    # Dirichlet's vector concentration, MultivariateNormal's
    # covariance, etc.). Populated as functions by the registering
    # module if needed.
    fixed_factory_override: Callable | None = field(default=None, repr=False)
    """Override for the all-literal ``FixedDistribution`` factory."""

    mixed_builder_override: Callable | None = field(default=None, repr=False)
    """Override for the mixed-mode dist builder
    ``(list[Tensor]) -> Distribution``."""

    conditional_class_override: type | None = field(default=None, repr=False)
    """Override for the conditional class (used by Categorical,
    Bernoulli, MultivariateNormal, etc.)."""

    @property
    def param_names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.params)


# Global mutable registry.  Populated by `register` at import time.
FAMILY_REGISTRY: dict[str, FamilySpec] = {}


def register(spec: FamilySpec) -> FamilySpec:
    """Register a :class:`FamilySpec` under its name; idempotent on
    re-registration of the same name to support hot-reload during
    development.
    """
    FAMILY_REGISTRY[spec.name] = spec
    return spec


def get(name: str) -> FamilySpec | None:
    """Look up a registered family by DSL name; ``None`` if absent."""
    return FAMILY_REGISTRY.get(name)


def names() -> tuple[str, ...]:
    """Sorted tuple of every registered family name."""
    return tuple(sorted(FAMILY_REGISTRY))


# ---------------------------------------------------------------------------
# Default builders driven by FamilySpec.  Imported and used by
# inline.py to construct FixedDistribution / MixedInlineDistribution
# without per-family boilerplate.
# ---------------------------------------------------------------------------


def build_torch_distribution(
    spec: FamilySpec,
    params: list[torch.Tensor] | dict[str, torch.Tensor],
) -> D.Distribution:
    """Instantiate the underlying torch distribution from a list /
    dict of parameter tensors that have already been clamped to the
    family's constraints.
    """
    if isinstance(params, dict):
        kwargs = params
    else:
        kwargs = {p.name: t for p, t in zip(spec.params, params)}
    return spec.dist_class(**kwargs)


def clamp_param(spec: FamilySpec, name: str, value: torch.Tensor) -> torch.Tensor:
    """Apply the inline safety clamp for the named parameter of
    ``spec``."""
    for p in spec.params:
        if p.name == name:
            return p.inline_clamp(value)
    raise KeyError(f"family {spec.name!r} has no parameter named {name!r}")


__all__ = [
    "FAMILY_REGISTRY",
    "FamilySpec",
    "ParamKind",
    "ParamSpec",
    "OutputKind",
    "build_torch_distribution",
    "clamp_param",
    "get",
    "names",
    "register",
]
