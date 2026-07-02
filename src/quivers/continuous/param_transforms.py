"""Bijector-typed parameter transforms for continuous families.

The parameter-transform vocabulary is the mapping from raw
network / lookup output (unconstrained real) to a constrained
value in the parametric family's parameter space (a positive rate,
a probability in the unit interval, a positive-with-shift degrees
of freedom, and so on). Historically the vocabulary lived as a
string-keyed table of `Callable[[Tensor], Tensor]`, with no
inverse, no Jacobian, and no composition surface. The
[`Bijector`][quivers.continuous.bijectors.Bijector] library
already carries all four of those, so the two vocabularies should
be one.

This module exposes the bijector-typed transform table:

* [`TRANSFORM_TO_BIJECTOR`][quivers.continuous.param_transforms.TRANSFORM_TO_BIJECTOR]
  maps every string key accepted by `_RAW_TRANSFORMS` to its
  [`Bijector`][quivers.continuous.bijectors.Bijector] equivalent.
  The two forms are behaviourally identical on the forward map,
  and the bijector form additionally exposes `inverse`,
  `forward_log_det_jacobian`, and `inverse_log_det_jacobian` for
  guide / pushforward machinery.
* [`resolve_transform`][quivers.continuous.param_transforms.resolve_transform]
  accepts either a registered string or a
  [`Bijector`][quivers.continuous.bijectors.Bijector] instance and
  returns the bijector. Family internals route the raw parameter
  tensor through the bijector's `forward` to produce the
  constrained parameter.

The registry is open: users may pass a custom bijector directly
wherever a `ParamSpec.transform` is accepted, or extend the
string table by mutating
[`TRANSFORM_TO_BIJECTOR`][quivers.continuous.param_transforms.TRANSFORM_TO_BIJECTOR]
at import time.
"""

from __future__ import annotations

import torch
from torch import Tensor

from quivers.continuous.bijectors import (
    Affine,
    Bijector,
    Compose,
    Identity,
    Sigmoid,
    Softplus,
)
from quivers.core._util import EPS


class _ClampAbove(Bijector):
    """Forward map $f(x) = \\max(x, \\text{floor})$ used as a safety
    net for numerically-degenerate inline parameter values.

    Not strictly invertible on values below the floor; the inverse
    is the identity on the image $(\\text{floor}, \\infty)$. The
    log-det-Jacobian is zero on the interior. Used only as an
    inline clamp; not registered in
    [`TRANSFORM_TO_BIJECTOR`][quivers.continuous.param_transforms.TRANSFORM_TO_BIJECTOR].
    """

    def __init__(self, floor: float) -> None:
        self._floor = float(floor)

    def forward(self, x: Tensor) -> Tensor:
        return x.clamp(min=self._floor)

    def inverse(self, y: Tensor) -> Tensor:
        return y

    def forward_log_det_jacobian(self, x: Tensor) -> Tensor:
        return torch.zeros_like(x)

    def inverse_log_det_jacobian(self, y: Tensor) -> Tensor:
        return torch.zeros_like(y)


class _ClampInterval(Bijector):
    """Forward map $f(x) = \\text{clamp}(x, \\text{low},
    \\text{high})$, the inline safety clamp for unit-interval
    parameters (the Sigmoid family).
    """

    def __init__(self, low: float, high: float) -> None:
        self._low = float(low)
        self._high = float(high)

    def forward(self, x: Tensor) -> Tensor:
        return x.clamp(min=self._low, max=self._high)

    def inverse(self, y: Tensor) -> Tensor:
        return y

    def forward_log_det_jacobian(self, x: Tensor) -> Tensor:
        return torch.zeros_like(x)

    def inverse_log_det_jacobian(self, y: Tensor) -> Tensor:
        return torch.zeros_like(y)


class _ExpClamped(Bijector):
    """Forward map $f(x) = \\max(\\exp(x), \\varepsilon)$.

    Behaviourally matches the historical ``exp`` string transform
    that clamps the exponentiated value at
    [`EPS`][quivers.core._util.EPS] to protect downstream torch
    distribution validation. Inverse is the plain log; the
    log-det-Jacobian matches [`Exp`][quivers.continuous.bijectors.Exp]
    on the image $(\\varepsilon, \\infty)$.
    """

    def forward(self, x: Tensor) -> Tensor:
        return x.exp().clamp(min=EPS)

    def inverse(self, y: Tensor) -> Tensor:
        return torch.log(y)

    def forward_log_det_jacobian(self, x: Tensor) -> Tensor:
        return x

    def inverse_log_det_jacobian(self, y: Tensor) -> Tensor:
        return -torch.log(y)


def _softplus_shifted(shift: float) -> Bijector:
    """Softplus composed with a downstream additive shift.

    Composition order: `Compose(outer=Affine(1, shift),
    inner=Softplus())` so `forward(x) = softplus(x) + shift`.
    Backs the ``softplus`` (`shift = EPS`) and
    ``softplus_shifted`` (`shift = 0.1`) entries of the transform
    registry.
    """
    return Compose(Affine(scale=1.0, shift=shift), Softplus())


TRANSFORM_TO_BIJECTOR: dict[str, Bijector] = {
    "id": Identity(),
    "sigmoid": Sigmoid(),
    "softplus": _softplus_shifted(EPS),
    "softplus_shifted": _softplus_shifted(0.1),
    "exp": _ExpClamped(),
}
"""Mapping from historical string-keyed transform names to their
`Bijector` equivalents.

The forward map of each bijector matches the historical
`Callable[[Tensor], Tensor]` in
`quivers.continuous.family_spec._RAW_TRANSFORMS` to numerical
precision. The composition
[`Compose`][quivers.continuous.bijectors.Compose] of
[`Softplus`][quivers.continuous.bijectors.Softplus] with an
[`Affine`][quivers.continuous.bijectors.Affine] shift replaces the
ad hoc `F.softplus(x) + shift` pattern; the identity, sigmoid, and
exponential entries map to their obvious bijector counterparts.
"""


INLINE_CLAMP_TO_BIJECTOR: dict[str, Bijector] = {
    "id": Identity(),
    "softplus": _ClampAbove(EPS),
    "softplus_shifted": _ClampAbove(0.1),
    "exp": _ClampAbove(EPS),
    "sigmoid": _ClampInterval(EPS, 1.0 - EPS),
}
"""Per-transform safety clamps used on the inline call path.

A `Bijector`-typed mirror of the historical inline-clamp table.
The inline path applies the clamp to user-supplied literal or
runtime parameters at construction time; the clamp is a
projection onto the constraint set, so its
`forward_log_det_jacobian` is zero.
"""


def resolve_transform(transform: str | Bijector) -> Bijector:
    """Return the bijector for a string key or pass through a bijector.

    Parameters
    ----------
    transform
        Either a string key registered in
        [`TRANSFORM_TO_BIJECTOR`][quivers.continuous.param_transforms.TRANSFORM_TO_BIJECTOR]
        or a `Bijector` instance.

    Returns
    -------
    Bijector
        The resolved bijector.

    Raises
    ------
    KeyError
        If `transform` is a string that is not registered.
    TypeError
        If `transform` is neither a string nor a `Bijector`.
    """
    if isinstance(transform, Bijector):
        return transform
    if isinstance(transform, str):
        if transform not in TRANSFORM_TO_BIJECTOR:
            raise KeyError(
                f"unknown transform {transform!r}; "
                f"valid string keys: {sorted(TRANSFORM_TO_BIJECTOR)}"
            )
        return TRANSFORM_TO_BIJECTOR[transform]
    raise TypeError(
        "resolve_transform: expected a string key or a Bijector instance; "
        f"got {type(transform).__name__}"
    )


def resolve_inline_clamp(transform: str | Bijector) -> Bijector:
    """Return the inline safety-clamp bijector for a transform.

    String transforms look up in
    [`INLINE_CLAMP_TO_BIJECTOR`][quivers.continuous.param_transforms.INLINE_CLAMP_TO_BIJECTOR];
    a user-supplied `Bijector` transform receives the identity
    clamp because the bijector's forward map is assumed to already
    land in its constraint set.

    Parameters
    ----------
    transform
        The transform whose inline clamp is requested.

    Returns
    -------
    Bijector
        The clamp bijector. Its `forward_log_det_jacobian` is zero
        because a projection onto a constraint set is measure-zero
        away from the constraint boundary.
    """
    if isinstance(transform, Bijector):
        return Identity()
    if isinstance(transform, str):
        if transform not in INLINE_CLAMP_TO_BIJECTOR:
            raise KeyError(
                f"unknown transform {transform!r}; "
                f"valid string keys: {sorted(INLINE_CLAMP_TO_BIJECTOR)}"
            )
        return INLINE_CLAMP_TO_BIJECTOR[transform]
    raise TypeError(
        "resolve_inline_clamp: expected a string key or a Bijector instance; "
        f"got {type(transform).__name__}"
    )


__all__ = [
    "INLINE_CLAMP_TO_BIJECTOR",
    "TRANSFORM_TO_BIJECTOR",
    "resolve_inline_clamp",
    "resolve_transform",
]
