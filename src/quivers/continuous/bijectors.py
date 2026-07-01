"""Bijector library for the compositional measure algebra.

A `Bijector` is a measurable bijection $f: X \\to Y$ between Borel
spaces, together with its inverse $f^{-1}$ and the log-determinant
of its Jacobian. Pushing forward a measure $\\mu$ along $f$ produces
a new measure $f_*\\mu$ with density

$$p_{f_*\\mu}(y) = p_\\mu(f^{-1}(y))\\,\\bigl|\\det J_{f^{-1}}(y)\\bigr|.$$

In log-space (the only sane computational discipline; see
[TensorFlow Probability's `Bijector` rationale][tfp-bijector]):

$$\\log p_{f_*\\mu}(y) = \\log p_\\mu(f^{-1}(y)) + \\log\\bigl|\\det
J_{f^{-1}}(y)\\bigr|.$$

Every bijector exposes:

* `forward(x)` evaluating $f$,
* `inverse(y)` evaluating $f^{-1}$,
* `forward_log_det_jacobian(x)` evaluating
  $\\log\\bigl|\\det J_f(x)\\bigr|$,
* `inverse_log_det_jacobian(y)` evaluating
  $\\log\\bigl|\\det J_{f^{-1}}(y)\\bigr|$ (equal to
  $-\\text{forward\\_log\\_det\\_jacobian}(f^{-1}(y))$ when both are
  finite).

The library is closed under composition via `Compose(b1, b2)` and
under inversion via `Inverse(b)`, so the family forms a groupoid
of measurable isomorphisms.

[tfp-bijector]: https://www.tensorflow.org/probability/api_docs/python/tfp/bijectors/Bijector
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor


class Bijector(ABC):
    """Abstract base for measurable bijections used in
    [`Pushforward`][quivers.continuous.measure.Pushforward].

    Implementations override the four primitive methods; the
    `__call__`, `Compose`, and `inv` helpers compose them.
    """

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor: ...

    @abstractmethod
    def inverse(self, y: Tensor) -> Tensor: ...

    @abstractmethod
    def forward_log_det_jacobian(self, x: Tensor) -> Tensor: ...

    def inverse_log_det_jacobian(self, y: Tensor) -> Tensor:
        """Default implementation: invert the forward Jacobian's
        log-determinant. Subclasses with a more numerically stable
        closed form should override.
        """
        return -self.forward_log_det_jacobian(self.inverse(y))

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def inv(self) -> Bijector:
        """Return the bijector's inverse, swapping `forward` /
        `inverse` and negating the Jacobian log-determinant.
        """
        return Inverse(self)


class Inverse(Bijector):
    """The inverse of a bijector, deferring all four primitives to
    the wrapped instance with their roles swapped.
    """

    def __init__(self, base: Bijector) -> None:
        self.base = base

    def forward(self, x: Tensor) -> Tensor:
        return self.base.inverse(x)

    def inverse(self, y: Tensor) -> Tensor:
        return self.base.forward(y)

    def forward_log_det_jacobian(self, x: Tensor) -> Tensor:
        return self.base.inverse_log_det_jacobian(x)

    def inverse_log_det_jacobian(self, y: Tensor) -> Tensor:
        return self.base.forward_log_det_jacobian(y)


class Compose(Bijector):
    """Composition `outer ∘ inner`. Forward applies `inner` first
    then `outer`; inverse applies `outer.inverse` first then
    `inner.inverse`. Log-Jacobians add by the chain rule.
    """

    def __init__(self, outer: Bijector, inner: Bijector) -> None:
        self.outer = outer
        self.inner = inner

    def forward(self, x: Tensor) -> Tensor:
        return self.outer.forward(self.inner.forward(x))

    def inverse(self, y: Tensor) -> Tensor:
        return self.inner.inverse(self.outer.inverse(y))

    def forward_log_det_jacobian(self, x: Tensor) -> Tensor:
        inner_x = self.inner.forward(x)
        return self.inner.forward_log_det_jacobian(
            x
        ) + self.outer.forward_log_det_jacobian(inner_x)

    def inverse_log_det_jacobian(self, y: Tensor) -> Tensor:
        outer_inv = self.outer.inverse(y)
        return self.outer.inverse_log_det_jacobian(
            y
        ) + self.inner.inverse_log_det_jacobian(outer_inv)


class Identity(Bijector):
    """The identity map. Useful as a neutral element in compositions
    and as the trivial pushforward.
    """

    def forward(self, x: Tensor) -> Tensor:
        return x

    def inverse(self, y: Tensor) -> Tensor:
        return y

    def forward_log_det_jacobian(self, x: Tensor) -> Tensor:
        return torch.zeros_like(x)

    def inverse_log_det_jacobian(self, y: Tensor) -> Tensor:
        return torch.zeros_like(y)


class Exp(Bijector):
    """The exponential map $f(x) = e^x$ from $\\mathbb{R}$ to
    $(0, \\infty)$. Jacobian log-determinant is $x$.
    """

    def forward(self, x: Tensor) -> Tensor:
        return torch.exp(x)

    def inverse(self, y: Tensor) -> Tensor:
        return torch.log(y)

    def forward_log_det_jacobian(self, x: Tensor) -> Tensor:
        return x

    def inverse_log_det_jacobian(self, y: Tensor) -> Tensor:
        return -torch.log(y)


class Log(Bijector):
    """The logarithm $f(x) = \\log x$ from $(0, \\infty)$ to
    $\\mathbb{R}$. Inverse of [`Exp`][quivers.continuous.bijectors.Exp];
    Jacobian log-determinant is $-\\log x$.
    """

    def forward(self, x: Tensor) -> Tensor:
        return torch.log(x)

    def inverse(self, y: Tensor) -> Tensor:
        return torch.exp(y)

    def forward_log_det_jacobian(self, x: Tensor) -> Tensor:
        return -torch.log(x)

    def inverse_log_det_jacobian(self, y: Tensor) -> Tensor:
        return y


class Sigmoid(Bijector):
    """The logistic sigmoid $f(x) = 1/(1 + e^{-x})$ from
    $\\mathbb{R}$ to $(0, 1)$.

    The Jacobian log-determinant is the log-density of the standard
    logistic distribution, $-x - 2\\log(1 + e^{-x})$, in a form that
    is stable in both tails.
    """

    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(x)

    def inverse(self, y: Tensor) -> Tensor:
        return torch.logit(y)

    def forward_log_det_jacobian(self, x: Tensor) -> Tensor:
        return -torch.nn.functional.softplus(x) - torch.nn.functional.softplus(-x)

    def inverse_log_det_jacobian(self, y: Tensor) -> Tensor:
        return -torch.log(y) - torch.log1p(-y)


class Logit(Bijector):
    """The logit $f(x) = \\log(x / (1 - x))$ from $(0, 1)$ to
    $\\mathbb{R}$. Inverse of [`Sigmoid`][quivers.continuous.bijectors.Sigmoid].
    """

    def forward(self, x: Tensor) -> Tensor:
        return torch.logit(x)

    def inverse(self, y: Tensor) -> Tensor:
        return torch.sigmoid(y)

    def forward_log_det_jacobian(self, x: Tensor) -> Tensor:
        return -torch.log(x) - torch.log1p(-x)

    def inverse_log_det_jacobian(self, y: Tensor) -> Tensor:
        return -torch.nn.functional.softplus(y) - torch.nn.functional.softplus(-y)


class Softplus(Bijector):
    """The softplus map $f(x) = \\log(1 + e^x)$ from $\\mathbb{R}$
    to $(0, \\infty)$. Smooth alternative to
    [`Exp`][quivers.continuous.bijectors.Exp] with linear tail growth
    on the positive side.

    Forward Jacobian: $df/dx = \\sigma(x)$, so
    $\\log|df/dx| = -\\text{softplus}(-x)$.

    Inverse: $f^{-1}(y) = \\log(e^y - 1)$.
    Inverse Jacobian: $df^{-1}/dy = 1/(1 - e^{-y})$, so
    $\\log|df^{-1}/dy| = -\\log(1 - e^{-y}) = -\\log(-\\mathrm{expm1}(-y))$.
    """

    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.softplus(x)

    def inverse(self, y: Tensor) -> Tensor:
        return y + torch.log(-torch.expm1(-y))

    def forward_log_det_jacobian(self, x: Tensor) -> Tensor:
        return -torch.nn.functional.softplus(-x)

    def inverse_log_det_jacobian(self, y: Tensor) -> Tensor:
        return -torch.log(-torch.expm1(-y))


class Affine(Bijector):
    """The affine map $f(x) = \\text{scale} \\cdot x + \\text{shift}$.
    `scale` must be strictly positive; the Jacobian log-determinant
    is $\\log\\,\\text{scale}$, broadcast to the input shape.
    """

    def __init__(
        self,
        scale: Tensor | float,
        shift: Tensor | float,
    ) -> None:
        self.scale = torch.as_tensor(scale)
        self.shift = torch.as_tensor(shift)
        if torch.any(self.scale <= 0):
            raise ValueError(
                f"Affine: scale must be strictly positive; got {self.scale!r}"
            )

    def forward(self, x: Tensor) -> Tensor:
        return self.scale * x + self.shift

    def inverse(self, y: Tensor) -> Tensor:
        return (y - self.shift) / self.scale

    def forward_log_det_jacobian(self, x: Tensor) -> Tensor:
        log_scale = torch.log(self.scale)
        return log_scale.expand_as(x)

    def inverse_log_det_jacobian(self, y: Tensor) -> Tensor:
        log_scale = torch.log(self.scale)
        return (-log_scale).expand_as(y)


class StickBreaking(Bijector):
    """The stick-breaking map from $\\mathbb{R}^{K-1}$ to the
    open $K$-simplex $\\{p \\in \\mathbb{R}^K_{>0} : \\sum_k p_k = 1\\}$.

    Forward: $y_k = z_k \\cdot \\prod_{j<k}(1 - z_j)$ for
    $k < K-1$, where $z_k = \\sigma(x_k - \\log(K - k - 1))$, and
    $y_{K-1} = 1 - \\sum_{k<K-1} y_k$. The shift by
    $\\log(K - k - 1)$ centres the prior on the uniform simplex
    when $x \\sim \\mathrm{Normal}(0, 1)$, matching the convention
    used by Stan and PyMC.

    Inverse: $x_k = \\sigma^{-1}(z_k) + \\log(K - k - 1)$ where
    $z_k = y_k / (1 - \\sum_{j<k} y_j)$.

    The Jacobian log-determinant in the forward direction sums
    `log(z_k) + log(1 - z_k) + log(remaining)` across the K-1
    stick-breaks. Used in the desugaring of `LogisticNormal` and
    other simplex-valued transforms.
    """

    def forward(self, x: Tensor) -> Tensor:
        K_minus_1 = x.shape[-1]
        offsets = torch.log(
            torch.arange(K_minus_1, 0, -1, dtype=x.dtype, device=x.device)
        )
        z = torch.sigmoid(x - offsets)
        log_remaining = torch.log1p(-z).cumsum(dim=-1)
        first = z[..., :1]
        rest = z[..., 1:] * torch.exp(log_remaining[..., :-1])
        y_first_K_minus_1 = torch.cat([first, rest], dim=-1)
        y_last = (1.0 - y_first_K_minus_1.sum(dim=-1, keepdim=True)).clamp_min(
            torch.finfo(x.dtype).tiny
        )
        return torch.cat([y_first_K_minus_1, y_last], dim=-1)

    def inverse(self, y: Tensor) -> Tensor:
        K = y.shape[-1]
        K_minus_1 = K - 1
        offsets = torch.log(
            torch.arange(K_minus_1, 0, -1, dtype=y.dtype, device=y.device)
        )
        cumulative = y[..., :K_minus_1].cumsum(dim=-1)
        first = y[..., :1]
        denom_rest = (1.0 - cumulative[..., :-1]).clamp_min(torch.finfo(y.dtype).tiny)
        rest = y[..., 1:K_minus_1] / denom_rest
        z = torch.cat([first, rest], dim=-1)
        return (
            torch.logit(
                z.clamp(
                    torch.finfo(y.dtype).tiny,
                    1.0 - torch.finfo(y.dtype).tiny,
                )
            )
            + offsets
        )

    def forward_log_det_jacobian(self, x: Tensor) -> Tensor:
        K_minus_1 = x.shape[-1]
        offsets = torch.log(
            torch.arange(K_minus_1, 0, -1, dtype=x.dtype, device=x.device)
        )
        shifted = x - offsets
        log_z = torch.nn.functional.logsigmoid(shifted)
        log_1m_z = torch.nn.functional.logsigmoid(-shifted)
        cum_log_1m_z = log_1m_z.cumsum(dim=-1)
        leading = cum_log_1m_z[..., :-1].sum(dim=-1)
        return log_z.sum(dim=-1) + log_1m_z.sum(dim=-1) + leading


__all__ = [
    "Affine",
    "Bijector",
    "Compose",
    "Exp",
    "Identity",
    "Inverse",
    "Log",
    "Logit",
    "Sigmoid",
    "Softplus",
    "StickBreaking",
]
