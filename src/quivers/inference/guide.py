"""Variational guide families for approximate posterior inference.

A guide is a parameterized distribution q(z | x) over latent variables
that approximates the true posterior p(z | x, y_obs). Guides are used
by SVI to optimize the ELBO.

This module provides:

- ``Guide`` — abstract base class for all guides
- ``AutoNormalGuide`` — mean-field Normal over all continuous latents,
  with per-site bijector that maps unconstrained Normal samples to
  the prior's constrained support (Pyro's ``AutoNormal`` semantics).
- ``AutoDeltaGuide`` — point-estimate (MAP) guide, with the same
  per-site bijector handling so the point estimate always lies inside
  the prior's support.

Implementation notes
====================

Both ``AutoNormalGuide`` and ``AutoDeltaGuide`` operate on the model's
*unconstrained* latent parameters. For each latent site they
introspect the underlying morphism's ``support`` property (a
:class:`torch.distributions.constraints.Constraint`) and compose a
matching bijector via :func:`torch.distributions.transforms.biject_to`.

* Forward: sample / look up an unconstrained value ``z``, return
  ``bijector(z)`` so the constrained value lies in the prior's
  support — the prior's ``log_prob`` then evaluates without raising
  ``ValueError: Expected value … to be within the support of …``.
* Backward (``log_prob``): invert through the bijector to get ``z``,
  evaluate the unconstrained Normal density at ``z``, and add the
  inverse Jacobian to obtain the density of the constrained value
  ``v`` under the pushforward measure.

This is the same construction that Pyro's ``AutoNormal`` uses;
see Pyro `pyro/infer/autoguide/guides.py
<https://docs.pyro.ai/en/stable/_modules/pyro/infer/autoguide/guides.html>`_.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.distributions as D
from torch.distributions import constraints as _constraints
from torch.distributions.transforms import Transform
from torch.distributions.constraint_registry import biject_to
from typing import cast

from quivers.continuous.morphisms import ContinuousMorphism
from quivers.continuous.programs import MonadicProgram, _LetSpec
from quivers.continuous.spaces import ContinuousSpace


def _support_for_morphism(morph: ContinuousMorphism) -> _constraints.Constraint:
    """Return the support of a morphism's underlying distribution.

    Every :class:`ContinuousMorphism` exposes a :attr:`support`
    property whose default is :data:`constraints.real`. Inline
    distributions (:class:`quivers.continuous.inline.FixedDistribution`,
    :class:`~quivers.continuous.inline.MixedInlineDistribution`,
    :class:`~quivers.continuous.inline.DirectBernoulli`,
    :class:`~quivers.continuous.inline.DirectTruncatedNormal`) and the
    family-conditional distributions in
    :mod:`quivers.continuous.families` override it with the correct
    constraint (``unit_interval`` for Beta / LogitNormal, ``positive``
    for HalfNormal / Gamma / Exponential, ``simplex`` for Dirichlet,
    ``positive_definite`` for Wishart, …). The variational guides
    consult this to choose the right bijector.
    """
    return morph.support


def _unconstrained_dim(support: _constraints.Constraint, constrained_dim: int) -> int:
    """Return the unconstrained-side dimension after applying
    :func:`biject_to(support)` to a tensor of declared dimension
    ``constrained_dim``.

    Most bijectors are dim-preserving (:class:`ExpTransform`,
    :class:`SigmoidTransform`, :class:`AffineTransform`). The
    stick-breaking bijector for the simplex is the notable
    exception: the constrained side is the :math:`(d-1)`-simplex
    embedded in :math:`\\mathbb{R}^d`, while the unconstrained side
    has :math:`d-1` real coordinates.
    """
    if support is _constraints.simplex:
        return max(1, constrained_dim - 1)
    # CorrCholesky's unconstrained dim is d*(d-1)/2 ; LKJ-Cholesky
    # bijectors live in the same family. We surface what we know;
    # adding more transforms is straightforward as we encounter them.
    return constrained_dim


def _apply_bijector(
    bijector: Transform, z: torch.Tensor, constrained_dim: int
) -> torch.Tensor:
    """Apply ``bijector`` to an unconstrained tensor ``z``.

    Handles the leading batch axis explicitly so the user-facing
    shape contract (the constrained value has shape ``(batch, dim)``
    or ``(batch,)`` for ``dim == 1``) is preserved.
    """
    v = bijector(z)
    # Some bijectors (StickBreakingTransform) expand the trailing
    # event dimension; the constrained tensor already has the right
    # trailing shape. Normalise to (batch, dim) or (batch,) for
    # downstream consumers.
    if v.dim() == 1 and constrained_dim == 1:
        return v
    if v.dim() == 1 and constrained_dim > 1:
        return v.unsqueeze(0)
    return v


class Guide(nn.Module, ABC):
    """Abstract variational guide.

    A guide provides a parameterized approximate posterior q(z | x)
    over latent variables. It must support reparameterized sampling
    and log-density evaluation.
    """

    @abstractmethod
    def rsample(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Sample latent variables from the guide.

        Parameters
        ----------
        x : torch.Tensor
            Program input. Shape (batch, ...).

        Returns
        -------
        dict[str, torch.Tensor]
            Sampled values for each latent variable.
        """
        ...

    @abstractmethod
    def log_prob(
        self,
        x: torch.Tensor,
        sites: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Log-density of latent values under the guide.

        Parameters
        ----------
        x : torch.Tensor
            Program input. Shape (batch, ...).
        sites : dict[str, torch.Tensor]
            Values for each latent variable.

        Returns
        -------
        torch.Tensor
            Total log-density. Shape (batch,).
        """
        ...

    @property
    @abstractmethod
    def latent_names(self) -> list[str]:
        """Names of latent variables this guide covers."""
        ...


class AutoNormalGuide(Guide):
    """Mean-field Normal guide with per-site support-constrained bijector.

    For each continuous latent site the guide maintains a pair
    ``(loc, log_scale)`` in the *unconstrained* space induced by the
    site's prior support. Sampling proceeds in two steps:

    1. Draw an unconstrained sample ``z ~ Normal(loc, exp(log_scale))``.
    2. Apply the bijector :func:`biject_to(prior.support)` to obtain
       a constrained sample ``v = bijector(z)`` that lies in the
       prior's support (``[0, 1]`` for ``Beta``/``LogitNormal``,
       :math:`(0, \\infty)` for ``HalfNormal`` / ``Gamma``, the
       simplex for ``Dirichlet``, etc.).

    Without this construction, sampling from a real-valued Normal
    and feeding the result into a constrained-support prior's
    ``log_prob`` raises ``ValueError: Expected value … to be within
    the support of the distribution``. With it, the same
    ``AutoNormalGuide`` works uniformly across every distribution
    family in :mod:`quivers.continuous.families` and
    :mod:`quivers.continuous.inline`.

    Parameters
    ----------
    model : MonadicProgram
        The generative model to build a guide for.
    observed_names : set[str]
        Names of observed variables (excluded from the guide).
    init_scale : float
        Initial scale for all latent sites (in unconstrained space).
    """

    def __init__(
        self,
        model: MonadicProgram,
        observed_names: set[str],
        init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self._latent_names: list[str] = []
        self._supports: dict[str, _constraints.Constraint] = {}
        self._constrained_dims: dict[str, int] = {}
        self._unconstrained_dims: dict[str, int] = {}

        for spec in model._step_specs:
            if isinstance(spec, _LetSpec):
                continue

            for var in spec.vars:
                if var in observed_names:
                    continue

                self._latent_names.append(var)

                assert model._modules[spec.morphism_name] is not None
                morph = cast(ContinuousMorphism, model._modules[spec.morphism_name])
                constrained_dim = self._infer_dim(morph, len(spec.vars))
                support = _support_for_morphism(morph)
                unconstrained_dim = _unconstrained_dim(support, constrained_dim)

                self._supports[var] = support
                self._constrained_dims[var] = constrained_dim
                self._unconstrained_dims[var] = unconstrained_dim

                self.register_parameter(
                    f"loc_{var}",
                    nn.Parameter(torch.zeros(unconstrained_dim)),
                )
                self.register_parameter(
                    f"log_scale_{var}",
                    nn.Parameter(
                        torch.full(
                            (unconstrained_dim,),
                            torch.tensor(init_scale).log().item(),
                        )
                    ),
                )

    @staticmethod
    def _infer_dim(morph: ContinuousMorphism, n_vars: int) -> int:
        """Infer the per-variable constrained-side dimension of a morphism."""
        cod = morph.codomain
        if isinstance(cod, ContinuousSpace):
            total_dim: int = cod.dim
            return max(1, total_dim // n_vars)
        return 1

    def _bijector(self, name: str) -> Transform:
        return biject_to(self._supports[name])

    def rsample(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Sample from the per-site Normal-then-bijector guide."""
        batch = x.shape[0]
        result: dict[str, torch.Tensor] = {}

        for name in self._latent_names:
            loc = getattr(self, f"loc_{name}")
            log_scale = getattr(self, f"log_scale_{name}")
            scale = log_scale.exp().clamp(min=1e-6)
            uncon_dim = self._unconstrained_dims[name]
            con_dim = self._constrained_dims[name]

            loc_batch = loc.unsqueeze(0).expand(batch, uncon_dim)
            scale_batch = scale.unsqueeze(0).expand(batch, uncon_dim)
            z = D.Normal(loc_batch, scale_batch).rsample()
            v = _apply_bijector(self._bijector(name), z, con_dim)

            # Match the tracer's shape convention: 1-dim sites are
            # represented as (batch,), not (batch, 1).
            if v.dim() == 2 and v.shape[-1] == 1:
                v = v.squeeze(-1)

            result[name] = v

        return result

    def log_prob(
        self,
        x: torch.Tensor,
        sites: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Pushforward log-density under the per-site Normal+bijector.

        For each latent site, the constrained value ``v`` has density

        .. math::

            \\log q(v) \\;=\\; \\log\\mathcal{N}(z;\\, \\mathrm{loc},\\, \\mathrm{scale})
                          \\;+\\; \\log\\bigl|\\det J_{T^{-1}}(v)\\bigr|,

        where ``T = bijector`` and ``z = T^{-1}(v)``. This is the same
        change-of-variables identity Pyro's ``AutoNormal`` evaluates.
        """
        batch = x.shape[0]
        total = torch.zeros(batch, device=x.device)

        for name in self._latent_names:
            if name not in sites:
                continue

            v = sites[name]
            if v.dim() == 1 and self._constrained_dims[name] == 1:
                v = v.unsqueeze(-1)

            bij = self._bijector(name)
            z = bij.inv(v)

            loc = getattr(self, f"loc_{name}")
            log_scale = getattr(self, f"log_scale_{name}")
            scale = log_scale.exp().clamp(min=1e-6)
            uncon_dim = self._unconstrained_dims[name]
            loc_batch = loc.unsqueeze(0).expand(batch, uncon_dim)
            scale_batch = scale.unsqueeze(0).expand(batch, uncon_dim)

            log_q_z = D.Normal(loc_batch, scale_batch).log_prob(z)
            if log_q_z.dim() > 1:
                log_q_z = log_q_z.sum(dim=-1)

            # Jacobian of the inverse transform (constrained -> unconstrained).
            log_abs_det = bij.inv.log_abs_det_jacobian(v, z)
            while log_abs_det.dim() > 1:
                log_abs_det = log_abs_det.sum(dim=-1)

            total = total + log_q_z + log_abs_det

        return total

    @property
    def latent_names(self) -> list[str]:
        """Names of latent variables this guide covers."""
        return list(self._latent_names)


class AutoDeltaGuide(Guide):
    """Point-estimate (MAP) guide with per-site support-constrained bijector.

    For each continuous latent site the guide maintains a single
    learnable point in the *unconstrained* space; the constrained
    point estimate is obtained by applying the bijector
    :func:`biject_to(prior.support)`. This ensures every site's
    estimate lies inside the prior's support (positive for
    ``HalfNormal``, on the simplex for ``Dirichlet``, etc.), so
    ``prior.log_prob(estimate)`` evaluates without raising.

    The log-density returned by :meth:`log_prob` is zero (the delta
    contribution cancels in the ELBO).

    Parameters
    ----------
    model : MonadicProgram
        The generative model.
    observed_names : set[str]
        Names of observed variables.
    init_value : float
        Initial value for each unconstrained coordinate (small Gaussian
        noise around this is added at construction). Default 0.0,
        which under the standard bijectors maps to a sensible interior
        point of every support (the median of a HalfNormal, the centre
        of the unit interval, the uniform Dirichlet, etc.).
    """

    def __init__(
        self,
        model: MonadicProgram,
        observed_names: set[str],
        init_value: float = 0.0,
    ) -> None:
        super().__init__()
        self._latent_names: list[str] = []
        self._supports: dict[str, _constraints.Constraint] = {}
        self._constrained_dims: dict[str, int] = {}
        self._unconstrained_dims: dict[str, int] = {}

        for spec in model._step_specs:
            if isinstance(spec, _LetSpec):
                continue

            for var in spec.vars:
                if var in observed_names:
                    continue

                self._latent_names.append(var)

                assert model._modules[spec.morphism_name] is not None
                morph = cast(ContinuousMorphism, model._modules[spec.morphism_name])
                constrained_dim = AutoNormalGuide._infer_dim(morph, len(spec.vars))
                support = _support_for_morphism(morph)
                unconstrained_dim = _unconstrained_dim(support, constrained_dim)

                self._supports[var] = support
                self._constrained_dims[var] = constrained_dim
                self._unconstrained_dims[var] = unconstrained_dim

                self.register_parameter(
                    f"unconstrained_{var}",
                    nn.Parameter(
                        torch.full((unconstrained_dim,), init_value)
                        + torch.randn(unconstrained_dim) * 0.01
                    ),
                )

    def _bijector(self, name: str) -> Transform:
        return biject_to(self._supports[name])

    def rsample(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return the learned point estimates in the prior's support."""
        batch = x.shape[0]
        result: dict[str, torch.Tensor] = {}

        for name in self._latent_names:
            z = getattr(self, f"unconstrained_{name}")
            uncon_dim = self._unconstrained_dims[name]
            con_dim = self._constrained_dims[name]
            z_batch = z.unsqueeze(0).expand(batch, uncon_dim)
            v = _apply_bijector(self._bijector(name), z_batch, con_dim)
            if v.dim() == 2 and v.shape[-1] == 1:
                v = v.squeeze(-1)
            result[name] = v

        return result

    def log_prob(
        self,
        x: torch.Tensor,
        sites: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Log-density under the delta guide (always zero).

        The delta-mass contribution and its Jacobian cancel in the
        ELBO under the standard score-function trick.
        """
        return torch.zeros(x.shape[0], device=x.device)

    @property
    def latent_names(self) -> list[str]:
        """Names of latent variables this guide covers."""
        return list(self._latent_names)
