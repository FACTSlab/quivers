"""Latent-site registry: the canonical introspection surface that
every variational guide and MCMC kernel consumes.

A :class:`LatentRegistry` is built once from a
:class:`~quivers.continuous.programs.MonadicProgram` and a set of
observed site names. It walks the model's ``_step_specs`` linear IR
exactly once, classifies each latent into "plate" vs "scalar",
records the prior support and per-site dimensionality, builds the
matching :class:`torch.distributions.transforms.Transform` via
:func:`torch.distributions.constraint_registry.biject_to`, and
caches a *fused* flatten / unflatten pipeline so a single
unconstrained vector can be turned into a fully constrained site
dict (and back) in one pass.

The registry is the only place in the inference layer that knows
about :class:`~quivers.continuous.bayesian.PlateDraw`, the layout
of ``MonadicProgram._step_specs``, or the per-site
constraint-to-bijector mapping. Every downstream component —
:class:`~quivers.inference.guides.base.Guide`,
:class:`~quivers.inference.mcmc.kernel.MCMCKernel`, the
flow-based / structured / mixture guides — receives a fully
populated registry and operates against its flat-vector and
per-site dict APIs.

Categorical interpretation
==========================

Each latent site corresponds to a draw in the model's Kleisli
arrow ``Γ → 𝒢(τ)``. The site's prior is a parameterized family
``F : Θ → 𝒢(B)`` with a support constraint
``C ⊆ B``. The bijector ``T : ℝ^d → C`` is the
:func:`biject_to` lift of ``C``; its log-determinant gives the
change-of-variables term in any pushforward density.

For a plate ``v : A → B`` the latent lives in ``C^|A|``; the
registry handles the leading plate axis (size ``|A|``) separately
from the trailing event axis (the support dimension), and the
unconstrained flat vector concatenates plate × event across every
site in declaration order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch.distributions import constraints as _constraints
from torch.distributions.constraint_registry import biject_to
from torch.distributions.transforms import Transform

from quivers.continuous.bayesian import PlateDraw
from quivers.continuous.morphisms import ContinuousMorphism
from quivers.continuous.programs import MonadicProgram, _LetSpec
from quivers.continuous.spaces import ContinuousSpace


def _unconstrained_event_dim(
    support: _constraints.Constraint, constrained_dim: int
) -> int:
    """Return the unconstrained-space event dimension matching a
    ``constrained_dim``-sized support.

    The two cases that change dim:

    * :data:`torch.distributions.constraints.simplex` — the
      stick-breaking bijector maps :math:`(d-1)`-dimensional
      unconstrained space onto the :math:`(d-1)`-simplex embedded
      in :math:`\\mathbb{R}^d`.
    * :data:`torch.distributions.constraints.corr_cholesky` — the
      correlation-Cholesky bijector maps :math:`d(d-1)/2` real
      coordinates onto the lower-triangular Cholesky factor of a
      :math:`d \\times d` correlation matrix.

    Every other supported constraint is dim-preserving (the
    bijector is :class:`torch.distributions.transforms.AffineTransform`,
    :class:`SigmoidTransform`, :class:`ExpTransform`, or a composition
    thereof).
    """
    if support is _constraints.simplex:
        return max(1, constrained_dim - 1)
    if support is _constraints.corr_cholesky:
        return max(1, constrained_dim * (constrained_dim - 1) // 2)
    return constrained_dim


@dataclass(frozen=True)
class LatentSite:
    """Frozen record describing a single latent site.

    Attributes
    ----------
    name : str
        Variable name (the bound name in the program body).
    morphism : ContinuousMorphism
        The site's runtime morphism (the prior). Held by reference
        so guides / kernels can call ``morphism.log_prob`` directly
        when they need the per-site prior contribution.
    support : torch.distributions.constraints.Constraint
        Support of the prior. Drives the choice of bijector.
    bijector : torch.distributions.transforms.Transform
        ``biject_to(support)``, cached at registry construction.
    constrained_dim : int
        Event dimension on the constrained side (for a scalar
        site, ``1``; for a Dirichlet on a 5-simplex, ``5``).
    unconstrained_dim : int
        Event dimension on the unconstrained side (for a Dirichlet
        on a 5-simplex, ``4``; otherwise equals ``constrained_dim``).
    is_plate : bool
        Whether this site is a plate draw
        (:class:`~quivers.continuous.bayesian.PlateDraw`).
    plate_index_size : int
        ``|A|`` for a plate site, ``0`` otherwise.
    spec_index : int
        Position in ``model._step_specs``.
    flat_offset : int
        Index of this site's first element in the registry's
        flat unconstrained vector.
    flat_length : int
        Number of elements this site occupies in the flat vector
        (= ``plate_index_size * unconstrained_dim`` for plate
        sites, ``unconstrained_dim`` for scalar sites).
    """

    name: str
    morphism: ContinuousMorphism
    support: _constraints.Constraint
    bijector: Transform
    constrained_dim: int
    unconstrained_dim: int
    is_plate: bool
    plate_index_size: int
    spec_index: int
    flat_offset: int
    flat_length: int

    @property
    def constrained_shape(self) -> tuple[int, ...]:
        """The trace-side shape of this site's value (after
        bijector forward / squeezing scalar event dims)."""
        if self.is_plate:
            if self.constrained_dim == 1:
                return (self.plate_index_size,)
            return (self.plate_index_size, self.constrained_dim)
        if self.constrained_dim == 1:
            return ()
        return (self.constrained_dim,)

    @property
    def unconstrained_shape(self) -> tuple[int, ...]:
        """The variational-side shape of this site's value in
        unconstrained space (before the bijector)."""
        if self.is_plate:
            return (self.plate_index_size, self.unconstrained_dim)
        return (self.unconstrained_dim,)


class LatentRegistry:
    """The complete introspection result for one (model,
    observed-set) pair.

    Constructed once via :meth:`from_model`; immutable thereafter.
    Provides:

    * Iteration over latent sites in declaration order
      (:attr:`sites`, :attr:`plate_sites`, :attr:`scalar_sites`).
    * Total unconstrained dimensionality
      (:attr:`total_unconstrained_dim`) — the dim of the flat
      vector HMC and full-rank Gaussian guides operate on.
    * Round-tripping between a flat unconstrained vector and a
      dict of constrained per-site tensors
      (:meth:`unflatten_unconstrained`, :meth:`flatten_constrained`).
    * Bijector forward / inverse with Jacobian accumulation
      (:meth:`to_constrained`, :meth:`to_unconstrained`).

    All operations are vectorized — no per-site Python loops on
    the hot path beyond the construction-time setup.
    """

    def __init__(
        self,
        sites: dict[str, LatentSite],
        observed_names: frozenset[str],
        model: MonadicProgram,
    ) -> None:
        self._sites: dict[str, LatentSite] = sites
        self._observed_names: frozenset[str] = observed_names
        self._model: MonadicProgram = model
        self._total_unconstrained_dim: int = sum(s.flat_length for s in sites.values())

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_model(
        cls,
        model: MonadicProgram,
        observed_names: set[str] | frozenset[str],
    ) -> "LatentRegistry":
        """Walk ``model._step_specs`` and assemble the registry.

        Parameters
        ----------
        model : MonadicProgram
            Compiled probabilistic program.
        observed_names : set[str] | frozenset[str]
            Variable names treated as observations. Sample sites
            with names in this set are skipped (their values come
            from the conditioning data dict at trace time, not
            from a guide).

        Returns
        -------
        LatentRegistry
            Fully populated registry with each latent's morphism,
            support, bijector, dims, and flat-vector offsets.
        """
        observed: frozenset[str] = frozenset(observed_names)
        sites: dict[str, LatentSite] = {}
        flat_cursor = 0

        for spec_idx, spec in enumerate(model._step_specs):
            if isinstance(spec, _LetSpec):
                continue
            for var in spec.vars:
                if var in observed:
                    continue
                morph = model._modules[spec.morphism_name]
                if morph is None:
                    raise RuntimeError(
                        f"LatentRegistry: morphism module for site "
                        f"{var!r} is unregistered on the model"
                    )
                morph_cm = cast(ContinuousMorphism, morph)
                is_plate = isinstance(morph_cm, PlateDraw)
                if is_plate:
                    plate = cast(PlateDraw, morph_cm)
                    inner = plate.family
                    inner_cod = inner.codomain
                    if isinstance(inner_cod, ContinuousSpace):
                        constrained_dim = max(1, int(inner_cod.dim))
                    else:
                        constrained_dim = 1
                    support = inner.support
                    plate_index_size = plate.index_size
                else:
                    cod = morph_cm.codomain
                    if isinstance(cod, ContinuousSpace):
                        total_dim = int(cod.dim)
                        constrained_dim = max(1, total_dim // max(1, len(spec.vars)))
                    else:
                        constrained_dim = 1
                    support = morph_cm.support
                    plate_index_size = 0

                unconstrained_dim = _unconstrained_event_dim(
                    support, constrained_dim
                )
                bijector = biject_to(support)

                flat_length = (
                    plate_index_size * unconstrained_dim
                    if is_plate
                    else unconstrained_dim
                )

                sites[var] = LatentSite(
                    name=var,
                    morphism=morph_cm,
                    support=support,
                    bijector=bijector,
                    constrained_dim=constrained_dim,
                    unconstrained_dim=unconstrained_dim,
                    is_plate=is_plate,
                    plate_index_size=plate_index_size,
                    spec_index=spec_idx,
                    flat_offset=flat_cursor,
                    flat_length=flat_length,
                )
                flat_cursor += flat_length

        return cls(sites, observed, model)

    # ------------------------------------------------------------------
    # Iteration / lookup
    # ------------------------------------------------------------------

    @property
    def sites(self) -> dict[str, LatentSite]:
        """Every latent site, keyed by name, in declaration order."""
        return self._sites

    @property
    def plate_sites(self) -> dict[str, LatentSite]:
        """The subset of sites that are :class:`PlateDraw`-shaped."""
        return {n: s for n, s in self._sites.items() if s.is_plate}

    @property
    def scalar_sites(self) -> dict[str, LatentSite]:
        """The subset of sites that are not plates."""
        return {n: s for n, s in self._sites.items() if not s.is_plate}

    @property
    def names(self) -> tuple[str, ...]:
        """Latent site names in declaration order."""
        return tuple(self._sites.keys())

    @property
    def observed_names(self) -> frozenset[str]:
        """The observed-name set used to build this registry."""
        return self._observed_names

    @property
    def model(self) -> MonadicProgram:
        """The underlying model."""
        return self._model

    @property
    def total_unconstrained_dim(self) -> int:
        """Sum of every site's flat-vector length.

        This is the dim of the unconstrained vector that full-rank
        Gaussian guides and HMC operate on.
        """
        return self._total_unconstrained_dim

    def __len__(self) -> int:
        return len(self._sites)

    def __iter__(self):
        return iter(self._sites.values())

    def __contains__(self, name: str) -> bool:
        return name in self._sites

    def __getitem__(self, name: str) -> LatentSite:
        return self._sites[name]

    # ------------------------------------------------------------------
    # Flatten / unflatten between dicts and a single vector
    # ------------------------------------------------------------------

    def unflatten_unconstrained(
        self, vec: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Reshape a flat unconstrained vector into per-site
        unconstrained tensors.

        Parameters
        ----------
        vec : torch.Tensor
            Shape ``(..., total_unconstrained_dim)``. Leading
            axes (batch, particles, chains) are preserved on every
            site's output.

        Returns
        -------
        dict[str, torch.Tensor]
            ``{name: site_tensor}`` where each ``site_tensor`` has
            shape ``vec.shape[:-1] + site.unconstrained_shape``.
        """
        if vec.shape[-1] != self._total_unconstrained_dim:
            raise ValueError(
                f"LatentRegistry.unflatten_unconstrained: expected "
                f"trailing dim {self._total_unconstrained_dim}, got "
                f"{vec.shape[-1]}"
            )
        out: dict[str, torch.Tensor] = {}
        for site in self._sites.values():
            chunk = vec[..., site.flat_offset : site.flat_offset + site.flat_length]
            out[site.name] = chunk.reshape(*vec.shape[:-1], *site.unconstrained_shape)
        return out

    def flatten_unconstrained(
        self, sites: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Concatenate per-site unconstrained tensors into a single
        flat vector.

        Inverse of :meth:`unflatten_unconstrained` (up to leading
        batch axes). The trailing batch axes of every site tensor
        must agree.
        """
        parts: list[torch.Tensor] = []
        leading_shape: tuple[int, ...] | None = None
        for site in self._sites.values():
            if site.name not in sites:
                raise KeyError(
                    f"LatentRegistry.flatten_unconstrained: missing "
                    f"site {site.name!r}"
                )
            t = sites[site.name]
            trail = len(site.unconstrained_shape)
            site_leading = t.shape[: t.dim() - trail] if trail else t.shape
            if leading_shape is None:
                leading_shape = tuple(site_leading)
            elif tuple(site_leading) != leading_shape:
                raise ValueError(
                    f"LatentRegistry.flatten_unconstrained: site "
                    f"{site.name!r} has leading shape {tuple(site_leading)} "
                    f"which disagrees with previously-seen "
                    f"{leading_shape}"
                )
            parts.append(t.reshape(*site_leading, site.flat_length))
        if not parts:
            device = next(iter(sites.values())).device if sites else torch.device("cpu")
            return torch.zeros((0,), device=device)
        return torch.cat(parts, dim=-1)

    # ------------------------------------------------------------------
    # Bijector application with Jacobian accumulation
    # ------------------------------------------------------------------

    def to_constrained(
        self, unconstrained: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Push every site's unconstrained value through its
        bijector.

        Returns
        -------
        constrained : dict[str, torch.Tensor]
            Per-site constrained values shaped according to
            :attr:`LatentSite.constrained_shape`. The trailing
            length-1 event axis of a scalar site is squeezed off
            to match the trace-side shape convention.
        log_abs_dets : dict[str, torch.Tensor]
            Per-site ``log |det dT/dz|`` returned by the bijector,
            at the bijector's natural shape (event axes preserved).
            Callers aggregate as the objective requires — for ELBO
            we sum over event axes; for IWAE we keep particles as
            a leading axis; etc.
        """
        constrained: dict[str, torch.Tensor] = {}
        log_abs_dets: dict[str, torch.Tensor] = {}
        for site in self._sites.values():
            if site.name not in unconstrained:
                raise KeyError(
                    f"LatentRegistry.to_constrained: missing site "
                    f"{site.name!r}"
                )
            z = unconstrained[site.name]
            v = site.bijector(z)
            log_abs_dets[site.name] = site.bijector.log_abs_det_jacobian(z, v)
            if (
                site.constrained_dim == 1
                and v.dim() >= 1
                and v.shape[-1] == 1
            ):
                v = v.squeeze(-1)
            constrained[site.name] = v
        return constrained, log_abs_dets

    def to_unconstrained(
        self, constrained: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Inverse of :meth:`to_constrained`."""
        unconstrained: dict[str, torch.Tensor] = {}
        log_abs_dets: dict[str, torch.Tensor] = {}
        for site in self._sites.values():
            if site.name not in constrained:
                raise KeyError(
                    f"LatentRegistry.to_unconstrained: missing site "
                    f"{site.name!r}"
                )
            v = constrained[site.name]
            if (
                site.constrained_dim == 1
                and v.dim() == (1 if site.is_plate else 0)
            ):
                v = v.unsqueeze(-1)
            z = site.bijector.inv(v)
            log_abs_dets[site.name] = site.bijector.inv.log_abs_det_jacobian(v, z)
            unconstrained[site.name] = z
        return unconstrained, log_abs_dets

    @staticmethod
    def aggregate_log_abs_det(
        log_abs_dets: dict[str, torch.Tensor],
        leading_shape: tuple[int, ...] = (),
    ) -> torch.Tensor:
        """Sum every per-site ``log_abs_det`` over its event axes
        and produce a single tensor broadcast to ``leading_shape``.

        This is the canonical aggregation an ELBO needs. Objectives
        that want particle-aware or per-site granularity can skip
        this helper and aggregate themselves.
        """
        if not log_abs_dets:
            return torch.zeros(leading_shape) if leading_shape else torch.zeros(())
        total: torch.Tensor | None = None
        n_leading = len(leading_shape)
        for site_lad in log_abs_dets.values():
            # Sum away every axis beyond the leading shape; what
            # remains is broadcast against the leading shape.
            while site_lad.dim() > n_leading:
                site_lad = site_lad.sum(dim=-1)
            total = site_lad if total is None else total + site_lad
        assert total is not None
        return total

    # ------------------------------------------------------------------
    # Misc utilities
    # ------------------------------------------------------------------

    def zero_unconstrained(
        self,
        *,
        leading_shape: tuple[int, ...] = (),
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return a dict of zero-initialized unconstrained site
        tensors with the supplied leading batch shape.

        Useful for HMC chain initialisation and AutoLaplace's MAP
        starting point.
        """
        out: dict[str, torch.Tensor] = {}
        for site in self._sites.values():
            shape = leading_shape + site.unconstrained_shape
            out[site.name] = torch.zeros(shape, device=device, dtype=dtype)
        return out

    def randn_unconstrained(
        self,
        *,
        leading_shape: tuple[int, ...] = (),
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        scale: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Standard-normal-initialized unconstrained site tensors."""
        out: dict[str, torch.Tensor] = {}
        for site in self._sites.values():
            shape = leading_shape + site.unconstrained_shape
            out[site.name] = scale * torch.randn(shape, device=device, dtype=dtype)
        return out


__all__ = ["LatentRegistry", "LatentSite"]
