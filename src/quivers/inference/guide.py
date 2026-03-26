"""Variational guide families for approximate posterior inference.

A guide is a parameterized distribution q(z | x) over latent variables
that approximates the true posterior p(z | x, y_obs). Guides are used
by SVI to optimize the ELBO.

This module provides:

- ``Guide`` — abstract base class for all guides
- ``AutoNormalGuide`` — mean-field Normal over all continuous latents
- ``AutoDeltaGuide`` — point-estimate (MAP) guide
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.distributions as D

from quivers.continuous.programs import MonadicProgram, _LetSpec
from quivers.continuous.spaces import ContinuousSpace


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
    """Mean-field Normal guide with learnable loc and scale per latent.

    Inspects the model's step specs to discover latent (non-observed)
    sites and creates a pair of parameters (loc, log_scale) for each.

    Parameters
    ----------
    model : MonadicProgram
        The generative model to build a guide for.
    observed_names : set[str]
        Names of observed variables (excluded from the guide).
    init_scale : float
        Initial scale for all latent sites.
    """

    def __init__(
        self,
        model: MonadicProgram,
        observed_names: set[str],
        init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self._latent_names = []

        for spec in model._step_specs:
            if isinstance(spec, _LetSpec):
                continue

            for var in spec.vars:
                if var in observed_names:
                    continue

                self._latent_names.append(var)

                # determine dimension from morphism codomain
                morph = model._modules[spec.morphism_name]
                dim = self._infer_dim(morph, len(spec.vars))

                # register learnable parameters
                self.register_parameter(
                    f"loc_{var}",
                    nn.Parameter(torch.zeros(dim)),
                )
                self.register_parameter(
                    f"log_scale_{var}",
                    nn.Parameter(torch.full((dim,), torch.tensor(init_scale).log())),
                )

    @staticmethod
    def _infer_dim(morph: nn.Module, n_vars: int) -> int:
        """Infer the per-variable output dimension of a morphism.

        Parameters
        ----------
        morph : nn.Module
            The morphism module.
        n_vars : int
            Number of variables bound in this step.

        Returns
        -------
        int
            Per-variable dimension.
        """
        cod = morph.codomain

        if isinstance(cod, ContinuousSpace):
            total_dim = cod.dim
            return max(1, total_dim // n_vars)

        # discrete codomain: 1-dimensional
        return 1

    def rsample(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Sample from mean-field Normal for each latent.

        Parameters
        ----------
        x : torch.Tensor
            Program input (used for batch size).

        Returns
        -------
        dict[str, torch.Tensor]
            Sampled latent values.
        """
        batch = x.shape[0]
        result = {}

        for name in self._latent_names:
            loc = getattr(self, f"loc_{name}")
            log_scale = getattr(self, f"log_scale_{name}")
            scale = log_scale.exp().clamp(min=1e-6)

            # expand to batch
            loc_batch = loc.unsqueeze(0).expand(batch, -1)
            scale_batch = scale.unsqueeze(0).expand(batch, -1)

            dist = D.Normal(loc_batch, scale_batch)
            sample = dist.rsample()

            # squeeze if 1-d
            if sample.shape[-1] == 1:
                sample = sample.squeeze(-1)

            result[name] = sample

        return result

    def log_prob(
        self,
        x: torch.Tensor,
        sites: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Log-density under the mean-field Normal guide.

        Parameters
        ----------
        x : torch.Tensor
            Program input (used for batch size).
        sites : dict[str, torch.Tensor]
            Latent variable values.

        Returns
        -------
        torch.Tensor
            Total log-density. Shape (batch,).
        """
        batch = x.shape[0]
        total = torch.zeros(batch, device=x.device)

        for name in self._latent_names:
            if name not in sites:
                continue

            loc = getattr(self, f"loc_{name}")
            log_scale = getattr(self, f"log_scale_{name}")
            scale = log_scale.exp().clamp(min=1e-6)

            val = sites[name]

            if val.dim() == 1:
                val = val.unsqueeze(-1)

            loc_batch = loc.unsqueeze(0).expand(batch, -1)
            scale_batch = scale.unsqueeze(0).expand(batch, -1)

            dist = D.Normal(loc_batch, scale_batch)
            # sum over event dims
            total = total + dist.log_prob(val).sum(dim=-1)

        return total

    @property
    def latent_names(self) -> list[str]:
        """Names of latent variables this guide covers."""
        return list(self._latent_names)


class AutoDeltaGuide(Guide):
    """Point-estimate (MAP) guide with a learnable value per latent.

    The guide distribution is a delta at the learned point, so
    log_prob returns 0 for all sites (the delta contribution cancels
    in the ELBO).

    Parameters
    ----------
    model : MonadicProgram
        The generative model.
    observed_names : set[str]
        Names of observed variables.
    """

    def __init__(
        self,
        model: MonadicProgram,
        observed_names: set[str],
    ) -> None:
        super().__init__()
        self._latent_names = []

        for spec in model._step_specs:
            if isinstance(spec, _LetSpec):
                continue

            for var in spec.vars:
                if var in observed_names:
                    continue

                self._latent_names.append(var)

                morph = model._modules[spec.morphism_name]
                dim = AutoNormalGuide._infer_dim(morph, len(spec.vars))

                self.register_parameter(
                    f"value_{var}",
                    nn.Parameter(torch.randn(dim) * 0.1),
                )

    def rsample(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return the learned point estimates.

        Parameters
        ----------
        x : torch.Tensor
            Program input (used for batch size).

        Returns
        -------
        dict[str, torch.Tensor]
            Point-estimate values for each latent.
        """
        batch = x.shape[0]
        result = {}

        for name in self._latent_names:
            val = getattr(self, f"value_{name}")
            expanded = val.unsqueeze(0).expand(batch, -1)

            if expanded.shape[-1] == 1:
                expanded = expanded.squeeze(-1)

            result[name] = expanded

        return result

    def log_prob(
        self,
        x: torch.Tensor,
        sites: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Log-density under the delta guide (always zero).

        Parameters
        ----------
        x : torch.Tensor
            Program input.
        sites : dict[str, torch.Tensor]
            Latent variable values (ignored).

        Returns
        -------
        torch.Tensor
            Zeros. Shape (batch,).
        """
        return torch.zeros(x.shape[0], device=x.device)

    @property
    def latent_names(self) -> list[str]:
        """Names of latent variables this guide covers."""
        return list(self._latent_names)
