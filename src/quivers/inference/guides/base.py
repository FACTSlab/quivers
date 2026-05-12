"""Variational guide abstract base.

A :class:`Guide` is a parameterised distribution
:math:`q_\\phi(z \\mid x)` over a model's latent variables. The
ABC fixes a uniform contract:

* :meth:`Guide.rsample` returns a dict ``{site_name: tensor}``
  whose per-site shapes match the model's trace-side convention
  (plate latents at ``(|A|, *B.shape)``, scalar latents at
  ``(batch, *B.shape)`` or ``(batch,)`` for scalar event shape).
* :meth:`Guide.log_prob` evaluates the variational density of a
  per-site dict and returns a ``(batch,)``-shaped tensor.

Every concrete subclass under :mod:`quivers.inference.guides` is
constructed against a :class:`~quivers.inference.registry.LatentRegistry`
— no per-guide model walk. New guides supply the *variational
family's structure* (Normal vs MVN vs flow), not its
introspection.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from quivers.continuous.programs import MonadicProgram
from quivers.inference.registry import LatentRegistry


class Guide(nn.Module, ABC):
    """Abstract variational guide.

    Subclasses MUST implement :meth:`rsample` and :meth:`log_prob`
    and expose :attr:`latent_names`. They MAY override
    :attr:`registry` if they construct their registry lazily, but
    the default implementation expects ``self._registry`` to be
    set in ``__init__``.
    """

    _registry: LatentRegistry

    @classmethod
    def build_registry(
        cls,
        model: MonadicProgram,
        observed_names: set[str] | frozenset[str],
    ) -> LatentRegistry:
        """Convenience wrapper around
        :meth:`LatentRegistry.from_model` so guide constructors
        can do ``self._registry = self.build_registry(model, obs)``
        without an extra import."""
        return LatentRegistry.from_model(model, observed_names)

    @property
    def registry(self) -> LatentRegistry:
        """The :class:`LatentRegistry` this guide was built
        against."""
        return self._registry

    @abstractmethod
    def rsample(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Reparameterised sample from :math:`q_\\phi(z \\mid x)`.

        Parameters
        ----------
        x : torch.Tensor
            Program input. Shape ``(batch, ...)``. Used only for
            its batch dim and device; the variational parameters
            are stored on the guide itself.

        Returns
        -------
        dict[str, torch.Tensor]
            Per-site constrained samples shaped to match the
            model's trace-side convention.
        """
        ...

    @abstractmethod
    def log_prob(
        self,
        x: torch.Tensor,
        sites: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Log density of ``sites`` under :math:`q_\\phi(z \\mid x)`,
        with the change-of-variables Jacobian correction baked in.

        Returns
        -------
        torch.Tensor
            Shape ``(batch,)``.
        """
        ...

    @property
    @abstractmethod
    def latent_names(self) -> list[str]:
        """Names of latent variables this guide covers."""
        ...


__all__ = ["Guide"]
