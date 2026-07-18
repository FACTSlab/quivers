"""Program: compile a morphism expression into a trainable nn.Module."""

from __future__ import annotations


from collections.abc import Callable
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from quivers.core.morphisms import Morphism
from quivers.continuous.morphisms import ContinuousMorphism


class Program(nn.Module):
    """Wraps a morphism expression as a trainable nn.Module.

    Traverses the morphism DAG, collects all learnable parameters
    and observed buffers, and provides a forward() that materializes
    the composed tensor.

    Supports both discrete ``Morphism`` instances (which produce a
    membership tensor via ``forward()``) and ``ContinuousMorphism``
    instances (which expose ``rsample`` / ``log_prob``).

    Parameters
    ----------
    morphism : Morphism or ContinuousMorphism
        The root morphism expression (possibly a composition tree).

    Examples
    --------
    >>> from quivers import FinSet, morphism, Program
    >>> X = FinSet(name="X", cardinality=3)
    >>> Y = FinSet(name="Y", cardinality=4)
    >>> f = morphism(X, Y)
    >>> prog = Program(f)
    >>> out = prog()  # shape (3, 4)
    """

    def __init__(
        self,
        morphism: Morphism | ContinuousMorphism | nn.Module | None = None,
    ) -> None:
        super().__init__()
        # Parametric-program-template invokers, attached by the DSL
        # compiler when the module's export names a template rather
        # than a concrete morphism. Keyed by template name; each
        # invoker instantiates the template at concrete arguments
        # and returns the resulting Program.
        self.templates: dict[str, Callable[..., Program]] = {}
        self._morphism = morphism
        self._is_continuous = isinstance(morphism, ContinuousMorphism)
        self._is_callable_module = (
            isinstance(morphism, nn.Module)
            and not isinstance(morphism, ContinuousMorphism)
            and not isinstance(morphism, Morphism)
        )

        # registering the module tree makes all parameters visible
        # to optimizer via self.parameters()
        if morphism is None:
            # A morphism-less Program is a container for structural
            # artifacts (signatures, encoders, decoders, losses)
            # in a module that declares no top-level exported morphism.
            self._root = None
        elif self._is_continuous or self._is_callable_module:
            # continuous morphisms and parser modules are already nn.Modules
            self._root = morphism
        else:
            self._root = cast(Morphism, morphism).module()

    @property
    def morphism(self) -> Morphism | ContinuousMorphism | nn.Module | None:
        """The underlying morphism expression, or ``None`` for a
        morphism-less module (one declaring only signatures /
        encoders / decoders / losses, or exporting a parametric
        program template that has not been instantiated)."""
        return self._morphism

    def _missing_morphism_error(self, attribute: str) -> TypeError:
        """Build the error for ``domain`` / ``codomain`` access on a
        Program that wraps no morphism.

        A ``TypeError`` (rather than ``AttributeError``) is essential
        here: a property that raises ``AttributeError`` is re-routed
        through ``nn.Module.__getattr__``, which replaces the message
        with a generic missing-attribute one.
        """
        if self.templates:
            names = sorted(self.templates)
            listed = ", ".join(repr(n) for n in names)
            noun = "templates" if len(names) > 1 else "template"
            return TypeError(
                f"this Program has no {attribute}: it exports the "
                f"parametric program {noun} {listed}, which denotes a "
                f"parameter-indexed family of kernels rather than a "
                f"single morphism. Instantiate the template at concrete "
                f"arguments (e.g. ``program.{names[0]}(...)``) and read "
                f".{attribute} on the returned Program."
            )
        return TypeError(
            f"this Program has no exported morphism, so its {attribute} "
            f"is undefined; the module declares structural artifacts "
            f"(signatures, encoders, decoders, losses) only"
        )

    @property
    def domain(self):
        """Domain of the underlying morphism.

        Raises
        ------
        TypeError
            If the Program wraps no morphism (an uninstantiated
            parametric template or a structural-artifact container).
        """
        if self._morphism is None:
            raise self._missing_morphism_error("domain")
        return self._morphism.domain

    @property
    def codomain(self):
        """Codomain of the underlying morphism.

        Raises
        ------
        TypeError
            If the Program wraps no morphism (an uninstantiated
            parametric template or a structural-artifact container).
        """
        if self._morphism is None:
            raise self._missing_morphism_error("codomain")
        return self._morphism.codomain

    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        """Reparameterized sample (continuous programs only).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        sample_shape : torch.Size
            Extra leading sample dimensions.

        Returns
        -------
        torch.Tensor
            Sampled output.

        Raises
        ------
        TypeError
            If the underlying morphism is not continuous.
        """
        if not isinstance(self._morphism, ContinuousMorphism):
            raise TypeError("rsample is only available for continuous programs")

        return cast(torch.Tensor, self._morphism.rsample(x, sample_shape))

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Log-probability (continuous programs only).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        y : torch.Tensor
            Output tensor.

        Returns
        -------
        torch.Tensor
            Log-probability of y given x.

        Raises
        ------
        TypeError
            If the underlying morphism is not continuous.
        """
        if not isinstance(self._morphism, ContinuousMorphism):
            raise TypeError("log_prob is only available for continuous programs")

        return cast(torch.Tensor, self._morphism.log_prob(x, y))

    def forward(self, n_steps: int | None = None) -> torch.Tensor:
        """Materialize the composed tensor (discrete programs only).

        Parameters
        ----------
        n_steps : int or None
            If provided, sets the step count on all
            ``RepeatMorphism`` instances in the morphism tree
            before materializing. This allows runtime-variable
            sequence lengths for models using ``repeat(f)``.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(*domain.shape, *codomain.shape)`` with
            values in [0, 1].

        Raises
        ------
        TypeError
            If the underlying morphism is continuous.
        """
        if self._morphism is None:
            raise TypeError(
                "forward() is not supported on a Program with no exported "
                "morphism; this module declares structural artifacts only"
            )
        if self._is_continuous:
            raise TypeError(
                "forward() is not supported for continuous programs; "
                "use rsample() or log_prob() instead"
            )

        if n_steps is not None:
            self._set_repeat_steps(n_steps)

        return cast(Morphism, self._morphism).tensor

    def _set_repeat_steps(self, n: int) -> None:
        """Set n_steps on all RepeatMorphism nodes in the tree.

        Parameters
        ----------
        n : int
            Number of repetition steps.
        """
        from quivers.core.morphisms import (
            RepeatMorphism,
            ComposedMorphism,
            ProductMorphism,
            MarginalizedMorphism,
            FunctorMorphism,
        )

        def _walk(m: Morphism) -> None:
            if isinstance(m, RepeatMorphism):
                m.n_steps = n

            # traverse composition tree
            if isinstance(m, (ComposedMorphism, ProductMorphism)):
                _walk(m._left)
                _walk(m._right)

            if isinstance(m, (MarginalizedMorphism, FunctorMorphism, RepeatMorphism)):
                _walk(m._inner)

        if isinstance(self._morphism, Morphism):
            _walk(self._morphism)

    def log_membership(self) -> torch.Tensor:
        """Log of the membership tensor.

        Returns
        -------
        torch.Tensor
            Log-membership values.  Entries near 0 map to large
            negative values.
        """
        t = cast(Morphism, self._morphism).tensor
        return torch.log(t.clamp(min=1e-7))

    def nll_loss(
        self,
        domain_indices: torch.Tensor,
        codomain_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Negative log-likelihood loss for observed (domain, codomain) pairs.

        For each pair (x, y), computes -log(tensor[x, y]).
        Suitable when the morphism represents fuzzy membership and
        observed pairs should have high membership.

        Parameters
        ----------
        domain_indices : torch.Tensor
            Integer tensor of shape (batch,) or (batch, n_domain_dims)
            indexing into the domain.
        codomain_indices : torch.Tensor
            Integer tensor of shape (batch,) or (batch, n_codomain_dims)
            indexing into the codomain.

        Returns
        -------
        torch.Tensor
            Scalar mean negative log-likelihood.
        """
        t = cast(Morphism, self._morphism).tensor

        # handle both flat and multi-dimensional indexing
        if domain_indices.ndim == 1:
            domain_indices = domain_indices.unsqueeze(-1)

        if codomain_indices.ndim == 1:
            codomain_indices = codomain_indices.unsqueeze(-1)

        # combine indices
        indices = torch.cat([domain_indices, codomain_indices], dim=-1)

        # index into the tensor
        idx_tuple = tuple(indices[:, i] for i in range(indices.shape[1]))
        values = t[idx_tuple]

        return -torch.log(values.clamp(min=1e-7)).mean()

    def bce_loss(self, target: torch.Tensor) -> torch.Tensor:
        """Binary cross-entropy between the morphism tensor and a target.

        Parameters
        ----------
        target : torch.Tensor
            Target tensor of the same shape as the morphism tensor,
            with values in [0, 1].

        Returns
        -------
        torch.Tensor
            Scalar BCE loss.
        """
        t = cast(Morphism, self._morphism).tensor
        return F.binary_cross_entropy(t, target)
