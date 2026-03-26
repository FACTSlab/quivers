"""Continuous morphisms: Markov kernels on continuous and mixed spaces.

A ContinuousMorphism represents a conditional probability distribution
p(y | x) where x and y may live in either discrete (FinSet) or
continuous (ContinuousSpace) spaces. The morphism is defined by two
operations:

    log_prob(x, y) — log-density/probability of y given x
    rsample(x)     — reparameterized samples from p(· | x)

Composition uses ancestral sampling:

    (g . f)(x, z) = integral f(x, y) g(y, z) dy
                   ~ E_{y~f(x,.)}[g(y, z)]

This module provides:

    ContinuousMorphism         — abstract base with >> and @ operators
    SampledComposition         — f >> g via ancestral sampling
    ProductContinuousMorphism  — f @ g (independent product)
    DiscreteAsContinuous       — wrap a discrete Morphism as continuous

Convention for input shapes
---------------------------
- Discrete domain (SetObject): x is LongTensor of shape (batch,)
- Continuous domain (ContinuousSpace): x is FloatTensor of shape (batch, dim)
- Discrete codomain: y is LongTensor of shape (batch,)
- Continuous codomain: y is FloatTensor of shape (batch, dim)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union, cast

import torch
import torch.nn as nn

from quivers.core.objects import SetObject
from quivers.continuous.spaces import ContinuousSpace

# union type for all spaces (discrete or continuous)
AnySpace = Union[SetObject, ContinuousSpace]


def _is_discrete(space: AnySpace) -> bool:
    """Check whether a space is discrete (SetObject)."""
    return isinstance(space, SetObject)


class ContinuousMorphism(nn.Module, ABC):
    """Abstract base for morphisms involving continuous spaces.

    Subclasses must implement ``log_prob`` and ``rsample``. The
    composition operator ``>>`` and product operator ``@`` are
    provided and dispatch to SampledComposition and
    ProductContinuousMorphism respectively.

    Unlike discrete Morphism (which materializes a full tensor),
    ContinuousMorphism is defined operationally: it can evaluate
    log-densities and generate reparameterized samples.

    Parameters
    ----------
    domain : SetObject or ContinuousSpace
        Source space.
    codomain : SetObject or ContinuousSpace
        Target space.
    """

    def __init__(self, domain: AnySpace, codomain: AnySpace) -> None:
        super().__init__()
        self._domain = domain
        self._codomain = codomain

    @property
    def domain(self) -> AnySpace:
        """Source space."""
        return self._domain

    @property
    def codomain(self) -> AnySpace:
        """Target space."""
        return self._codomain

    @abstractmethod
    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Log-probability (density) of y given x.

        Parameters
        ----------
        x : torch.Tensor
            Inputs. Shape (batch,) for discrete domain or
            (batch, domain_dim) for continuous domain.
        y : torch.Tensor
            Outputs. Shape (batch,) for discrete codomain or
            (batch, codomain_dim) for continuous codomain.

        Returns
        -------
        torch.Tensor
            Log-probabilities/densities. Shape (batch,).
        """
        ...

    @abstractmethod
    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        """Reparameterized samples from p(. | x).

        Gradients flow through the returned samples back to the
        parameters of this morphism (and to x if the domain is
        continuous).

        Parameters
        ----------
        x : torch.Tensor
            Inputs. Shape (batch,) or (batch, domain_dim).
        sample_shape : torch.Size
            Additional leading sample dimensions.

        Returns
        -------
        torch.Tensor
            Samples. Shape (*sample_shape, batch, codomain_dim) for
            continuous codomain, or (*sample_shape, batch) for discrete.
        """
        ...

    def sample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        """Non-reparameterized samples (no gradient through samples).

        Parameters
        ----------
        x : torch.Tensor
            Inputs.
        sample_shape : torch.Size
            Additional leading sample dimensions.

        Returns
        -------
        torch.Tensor
            Samples (detached from computation graph).
        """
        with torch.no_grad():
            return self.rsample(x, sample_shape)

    # -- categorical operators ------------------------------------------------

    def __rshift__(self, other: object) -> ContinuousMorphism:
        """Composition via ancestral sampling: self >> other."""
        if isinstance(other, ContinuousMorphism):
            return SampledComposition(self, other)

        # discrete morphism on the right
        from quivers.core.morphisms import Morphism

        if isinstance(other, Morphism):
            return SampledComposition(self, DiscreteAsContinuous(other))

        return NotImplemented

    def __rrshift__(self, other: object) -> ContinuousMorphism:
        """Handle discrete_morphism >> continuous_morphism."""
        from quivers.core.morphisms import Morphism

        if isinstance(other, Morphism):
            return SampledComposition(DiscreteAsContinuous(other), self)

        return NotImplemented

    def __matmul__(self, other: object) -> ProductContinuousMorphism:
        """Independent product: self @ other."""
        if isinstance(other, ContinuousMorphism):
            return ProductContinuousMorphism(self, other)

        from quivers.core.morphisms import Morphism

        if isinstance(other, Morphism):
            return ProductContinuousMorphism(self, DiscreteAsContinuous(other))

        return NotImplemented

    def __repr__(self) -> str:
        cls = type(self).__name__
        return f"{cls}({self.domain!r} -> {self.codomain!r})"


# -- parameter sources -------------------------------------------------------


class _LookupSource(nn.Module):
    """Parameter source for discrete domains: index into a table.

    For each domain element i, returns a parameter vector table[i].
    """

    def __init__(self, n_entries: int, param_dim: int) -> None:
        super().__init__()
        self.table = nn.Parameter(torch.randn(n_entries, param_dim) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Look up parameters.

        Parameters
        ----------
        x : torch.Tensor
            Integer indices. Shape (batch,).

        Returns
        -------
        torch.Tensor
            Parameter vectors. Shape (batch, param_dim).
        """
        return self.table[x.long()]


class _NeuralSource(nn.Module):
    """Parameter source for continuous domains: a small MLP.

    Maps continuous inputs to parameter vectors via a two-layer
    network with tanh activations.
    """

    def __init__(
        self,
        input_dim: int,
        param_dim: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, param_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute parameters from continuous input.

        Parameters
        ----------
        x : torch.Tensor
            Continuous inputs. Shape (batch, input_dim).

        Returns
        -------
        torch.Tensor
            Parameter vectors. Shape (batch, param_dim).
        """
        return self.net(x)


def _make_source(
    domain: AnySpace,
    param_dim: int,
    hidden_dim: int = 64,
) -> nn.Module:
    """Create an appropriate parameter source for the given domain.

    Parameters
    ----------
    domain : SetObject or ContinuousSpace
        The domain space.
    param_dim : int
        Output dimensionality of the parameter source.
    hidden_dim : int
        Hidden layer width for neural source (continuous domains).

    Returns
    -------
    nn.Module
        A _LookupSource or _NeuralSource.
    """
    if isinstance(domain, SetObject):
        return _LookupSource(domain.size, param_dim)

    else:
        return _NeuralSource(cast(ContinuousSpace, domain).dim, param_dim, hidden_dim)


# -- composition via sampling ------------------------------------------------


class SampledComposition(ContinuousMorphism):
    """Composition of morphisms via ancestral sampling.

    Given f: X -> Y and g: Y -> Z, the composition g . f satisfies:

        (g . f)(x, z) = integral f(x, y) g(y, z) dy

    This integral is computed:
    - Exactly (finite sum) when Y is discrete.
    - Approximately (Monte Carlo) when Y is continuous.

    For rsample: draw y ~ f(x, .), then draw z ~ g(y, .).
    For log_prob: sum/average g(z | y_i) weighted by f(y_i | x).

    Parameters
    ----------
    left : ContinuousMorphism
        First morphism (applied first).
    right : ContinuousMorphism
        Second morphism (applied second).
    n_intermediate : int
        Number of Monte Carlo samples for continuous intermediate
        spaces. Ignored when the intermediate space is discrete.
    """

    def __init__(
        self,
        left: ContinuousMorphism,
        right: ContinuousMorphism,
        n_intermediate: int = 100,
    ) -> None:
        super().__init__(left.domain, right.codomain)
        self.left = left
        self.right = right
        self.n_intermediate = n_intermediate

    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        """Ancestral sampling: y ~ f(x, .), then z ~ g(y, .).

        Parameters
        ----------
        x : torch.Tensor
            Inputs to the composition.
        sample_shape : torch.Size
            Additional sample dimensions.

        Returns
        -------
        torch.Tensor
            Samples from the composed morphism.
        """
        # sample intermediate values from left morphism
        y = self.left.rsample(x, sample_shape)

        # flatten sample_shape + batch into a single batch dim for right
        if len(sample_shape) > 0:
            leading = y.shape[: len(sample_shape)]
            batch = x.shape[0]
            flat_size = int(torch.tensor(leading).prod().item()) * batch

            if y.dim() > len(sample_shape) + 1:
                # continuous intermediate: (..., batch, dim)
                event_dims = y.shape[len(sample_shape) + 1 :]
                flat_y = y.reshape(flat_size, *event_dims)

            else:
                # discrete intermediate: (..., batch)
                flat_y = y.reshape(flat_size)

        else:
            flat_y = y

        # sample from right morphism
        z = self.right.rsample(flat_y)

        # reshape back to (*sample_shape, batch, ...)
        if len(sample_shape) > 0:
            batch = x.shape[0]

            if z.dim() > 1:
                event_dims = z.shape[1:]
                z = z.reshape(*sample_shape, batch, *event_dims)

            else:
                z = z.reshape(*sample_shape, batch)

        return z

    def log_prob(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Log-probability of y given x through the composition.

        When the intermediate space is discrete, computes the exact
        marginalization. When continuous, uses Monte Carlo estimation.

        Parameters
        ----------
        x : torch.Tensor
            Inputs. Shape (batch,) or (batch, dom_dim).
        y : torch.Tensor
            Outputs. Shape (batch,) or (batch, cod_dim).

        Returns
        -------
        torch.Tensor
            Log-probabilities. Shape (batch,).
        """
        intermediate = self.left.codomain

        if isinstance(intermediate, SetObject):
            return self._log_prob_exact(x, y, intermediate)

        else:
            return self._log_prob_mc(x, y)

    def _log_prob_exact(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        intermediate: SetObject,
    ) -> torch.Tensor:
        """Exact log-prob via finite summation over discrete intermediate."""
        batch = x.shape[0]
        n_y = intermediate.size

        # all possible intermediate values
        all_y = torch.arange(n_y, device=x.device)

        # log f(y | x) for all y: (batch, n_y)
        x.unsqueeze(1).expand(
            batch if x.dim() == 1 else x.shape[0],
            n_y,
            *(() if x.dim() == 1 else x.shape[1:]),
        )

        if x.dim() == 1:
            x_flat = x.unsqueeze(1).expand(batch, n_y).reshape(-1)

        else:
            x_flat = (
                x.unsqueeze(1)
                .expand(
                    batch,
                    n_y,
                    x.shape[-1],
                )
                .reshape(-1, x.shape[-1])
            )

        y_flat = all_y.unsqueeze(0).expand(batch, n_y).reshape(-1)

        log_f = self.left.log_prob(x_flat, y_flat).reshape(batch, n_y)

        # log g(z | y) for all y: (batch, n_y)
        if z.dim() == 1:
            z_flat = z.unsqueeze(1).expand(batch, n_y).reshape(-1)

        else:
            z_flat = (
                z.unsqueeze(1)
                .expand(
                    batch,
                    n_y,
                    z.shape[-1],
                )
                .reshape(-1, z.shape[-1])
            )

        log_g = self.right.log_prob(y_flat, z_flat).reshape(batch, n_y)

        # log p(z | x) = log sum_y exp(log f(y|x) + log g(z|y))
        return torch.logsumexp(log_f + log_g, dim=1)

    def _log_prob_mc(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Monte Carlo estimate of log-prob via importance sampling."""
        n = self.n_intermediate
        batch = x.shape[0]

        # draw intermediate samples: (n, batch, ...)
        y = self.left.rsample(x, torch.Size([n]))

        # evaluate g(z | y_i) for each sample
        if y.dim() == 2:
            # discrete or 1d intermediate: (n, batch)
            y_flat = y.reshape(n * batch)

        else:
            y_flat = y.reshape(n * batch, -1)

        if z.dim() == 1:
            z_flat = z.unsqueeze(0).expand(n, batch).reshape(n * batch)

        else:
            z_flat = (
                z.unsqueeze(0)
                .expand(
                    n,
                    *z.shape,
                )
                .reshape(n * batch, -1)
            )

        log_g = self.right.log_prob(y_flat, z_flat).reshape(n, batch)

        # log-mean-exp for numerical stability
        return (
            torch.logsumexp(log_g, dim=0)
            - torch.tensor(float(n), device=x.device).log()
        )


# -- product morphism --------------------------------------------------------


class ProductContinuousMorphism(ContinuousMorphism):
    """Independent product of two continuous morphisms.

    Given f: A -> B and g: C -> D, produces f @ g: (A, C) -> (B, D)
    where p_{f@g}((y,z) | (x,w)) = f(y | x) * g(z | w).

    Domain inputs are concatenated: (x, w) as a single vector.
    Codomain outputs are concatenated: (y, z) as a single vector.
    For discrete components, indices are embedded as 1-d floats.

    Parameters
    ----------
    left : ContinuousMorphism
        Left factor morphism.
    right : ContinuousMorphism
        Right factor morphism.
    """

    def __init__(
        self,
        left: ContinuousMorphism,
        right: ContinuousMorphism,
    ) -> None:

        dom = _combine_spaces(left.domain, right.domain)
        cod = _combine_spaces(left.codomain, right.codomain)
        super().__init__(dom, cod)

        self.left = left
        self.right = right
        self._left_dom_dim = _event_dim(left.domain)
        self._right_dom_dim = _event_dim(right.domain)
        self._left_cod_dim = _event_dim(left.codomain)
        self._right_cod_dim = _event_dim(right.codomain)

    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        x_left, x_right = self._split_input(x)

        y_left = self.left.rsample(x_left, sample_shape)
        y_right = self.right.rsample(x_right, sample_shape)

        # ensure both are at least 2d for concatenation
        if y_left.dim() < y_right.dim():
            y_left = y_left.unsqueeze(-1)

        elif y_right.dim() < y_left.dim():
            y_right = y_right.unsqueeze(-1)

        return torch.cat([y_left, y_right], dim=-1)

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_left, x_right = self._split_input(x)
        y_left = y[..., : self._left_cod_dim]
        y_right = y[..., self._left_cod_dim :]

        # reconstruct discrete indices if needed
        if _is_discrete(self.left.codomain):
            y_left = y_left.squeeze(-1).long()

        if _is_discrete(self.right.codomain):
            y_right = y_right.squeeze(-1).long()

        return self.left.log_prob(x_left, y_left) + self.right.log_prob(
            x_right, y_right
        )

    def _split_input(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split concatenated domain input into left and right parts."""
        d = self._left_dom_dim
        x_left = x[..., :d]
        x_right = x[..., d:]

        # reconstruct discrete indices if needed
        if _is_discrete(self.left.domain):
            x_left = x_left.squeeze(-1).long()

        if _is_discrete(self.right.domain):
            x_right = x_right.squeeze(-1).long()

        return x_left, x_right


# -- fan-out (diagonal) morphism -----------------------------------------------


class FanOutMorphism(ContinuousMorphism):
    """Fan-out morphism: copy input to N morphisms, concatenate outputs.

    Given f_1: A -> B_1, f_2: A -> B_2, ..., f_N: A -> B_N,
    produces fan(f_1, ..., f_N): A -> B_1 * B_2 * ... * B_N
    where the input A is copied to all N morphisms.

    Unlike the tensor product (f @ g), which takes a product domain
    (A * C), fan-out feeds the same input to all morphisms. This
    implements the diagonal morphism Delta: A -> A^N followed by
    the product f_1 @ f_2 @ ... @ f_N.

    Parameters
    ----------
    components : list[ContinuousMorphism]
        The morphisms to fan out to. All must share the same domain.
    """

    def __init__(self, components: list[ContinuousMorphism]) -> None:
        if not components:
            raise ValueError("fan-out requires at least one component")

        domain = components[0].domain

        # all components must share the same domain
        for i, c in enumerate(components[1:], 1):
            dom_dim = _event_dim(domain)
            c_dim = _event_dim(c.domain)

            if dom_dim != c_dim:
                raise TypeError(
                    f"fan-out: component {i} domain dim {c_dim} "
                    f"!= component 0 domain dim {dom_dim}"
                )

        # build product codomain
        codomain = components[0].codomain

        for c in components[1:]:
            codomain = _combine_spaces(codomain, c.codomain)

        super().__init__(domain, codomain)
        self._components = torch.nn.ModuleList(components)
        self._cod_dims = [_event_dim(c.codomain) for c in components]

    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        """Sample from all components and concatenate outputs.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor (broadcast to all components).
        sample_shape : torch.Size
            Additional leading sample dimensions.

        Returns
        -------
        torch.Tensor
            Concatenated outputs from all components.
        """
        outs = []

        for comp in self._components:
            y = cast(ContinuousMorphism, comp).rsample(x, sample_shape)

            # ensure at least 2d for concatenation
            if y.dim() == 1:
                y = y.unsqueeze(-1)

            outs.append(y)

        return torch.cat(outs, dim=-1)

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Log-probability: sum of component log-probs.

        Parameters
        ----------
        x : torch.Tensor
            Input (same for all components).
        y : torch.Tensor
            Concatenated output values.

        Returns
        -------
        torch.Tensor
            Sum of log-probabilities. Shape ``(batch,)``.
        """
        lp = torch.zeros(x.shape[0], device=x.device)
        offset = 0

        for comp_mod, d in zip(self._components, self._cod_dims):
            comp = cast(ContinuousMorphism, comp_mod)
            y_slice = y[..., offset : offset + d]

            if _is_discrete(comp.codomain):
                y_slice = y_slice.squeeze(-1).long()

            lp = lp + comp.log_prob(x, y_slice)
            offset += d

        return lp


# -- discrete-continuous bridge -----------------------------------------------


class DiscreteAsContinuous(ContinuousMorphism):
    """Wrap a discrete Morphism as a ContinuousMorphism.

    Enables composition between discrete and continuous morphisms
    via the >> operator. The wrapped morphism's tensor is used for
    both log_prob evaluation and sampling.

    Note
    ----
    Sampling from a discrete distribution is NOT reparameterizable.
    Gradients do not flow through the discrete samples back to the
    left morphism's parameters. Use score function estimators
    (REINFORCE) if gradients through discrete choices are needed.

    Parameters
    ----------
    inner : Morphism
        The discrete morphism to wrap.
    """

    def __init__(self, inner: object) -> None:
        from quivers.core.morphisms import Morphism

        if not isinstance(inner, Morphism):
            raise TypeError(f"expected a discrete Morphism, got {type(inner).__name__}")

        super().__init__(inner.domain, inner.codomain)
        self._inner = inner

        # register the inner morphism's module for parameter collection
        self._inner_module = inner.module()

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Log-probability from the discrete tensor.

        Parameters
        ----------
        x : torch.Tensor
            Domain indices. Shape (batch,).
        y : torch.Tensor
            Codomain indices. Shape (batch,).

        Returns
        -------
        torch.Tensor
            Log-probabilities. Shape (batch,).
        """
        t = self._inner.tensor
        probs = t[x.long(), y.long()]
        return torch.log(probs.clamp(min=1e-7))

    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        """Sample from the categorical distribution defined by the tensor.

        Note: not reparameterizable. Gradients do not flow through
        the returned samples.

        Parameters
        ----------
        x : torch.Tensor
            Domain indices. Shape (batch,).
        sample_shape : torch.Size
            Additional sample dimensions.

        Returns
        -------
        torch.Tensor
            Sampled codomain indices. Shape (*sample_shape, batch).
        """
        t = self._inner.tensor
        probs = t[x.long()]  # (batch, codomain_size)

        n_samples = (
            int(torch.Size(sample_shape).numel()) if len(sample_shape) > 0 else 1
        )

        # sample with replacement
        samples = torch.multinomial(
            probs,
            n_samples,
            replacement=True,
        )  # (batch, n_samples)

        if len(sample_shape) == 0:
            return samples.squeeze(-1)

        else:
            # reshape to (*sample_shape, batch)
            return samples.T.reshape(*sample_shape, -1)


# -- helpers ------------------------------------------------------------------


def _event_dim(space: AnySpace) -> int:
    """Get the event dimensionality of a space.

    Discrete spaces are treated as 1-dimensional (index encoded as float).
    """
    if isinstance(space, ContinuousSpace):
        return space.dim

    return 1


def _combine_spaces(a: AnySpace, b: AnySpace) -> AnySpace:
    """Create a product of two spaces (possibly mixed types).

    For two continuous spaces, returns a ProductSpace.
    For mixed types, wraps discrete spaces as Euclidean(1).
    """
    from quivers.continuous.spaces import ProductSpace, Euclidean

    def _as_continuous(s: AnySpace) -> ContinuousSpace:
        if isinstance(s, ContinuousSpace):
            return s

        return Euclidean(f"idx({s!r})", 1)

    return ProductSpace(_as_continuous(a), _as_continuous(b))
