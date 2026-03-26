"""Probability queries on stochastic morphisms.

Provides functions to extract probabilities, compute marginal
probabilities, and compute expected values under stochastic morphisms.
"""

from __future__ import annotations

import torch

from quivers.core.morphisms import Morphism


def prob(
    f: Morphism,
    domain_indices: torch.Tensor,
    codomain_indices: torch.Tensor,
) -> torch.Tensor:
    """Extract probabilities at specific (domain, codomain) index pairs.

    Parameters
    ----------
    f : Morphism
        A (stochastic) morphism A → B.
    domain_indices : torch.Tensor
        Integer indices into the domain. Shape (batch,) or (batch, n_dom_dims).
    codomain_indices : torch.Tensor
        Integer indices into the codomain. Shape (batch,) or (batch, n_cod_dims).

    Returns
    -------
    torch.Tensor
        Probability values at the specified index pairs. Shape (batch,).
    """
    t = f.tensor

    if domain_indices.ndim == 1:
        domain_indices = domain_indices.unsqueeze(-1)

    if codomain_indices.ndim == 1:
        codomain_indices = codomain_indices.unsqueeze(-1)

    indices = torch.cat([domain_indices, codomain_indices], dim=-1)
    idx_tuple = tuple(indices[:, i] for i in range(indices.shape[1]))
    return t[idx_tuple]


def marginal_prob(f: Morphism, codomain_indices: torch.Tensor) -> torch.Tensor:
    """Compute marginal probability over the domain for specific codomain indices.

    Marginalizes (sums) over all domain elements:
        P(b) = Σ_a f(a, b) / |A|

    This assumes a uniform prior over the domain.

    Parameters
    ----------
    f : Morphism
        A (stochastic) morphism A → B.
    codomain_indices : torch.Tensor
        Integer indices into the codomain. Shape (batch,).

    Returns
    -------
    torch.Tensor
        Marginal probabilities. Shape (batch,).
    """
    t = f.tensor
    n_dom = f.domain.ndim
    dom_dims = tuple(range(n_dom))

    # average over domain (uniform prior)
    marginal = t.mean(dim=dom_dims)

    if codomain_indices.ndim == 1:
        codomain_indices = codomain_indices.unsqueeze(-1)

    idx_tuple = tuple(codomain_indices[:, i] for i in range(codomain_indices.shape[1]))
    return marginal[idx_tuple]


def expectation(
    f: Morphism,
    values: torch.Tensor,
) -> torch.Tensor:
    """Compute expected value of a function under a stochastic morphism.

    For each domain element a, computes:
        E_f[v | a] = Σ_b f(a, b) · values(b)

    Parameters
    ----------
    f : Morphism
        A (stochastic) morphism A → B.
    values : torch.Tensor
        Real-valued function on the codomain. Shape (*codomain.shape).

    Returns
    -------
    torch.Tensor
        Expected values. Shape (*domain.shape).
    """
    t = f.tensor
    n_dom = f.domain.ndim
    n_cod = f.codomain.ndim

    # expand values to broadcast: (1, ..., 1, *cod_shape)
    v = values

    for _ in range(n_dom):
        v = v.unsqueeze(0)

    weighted = t * v
    cod_dims = tuple(range(n_dom, n_dom + n_cod))
    return weighted.sum(dim=cod_dims)
