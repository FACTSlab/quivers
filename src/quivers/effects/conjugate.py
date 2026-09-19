"""Closed-form marginals of the standard conjugate pairs.

A conjugate pair is a parent site whose value is the parameter of a
child site: ``z ~ Beta(a, b); y ~ Bernoulli(z)`` and its kin. The child's
marginal, with the parent integrated out, has a closed form for the four
pairs documented in
[Bernardo and Smith (1994)](https://doi.org/10.1002/9780470316870):
Normal-Normal, Beta-Bernoulli, Gamma-Poisson, and Dirichlet-Categorical.
The solvers here compute those marginals from the parent's distribution
parameters and the child's value, and check that the child is the
identity-link form the marginal is derived for.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import math

import torch

from quivers.effects.sites import TorchSampleable
from quivers.transpile.family_meta import FAMILY_META

type Solver = Callable[
    [Mapping[str, torch.Tensor], torch.Tensor, TorchSampleable, torch.Tensor],
    torch.Tensor,
]
"""A closed-form marginal: given the parent's parameters, the parent's
drawn value, the child's sampleable at that value, and the child's value,
the log marginal density of the child's value."""


def family_of(sampleable: TorchSampleable) -> str:
    """The registry family a sampleable's morphism draws from.

    Parameters
    ----------
    sampleable : TorchSampleable
        The sampleable.

    Returns
    -------
    str
        The family's registry name.

    Raises
    ------
    TypeError
        If the morphism's class is registered for no family.
    """
    inner = sampleable.declared
    assert inner is not None
    for name, meta in FAMILY_META.items():
        if meta.quivers_class is not None and isinstance(inner, meta.quivers_class):
            return name
    get_dist = getattr(inner, "_get_dist", None)
    if callable(get_dist):
        distribution = get_dist(sampleable.input)
        for name, meta in FAMILY_META.items():
            if type(distribution) is meta.distribution_class:
                return name
    raise TypeError(
        f"{type(inner).__name__} draws from no registered family; a conjugate "
        "marginal needs the parent's and the child's families"
    )


def _identity_link(child: TorchSampleable, parameter: str, value: torch.Tensor) -> None:
    """Require a child's parameter to be the parent's value.

    Parameters
    ----------
    child : TorchSampleable
        The child's sampleable, built at the parent's drawn value.
    parameter : str
        The child's parameter the parent's value must fill.
    value : torch.Tensor
        The parent's drawn value.

    Raises
    ------
    ValueError
        If the child's parameter differs from the parent's value, so the
        child is not the identity-link form the marginal is derived for.
    """
    actual = child.parameters()[parameter]
    rows = actual.shape[0] if actual.dim() > 0 else 1
    flat_actual = actual.reshape(rows, -1)
    flat_value = value.reshape(rows, -1) if value.numel() == actual.numel() else None
    if flat_value is None or not torch.allclose(
        flat_actual, flat_value.to(actual.dtype), atol=1e-6, rtol=1e-5
    ):
        raise ValueError(
            f"the child's {parameter!r} is not the parent's value; the conjugate "
            "marginal holds for the identity link only"
        )


def normal_normal(
    parent: Mapping[str, torch.Tensor],
    parent_value: torch.Tensor,
    child: TorchSampleable,
    value: torch.Tensor,
) -> torch.Tensor:
    """The marginal of ``y ~ Normal(z, tau)`` under ``z ~ Normal(mu, sigma)``.

    Parameters
    ----------
    parent : Mapping[str, torch.Tensor]
        The parent's ``loc`` and ``scale``.
    parent_value : torch.Tensor
        The parent's drawn value.
    child : TorchSampleable
        The child at the parent's drawn value, whose ``scale`` is ``tau``.
    value : torch.Tensor
        The child's value.

    Returns
    -------
    torch.Tensor
        ``log Normal(y; mu, sqrt(sigma^2 + tau^2))`` summed over the event.

    Raises
    ------
    ValueError
        If the child's location is not the parent's value.
    """
    _identity_link(child, "loc", parent_value)
    tau = child.parameters()["scale"].reshape(parent["scale"].shape)
    scale = torch.sqrt(parent["scale"] ** 2 + tau**2)
    residual = (value - parent["loc"]) / scale
    log_density = -0.5 * residual**2 - scale.log() - 0.5 * math.log(2 * math.pi)
    return log_density.sum(dim=-1) if log_density.dim() > 1 else log_density


def beta_bernoulli(
    parent: Mapping[str, torch.Tensor],
    parent_value: torch.Tensor,
    child: TorchSampleable,
    value: torch.Tensor,
) -> torch.Tensor:
    """The marginal of ``y ~ Bernoulli(z)`` under ``z ~ Beta(a, b)``.

    Parameters
    ----------
    parent : Mapping[str, torch.Tensor]
        The parent's ``concentration1`` and ``concentration0``.
    parent_value : torch.Tensor
        The parent's drawn value.
    child : TorchSampleable
        The child at the parent's drawn value.
    value : torch.Tensor
        The child's value, zero or one.

    Returns
    -------
    torch.Tensor
        ``log(a / (a + b))`` for one and ``log(b / (a + b))`` for zero.

    Raises
    ------
    ValueError
        If the child's probability is not the parent's value.
    """
    _identity_link(child, "probs", parent_value)
    a = parent["concentration1"]
    b = parent["concentration0"]
    probability = a / (a + b)
    log_density = torch.where(
        value.to(probability.dtype) > 0.5, probability.log(), (1 - probability).log()
    )
    return log_density.sum(dim=-1) if log_density.dim() > 1 else log_density


def gamma_poisson(
    parent: Mapping[str, torch.Tensor],
    parent_value: torch.Tensor,
    child: TorchSampleable,
    value: torch.Tensor,
) -> torch.Tensor:
    """The marginal of ``y ~ Poisson(z)`` under ``z ~ Gamma(alpha, beta)``.

    Parameters
    ----------
    parent : Mapping[str, torch.Tensor]
        The parent's ``concentration`` and ``rate``.
    parent_value : torch.Tensor
        The parent's drawn value.
    child : TorchSampleable
        The child at the parent's drawn value.
    value : torch.Tensor
        The child's count.

    Returns
    -------
    torch.Tensor
        The negative binomial log mass with ``alpha`` successes and
        success probability ``beta / (beta + 1)``.

    Raises
    ------
    ValueError
        If the child's rate is not the parent's value.
    """
    _identity_link(child, "rate", parent_value)
    alpha = parent["concentration"]
    beta = parent["rate"]
    count = value.to(alpha.dtype)
    log_density = (
        torch.lgamma(alpha + count)
        - torch.lgamma(alpha)
        - torch.lgamma(count + 1)
        + alpha * (beta.log() - torch.log1p(beta))
        - count * torch.log1p(beta)
    )
    return log_density.sum(dim=-1) if log_density.dim() > 1 else log_density


def dirichlet_categorical(
    parent: Mapping[str, torch.Tensor],
    parent_value: torch.Tensor,
    child: TorchSampleable,
    value: torch.Tensor,
) -> torch.Tensor:
    """The marginal of ``y ~ Categorical(z)`` under ``z ~ Dirichlet(alpha)``.

    Parameters
    ----------
    parent : Mapping[str, torch.Tensor]
        The parent's ``concentration``.
    parent_value : torch.Tensor
        The parent's drawn value.
    child : TorchSampleable
        The child at the parent's drawn value.
    value : torch.Tensor
        The child's category.

    Returns
    -------
    torch.Tensor
        ``log(alpha_y / sum(alpha))``.

    Raises
    ------
    ValueError
        If the child's probabilities are not the parent's value.
    """
    _identity_link(child, "probs", parent_value)
    alpha = parent["concentration"]
    probabilities = alpha / alpha.sum(dim=-1, keepdim=True)
    index = value.long().unsqueeze(-1)
    return probabilities.gather(-1, index).squeeze(-1).log()


#: The registered closed-form marginals, by parent and child family.
CONJUGATE_SOLVERS: Mapping[tuple[str, str], Solver] = {
    ("Normal", "Normal"): normal_normal,
    ("Beta", "Bernoulli"): beta_bernoulli,
    ("Gamma", "Poisson"): gamma_poisson,
    ("Dirichlet", "Categorical"): dirichlet_categorical,
}


def conjugate_solver(parent_family: str, child_family: str) -> Solver:
    """The solver of a conjugate pair.

    Parameters
    ----------
    parent_family : str
        The parent's family.
    child_family : str
        The child's family.

    Returns
    -------
    Solver
        The closed-form marginal.

    Raises
    ------
    KeyError
        If the pair has no registered marginal.
    """
    try:
        return CONJUGATE_SOLVERS[(parent_family, child_family)]
    except KeyError as error:
        registered = ", ".join(f"{a}-{b}" for a, b in CONJUGATE_SOLVERS)
        raise KeyError(
            f"no conjugate marginal is registered for ({parent_family}, "
            f"{child_family}); the registered pairs are {registered}"
        ) from error


__all__ = [
    "CONJUGATE_SOLVERS",
    "Solver",
    "beta_bernoulli",
    "conjugate_solver",
    "dirichlet_categorical",
    "family_of",
    "gamma_poisson",
    "normal_normal",
]
