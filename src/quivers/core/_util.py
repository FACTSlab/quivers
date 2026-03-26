"""Numerical utilities for stable fuzzy-logic operations."""

from __future__ import annotations

import torch

EPS: float = 1e-7


def clamp_probs(x: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Clamp tensor values to the open interval (eps, 1 - eps).

    Parameters
    ----------
    x : torch.Tensor
        Input tensor with values nominally in [0, 1].
    eps : float
        Clamping margin.

    Returns
    -------
    torch.Tensor
        Clamped tensor.
    """
    return x.clamp(min=eps, max=1.0 - eps)


def safe_log1p_neg(x: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Compute log(1 - x) in a numerically stable way.

    Uses torch.log1p(-x) after clamping x away from 1.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor with values in [0, 1].
    eps : float
        Clamping margin.

    Returns
    -------
    torch.Tensor
        log(1 - x), with values in (-inf, 0].
    """
    return torch.log1p(-clamp_probs(x, eps))
