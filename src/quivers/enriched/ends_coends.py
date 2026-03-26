"""Ends and coends: general contraction principles for V-enriched categories.

The coend ∫^X F(X,X) computes the join (existential) over the diagonal of
matched contravariant/covariant dimensions. The end ∫_X F(X,X) computes
the meet (universal) over the diagonal.

These generalize composition (coend over shared object) and the hom in
the functor category (end over all objects).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from quivers.core.quantales import Quantale


def coend(
    tensor: torch.Tensor,
    contra_dims: tuple[int, ...],
    co_dims: tuple[int, ...],
    quantale: Quantale | None = None,
) -> torch.Tensor:
    """Compute coend ∫^X F(X,X) via diagonal extraction and join.

    For each matched pair (contra_dims[i], co_dims[i]), restricts the
    tensor to the diagonal (where the contravariant and covariant
    indices agree) and then joins (existential quantification) over
    those matched dimensions.

    Parameters
    ----------
    tensor : torch.Tensor
        The input tensor representing a bifunctor F(X, Y) with both
        contravariant and covariant occurrences of X.
    contra_dims : tuple[int, ...]
        Dimension indices for the contravariant occurrences of X.
    co_dims : tuple[int, ...]
        Dimension indices for the covariant occurrences of X.
        Must be same length as contra_dims, with matching sizes.
    quantale : Quantale or None
        The enrichment algebra. Defaults to PRODUCT_FUZZY.

    Returns
    -------
    torch.Tensor
        Result with matched dimension pairs removed via join.
    """
    from quivers.core.quantales import PRODUCT_FUZZY

    if quantale is None:
        quantale = PRODUCT_FUZZY

    if len(contra_dims) != len(co_dims):
        raise ValueError(
            f"contra_dims and co_dims must have same length, "
            f"got {len(contra_dims)} and {len(co_dims)}"
        )

    if len(contra_dims) == 0:
        return tensor

    return _trace_and_reduce(tensor, contra_dims, co_dims, quantale.join)


def end(
    tensor: torch.Tensor,
    contra_dims: tuple[int, ...],
    co_dims: tuple[int, ...],
    quantale: Quantale | None = None,
) -> torch.Tensor:
    """Compute end ∫_X F(X,X) via diagonal extraction and meet.

    Dual to coend: restricts to diagonal and applies meet (universal
    quantification) over matched dimensions.

    Parameters
    ----------
    tensor : torch.Tensor
        The input tensor.
    contra_dims : tuple[int, ...]
        Dimension indices for contravariant occurrences.
    co_dims : tuple[int, ...]
        Dimension indices for covariant occurrences.
    quantale : Quantale or None
        The enrichment algebra. Defaults to PRODUCT_FUZZY.

    Returns
    -------
    torch.Tensor
        Result with matched dimension pairs removed via meet.
    """
    from quivers.core.quantales import PRODUCT_FUZZY

    if quantale is None:
        quantale = PRODUCT_FUZZY

    if len(contra_dims) != len(co_dims):
        raise ValueError(
            f"contra_dims and co_dims must have same length, "
            f"got {len(contra_dims)} and {len(co_dims)}"
        )

    if len(contra_dims) == 0:
        return tensor

    return _trace_and_reduce(tensor, contra_dims, co_dims, quantale.meet)


def _trace_and_reduce(
    tensor: torch.Tensor,
    contra_dims: tuple[int, ...],
    co_dims: tuple[int, ...],
    reduce_fn: Callable[..., torch.Tensor],
) -> torch.Tensor:
    """Extract diagonal and reduce over matched dimension pairs.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor.
    contra_dims : tuple[int, ...]
        Contravariant dimension indices.
    co_dims : tuple[int, ...]
        Covariant dimension indices.
    reduce_fn : callable
        Reduction function (quantale.join or quantale.meet) taking
        (tensor, dim) arguments.

    Returns
    -------
    torch.Tensor
        Reduced tensor.
    """
    # validate dimension sizes match
    for i, (cd, cvd) in enumerate(zip(contra_dims, co_dims)):
        if tensor.shape[cd] != tensor.shape[cvd]:
            raise ValueError(
                f"dimension pair {i}: contra dim {cd} has size "
                f"{tensor.shape[cd]} but co dim {cvd} has size "
                f"{tensor.shape[cvd]}"
            )

    # process pairs one at a time, adjusting indices as we go.
    # strategy: for each pair, use torch.diagonal to extract the diagonal,
    # which moves the paired dimension to the end. then reduce over it.
    result = tensor

    # sort pairs by contra_dim descending so removals don't shift indices
    # we need to track both dims through removals
    pairs = list(zip(contra_dims, co_dims))

    for contra_d, co_d in pairs:
        result.shape[contra_d]

        # extract diagonal: moves the diagonal to the last dimension,
        # removes contra_d and co_d from their positions
        result = torch.diagonal(result, dim1=contra_d, dim2=co_d)

        # the diagonal is now the last dimension; reduce over it
        result = reduce_fn(result, dim=-1)

    return result
