"""Scan combinator: temporal recurrence over sequences.

A ScanMorphism wraps a recurrent cell and applies it across a
sequence, threading hidden state from one time step to the next.
This implements the standard RNN pattern:

    h_t = cell(x_t, h_{t-1})

where cell : A * H -> H is a morphism (either a plain
ContinuousMorphism or a MonadicProgram) whose domain is a
product of the per-timestep input space A and the hidden state
space H, and whose codomain is H.

Given a cell : A * H -> H, ``scan(cell)`` produces a morphism
A -> H that, at runtime:

1. Expects a 3D input tensor of shape (batch, seq_len, dim_A).
2. Initializes hidden state h_0 (zeros or a learned parameter).
3. At each step t, concatenates x[:, t, :] with h to form the
   cell input, then calls cell.rsample to produce the new h.
4. Returns the final hidden state h_T of shape (batch, dim_H).

The scan's type in the categorical framework is:

    scan(f : A x H -> H) : A -> H

where the sequence structure is implicit in the tensor's time
dimension, following standard neural network conventions.

Initialization strategies
-------------------------
- ``"zeros"``: h_0 = 0 (default).
- ``"learned"``: h_0 is a learnable nn.Parameter.

Examples
--------
>>> from quivers.continuous.spaces import Euclidean, ProductSpace
>>> from quivers.continuous.families import ConditionalNormal
>>> A = Euclidean("input", 32)
>>> H = Euclidean("hidden", 64)
>>> cell = ConditionalNormal(ProductSpace(A, H), H, scale=0.1)
>>> scanned = ScanMorphism(cell, init="zeros")
>>> scanned.domain   # Euclidean("input", 32)
>>> scanned.codomain # Euclidean("hidden", 64)
>>> x = torch.randn(8, 10, 32)  # batch=8, seq_len=10, input_dim=32
>>> h = scanned.rsample(x)      # (8, 64)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from quivers.continuous.morphisms import (
    ContinuousMorphism,
    _event_dim,
)
from quivers.continuous.spaces import ContinuousSpace, ProductSpace


def _extract_input_space(cell: ContinuousMorphism) -> ContinuousSpace:
    """Extract the per-timestep input space from a cell's product domain.

    The cell must have a product domain A * H where H matches the
    codomain. Returns A.

    Parameters
    ----------
    cell : ContinuousMorphism
        A recurrent cell with product domain.

    Returns
    -------
    ContinuousSpace
        The input component of the product domain.

    Raises
    ------
    TypeError
        If the domain is not a ProductSpace or the last component
        does not match the codomain.
    """
    domain = cell.domain
    codomain = cell.codomain

    if not isinstance(domain, ProductSpace):
        raise TypeError(
            f"scan cell must have a ProductSpace domain, "
            f"got {type(domain).__name__}: {domain!r}"
        )

    components = domain.components

    if len(components) < 2:
        raise TypeError(
            f"scan cell product domain must have at least 2 "
            f"components, got {len(components)}"
        )

    # the last component must match the codomain (the hidden state)
    hidden_component = components[-1]
    cod_dim = _event_dim(codomain)
    hid_dim = _event_dim(hidden_component)

    if hid_dim != cod_dim:
        raise TypeError(
            f"scan cell: last domain component dim ({hid_dim}) "
            f"does not match codomain dim ({cod_dim}); the cell "
            f"must have type A * H -> H"
        )

    # the input space is everything except the last component
    if len(components) == 2:
        return components[0]

    else:
        # rebuild a product of the non-hidden components
        result = components[0]

        for c in components[1:-1]:
            result = ProductSpace(result, c)

        return result


class ScanMorphism(ContinuousMorphism):
    """Temporal scan: apply a recurrent cell across a sequence.

    Wraps a cell morphism ``f : A * H -> H`` and produces a morphism
    ``A -> H`` that iterates over the time dimension of a 3D input
    tensor, threading hidden state forward.

    This implements standard RNN-style recurrence::

        h_0 = init
        h_t = cell(concat(x_t, h_{t-1}))  for t = 1..T

    The scan returns the final hidden state h_T.

    Parameters
    ----------
    cell : ContinuousMorphism
        The recurrent cell. Must have a product domain ``A * H``
        and codomain ``H``, where ``H`` matches the last component
        of the product domain.
    init : str
        Initialization strategy for h_0. One of ``"zeros"``
        (default) or ``"learned"`` (trainable initial state).
    """

    def __init__(
        self,
        cell: ContinuousMorphism,
        init: str = "zeros",
    ) -> None:
        input_space = _extract_input_space(cell)
        hidden_space = cell.codomain

        super().__init__(input_space, hidden_space)

        self._cell = cell
        self._init_strategy = init
        self._input_dim = _event_dim(input_space)
        self._hidden_dim = _event_dim(hidden_space)

        if init == "learned":
            self._h0 = nn.Parameter(torch.zeros(self._hidden_dim))

        elif init != "zeros":
            raise ValueError(
                f"unknown init strategy {init!r}; expected 'zeros' or 'learned'"
            )

    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        """Run the cell across the time dimension of x.

        Parameters
        ----------
        x : torch.Tensor
            Input sequence. Shape ``(batch, seq_len, input_dim)``.
        sample_shape : torch.Size
            Additional leading sample dimensions (applied to the
            cell's rsample at the first time step only).

        Returns
        -------
        torch.Tensor
            Final hidden state. Shape ``(batch, hidden_dim)``,
            or ``(*sample_shape, batch, hidden_dim)`` if
            sample_shape is non-empty.
        """
        if x.dim() == 2:
            # single time step: (batch, input_dim) -> (batch, 1, input_dim)
            x = x.unsqueeze(1)

        batch, seq_len, _ = x.shape

        # initialize hidden state
        if self._init_strategy == "learned":
            h = self._h0.unsqueeze(0).expand(batch, -1)

        else:
            h = torch.zeros(
                batch,
                self._hidden_dim,
                device=x.device,
                dtype=x.dtype,
            )

        # iterate over time
        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch, input_dim)
            cell_input = torch.cat([x_t, h], dim=-1)  # (batch, input_dim + hidden_dim)

            # only pass sample_shape on the first step
            if t == 0 and len(sample_shape) > 0:
                h = self._cell.rsample(cell_input, sample_shape)
                h = self._flatten_cell_output(h)

                # if sample_shape introduced extra dims, reshape x
                # for subsequent steps
                if len(sample_shape) > 0 and h.dim() > 2:
                    # h is (*sample_shape, batch, hidden_dim)
                    # expand x to match: (*sample_shape, batch, seq_len, input_dim)
                    x = x.unsqueeze(0).expand(*sample_shape, *x.shape)

            else:
                if h.dim() > 2:
                    # h has sample dims: (*sample_shape, batch, hidden_dim)
                    # x_t needs matching: (*sample_shape, batch, input_dim)
                    x_t = x[..., t, :]
                    cell_input = torch.cat([x_t, h], dim=-1)

                h = self._cell.rsample(cell_input)
                h = self._flatten_cell_output(h)

        return h

    @staticmethod
    def _flatten_cell_output(
        result: torch.Tensor | dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Flatten a cell output to a single tensor.

        Monadic programs with tuple returns produce dicts. For scan,
        we concatenate the dict values along the feature dimension
        to reconstruct the full hidden state vector.

        Parameters
        ----------
        result : torch.Tensor or dict
            Cell output (tensor or dict from tuple-returning program).

        Returns
        -------
        torch.Tensor
            Flattened hidden state.
        """
        if isinstance(result, dict):
            return torch.cat(list(result.values()), dim=-1)

        return result

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Log-probability is not directly supported for scan.

        Computing log p(h_T | x_{1:T}) requires marginalizing
        over all intermediate hidden states h_1, ..., h_{T-1},
        which is intractable in general.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError(
            "log_prob is not supported for scan morphisms; "
            "computing p(h_T | x_{1:T}) requires marginalizing "
            "over all intermediate hidden states. use rsample() "
            "for forward sampling, or log_joint() for scoring "
            "given all intermediates."
        )

    def log_joint(
        self,
        x: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Joint log-density given all intermediate hidden states.

        Computes:
            log p(h_1, ..., h_T | x_{1:T}) =
                sum_t log p(h_t | x_t, h_{t-1})

        Parameters
        ----------
        x : torch.Tensor
            Input sequence. Shape ``(batch, seq_len, input_dim)``.
        hidden_states : torch.Tensor
            All hidden states including final. Shape
            ``(batch, seq_len, hidden_dim)``.

        Returns
        -------
        torch.Tensor
            Joint log-density. Shape ``(batch,)``.
        """
        batch, seq_len, _ = x.shape
        total = torch.zeros(batch, device=x.device)

        # initial hidden state
        if self._init_strategy == "learned":
            h = self._h0.unsqueeze(0).expand(batch, -1)

        else:
            h = torch.zeros(
                batch,
                self._hidden_dim,
                device=x.device,
                dtype=x.dtype,
            )

        for t in range(seq_len):
            x_t = x[:, t, :]
            h_t = hidden_states[:, t, :]
            cell_input = torch.cat([x_t, h], dim=-1)
            total = total + self._cell.log_prob(cell_input, h_t)
            h = h_t

        return total

    def __repr__(self) -> str:
        init = f", init={self._init_strategy}" if self._init_strategy != "zeros" else ""
        return f"ScanMorphism({self._cell!r}{init})"
