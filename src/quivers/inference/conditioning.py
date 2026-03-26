"""Conditioning: pair a program with observed data.

The ``condition`` function wraps a MonadicProgram with a dict of
observations, producing a ``Conditioned`` object that threads those
observations through tracing and log-joint evaluation. This is the
primary interface for specifying observed data in inference.
"""

from __future__ import annotations

import torch

from quivers.continuous.programs import MonadicProgram

from quivers.inference.trace import Trace, trace


class Conditioned:
    """A monadic program paired with observed data.

    Parameters
    ----------
    model : MonadicProgram
        The generative model.
    data : dict[str, torch.Tensor]
        Observed variable names mapped to their observed values.
    """

    def __init__(
        self,
        model: MonadicProgram,
        data: dict[str, torch.Tensor],
    ) -> None:
        self.model = model
        self.data = data

    def trace(self, x: torch.Tensor) -> Trace:
        """Execute the model with observations clamped.

        Parameters
        ----------
        x : torch.Tensor
            Program input. Shape (batch, ...).

        Returns
        -------
        Trace
            Execution trace with observed sites clamped.
        """
        return trace(self.model, x, observations=self.data)

    @property
    def observed_names(self) -> set[str]:
        """Names of all observed variables."""
        return set(self.data.keys())

    def __repr__(self) -> str:
        obs = ", ".join(sorted(self.data.keys()))
        return f"Conditioned({self.model!r}, observed=[{obs}])"


def condition(
    model: MonadicProgram,
    data: dict[str, torch.Tensor],
) -> Conditioned:
    """Condition a model on observed data.

    Parameters
    ----------
    model : MonadicProgram
        The generative model.
    data : dict[str, torch.Tensor]
        Observed variable names mapped to their observed values.

    Returns
    -------
    Conditioned
        Wrapped model with observations.
    """
    return Conditioned(model, data)
