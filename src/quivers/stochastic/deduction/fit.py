"""Point-estimate fitting of a weighted deduction system.

`adam_fit_deduction` runs gradient descent on the
deduction's learnable log-weights to maximise the corpus
log-marginal :math:`\\sum_n \\log Z(s_n; \\mathbf{w})`, optionally
under an isotropic Normal prior (MAP). Each :math:`\\log Z(s;
\\mathbf{w})` is computed exactly by the chart's LogProb-semiring
fixed point; autograd through the agenda's semiring operations
yields the exact gradient
:math:`\\nabla_{\\mathbf{w}} \\log Z(s; \\mathbf{w})
= \\mathbb{E}_{d \\mid s}[\\phi(d)]` (the standard inside-outside
identity).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from quivers.stochastic.deduction._internal import materialise_parameters
from quivers.stochastic.agenda import DeductionSystem


__all__ = ["adam_fit_deduction"]


def adam_fit_deduction(
    ded: DeductionSystem,
    corpus: Sequence[Sequence[str]],
    *,
    steps: int = 300,
    lr: float = 5e-2,
    prior_scale: float | None = None,
) -> list[float]:
    """Maximise the corpus log-marginal under an optional Normal
    prior on the parameters.

    Parameters
    ----------
    ded : DeductionSystem
        Deduction whose ``_axiom_module`` and ``_rule_module``
        parameters are optimised.
    corpus : sequence of sentences
        Each sentence is a sequence of token strings the
        deduction's axiom injector accepts.
    steps : int
        Adam steps.
    lr : float
        Adam learning rate.
    prior_scale : float, optional
        If supplied, adds a Gaussian regulariser
        :math:`\\tfrac{1}{2\\sigma^2}\\lVert \\mathbf{w} \\rVert^2`
        to the loss (MAP). Defaults to ``None`` (MLE).

    Returns
    -------
    list[float]
        The loss trajectory; length == ``steps``.
    """
    materialise_parameters(ded, corpus)
    params = list(ded.parameters())
    if not params:
        return []
    optim = torch.optim.Adam(params, lr=lr)
    history: list[float] = []
    for _ in range(steps):
        optim.zero_grad()
        log_z = torch.zeros(())
        for sentence in corpus:
            log_z = log_z + ded(list(sentence)).goal_weight()
        loss = -log_z
        if prior_scale is not None:
            inv_var = 1.0 / (prior_scale ** 2)
            for p in params:
                loss = loss + 0.5 * inv_var * (p ** 2).sum()
        loss.backward()
        optim.step()
        history.append(float(loss.detach()))
    return history
