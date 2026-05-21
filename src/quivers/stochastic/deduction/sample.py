"""Forward sampling of yields from a weighted deduction system.

`sample_corpus` draws length-fixed token sequences from the
length-conditional distribution induced by the chart's weights:

.. math::

    p(s \\mid \\text{length} = L,\\, \\mathbf{w})
    \\;=\\; \\frac{Z(s; \\mathbf{w})}
                 {\\sum_{s' \\text{ of length } L} Z(s'; \\mathbf{w})}.

It enumerates every length-:math:`L` sequence over the
deduction's surface vocabulary, evaluates :math:`\\log Z(s;
\\mathbf{w})` for each via the chart, softmaxes the log-weights,
and draws a multinomial. The procedure is *exact* (the chart
already marginalises over the derivation forest); the
:math:`|V|^L` enumeration cost is the fundamental cost of forward
sampling from a globally-normalised chart-defined distribution.
"""

from __future__ import annotations

import itertools

import torch

from quivers.stochastic.agenda import DeductionSystem


__all__ = ["sample_corpus"]


def sample_corpus(
    ded: DeductionSystem,
    *,
    length: int,
    n_samples: int,
    seed: int | None = None,
) -> list[list[str]]:
    """Sample ``n_samples`` yields of length ``length`` from the
    chart's length-conditional distribution under the deduction's
    current parameters.

    Parameters
    ----------
    ded : DeductionSystem
        The deduction with materialised parameters.
    length : int
        Length of yields to enumerate.
    n_samples : int
        Number of sentences to draw.
    seed : int, optional
        Seed for the multinomial draws.
    """
    vocab = _vocabulary(ded)
    if not vocab:
        raise ValueError(
            "sample_corpus: cannot determine the deduction's "
            "vocabulary; set ``ded._vocabulary`` explicitly or "
            "call ``materialise_parameters`` first"
        )

    yields: list[list[str]] = []
    log_weights: list[torch.Tensor] = []
    for combo in itertools.product(vocab, repeat=length):
        chart = ded(list(combo))
        w = chart.goal_weight()
        if torch.isfinite(w):
            yields.append(list(combo))
            log_weights.append(w)
    if not yields:
        raise ValueError(
            f"sample_corpus: no yield of length {length} parses under "
            f"the deduction's current parameters"
        )
    logw = torch.stack([w.detach() for w in log_weights])
    probs = torch.softmax(logw, dim=0)
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)
    idxs = torch.multinomial(probs, n_samples, replacement=True, generator=gen)
    return [yields[int(i.item())] for i in idxs]


def _vocabulary(ded: DeductionSystem) -> list[str]:
    """Pull the deduction's surface vocabulary from its axiom
    injector. Works for the standard lexicon-backed injector the
    compiler emits; users with custom injectors can attach a
    ``_vocabulary`` attribute directly on the deduction system."""
    explicit = getattr(ded, "_vocabulary", None)
    if explicit is not None:
        return list(explicit)
    inj = getattr(ded, "axiom_injector", None)
    if inj is None or not hasattr(inj, "__defaults__"):
        return []
    defaults = inj.__defaults__ or ()
    for default in defaults:
        if isinstance(default, tuple) and default and isinstance(default[0], tuple):
            words = [entry[0] for entry in default if isinstance(entry, tuple)]
            seen: set[str] = set()
            out: list[str] = []
            for w in words:
                if w not in seen:
                    seen.add(w)
                    out.append(w)
            return out
    return []
