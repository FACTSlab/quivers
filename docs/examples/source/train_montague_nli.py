"""End-to-end trainer for the Montague + Prover NLI pipeline.

Loads ``examples/montague_nli.qvr``, builds a tiny NLI dataset, and
runs SGD on the lexicon's learnable parameters so that the model
assigns higher chart-goal weight to entailment-consistent
(premise, hypothesis) pairs than to inconsistent ones.

The QVR side declares the constructor algebra and inference rules;
this driver wires up the optimizer + data loop.
"""

from __future__ import annotations

import torch

from quivers.dsl import load


def _enumerate_root(chart, n_tokens):
    """Yield (lf, log_weight) for every span(0, n, S, lf) item."""
    for item, w in chart.chart.items():
        if not (isinstance(item, tuple) and len(item) == 5):
            continue
        head, i, j, cat, lf = item
        if head == "span" and i == 0 and j == n_tokens:
            if cat == ("atom", "S"):
                yield lf, w


def main() -> None:
    program = load("examples/montague_nli.qvr")
    Montague = program.deductions["Montague"]
    Prover = program.deductions["Prover"]

    pairs = [
        (["dog", "barks"], ["animal", "barks"], 1.0),
        (["cat", "barks"], ["animal", "barks"], 1.0),
        (["dog", "barks"], ["dog", "barks"], 1.0),
        (["animal", "barks"], ["dog", "barks"], 0.0),
        (["cat", "barks"], ["dog", "barks"], 0.0),
        (["dog", "barks"], ["cat", "barks"], 0.0),
    ]

    background = [
        (("Claim", ("Implies", ("pred_dog",), ("pred_anim",))), torch.tensor(0.0)),
        (("Claim", ("Implies", ("pred_cat",), ("pred_anim",))), torch.tensor(0.0)),
    ]

    axmod = getattr(Montague, "_axiom_module", None)
    params = list(axmod.parameters()) if axmod is not None else []

    # Break the zero-initialization symmetry so the optimizer
    # starts in a state where every span has a distinct weight.
    with torch.no_grad():
        for p in params:
            p.normal_(mean=0.0, std=0.1)

    opt = torch.optim.Adam(params, lr=0.05)
    print(f"trainable lexicon params: {len(params)}")

    for step in range(30):
        opt.zero_grad()
        total_loss = torch.zeros(())

        for premise, hypothesis, label in pairs:
            p_chart = Montague(premise)
            h_chart = Montague(hypothesis)

            joint_weights = []
            for p_lf, p_w in _enumerate_root(p_chart, len(premise)):
                for h_lf, h_w in _enumerate_root(h_chart, len(hypothesis)):
                    proof_axioms = background + [
                        (("Claim", p_lf), p_w),
                    ]
                    proof_chart = Prover(proof_axioms)
                    target = ("Claim", h_lf)
                    proof_w = proof_chart.try_weight(target)
                    if proof_w is None:
                        continue
                    joint_weights.append(p_w + proof_w + h_w)

            # logsumexp over the joint proof weights. We include a
            # constant `-30` term so the logsumexp is well-defined
            # even when no proof is found, and so its gradient
            # always flows through the chart-parameter graph.
            sentinel = -30.0 * torch.ones(())
            stacked = torch.stack([sentinel] + joint_weights)
            log_p = torch.logsumexp(stacked, dim=0)

            target = torch.tensor(float(label))
            pair_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                log_p, target
            )
            total_loss = total_loss + pair_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        opt.step()

        if step % 5 == 0 or step == 39:
            print(f"step {step:3d}  loss {float(total_loss):.4f}")


if __name__ == "__main__":
    main()
