# Bilinear Tensor Contraction

## Overview

Bilinear scoring composes two embeddings through a third-order interaction tensor, the core of the neural tensor layer of [Socher et al. (2013)](https://papers.nips.cc/paper/2013/hash/b337e84de8752b27eda3a12363109e80-Abstract.html). Here the model scores predicate-argument pairs: each item $i$ carries a predicate embedding $P_{i,:}$ and an argument embedding $A_{i,:}$, and a shared interaction tensor $W$ maps each embedding pair to a score for every point $g$ on a judgment scale:

$$
s_{i,g} \;=\; \sum_{b}\sum_{c} P_{i,b}\, A_{i,c}\, W_{b,c,g}.
$$

In quivers this three-way join is a single `contraction` declaration: an [operad](https://ncatlab.org/nlab/show/operad)-style n-ary morphism that combines typed input arrows under a named composition rule. The einsum wiring is inferred from the typed signature, so the source spells out only which arrow plugs into which wire.

## QVR source

```qvr
composition real [level=algebra]

object Item : FinSet 4
object PredDim, ArgDim : FinSet 2
object Judgment : FinSet 3

morphism pred_embed : Item -> PredDim [role=latent]
morphism arg_embed : Item -> ArgDim [role=latent]
morphism interaction : (PredDim * ArgDim) -> Judgment [role=latent]

contraction bilinear_score (
    p : Item -> PredDim,
    a : Item -> ArgDim,
    w : (PredDim * ArgDim) -> Judgment,
) : Item -> Judgment [rule=real]

define plausibility = bilinear_score(pred_embed, arg_embed, interaction)

export plausibility
```

## Walkthrough

The three latent declarations introduce the two embedding maps and the interaction tensor as first-class arrows: `pred_embed` and `arg_embed` have tensors of shape `(4, 2)`, and `interaction`, whose domain is the product `PredDim * ArgDim`, has a tensor of shape `(2, 2, 3)`.

The `contraction` declaration types each input wire and names the composition rule:

<!-- compile: false -->
```qvr
contraction bilinear_score (
    p : Item -> PredDim,
    a : Item -> ArgDim,
    w : (PredDim * ArgDim) -> Judgment,
) : Item -> Judgment [rule=real]
```

The required `rule=` option references a registered composition rule; `real` is the ordinary sum-product [semiring](https://en.wikipedia.org/wiki/Semiring) on $\mathbb{R}$, so contracted axes are joined by multiply-then-sum, exactly a [`torch.einsum`](https://docs.pytorch.org/docs/stable/generated/torch.einsum.html). The compiler infers the wiring from the typed signature: `PredDim` and `ArgDim` each appear in two inputs but not in the output, so both are contracted; `Item` and `Judgment` appear in the output, so both propagate. The inferred spec is `ab, ac, bcd -> ad`. For contractions the inference cannot express, the option block also admits a `share=[...]` clause (keep a listed axis out of contraction) and a `wiring="..."` clause (verbatim einsum escape hatch); see the [contractions guide](../guides/dsl-contractions.md).

The `define` binding invokes the contraction on the three declared arrows. Each call site checks the argument count and per-argument shape against the declared wires, then runs the wiring and returns a fresh arrow with the declared `Item -> Judgment` typing; that arrow is what the module exports. Note that a multi-line input list takes a trailing comma before the closing parenthesis.

## Try it

> The short fits below demonstrate the API. Assess convergence with multiple chains and diagnostics before interpreting a posterior.

The exported arrow is a deterministic score tensor materialised from whatever the three latents currently hold, so the module fixes a mean surface rather than a measure. The snippets below supply the probabilistic surface around it: every entry of the two embeddings and of the interaction tensor carries an independent $\mathcal{N}(0, 1)$ prior, and each judgment is scored under $y_{i, g} \sim \mathcal{N}(s_{i, g}, \sigma^2)$. The likelihood re-runs the wiring on every evaluation rather than reading the materialised tensor, so gradients and Hamiltonian trajectories reach all three arrows. The compiled contraction is reachable through the [`Compiler`](../api/dsl/compiler.md#quivers.dsl.compiler.Compiler) environment's `contractions` mapping.

### Generating synthetic data

Draw ground-truth embeddings and an interaction tensor, push them through the bilinear form, and add Normal observation noise. The three `sample` sites of the [`MonadicProgram`](../api/continuous/programs.md#quivers.continuous.programs.MonadicProgram) built here are named after the arrows the source declares, and the judgments come from the very tensors bound as ground truth, so the snippet leaves one self-consistent point of the joint behind. Every shape is read off the compiled module rather than restated, so the block tracks the source if the source changes.

```python
import torch
import torch.distributions as D

from quivers.continuous.morphisms import ContinuousMorphism
from quivers.continuous.programs import MonadicProgram
from quivers.continuous.spaces import Euclidean
from quivers.core.objects import Unit
from quivers.dsl.compiler import Compiler
from quivers.dsl.parser import parse_file

torch.manual_seed(0)

compiler = Compiler(parse_file("docs/examples/source/tensor_contraction.qvr"))
compiler.compile()
wiring = compiler.contractions["bilinear_score"].wiring

pred_shape = tuple(compiler.morphisms["pred_embed"].tensor.shape)
arg_shape = tuple(compiler.morphisms["arg_embed"].tensor.shape)
inter_shape = tuple(compiler.morphisms["interaction"].tensor.shape)
n_judgment = inter_shape[-1]
sigma = 0.5


def bilinear_score(pred, arg, inter):
    """Run the declared `bilinear_score` contraction on three tensors."""
    return wiring.apply(pred, arg, inter)


class EntrywisePrior(ContinuousMorphism):
    """Independent Normal(0, 1) prior over one arrow's entries."""

    def __init__(self, shape):
        dim = 1
        for axis in shape:
            dim *= int(axis)
        super().__init__(Unit, Euclidean(name="Entries", dim=dim))
        self._shape = tuple(int(axis) for axis in shape)

    def rsample(self, x, sample_shape=torch.Size()):
        return D.Normal(torch.zeros(self._shape), 1.0).rsample()

    def log_prob(self, x, y):
        return D.Normal(0.0, 1.0).log_prob(y.reshape(self._shape)).sum()


class JudgmentLikelihood(ContinuousMorphism):
    """Normal judgment around the contracted bilinear score."""

    def __init__(self, sigma):
        super().__init__(
            Euclidean(name="Score", dim=n_judgment),
            Euclidean(name="Judgment", dim=n_judgment),
        )
        self._sigma = sigma

    def rsample(self, x, sample_shape=torch.Size()):
        return D.Normal(x, self._sigma).rsample()

    def log_prob(self, x, y):
        return D.Normal(x, self._sigma).log_prob(y.reshape(x.shape)).sum()


model = MonadicProgram(
    domain=Euclidean(name="Ix", dim=1),
    codomain=Euclidean(name="Judgment", dim=n_judgment),
    steps=[
        (("pred_embed",), EntrywisePrior(pred_shape), None, False),
        (("arg_embed",), EntrywisePrior(arg_shape), None, False),
        (("interaction",), EntrywisePrior(inter_shape), None, False),
        (
            ("mu",),
            None,
            lambda env: bilinear_score(
                env["pred_embed"].reshape(pred_shape),
                env["arg_embed"].reshape(arg_shape),
                env["interaction"].reshape(inter_shape),
            ),
        ),
        (("judgment",), JudgmentLikelihood(sigma), ("mu",), True),
    ],
    return_vars=("judgment",),
)

true_pred_embed = torch.randn(pred_shape)
true_arg_embed = torch.randn(arg_shape)
true_interaction = torch.randn(inter_shape)
score = bilinear_score(true_pred_embed, true_arg_embed, true_interaction)
judgment = D.Normal(score, sigma).sample()

observations = {"judgment": judgment}
x_in = torch.zeros(1, 1)

print("scores:", score.round(decimals=2).tolist())
```

The `mu` step is a deterministic `let`: it reads the three sampled tensors out of the trace environment, reshapes each back to its declared arrow shape, and returns the $4 \times 3$ contraction. Reshaping is what makes the step total, since a clamped site arrives flattened, and the third-order `interaction` is the one arrow whose shape a flat vector cannot be read off.

### SVI fit

Fit with [`AutoNormalGuide`](../api/inference/guide.md#quivers.inference.guides.AutoNormalGuide) + [`ELBO`](../api/inference/elbo.md#quivers.inference.objectives.ELBO) + [`SVI`](../api/inference/svi.md#svi). All three arrows are genuine `sample` sites, so the guide carries a mean-field Normal over each rather than degenerating to a point estimate.

```python
from quivers.inference import AutoNormalGuide, ELBO, SVI

torch.manual_seed(1)
guide = AutoNormalGuide(model, observed_names={"judgment"})
optim = torch.optim.Adam(
    list(model.parameters()) + list(guide.parameters()), lr=5e-2,
)
svi = SVI(model, guide, optim, ELBO(num_particles=1))

losses = [svi.step(x_in, observations) for _ in range(200)]
print(f"initial loss: {losses[0]:.2f}")
print(f"final loss:   {losses[-1]:.2f}")
```

The factorisation is identifiable only up to invertible reparameterisations of the two embedding spaces, so the fitted embeddings tend to differ from the truths even once the score tensor has converged.

### NUTS posterior

Because the priors are declared inside the program, [`NUTSKernel`](../api/inference/mcmc.md#quivers.inference.mcmc.NUTSKernel) targets it directly; a model that instead exposed the three arrows as bare `[role=latent]` parameters would first need [`bayesian_lift_parameters`](../api/inference/lifts.md#quivers.inference.lifts.bayesian_lift_parameters) to give them a prior.

```python
from quivers.inference import MCMC, NUTSKernel

torch.manual_seed(2)
kernel = NUTSKernel(step_size=0.05, max_tree_depth=3, target_accept=0.8)
mc     = MCMC(kernel, num_warmup=15, num_samples=15, num_chains=1)
result = mc.run(model, x_in, observations)

print(f"acceptance:  {float(result.acceptance_rates.mean()):.2f}")
print(f"divergences: {int(result.divergence_counts.sum())}")
```

## Categorical perspective

Binary composition `>>` is the 2-ary case of a wider operadic structure: a `contraction` denotes an n-ary morphism in the [multicategory](https://ncatlab.org/nlab/show/multicategory) of tensor spaces over the active composition rule, with the wiring specification fixing which axes are joined and which survive to the output (see [Composition Rules § 4](../semantics/composition-rules.md#4-operadic-contractions) for the denotation). The bilinear score is the case $n = 3$: two argument tensors over a shared `Item` axis and a kernel over the two embedding axes, folded under $\otimes$ and joined under $\bigoplus$ of the sum-product semiring. Chains of binary `>>` can express only tree-shaped contractions of matrices; the third-order `interaction` tensor makes this join genuinely operadic.

## See also

- [DSL Contractions](../guides/dsl-contractions.md) for the declaration surface, wiring inference, and the `share=` / `wiring=` clauses.
- [Composition Rules § 4](../semantics/composition-rules.md#4-operadic-contractions) for the categorical semantics of operadic contractions.
- [Probabilistic Matrix Factorization](pmf.md) for the 2-ary bilinear score expressed with `.dagger` and `>>`.

## References

- Richard Socher, Danqi Chen, Christopher D. Manning, and Andrew Y. Ng. 2013. [Reasoning With Neural Tensor Networks for Knowledge Base Completion](https://papers.nips.cc/paper/2013/hash/b337e84de8752b27eda3a12363109e80-Abstract.html). In *Advances in Neural Information Processing Systems 26*.
