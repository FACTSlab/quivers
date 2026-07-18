# Bilinear Tensor Contraction

## Overview

Bilinear scoring composes two embeddings through a third-order interaction tensor, the core of the neural tensor layer of [Socher et al. (2013)](https://papers.nips.cc/paper/2013/hash/b337e84de8752b27eda3a12363109e80-Abstract.html). Here the model scores predicate-argument pairs: each item $i$ carries a predicate embedding $P_{i,:}$ and an argument embedding $A_{i,:}$, and a shared interaction tensor $W$ maps each embedding pair to a score for every point $g$ on a judgment scale:

$$
s_{i,g} \;=\; \sum_{b}\sum_{c} P_{i,b}\, A_{i,c}\, W_{b,c,g}.
$$

In quivers this three-way join is a single `contraction` declaration: an [operad](https://ncatlab.org/nlab/show/operad)-style n-ary morphism that combines typed input arrows under a named composition rule. The einsum wiring is inferred from the typed signature, so the source spells out only which arrow plugs into which wire.

## QVR Source

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

> The SVI step counts and NUTS warmup, sample, and chain budgets in the snippets below are illustrative: each block is sized to run in tens of seconds and demonstrate the API surface. Production fits typically need 10x to 100x more SVI steps, longer NUTS warmup, and multiple chains to actually converge to the data-generating parameters.

The exported arrow is a deterministic score tensor materialized from the current latents. To fit the latents, the snippets below wrap the contraction in a Normal observation around the recomputed score: the likelihood re-runs the wiring on every evaluation, so gradients flow from the observed judgments back into the two embeddings and the interaction tensor. The compiled contraction is reachable through the [`Compiler`](../api/dsl/compiler.md#quivers.dsl.compiler.Compiler) environment's `contractions` mapping.

### Generating synthetic data

Draw ground-truth embeddings and an interaction tensor, push them through the bilinear form, and add Normal observation noise:

```python
import torch

torch.manual_seed(0)
n_item, d_pred, d_arg, n_judgment = 4, 2, 2, 3
P_true = torch.rand(n_item, d_pred)
A_true = torch.rand(n_item, d_arg)
W_true = torch.rand(d_pred, d_arg, n_judgment)
sigma  = 0.1
S_true = torch.einsum("ib,ic,bcd->id", P_true, A_true, W_true)
Y      = S_true + sigma * torch.randn(n_item, n_judgment)
print("scores:", S_true.round(decimals=2).tolist())
```

### SVI fit

Wrap the recomputed score in a Normal judgment likelihood inside a [`MonadicProgram`](../api/continuous/programs.md#quivers.continuous.programs.MonadicProgram), then fit with [`AutoNormalGuide`](../api/inference/guide.md#quivers.inference.guides.AutoNormalGuide) + [`ELBO`](../api/inference/elbo.md#quivers.inference.objectives.ELBO) + [`SVI`](../api/inference/svi.md#svi). The guide is degenerate here (no `sample` latent sites in the program), so the optimiser walks the embeddings and the interaction tensor as point variables.

```python
import torch
import torch.distributions as D
from quivers.dsl.parser import parse_file
from quivers.dsl.compiler import Compiler
from quivers.continuous.programs import MonadicProgram
from quivers.continuous.morphisms import ContinuousMorphism
from quivers.continuous.spaces import Euclidean
from quivers.core.objects import Unit
from quivers.inference import AutoNormalGuide, ELBO, SVI


class JudgmentLikelihood(ContinuousMorphism):
    """Normal observation around the recomputed bilinear score."""

    def __init__(self, contraction, pred, arg, inter, sigma=0.1):
        super().__init__(Unit, Euclidean(name="Judgment", dim=3))
        self._wiring = contraction.wiring
        self._pred, self._arg, self._inter = pred, arg, inter
        self._pred_mod = pred.module()
        self._arg_mod = arg.module()
        self._inter_mod = inter.module()
        self._sigma = sigma

    def _score(self):
        return self._wiring.apply(
            self._pred.tensor, self._arg.tensor, self._inter.tensor
        )

    def rsample(self, x, sample_shape=None):
        i = x[..., 0].long()
        mu = self._score()[i]
        return mu + self._sigma * torch.randn(mu.shape)

    def log_prob(self, x, y):
        i = x[..., 0].long()
        mu = self._score()[i]
        return D.Normal(mu, self._sigma).log_prob(y).sum(-1)


torch.manual_seed(0)
n_item, d_pred, d_arg, n_judgment = 4, 2, 2, 3
P_true = torch.rand(n_item, d_pred)
A_true = torch.rand(n_item, d_arg)
W_true = torch.rand(d_pred, d_arg, n_judgment)
sigma  = 0.1
S_true = torch.einsum("ib,ic,bcd->id", P_true, A_true, W_true)
Y      = S_true + sigma * torch.randn(n_item, n_judgment)

torch.manual_seed(1)
compiler = Compiler(parse_file("docs/examples/source/tensor_contraction.qvr"))
prog = compiler.compile()
score = compiler.contractions["bilinear_score"]
lik = JudgmentLikelihood(
    score,
    compiler.morphisms["pred_embed"],
    compiler.morphisms["arg_embed"],
    compiler.morphisms["interaction"],
    sigma=sigma,
)
inner = MonadicProgram(
    domain=Euclidean(name="Ix", dim=1),
    codomain=Euclidean(name="Judgment", dim=3),
    steps=[(("judgment",), lik, None, True)],
    return_vars=("judgment",),
)
x = torch.arange(n_item, dtype=torch.float32).unsqueeze(-1)
obs = {"judgment": Y}

guide = AutoNormalGuide(inner, observed_names={"judgment"})
optim = torch.optim.Adam(
    list(inner.parameters()) + list(guide.parameters()), lr=5e-2,
)
svi = SVI(inner, guide, optim, ELBO())
loss0 = svi.step(x, obs)
for _ in range(200):
    loss = svi.step(x, obs)
print(f"ELBO loss: {loss0:.2f} -> {loss:.2f}")
```

The factorisation is identifiable only up to invertible reparameterisations of the two embedding spaces, so the fitted embeddings may differ from the truths even when the score tensor has converged.

### NUTS posterior

The model exposes its latents as `[role=latent]` parameters with no explicit prior. To run NUTS we lift them into Normal-prior sample sites via [`bayesian_lift_parameters`](../api/inference/lifts.md#quivers.inference.lifts.bayesian_lift_parameters) and target the lifted Bayesian model:

```python
import torch
import torch.distributions as D
from quivers.dsl.parser import parse_file
from quivers.dsl.compiler import Compiler
from quivers.continuous.programs import MonadicProgram
from quivers.continuous.morphisms import ContinuousMorphism
from quivers.continuous.spaces import Euclidean
from quivers.core.objects import Unit
from quivers.inference import MCMC, NUTSKernel
from quivers.inference import bayesian_lift_parameters


class JudgmentLikelihood(ContinuousMorphism):
    def __init__(self, contraction, pred, arg, inter, sigma=0.1):
        super().__init__(Unit, Euclidean(name="Judgment", dim=3))
        self._wiring = contraction.wiring
        self._pred, self._arg, self._inter = pred, arg, inter
        self._pred_mod = pred.module()
        self._arg_mod = arg.module()
        self._inter_mod = inter.module()
        self._sigma = sigma

    def _score(self):
        return self._wiring.apply(
            self._pred.tensor, self._arg.tensor, self._inter.tensor
        )

    def rsample(self, x, sample_shape=None):
        i = x[..., 0].long()
        mu = self._score()[i]
        return mu + self._sigma * torch.randn(mu.shape)

    def log_prob(self, x, y):
        i = x[..., 0].long()
        mu = self._score()[i]
        return D.Normal(mu, self._sigma).log_prob(y).sum(-1)


torch.manual_seed(0)
n_item, d_pred, d_arg, n_judgment = 4, 2, 2, 3
P_true = torch.rand(n_item, d_pred)
A_true = torch.rand(n_item, d_arg)
W_true = torch.rand(d_pred, d_arg, n_judgment)
sigma  = 0.1
S_true = torch.einsum("ib,ic,bcd->id", P_true, A_true, W_true)
Y      = S_true + sigma * torch.randn(n_item, n_judgment)

torch.manual_seed(2)
compiler = Compiler(parse_file("docs/examples/source/tensor_contraction.qvr"))
prog = compiler.compile()
score = compiler.contractions["bilinear_score"]
lik = JudgmentLikelihood(
    score,
    compiler.morphisms["pred_embed"],
    compiler.morphisms["arg_embed"],
    compiler.morphisms["interaction"],
    sigma=sigma,
)
inner = MonadicProgram(
    domain=Euclidean(name="Ix", dim=1),
    codomain=Euclidean(name="Judgment", dim=3),
    steps=[(("judgment",), lik, None, True)],
    return_vars=("judgment",),
)
x = torch.arange(n_item, dtype=torch.float32).unsqueeze(-1)
obs = {"judgment": Y}

lifted, lx, lobs = bayesian_lift_parameters(inner, x, obs, prior_scale=1.0)
kernel = NUTSKernel(step_size=0.05, max_tree_depth=3, target_accept=0.8)
mc     = MCMC(kernel, num_warmup=10, num_samples=10, num_chains=1)
result = mc.run(lifted, lx, lobs)

print("acceptance:", float(result.acceptance_rates.mean()))
print("divergences:", int(result.divergence_counts.sum()))
```

## Categorical Perspective

Binary composition `>>` is the 2-ary case of a wider operadic structure: a `contraction` denotes an n-ary morphism in the [multicategory](https://ncatlab.org/nlab/show/multicategory) of tensor spaces over the active composition rule, with the wiring specification fixing which axes are joined and which survive to the output (see [Composition Rules § 4](../semantics/composition-rules.md#4-operadic-contractions) for the denotation). The bilinear score is the case $n = 3$: two argument tensors over a shared `Item` axis and a kernel over the two embedding axes, folded under $\otimes$ and joined under $\bigoplus$ of the sum-product semiring. Chains of binary `>>` can express only tree-shaped contractions of matrices; the third-order `interaction` tensor makes this join genuinely operadic.

## See Also

- [DSL Contractions](../guides/dsl-contractions.md) for the declaration surface, wiring inference, and the `share=` / `wiring=` clauses.
- [Composition Rules § 4](../semantics/composition-rules.md#4-operadic-contractions) for the categorical semantics of operadic contractions.
- [Probabilistic Matrix Factorization](pmf.md) for the 2-ary bilinear score expressed with `.dagger` and `>>`.

## References

- Richard Socher, Danqi Chen, Christopher D. Manning, and Andrew Y. Ng. 2013. [Reasoning With Neural Tensor Networks for Knowledge Base Completion](https://papers.nips.cc/paper/2013/hash/b337e84de8752b27eda3a12363109e80-Abstract.html). In *Advances in Neural Information Processing Systems 26*.
