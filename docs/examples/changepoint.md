# Bayesian Change-Point Detection

## Overview

A canonical Bayesian [change-point model](https://doi.org/10.2307/2347570) in the rate of a [Poisson](https://en.wikipedia.org/wiki/Poisson_distribution) series, with a single switch at an unknown time `tau`. The classical application is the British coal-mining disaster series: the rate of accidents per year drops at an estimated change-point in the late nineteenth century. Two rates, one before and one after the switch, are tied together by a soft indicator parameterized as $s(t) = \sigma(k \cdot (t - \tau))$, which is the standard differentiable relaxation of the indicator $\mathbf{1}\{t > \tau\}$ and lets gradient-based variational inference fit the change-point location directly.

## QVR Source

```qvr
object Step : FinSet 100

program changepoint : Step -> Step
    sample tau <- Uniform(0.0, 100.0)
    sample rate_before <- Gamma(2.0, 1.0)
    sample rate_after <- Gamma(2.0, 1.0)

    let s = sigmoid(20.0 * (t - tau))
    let rate = (1.0 - s) * rate_before + s * rate_after

    observe y : Step <- Poisson(rate)
    return tau

export changepoint
```

## Walkthrough

`tau` is the change-point time with a uniform prior on the observation window. `rate_before` and `rate_after` are Gamma-prior intensities, each centered on a prior mean of 2 events per period. The identifier `t` is exogenous host-data: it is never declared inside the program, so the runtime resolves it from the observations dict at trace time, where the caller supplies the per-step time vector. The soft indicator $s = \sigma(20 (t - \tau))$ with steepness 20 gives a sharp but differentiable switch; the per-step rate is the convex combination of the two regime rates, and the observation `y : Step <- Poisson(rate)` is the per-step Poisson likelihood.

For a strict hard change-point with integer-valued `tau`, replace the soft indicator with a scoped `marginalize tau_idx : Tau <- Categorical(uniform) [reduction=logsumexp]` block whose indented body emits the per-step likelihood. The runtime then log-sum-exps the per-step likelihood over the `Tau` axis, realizing the exact Bayesian change-point posterior at the cost of a $|\mathsf{Tau}|$-fold widening of the inner loop.

## Try it

> The SVI step counts and NUTS warmup, sample, and chain budgets in the snippets below are illustrative: each block is sized to run in tens of seconds and demonstrate the API surface. Production fits typically need 10x to 100x more SVI steps, longer NUTS warmup, and multiple chains to actually converge to the data-generating parameters.


### Generating synthetic data

Pick concrete ground-truth rates and a change-point location, then forward-sample a Poisson count series whose intensity jumps from `rate_before` to `rate_after` at `tau_true`. The per-step time vector `t` is the host-data the program reads from the observations dict.

```python
import torch
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/changepoint.qvr")
model = prog.morphism

T = 64
tau_true = 32
rate_before, rate_after = 2.0, 5.0
y = torch.cat([
    torch.poisson(torch.ones(tau_true) * rate_before),
    torch.poisson(torch.ones(T - tau_true) * rate_after),
])
t_vec = torch.arange(T, dtype=torch.float32)
x_in = torch.zeros(T, 1)
observations = {"y": y, "t": t_vec}
```

### SVI fit

Re-initialise the program and fit `tau`, `rate_before`, and `rate_after` by maximising the ELBO. The oracle log-likelihood at the true rates (piecewise-constant indicator at `tau_true`) is reported for reference.

```python
import torch.distributions as D
from quivers.inference import AutoNormalGuide, ELBO, SVI

torch.manual_seed(0)
prog = load("docs/examples/source/changepoint.qvr")
model = prog.morphism

guide = AutoNormalGuide(model, observed_names={"y"})
optim = torch.optim.Adam(
    list(model.parameters()) + list(guide.parameters()), lr=1e-2,
)
svi = SVI(model, guide, optim, ELBO())

loss0 = svi.step(x_in, observations)
losses = [svi.step(x_in, observations) for _ in range(800)]
loss_final = sum(losses[-20:]) / 20.0
rates_true = torch.where(
    t_vec < tau_true, torch.tensor(rate_before), torch.tensor(rate_after),
)
oracle_ll = D.Poisson(rates_true).log_prob(y).sum().item()
print(f"initial ELBO loss: {loss0:.1f}")
print(f"final ELBO loss:   {loss_final:.1f}")
print(f"oracle -log p(y):  {-oracle_ll:.1f}")
```

### NUTS posterior

The model declares explicit `sample` sites for `tau`, `rate_before`, and `rate_after`, so [`NUTSKernel`](../api/inference/mcmc.md#quivers.inference.mcmc.NUTSKernel) samples them directly. Small warmup and sample budgets keep the snippet inside tens of seconds while still producing a usable posterior summary.

```python
from quivers.inference import MCMC, NUTSKernel

torch.manual_seed(0)
prog = load("docs/examples/source/changepoint.qvr")
model = prog.morphism

kernel = NUTSKernel(step_size=0.05, max_tree_depth=3, target_accept=0.8)
mc = MCMC(kernel, num_warmup=15, num_samples=15, num_chains=1)
result = mc.run(model, x_in, observations)

print("acceptance:", float(result.acceptance_rates.mean()))
print("divergences:", int(result.divergence_counts.sum()))
for name, samples in result.samples.items():
    print(f"{name}: posterior mean = {samples.mean().item():.3f}")
```


## Categorical Perspective

The model is a Kleisli morphism in the [Giry monad](https://doi.org/10.1007/BFb0092872)'s Kleisli category whose denotation is the joint $p(\tau, r_1, r_2) \prod_t \mathrm{Poisson}(y_t \mid r(t; \tau, r_1, r_2))$. The soft indicator is the [reparameterisation gradient](https://doi.org/10.48550/arXiv.1312.6114)-friendly relaxation of an indicator function; in the hard-change-point variant, the discrete `tau_idx` is integrated out by `marginalize`, realizing the right Kan extension along the trivial projection from the time axis to a singleton.


## References

- Bradley P. Carlin, Alan E. Gelfand, and Adrian F. M. Smith. 1992. Hierarchical Bayesian analysis of changepoint problems. *Journal of the Royal Statistical Society Series C: Applied Statistics*, 41(2):389–405.
- Diederik P. Kingma and Max Welling. 2013. Auto-Encoding Variational Bayes. arXiv preprint arXiv:1312.6114.
- Michèle Giry. 1982. A categorical approach to probability theory. In Bernhard Banaschewski, editor, *Categorical Aspects of Topology and Analysis*, volume 915 of *Lecture Notes in Mathematics*, pages 68–85. Springer, Berlin, Heidelberg.
