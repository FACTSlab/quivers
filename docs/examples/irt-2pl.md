# Two-Parameter Logistic IRT

## Overview

The 2PL [item response theory](https://en.wikipedia.org/wiki/Item_response_theory) model ([Birnbaum 1968](https://doi.org/10.1007/BF02288467)) for binary item responses `y_{ij}` of respondent i to item j. Each respondent carries a unidimensional ability `theta_i`, each item carries a difficulty `b_j` and a discrimination `a_j` (positive by construction via a [LogNormal](https://en.wikipedia.org/wiki/Log-normal_distribution) prior), and the probability of a correct response is `sigmoid(a_j * (theta_i - b_j))`.

## QVR Source

```qvr
object Person : 500
object Item : 30
object Resp : 15000

program irt_2pl : Resp -> Resp
    ability : Person <- Normal(0.0, 1.0)
    difficulty : Item <- Normal(0.0, 1.0)
    discrim : Item <- LogNormal(0.0, 1.0)

    let theta = ability[person_idx]
    let b = difficulty[item_idx]
    let a = discrim[item_idx]

    let eta = a * (theta - b)
    let p = sigmoid(eta)

    observe y : Resp <- Bernoulli(p)
    return p

export irt_2pl
```

## Walkthrough

`ability`, `difficulty`, and `discrim` are plate-bound on `Person` and `Item`; `discrim` carries a LogNormal prior so it's positive by construction. The runtime supplies `person_idx` and `item_idx` at fit time, and the gather idiom `ability[person_idx]` / `difficulty[item_idx]` / `discrim[item_idx]` realizes the standard cross-classified design. The Bernoulli link `sigmoid(a * (theta - b))` is the canonical [logistic](https://en.wikipedia.org/wiki/Logistic_function) form of the 2PL response function.

## Try it

```python
import torch
from quivers.dsl import load
from quivers.inference import AutoNormalGuide, ELBO, SVI

torch.manual_seed(0)

prog = load("docs/examples/source/irt_2pl.qvr")
model = prog.morphism

n_person, n_item, n_resp = 500, 30, 15000
ability_true = torch.randn(n_person)
difficulty_true = torch.randn(n_item)
discrim_true = torch.randn(n_item).exp()
person_idx = torch.randint(0, n_person, (n_resp,))
item_idx = torch.randint(0, n_item, (n_resp,))
eta = discrim_true[item_idx] * (ability_true[person_idx] - difficulty_true[item_idx])
y = torch.bernoulli(torch.sigmoid(eta))

guide = AutoNormalGuide(model, observed_names={"y"})
optim = torch.optim.Adam(
    list(model.parameters()) + list(guide.parameters()), lr=5e-3,
)
svi = SVI(model, guide, optim, ELBO())
for _ in range(2000):
    svi.step(torch.zeros(n_resp, 1), {
        "person_idx": person_idx, "item_idx": item_idx, "y": y,
    })

ability_fit = guide._loc("ability").detach()
print("ability corr:", torch.corrcoef(
    torch.stack([ability_fit.squeeze(-1), ability_true])
)[0, 1].item())
```

## Categorical Perspective

The 2PL is a Kleisli morphism over the Person + Item plate structure in the [Giry monad](https://doi.org/10.1007/BFb0092872)'s Kleisli category. The plate-gather operations are pullbacks of indexed kernels along the response-row index maps `person_idx : Resp -> Person` and `item_idx : Resp -> Item`.
