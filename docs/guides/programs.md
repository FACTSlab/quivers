# Monadic Programs

## What is a MonadicProgram?

A `MonadicProgram` is a probabilistic program specified as a sequence of draw and let steps. It defines a `ContinuousMorphism` from a domain to a codomain via monadic composition (Kleisli bind).

The program syntax mirrors probabilistic programming languages (PDS, Pyro):

```
program name : domain -> codomain
    draw x₁ ~ morphism_1
    draw x₂ ~ morphism_2(x₁)
    let y = x₁ + x₂
    observe z ~ morphism_3(y)
    return y
```

Each draw step samples from a conditional distribution, binding the result. The observe keyword conditions the program on an external observation.

## Program Structure

A program is an `nn.Module` that, when called, executes forward ancestral sampling:

```python
from quivers.continuous.programs import MonadicProgram
from quivers.continuous.families import ConditionalNormal
from quivers.core.objects import FinSet
import torch

# Build program manually
program = MonadicProgram(
    domain=FinSet("input", 5),
    codomain=FinSet("output", 3),
)

# Register morphisms
f = ConditionalNormal(...)
program.add_morphism("f", f)

# Add steps
program.add_draw("x", "f", args=None)
program.add_draw("y", "f", args=("x",))
program.add_return("y")

# Forward pass: sampling
samples = program.rsample(
    torch.randn(5), sample_shape=torch.Size([100])
)  # shape (100, 3)

# Log joint: sum_i log p(z_i | pa(z_i)) given every bound variable
log_joint = program.log_joint(input_data, {"x": x_val, "y": y_val})
```

## Draw Steps

A draw step `draw x ~ f` or `draw x ~ f(y, z)` samples from a morphism, optionally conditioned on previous variables.

Single draw:

```
draw x ~ prior_f
```

Conditioned draw:

```
draw y ~ likelihood_f(x)
```

Multiple arguments (stacked along feature dimension):

```
draw (x, y) ~ joint_f(z, w)
```

The variable names on the left side are bound in the environment.

## Let Steps

Deterministic binding:

```
let x = y + z
let weight = 0.5
```

Supports literals, variable references, and simple callable expressions.

## Observe Keyword

Condition the program on an observation:

```
observe y ~ likelihood(x)
```

This marks `y` as conditioned. During inference, observations clamp these variables to external values.

## Return Statement

Specify the program output. Single or tuple:

```
return x
```

```
return (x, y, z)
```

The return value's shape determines the codomain.

## Domains and Codomains

Domains can be:
- A single `FinSet` or `ContinuousSpace`
- A product of sets/spaces: `X * Y * Z`
- Named parameters: the domain is the product, but variables can refer to sub-components

Codomains are determined by the return statement shape.

## ReSample and Log Joint

Two key operations:

### rsample(x, sample_shape=(), observations=None)

Generate samples by executing the program:

```python
x = torch.randn(5)
samples = program.rsample(x, sample_shape=torch.Size([1000]))
# shape: (1000, codomain_dim)
```

Sequential ancestral sampling: each draw step samples, previous draws are available to subsequent steps. `observations` is an optional `dict[str, torch.Tensor]` clamping observed sites to runtime data.

### log_joint(x, intermediates)

Compute $\log p(z_1, \ldots, z_k | x) = \sum_i \log p(z_i \mid \mathrm{pa}(z_i))$ given all bound-variable values:

```python
x = torch.randn(5)
intermediates = {"z": z_value, "y": y_value}  # every bound variable

log_pjoint = program.log_joint(x, intermediates)
```

`log_joint` is the core kernel summed across the program's draw / plate-draw / observe steps, used inside `ELBO.forward` after the guide samples latents.

### The `observations` dict

Vectorised-observe steps (`observe r[n] ~ F(args) for n in N`) read their response buffers from a runtime `observations: dict[str, torch.Tensor]`, keyed by the observed-variable name. The dict is passed as the `observations` kwarg to `MonadicProgram.rsample` and as the final positional argument to `ELBO.forward` / `SVI.step`:

```python
observations = {
    "cloze_resp": cloze_tensor,    # shape (n_cloze_resp,)
    "prop_resp":  prop_tensor,     # shape (n_prop_resp,)
}

samples = program.rsample(x, observations=observations)
loss = elbo(model, guide, x, observations)
```

There is no `.qvr`-level data block; the tensor sources live in Python at the call site, and the keys must match the response identifiers declared in the program body.

## Named Parameters

If the domain is a product, define sub-domains:

```python
program = MonadicProgram(
    domain=FinSet("A", 3) * FinSet("B", 4),
    codomain=FinSet("Z", 5),
)

program.add_param("a", FinSet("A", 3))
program.add_param("b", FinSet("B", 4))

# Now steps can reference a, b by name
program.add_draw("x", "f", args=("a", "b"))
```

## Example: A Simple Model

```python
from quivers.continuous.programs import MonadicProgram
from quivers.continuous.families import (
    ConditionalNormal,
    ConditionalLogitNormal,
)
from quivers.core.objects import Unit
from quivers.continuous.spaces import Euclidean
import torch.nn as nn

# Build a linear regression model
prior_mu = nn.Linear(1, 1)
prior_sigma = nn.Linear(1, 1)
likelihood_sigma = nn.Linear(1, 1)

program = MonadicProgram(
    domain=Unit,
    codomain=Euclidean(1),
)

# Prior on μ
f_mu = ConditionalNormal(Unit, Euclidean(1))
program.add_morphism("prior_mu", f_mu)

# Prior on σ
f_sigma = ConditionalLogitNormal(Unit, Euclidean(1))
program.add_morphism("prior_sigma", f_sigma)

# Likelihood
f_like = ConditionalNormal(Euclidean(2), Euclidean(1))
program.add_morphism("likelihood", f_like)

# Steps
program.add_draw("mu", "prior_mu")
program.add_draw("sigma", "prior_sigma")
program.add_draw("y", "likelihood", args=("mu", "sigma"))
program.add_return("y")

# Use for inference
optimizer = torch.optim.Adam(program.parameters())
```

## Destructuring Draws

Extract multiple values from a tuple-returning sub-program:

```
program sub : X -> Y * Y
    draw (a, b) ~ some_morphism
    return (a, b)

program main : X -> Z
    draw (u, v) ~ sub
    draw w ~ g(u, v)
    return w
```

The pattern `(u, v) ~ sub` destructures the output.

## Observation Clamping

During inference, the condition() function clamps observations:

```python
from quivers.inference import condition

# Condition program on external observations
observed_y = torch.tensor([1.0, -0.5, 2.0])

conditioned = condition(program, {"y": observed_y})

# Forward pass on conditioned program uses the clamped value
log_pjoint = conditioned.log_joint(x, observed_y)
```

## Product Domains and Outputs

For multiple domain inputs, stack along the feature dimension:

```
program f : (X * Y) -> Z
    draw (x_val, y_val) from domain input
    draw z ~ g(x_val, y_val)
    return z
```

Internally, the domain tensor is reshaped to match.

## Integration with DSL

MonadicPrograms are the output of `.qvr` DSL compilation (see DSL guide). The DSL parser translates:

```qvr
object X : 3
object Y : 4

program my_prog : X -> Y
    draw mu ~ LogitNormal(0, 1)
    draw x ~ Normal(mu, 1)
    return x

output my_prog
```

into a MonadicProgram instance that can be trained.

## Hierarchical Models with Parametric Templates

A parametric program declares a reusable kernel template polymorphic over typed parameters (`FinSet`, `Space`, `Object`, `Real`, `Nat`, or `Mor[A, B]`). Each call site `draw v ~ template(...)` inlines a fresh α-renamed copy of the template's body, so call sites contribute distinct latents.

```qvr
object Subject : 200
object Verb : 100
object Resp : 5000

program random_intercepts (G : FinSet, scale : Real) : G -> 1
    draw sigma ~ HalfNormal(scale)
    draw v : G -> 1 ~ Normal(0.0, sigma)
    return v

program crossed : Resp -> Resp
    draw intercept ~ Normal(0.0, 1.0)

    draw by_subject ~ random_intercepts(Subject, 1.0)
    draw by_verb    ~ random_intercepts(Verb,    1.0)

    observe response[n] ~ Bernoulli(intercept) for n in Resp
    return intercept

output crossed
```

Each `random_intercepts` call inlines an independent `sigma` and per-level plate; the observed response is the runtime tensor supplied via `observations={"response": response_tensor}`. Monotone ordinal effects are expressed as `cumsum` of `HalfNormal` increments (positive support ⇒ monotone partial sums); discrete latent classes are integrated out with `marginalize`.

```python
from quivers.dsl import load
from quivers.inference import ELBO, AutoNormalGuide, SVI

program = load("crossed.qvr")
model = program.morphism  # underlying MonadicProgram
observations = {"response": response_tensor}

guide = AutoNormalGuide(model, observed_names={"response"})
elbo  = ELBO(num_particles=1)
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(guide.parameters()), lr=1e-2,
)
svi = SVI(model, guide, optimizer, elbo)

for _ in range(2000):
    svi.step(domain_input, observations)
```
