# Monadic Programs

## What is a MonadicProgram?

A
[`MonadicProgram`](../api/continuous/programs.md#quivers.continuous.programs.MonadicProgram)
is a probabilistic program specified as a sequence of bind and let
steps. It defines a
[`ContinuousMorphism`](../api/continuous/morphisms.md#quivers.continuous.morphisms.ContinuousMorphism)
from a domain to a codomain via [monadic
composition](https://ncatlab.org/nlab/show/Kleisli+category)
(Kleisli bind).

The program syntax mirrors probabilistic programming languages
(Pyro, NumPyro, Stan):

```
program name : domain -> codomain
    x_1 <- morphism_1
    x_2 <- morphism_2(x_1)
    let y = x_1 + x_2
    observe z <- morphism_3(y)
    return y
```

Each `<-` bind samples from a conditional distribution, binding the
result. The `observe` keyword conditions the program on an external
observation.

## Program structure

A program is an
[`nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)
that, when called, executes forward ancestral sampling:

```python
from quivers.continuous.programs import MonadicProgram
from quivers.continuous.families import ConditionalNormal
from quivers.continuous.spaces import Euclidean
from quivers.core.objects import FinSet
import torch

Unit = FinSet(name="Unit", cardinality=1)
R1 = Euclidean(name="R1", dim=1)

prior      = ConditionalNormal(Unit, R1)            # x ~ Normal(0, 1)
likelihood = ConditionalNormal(R1, R1)              # y ~ Normal(x, 1)

program = MonadicProgram(
    Unit,
    R1,
    steps=[
        (("x",), prior, None),                      # x <- prior
        (("y",), likelihood, ("x",)),               # y <- likelihood(x)
    ],
    return_vars=("y",),
)

# Forward pass: sampling
samples = program.rsample(
    torch.zeros(4, dtype=torch.long), sample_shape=torch.Size([100])
)  # shape (100, 4, 1)

# Log joint: sum_i log p(z_i | pa(z_i)) given every bound variable
log_joint = program.log_joint(
    torch.zeros(4, dtype=torch.long),
    {"x": x_val, "y": y_val},
)
```

## Bind steps

A bind step `x <- f` or `x <- f(y, z)` samples from a morphism,
optionally conditioned on previous variables.

Single bind:

```
x <- prior_f
```

Conditioned bind:

```
y <- likelihood_f(x)
```

Destructuring tuple bind (stacked along feature dimension):

```
(x, y) <- joint_f(z, w)
```

The variable names on the left side are bound in the environment.
An indexed bind `v : A <- F(args)` declares `v` as an
$A$-indexed plate of independent draws.

## Let steps

Deterministic binding:

```
let x = y + z
let weight = 0.5
```

Supports literals, variable references, arithmetic, and the
let-expression primitive surface; see
[DSL Programs and Let-Expressions](dsl-programs-and-lets.md#let-expressions-arithmetic-and-primitives)
for the full primitive list.

## Observe keyword

Condition the program on an observation:

```
observe y <- likelihood(x)
```

This marks `y` as conditioned. During inference, observations
clamp these variables to external values. An indexed-observe
`observe r : N <- F(args)` accumulates a batched likelihood over
the index set `N`, with the response buffer supplied via the
runtime `observations` dict.

## Return statement

Specify the program output. Single or tuple:

```
return x
```

```
return (x, y, z)
```

The return value's shape determines the codomain. Tuples are
bare-positional; the resulting product space's components are
ordered by tuple position.

## Domains and codomains

Domains can be:

- A single [`FinSet`](../api/core/objects.md) or
  [`ContinuousSpace`](../api/continuous/spaces.md).
- A product of sets / spaces: `X * Y * Z`.
- Named parameters: the domain is the product, but variables can
  refer to sub-components.

Codomains are determined by the return statement shape.

## rsample and log_joint

Two key operations on a compiled program:

### rsample(x, sample_shape=(), observations=None)

Generate samples by executing the program:

```python
x = torch.randn(5)
samples = program.rsample(x, sample_shape=torch.Size([1000]))
# shape: (1000, codomain_dim)
```

Sequential ancestral sampling: each draw step samples, previous
draws are available to subsequent steps. `observations` is an
optional `dict[str, torch.Tensor]` clamping observed sites to
runtime data.

### log_joint(x, intermediates)

Compute $\log p(z_1, \ldots, z_k \mid x) = \sum_i \log p(z_i \mid
\mathrm{pa}(z_i))$ given all bound-variable values:

```python
x = torch.randn(5)
intermediates = {"z": z_value, "y": y_value}  # every bound variable

log_pjoint = program.log_joint(x, intermediates)
```

`log_joint` is the core kernel summed across the program's draw /
plate-draw / observe steps, used inside
[`ELBO.forward`](../api/inference/elbo.md#quivers.inference.objectives.ELBO)
after the guide samples latents.

### The `observations` dict

Indexed-observe steps (`observe r : N <- F(args)`) read their
response buffers from a runtime
`observations: dict[str, torch.Tensor]`, keyed by the
observed-variable name. The dict is passed as the `observations`
kwarg to
[`MonadicProgram.rsample`](../api/continuous/programs.md#quivers.continuous.programs.MonadicProgram.rsample)
and as the final positional argument to
[`ELBO.forward`](../api/inference/elbo.md#quivers.inference.objectives.ELBO) /
[`SVI.step`](../api/inference/svi.md#quivers.inference.svi.SVI.step):

```python
observations = {
    "cloze_resp": cloze_tensor,    # shape (n_cloze_resp,)
    "prop_resp":  prop_tensor,     # shape (n_prop_resp,)
}

samples = program.rsample(x, observations=observations)
loss = elbo(model, guide, x, observations)
```

There is no `.qvr`-level data block; the tensor sources live in
Python at the call site, and the keys must match the response
identifiers declared in the program body.

## Named parameters

If the domain is a product, name the components via the `params`
argument so steps can reference them by name:

```python
A = FinSet(name="A", cardinality=3)
B = FinSet(name="B", cardinality=4)
Z = FinSet(name="Z", cardinality=5)

program = MonadicProgram(
    A * B,
    Z,
    steps=[
        (("x",), f, ("a", "b")),                    # x <- f(a, b)
    ],
    return_vars=("x",),
    params=("a", "b"),
)
```

The program splits the product input along the feature axis at
runtime and binds each slice to the corresponding name in `params`.

## Example: a simple model

```python
from quivers.continuous.programs import MonadicProgram
from quivers.continuous.families import (
    ConditionalNormal,
    ConditionalLogitNormal,
)
from quivers.continuous.spaces import Euclidean
from quivers.core.objects import FinSet

Unit = FinSet(name="Unit", cardinality=1)
R1 = Euclidean(name="R1", dim=1)
R2 = Euclidean(name="R2", dim=2)

prior_mu    = ConditionalNormal(Unit, R1)
prior_sigma = ConditionalLogitNormal(Unit, R1)
likelihood  = ConditionalNormal(R2, R1)

program = MonadicProgram(
    Unit,
    R1,
    steps=[
        (("mu",),    prior_mu,    None),
        (("sigma",), prior_sigma, None),
        (("y",),     likelihood,  ("mu", "sigma")),
    ],
    return_vars=("y",),
)

# Use for inference
optimizer = torch.optim.Adam(program.parameters())
```

## Destructuring binds

Extract multiple values from a tuple-returning sub-program:

```
program sub : X -> Y * Y
    (a, b) <- some_morphism
    return (a, b)

program main : X -> Z
    (u, v) <- sub
    w <- g(u, v)
    return w
```

The pattern `(u, v) <- sub` destructures the output.

## Observation clamping

During inference, the
[`condition()`](../api/inference/conditioning.md#quivers.inference.conditioning.condition)
function clamps observations:

```python
from quivers.inference import condition

# Condition program on external observations
observed_y = torch.tensor([1.0, -0.5, 2.0])

conditioned = condition(program, {"y": observed_y})

# Trace under the conditioning: observed sites are clamped to the data
tr = conditioned.trace(x)
```

## Product domains and outputs

For multiple domain inputs, stack along the feature dimension:

```
program f(x_val, y_val) : (X * Y) -> Z
    z <- g(x_val, y_val)
    return z
```

The bare-identifier parameters `x_val`, `y_val` name the
projections of the product domain. Internally, the domain tensor
is reshaped to match.

## Integration with the DSL

MonadicPrograms are the output of `.qvr` DSL compilation. The DSL
parser translates:

<!-- compile: false -->
```qvr
object X : 3
object Y : 4

program my_prog : X -> Y
    mu <- LogitNormal(0, 1)
    x <- Normal(mu, 1)
    return x

export my_prog
```

into a `MonadicProgram` instance that can be trained. The full DSL
surface for programs lives in
[DSL Programs and Let-Expressions](dsl-programs-and-lets.md).

## Where to next

- [Hierarchical Programs](programs-hierarchical.md): parametric
  templates for crossed random intercepts, monotone-spline
  coefficients, and the grouped marginalization construct for
  fibred discrete latents.
- [Variational Inference](inference-foundations.md): how programs
  feed into the variational training loop.
