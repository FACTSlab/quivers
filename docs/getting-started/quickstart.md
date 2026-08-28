# Quickstart

This guide walks through the core concepts of quivers with concrete examples: creating objects, building morphisms, composing them, and running them as differentiable programs.

## 1. Basic Morphisms

A **morphism** is a $\mathcal{V}$-enriched relation: a function from a pair of objects to an algebra (lattice of truth values). In quivers, morphisms are tensors.

Create finite sets:

```python
from quivers import FinSet, morphism, observed
import torch

X = FinSet(name="X", cardinality=3)
Y = FinSet(name="Y", cardinality=4)
```

Create a **latent** (learnable) morphism from X to Y. Its tensor entries are parameters:

```python
f = morphism(X, Y)
print(f.tensor.shape)  # torch.Size([3, 4])
print(f.domain.name, "->", f.codomain.name)  # X -> Y
```

Create an **observed** (fixed) morphism with a fixed tensor:

```python
data = torch.tensor([
    [1.0, 0.5, 0.0, 0.2],
    [0.0, 1.0, 0.8, 0.1],
    [0.3, 0.2, 1.0, 0.9],
])
g = observed(X, Y, data)
print(g.tensor)  # your data tensor
```

Get the underlying `nn.Module` for use in training:

```python
mod = f.module()
params = list(mod.parameters())
print(len(params))  # 1 parameter matrix
```

## 2. Composition

Compose morphisms with the `>>` operator. The domain of the second must match the codomain of the first:

```python
Z = FinSet(name="Z", cardinality=2)
h = morphism(Y, Z)

# Compose f: X -> Y and h: Y -> Z to get X -> Z
composed = f >> h
print(composed.tensor.shape)  # torch.Size([3, 2])
```

Composition is **lazy**: it builds a computation graph. The final tensor is only materialized on evaluation.

## 3. Programs

Wrap a morphism as a differentiable `nn.Module` with `Program`:

```python
from quivers import FinSet, morphism, Program

X = FinSet(name="X", cardinality=3)
Y = FinSet(name="Y", cardinality=4)
Z = FinSet(name="Z", cardinality=2)
composed = morphism(X, Y) >> morphism(Y, Z)

program = Program(composed)
output = program()
print(output.shape)  # torch.Size([3, 2])
```

Integrate with a training loop:

```python
import torch.optim as optim

optimizer = optim.Adam(program.parameters(), lr=0.01)

for epoch in range(10):
    optimizer.zero_grad()
    output = program()
    # Define a loss and backpropagate
    loss = output.sum()
    loss.backward()
    optimizer.step()
```

## 4. Monadic Programs (Continuous)

For probabilistic computations with continuous and discrete variables, use [**monadic programs**](../api/continuous/programs.md). A program is a tuple `(domain, codomain, steps, return_vars)` where each step binds a sample, observation, or let-expression into a named variable; the final `return_vars` names the variables returned in the program's output tuple.

```python
from quivers import FinSet
from quivers.continuous.spaces import Euclidean
from quivers.continuous.families import ConditionalNormal
from quivers.continuous.programs import MonadicProgram
import torch

# Define spaces.
X = FinSet(name="context", cardinality=3)
R = Euclidean(name="response", dim=2)

# Conditional normal family: each input x ∈ X yields a 2D normal on R.
family = ConditionalNormal(X, R, hidden_dim=16)

# A one-step program: bind y ∼ family(x), return y.
program = MonadicProgram(
    domain=X,
    codomain=R,
    steps=[(("y",), family, None)],
    return_vars=("y",),
)

x = torch.tensor([0, 1, 2])
output = program.rsample(x)        # reparameterized samples, shape (3, 2)
```

## 5. The DSL

Write categorical programs declaratively in `.qvr` files:

```qvr
composition product_fuzzy [level=algebra]
object X : FinSet 3
object Y : FinSet 4
object Z : FinSet 2
morphism f : X -> Y [role=latent]
morphism g : Y -> Z [role=latent]
define composed = f >> g
export composed
```

Load and run:

```python
from quivers.dsl import load

prog = load("docs/examples/source/bayesian_regression.qvr")
```

Or use `loads` for inline strings:

```python
from quivers.dsl import loads

source = """
composition product_fuzzy [level=algebra]
object X : FinSet 3
object Y : FinSet 4
object Z : FinSet 2
morphism f : X -> Y [role=latent]
morphism g : Y -> Z [role=latent]
define composed = f >> g
export composed
"""

prog = loads(source)
output = prog()
print(output.shape)  # torch.Size([3, 2])
```

Supported DSL operators:

| Operator | Meaning | Example |
|----------|---------|---------|
| `>>` | composition | `f >> g` |
| `>>>` | transformation composition | `softmax(B) >>> expectation` |
| `@` | tensor product | `f @ g` |
| `.marginalize(X)` | marginalize over object | `f.marginalize(X)` |
| `.change_base(t)` | change of base under transformation `t` | `f.change_base(softmax(B))` |
| `identity(X)` | identity morphism | `morphism id : X -> X [role=observed] ~ identity(X)` |

The DSL also has surface for monadic probabilistic programs (`program ... [effects = [Sample, Score]]`), composition rules at four algebraic levels via `composition NAME [level=<algebra | semigroupoid | bilinear_form | rule>]`, and operadic contractions (`contraction op : ... [rule = R, wiring = "..."]`). The [QVR tutorial](../tutorials/qvr/01-first-model.md) walks through the full API.

## 6. Stochastic Morphisms

The **FinStoch** category models Markov kernels on finite sets:

```python
from quivers import (
    stochastic, FinSet, DiscretizedNormal,
    condition, ConditionedMorphism
)
import torch

X = FinSet(name="X", cardinality=3)
Y = FinSet(name="Y", cardinality=4)

# Create a stochastic morphism (Markov kernel)
kern = stochastic(X, Y)
print(kern.tensor.shape)  # [3, 4], rows sum to 1
print(kern.tensor.sum(dim=-1))  # all 1.0

# Condition on an observation
obs_data = torch.tensor([0.1, 0.8, 0.05, 0.05])  # P(Y)
conditioned = condition(kern, obs_data)
print(conditioned.tensor.shape)  # [3, 4]
```

## 7. Enriched Structures

Work with more abstract categorical structures:

```python
from quivers import (
    FuzzyPowersetMonad, KleisliCategory, FinSet, morphism,
)

# Create a monad
monad = FuzzyPowersetMonad()

# Work in its Kleisli category
kleisli = KleisliCategory(monad)

X = FinSet(name="X", cardinality=3)
Y = FinSet(name="Y", cardinality=4)

f = morphism(X, Y)
g = morphism(Y, X)

# Kleisli composition
comp = kleisli.compose(f, g)
print(comp.tensor.shape)  # [3, 3]
```

## 8. Interactive exploration

`pip install 'quivers[repl,lsp]'` adds two interactive surfaces over
the same elaborator the library uses internally.

### The REPL

```bash
qvr repl docs/examples/source/seq2seq.qvr
```

A four-pane Textual TUI: input editor (top-left), output log (mid-left),
environment browser (right, click any leaf to `:info` it), watches and
diagnostics strips at the bottom (auto-hidden when empty), plus a
status bar showing the loaded file, active algebra, and binding
counts.

GHCi-shaped meta-commands:

```
qvr> :type seq2seq
seq2seq :: Source * Target -> Target

qvr> :info backbone
define backbone = (encoder @ decoder) >> cross
-- declared at docs/examples/source/seq2seq.qvr:59:0

qvr> :watch backbone           # pinned re-eval after every recompile
qvr> :save my_edits.qvr        # write the live module to disk
```

`Ctrl-G` (or `Ctrl-O`, `F8`) evaluates; `Tab` cycles completions;
`Ctrl-P` opens a fuzzy command palette; the loaded file auto-reloads
on save. The full reference is in
[Interactive surface](../guides/repl-and-lsp.md).

### The language server

```bash
pip install 'quivers[lsp]'    # provides `qvr-lsp` on your PATH
```

The
[`vscode-qvr`](https://github.com/FACTSlab/quivers/tree/main/editors/vscode-qvr)
and
[`zed-extension-qvr`](https://github.com/FACTSlab/quivers/tree/main/editors/zed-extension-qvr)
extensions auto-discover it. Capabilities: hover (QVR declaration plus
collapsible AST), go-to-definition, references, document symbols,
semantic highlighting, completion, formatting, live diagnostics. The
same `STYLE_TABLE` drives the REPL and the LSP so the colours never
disagree.

### The Jupyter kernel

```bash
qvr-kernel install --user
jupyter console --kernel quivers
```

Cell semantics match the REPL: leading-`:` lines run meta-commands,
other lines append statements to the live module.

## Next Steps

- **[Interactive surface](../guides/repl-and-lsp.md):** REPL, language server, and Jupyter kernel reference.
- **[Architecture](architecture.md):** learn the package structure and design principles.
- **[API Reference](../api/index.md):** detailed documentation of all classes and functions.
- **Guides:** core types, morphisms, categorical structures, stochastic and continuous morphisms, the DSL, and variational inference.
- **Tutorials:** end-to-end examples including probabilistic programs and inference.
