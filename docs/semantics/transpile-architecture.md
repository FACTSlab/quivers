# Transpilation architecture

The transpile layer in [`quivers.transpile`][quivers.transpile]
realizes every well-typed QVR module $M$ as source bytes for a
target probabilistic programming language $\mathsf{T}$. This page
describes the architecture of that realization: the intermediate
representation, the family-metadata registry, the per-target
renderer interface, and the dispatch pattern that lets one walker
serve eleven backends without family-name special-casing.

The companion page [Transpilation correctness](transpile-correctness.md)
proves that the architecture preserves the joint distribution. This
page tells you how the pieces fit together.

## 1. The three-stage pipeline

The transpile pipeline is a
[`didactic.api.Mapping`][didactic.api.Mapping]
composition:

$$
\mathrm{Module}
\;\xrightarrow{\;\mathsf{Compile}\;}\;
\mathrm{Program}
\;\xrightarrow{\;\mathsf{Lower}\;}\;
\mathrm{IR}
\;\xrightarrow{\;\mathsf{Render}_{\mathsf{T}}\;}\;
\mathrm{panproto.Schema}
\;\xrightarrow{\;\mathsf{Pretty}_{\mathsf{T}}\;}\;
\mathrm{bytes}
$$

Each arrow is a small pure transformation; the composition is
[Theorem 4.1][transpile-correctness.md]'s first structural handle,
because each arrow's correctness lemma is local to its file.

* **Compile** parses a `.qvr` source text into a
  [`Module`][quivers.dsl.ast_nodes.Module] AST and resolves
  declarations into a `Program` containing the program's draws,
  morphism table, and let table.
* **[`Lower`][quivers.transpile.lower.Lower]** is
  target-independent. It walks the `Program` and emits an
  [`IRProgram`][quivers.transpile.ir.IRProgram] whose nodes carry
  the structural intent (sample, observe, marginalize, ...) plus
  the support and plate shape derived from
  [`FAMILY_META`][quivers.transpile.family_meta.FAMILY_META] and
  [`torch.distributions.Distribution.arg_constraints`][torch.distributions.Distribution.arg_constraints].
* **[`Render[T]`][quivers.transpile.renderers._base.RendererBase]**
  is one subclass per backend
  ([`StanRenderer`][quivers.transpile.renderers.stan.StanRenderer],
  [`NumPyroRenderer`][quivers.transpile.renderers.numpyro.NumPyroRenderer],
  [`PyMCRenderer`][quivers.transpile.renderers.pymc.PyMCRenderer],
  ...). It consumes the IR and emits a target-specific
  [`panproto.Schema`][panproto.Schema] using only the support
  predicates of §2.2 and the `FAMILY_META` entries.
* **Pretty[T]** is
  [`panproto.AstParserRegistry.emit_pretty`][panproto.AstParserRegistry.emit_pretty]
  for the target's tree-sitter grammar. It renders the schema as
  the canonical source-byte serialization.

Each of these claims is structural, enforced by the IR's shape and the
renderer's interface; see §3.

## 2. The IR

The IR lives in
[`src/quivers/transpile/ir.py`][quivers.transpile.ir]. Every entry
is a [`dx.Model`][didactic.api.Model] or
[`dx.TaggedUnion`][didactic.api.TaggedUnion]. The IR is purely
structural: no target-language strings, no schema vertices, no
panproto types.

### 2.1 `Plate`: event versus batch axes

A draw's plate annotation decomposes into the event axes (the
family's joint structure) and the batch axes (replication).

```python
class Plate(dx.Model):
    event_dims: tuple[Dim, ...]
    batch_dims: tuple[Dim, ...]


class Dim(dx.TaggedUnion, discriminator="kind"):
    name: str


class DimStatic(Dim):
    size: int
    kind: Literal["static"] = "static"


class DimDynamic(Dim):
    size_name: str
    kind: Literal["dynamic"] = "dynamic"
```

`event_dims` comes from `AxisSpec.over` (or the deprecated
`step.index` shorthand); `batch_dims` from `AxisSpec.iid_over`.
`Lower` preserves the source-declaration order; each renderer
walks `batch_dims` to emit nested `for` loops (Stan), nested
[plate contexts][numpyro.plate] (NumPyro / Pyro),
[`dims=(...)`][pymc.Distribution] declarations (PyMC), or
[`filldist`][turing.distributions.filldist] / `arraydist` wrappers
(Turing.jl / Gen.jl) per its native idiom.

### 2.2 Support classification

`src/quivers/transpile/ir.py` exports a small set of predicates
over
[`torch.distributions.constraints.Constraint`][torch.distributions.constraints]:

```python
def is_real_scalar(c: Constraint) -> bool: ...
def is_real_positive(c: Constraint) -> bool: ...
def is_real_unit_interval(c: Constraint) -> bool: ...
def is_real_vector(c: Constraint) -> bool: ...
def is_real_simplex(c: Constraint) -> bool: ...
def is_real_cov_matrix(c: Constraint) -> bool: ...
def is_real_corr_chol(c: Constraint) -> bool: ...
def is_real_matrix(c: Constraint) -> bool: ...
def is_real_one_hot(c: Constraint) -> bool: ...
def is_int_bit(c: Constraint) -> bool: ...
def is_int_category(c: Constraint) -> bool: ...
def is_int_count(c: Constraint) -> bool: ...
```

These are the only typeclass operations a renderer performs.
Renderers never `isinstance(c, _Simplex)` directly; they call
[`is_real_simplex`][quivers.transpile.ir.is_real_simplex]. Adding
a new support kind (ordered vectors, say) means adding one
predicate. The predicates dispatch on torch's existing
[`Constraint`][torch.distributions.constraints.Constraint]
taxonomy.

Because [Torch][torch.distributions] does not survive
didactic's tagged-union encode / decode round trip, the IR
actually stores constraints as a structural mirror
[`ConstraintSpec`][quivers.transpile.ir.ConstraintSpec] that
materializes to the underlying `Constraint` via
`.to_constraint()` when a renderer needs the real value. The
mirror has one variant per kind the predicates distinguish; the
[`from_constraint`][quivers.transpile.ir.from_constraint]
converter goes the other way at Lower time.

### 2.3 `IRArg`: typed argument tree

The parser produces stringly-typed bracket args like `"phi[z]"`;
Lower parses them into a typed tree so renderers don't re-parse
strings.

```python
class IRArg(dx.TaggedUnion, discriminator="kind"):
    ...

class IRArgNumber(IRArg):
    value: float
    kind: Literal["number"] = "number"

class IRArgRef(IRArg):
    name: str
    indices: tuple[IRArg, ...] = ()
    kind: Literal["ref"] = "ref"

class IRArgBroadcast(IRArg):
    """A scalar broadcast to satisfy an arg's expected constraint."""
    value: IRArg
    target_shape: tuple[int, ...]
    kind: Literal["broadcast"] = "broadcast"

class IRArgList(IRArg):
    elements: tuple[IRArg, ...]
    kind: Literal["list"] = "list"

class IRArgMatrix(IRArg):
    rows: tuple[IRArgList, ...]
    kind: Literal["matrix"] = "matrix"

class IRArgFamilyRef(IRArg):
    """A reference to a morphism whose ``~ Family(...)`` init clause
    names the wrapped distribution (used by Truncated, Mixture,
    Independent, Transformed, LKJCorrelationFactor)."""
    name: str
    kind: Literal["family_ref"] = "family_ref"
```

Lower wraps a scalar arg in `IRArgBroadcast` when the matched
[`arg_constraints[name]`][torch.distributions.Distribution.arg_constraints]
is `IndependentConstraint(base, n>=1)`. Each renderer translates
the broadcast to its native op:
[`rep_vector(x, K)`][stan.functions.rep_vector] in Stan,
[`jnp.full((K,), x)`][jax.numpy.full] in NumPyro,
[`torch.full((K,), x)`][torch.full] in Pyro,
[`np.full((K,), x)`][numpy.full] in PyMC,
[`fill(x, K)`][julia.fill] in Turing / Gen,
`repeat(K, function() { return x; })` in WebPPL,
`(make-list K x)` in Church. The translation lives in each
renderer's `broadcast(value, target_shape)` method.

### 2.4 `IRNode`: program-body statements

```python
class IRNode(dx.TaggedUnion, discriminator="kind"):
    ...

class IRDataInput(IRNode):
    name: str
    constraint: ConstraintSpec
    plate: Plate
    kind: Literal["data_input"] = "data_input"

class IRSample(IRNode):
    name: str
    family: str
    args: tuple[IRArg, ...]
    arg_names: tuple[str, ...]
    constraint: ConstraintSpec
    plate: Plate
    kind: Literal["sample"] = "sample"

class IRObserve(IRNode):
    name: str
    family: str
    args: tuple[IRArg, ...]
    arg_names: tuple[str, ...]
    constraint: ConstraintSpec
    plate: Plate
    via: str | None
    kind: Literal["observe"] = "observe"

class IRDeterministic(IRNode):
    name: str
    expr: IRExpr
    constraint: ConstraintSpec
    plate: Plate
    kind: Literal["deterministic"] = "deterministic"

class IRScore(IRNode):
    name: str
    expr: IRExpr
    kind: Literal["score"] = "score"

class IRMarginalize(IRNode):
    """A discrete-latent integration scope. Each renderer decides
    how to emit this: Stan as `log_sum_exp` per-group enumeration,
    every other backend by inline lowering to `IRSample(latent) +
    scope body`."""
    latent: str
    family: str
    args: tuple[IRArg, ...]
    arg_names: tuple[str, ...]
    constraint: ConstraintSpec
    plate: Plate
    reduction: Literal["logsumexp"]
    scope: tuple[IRNode, ...]
    kind: Literal["marginalize"] = "marginalize"

class IRReturn(IRNode):
    names: tuple[str, ...]
    kind: Literal["return"] = "return"
```

`arg_names` parallels `args` and carries the keyword names from
torch's `arg_constraints` (`"loc"`, `"scale"`,
`"concentration"`, ...). Renderers that prefer keyword calls
(NumPyro, Pyro, PyMC, Edward2) read from `arg_names`; positional
renderers (Stan, BUGS, JAGS) ignore it.

```python
class IRProgram(dx.Model):
    name: str
    inputs: tuple[IRDataInput, ...]
    body: tuple[IRNode, ...]
```

## 3. `FamilyMeta`: the registry for transpile-only facts

One registry, in
[`src/quivers/transpile/family_meta.py`][quivers.transpile.family_meta]:

```python
class FamilyMeta(dx.Model):
    qvr_name: str
    distribution_class: type[Distribution]
    quivers_class: type[ContinuousMorphism] | None
    target_names: dict[str, str]
    arg_aliases: dict[str, dict[str, str]]
```

* `qvr_name`: the DSL-facing family name (`"Normal"`,
  `"Dirichlet"`).
* `distribution_class`: the underlying
  [`torch.distributions.Distribution`][torch.distributions.Distribution]
  subclass (or a thin shim exposing the right `arg_constraints` +
  `.support` surface for families with no native torch
  counterpart, like `OrderedLogistic` and `HalfStudentT`). Source
  of truth for the family's argument constraints and output
  support.
* `quivers_class`: the
  [`ContinuousMorphism`][quivers.continuous.morphisms.ContinuousMorphism]
  subclass the inference layer instantiates at runtime
  ([`ConditionalNormal`][quivers.continuous.families.ConditionalNormal],
  etc.). Empty for wrapper families whose runtime morphism is
  constructed from a referenced inner morphism.
* `target_names`: per-backend distribution-name mapping. The
  single source of truth for backend-to-distribution-name
  resolution. Renderers look up
  `FAMILY_META[family].target_names[backend]`; no per-renderer
  `_FAMILIES` dict exists.
* `arg_aliases`: per-backend per-arg renames. Most families have
  empty `arg_aliases`. Renderers that apply parameterisation-
  converting arithmetic (BUGS / JAGS Normal mean+scale to
  mean+precision; PyMC's `concentration → a` rename for
  Dirichlet) key the arithmetic on the alias's target name.

The marginalize-eligibility check is a per-call function rather
than a per-family flag:

```python
def finite_enumerable_at_call_site(
    family_meta: FamilyMeta,
    args: tuple[IRArg, ...],
) -> bool: ...
```

Returns `True` for Bernoulli, Categorical, OrderedLogistic, and
OrderedProbit unconditionally. For Binomial returns `True` only
when `args[0]` (total_count) is a literal `IRArgNumber`; the
Stan renderer's `marginalize` raises
[`UnsupportedConstruct`][quivers.transpile.UnsupportedConstruct]
when the check returns `False`.

### 3.1 What `FAMILY_META` does not carry

* **Argument shapes / constraints.** Lives in
  `distribution_class.arg_constraints`. Lower reads from there.
* **Output support.** Lives in `distribution_class.support` (the
  class-level support, or its evaluation on a sentinel parameter
  set for instance-dependent supports like `Uniform(low, high)`).
* **Event rank.** Derived from
  `distribution_class().event_shape` on the sentinel.

This separation keeps `FAMILY_META` small (under a hundred lines
per family) and ties the structural classification to torch's
existing implementation. Adding a new family is one
[`Conditional*`][quivers.continuous.families.ConditionalNormal]
class plus one `FamilyMeta` entry; no renderer touches.

## 4. `Lower`: Program → IR

[`Lower`][quivers.transpile.lower.Lower] is a single class
implementing `dx.Mapping[Program, IRProgram]`. Its `forward`:

1. Runs
   [`expand_composite_lets`][quivers.transpile._expand_composites.expand_composite_lets]
   on the program. Composite-let bindings (`let chain = prior >>
   likelihood`) flatten into atomic sample chains so each
   program-step the IR sees references a single morphism.
2. Resolves every step's morphism slot to a `(family, args)`
   pair via
   [`resolve_step_dist`][quivers.transpile._resolve.resolve_step_dist].
3. Looks up `meta = FAMILY_META[family]`.
4. Reads `arg_constraints = meta.distribution_class.arg_constraints`
   and resolves the output support, instantiating with sentinel
   args when the support is parameter-dependent.
5. Computes `Plate` from `(AxisSpec, step.index, cards)`. `over`
   axes become `event_dims`; `iid_over` axes become `batch_dims`.
6. Matches user args against `arg_constraints` positionally,
   wrapping scalars in `IRArgBroadcast` when the constraint is
   `IndependentConstraint(base, n>=1)` and the user supplied a
   scalar. Wrapper-family arguments wrap in `IRArgFamilyRef` when
   they reference a morphism with a `~ Family(...)` init clause.
7. Discovers exogenous identifiers: free names in let / score
   bodies, free names in bracket-indexed args, `via=`
   fibrations, scalar program parameters. Each surfaces as
   `IRDataInput` with a constraint derived from how it is used.

Lower is target-independent. It never imports any renderer or
backend-specific module.

## 5. `Renderer[T]`: IR → panproto.Schema

Each backend implements a
[`Renderer`][quivers.transpile.renderers._base.RendererBase]
subclass with one public method `render(ir: IRProgram) ->
panproto.Schema` and four private dispatch points:

```python
class Renderer(Protocol):
    @abstractmethod
    def render(self, ir: IRProgram) -> panproto.Schema: ...

    @abstractmethod
    def declare(self, name, constraint, plate, *, block) -> SchemaFragment: ...

    @abstractmethod
    def sample(self, name, family, args, arg_names, constraint,
               plate, observed) -> SchemaFragment: ...

    @abstractmethod
    def marginalize(self, node: IRMarginalize) -> SchemaFragment: ...

    @abstractmethod
    def broadcast(self, value, target_shape) -> SchemaFragment: ...
```

`BlockKind` is the renderer-side notion of where a declaration
lands (`"data"`, `"parameters"`, `"transformed_parameters"`,
`"generated_quantities"`, `"function_body"`). Each backend
interprets it per its own program structure: Stan has actual
blocks; NumPyro's "block" is the function body; PyMC's is the
`with pymc.Model() as model:` scope; BUGS / JAGS have only a
single `model { ... }` enclosure.

[`RendererBase`][quivers.transpile.renderers._base.RendererBase]
provides the IR walk (`IRDataInput → declare`, `IRSample
(non-observed) → declare + sample`, `IRObserve → declare +
sample(observed=True)`, `IRDeterministic → declare + assignment`,
`IRScore → declare scalar + log-density increment`, `IRMarginalize
→ marginalize`, `IRReturn → backend return idiom`), index-
substitution helpers consumed by both `sample` and `marginalize`,
and the explicit-latent rewrite helper shared by every backend
whose `marginalize` lowers `IRMarginalize` to `IRSample` plus the
scope inline.

`declare` dispatches on the predicates of §2.2. The Stan
renderer's table:

| predicate | event | batch | declaration |
|---|---|---|---|
| `is_real_scalar(c)` | () | () | `real <name>;` |
| `is_real_scalar(c)` | () | (B,) | `vector[B] <name>;` |
| `is_real_positive(c)` | () | () | `real<lower=0> <name>;` |
| `is_real_positive(c)` | () | (B,) | `vector<lower=0>[B] <name>;` |
| `is_real_unit_interval(c)` | () | () | `real<lower=0, upper=1> <name>;` |
| `is_real_vector(c)` | (E,) | () | `vector[E] <name>;` |
| `is_real_vector(c)` | (E,) | (B,) | `array[B] vector[E] <name>;` |
| `is_real_simplex(c)` | (E,) | () | `simplex[E] <name>;` |
| `is_real_simplex(c)` | (E,) | (B,) | `array[B] simplex[E] <name>;` |
| `is_real_cov_matrix(c)` | (E,) | () | `cov_matrix[E] <name>;` |
| `is_real_corr_chol(c)` | (E,) | () | `cholesky_factor_corr[E] <name>;` |
| `is_real_matrix(c)` | (R,C) | () | `matrix[R, C] <name>;` |
| `is_int_bit(c)` | () | () | `int<lower=0, upper=1> <name>;` |
| `is_int_bit(c)` | () | (B,) | `array[B] int<lower=0, upper=1> <name>;` |
| `is_int_category(c)` | () | () | `int<lower=1, upper=K> <name>;` |
| `is_int_count(c)` | () | () | `int<lower=0> <name>;` |

Other backends have analogous tables. The grammar of the table is
the same: `(predicate, event_dims, batch_dims) → target-language
declaration`. No row references a family name.

### 5.1 Backend idioms

The eleven backends fall into three idiomatic families:

* **Block-structured static-type** (Stan):
  `data { ... } parameters { ... } model { ... } generated
  quantities { ... }`. The renderer threads the per-block
  declarations through panproto schema vertices for each block.
* **Trace-based** (NumPyro, Pyro, Turing.jl, Gen.jl, Church,
  WebPPL). The renderer emits a `def model(...)` (or `@model
  function`, or `(define (model ...))`) and uses the target's
  native plate primitive
  ([`numpyro.plate`][numpyro.plate],
  [`pyro.plate`][pyro.plate],
  [`filldist`][turing.distributions.filldist],
  [`@trace`][gen.trace],
  `map` over `iota`, `repeat`)
  to express batch dimensions.
* **Graphical-model relational** (PyMC, Edward2, BUGS, JAGS).
  PyMC and Edward2 use named-distribution constructors with
  `dims=(...)` / `sample_shape=[...]` carrying the batch shape;
  BUGS and JAGS use `for (m in 1:N) { name[m] ~ d<family>(args)
  }` row-loops. The BUGS Normal mean+scale → mean+precision
  conversion (`tau = 1 / (scale * scale)`) lives in
  `FAMILY_META.arg_aliases["bugs"]` plus a renderer-internal
  arithmetic-transform table keyed on the alias target name.

Each backend's renderer is roughly one file of 700 to 1400 lines.
None imports from any other.

## 6. LDA end-to-end

The canonical Latent Dirichlet Allocation source:

```qvr
program lda(alpha : Real, beta : Real) : Word -> Word
    sample theta : Doc <- Dirichlet(alpha) [over=Topic, iid_over=Doc]
    sample phi : Topic <- Dirichlet(beta) [over=Word, iid_over=Topic]
    marginalize z : Topic <- Categorical(theta) [over=Doc, reduction=logsumexp]
        observe w : Word <- Categorical(phi[z]) [via=word_idx]
    return theta
```

Cardinalities: Doc=20, Topic=3, Word=200.

After `Lower`, the IR carries:

* Inputs: `alpha` and `beta` as `IRDataInput`s with `CSReal()`
  constraints; `word_idx` and `w` as `IRDataInput`s with
  `IntegerInterval` constraints and `DimDynamic(size_name="N_w")`
  batch dimensions.
* Body: `IRSample(theta)` with `support=CSSimplex(event_dim=3)`
  and `plate=Plate(event_dims=(DimStatic(3, "Topic"),),
  batch_dims=(DimStatic(20, "Doc"),))`. Its only arg is
  `IRArgBroadcast(value=IRArgRef("alpha"), target_shape=(3,))`.
* `IRSample(phi)` analogous, transposed dims.
* `IRMarginalize(z)` with `args=(IRArgRef("theta"),)`, scope
  containing one `IRObserve(w)` whose args are
  `IRArgRef("phi", indices=(IRArgRef("z"),))` and whose
  `via="word_idx"`.
* `IRReturn(names=("theta",))`.

`StanRenderer.render` produces:

```stan
data {
  real alpha;
  real beta;
  int N_w;
  array[N_w] int<lower=1, upper=20> word_idx;
  array[N_w] int<lower=1, upper=200> w;
}
parameters {
  array[20] simplex[3] theta;
  array[3] simplex[200] phi;
}
model {
  for (m_Doc in 1:20)
    theta[m_Doc] ~ dirichlet(rep_vector(alpha, 3));
  for (m_Topic in 1:3)
    phi[m_Topic] ~ dirichlet(rep_vector(beta, 200));
  {
    array[20] vector[3] lps_z;
    for (g_Doc in 1:20)
      for (k in 1:3)
        lps_z[g_Doc, k] = categorical_lpmf(k | theta[g_Doc]);
    for (n in 1:N_w)
      for (k in 1:3)
        lps_z[word_idx[n], k] += categorical_lpmf(w[n] | phi[k]);
    for (g_Doc in 1:20)
      target += log_sum_exp(lps_z[g_Doc]);
  }
}
generated quantities {
  array[20] simplex[3] theta_value = theta;
}
```

`NumPyroRenderer.render` produces:

```python
import jax.numpy as jnp
import numpyro
import numpyro.distributions

def model(alpha, beta, word_idx, w=None):
    with numpyro.plate("Doc", 20):
        theta = numpyro.sample(
            "theta",
            numpyro.distributions.Dirichlet(jnp.full((3,), alpha)),
        )
    with numpyro.plate("Topic", 3):
        phi = numpyro.sample(
            "phi",
            numpyro.distributions.Dirichlet(jnp.full((200,), beta)),
        )
    with numpyro.plate("Doc_z", 20):
        z = numpyro.sample(
            "z",
            numpyro.distributions.Categorical(theta),
        )
    with numpyro.plate("Word_obs", w.shape[0]):
        numpyro.sample(
            "w",
            numpyro.distributions.Categorical(phi[z[word_idx]]),
            obs=w,
        )
    return theta
```

Same IR, different renderer. The Stan renderer's `marginalize`
emits the `log_sum_exp` enumeration; the NumPyro renderer's
`marginalize` lowers the construct to `IRSample(z) + scope` and
the scope's `IRObserve(w)` becomes a `numpyro.sample(..., obs=w)`
inside a per-word
[`plate`][numpyro.plate]. Neither renderer's code
references the family name `Dirichlet` or `Categorical`; both
dispatch on `is_real_simplex` (for the Dirichlet declaration) and
`is_int_category` (for the Categorical observation).

## 7. Adding a new family

1. Implement a [`ContinuousMorphism`][quivers.continuous.morphisms.ContinuousMorphism]
   subclass in
   [`src/quivers/continuous/families.py`][quivers.continuous.families]
   (or a new file under `src/quivers/continuous/` if the family
   has its own structural shape, like the cutpoint-parameterized
   ordered families in `src/quivers/continuous/ordered.py`).
2. Add a
   [`FamilyMeta`][quivers.transpile.family_meta.FamilyMeta]
   entry to `FAMILY_META`. Populate `qvr_name`,
   `distribution_class` (the torch class or a thin shim
   exposing the right `arg_constraints` and `.support`),
   `quivers_class`, `target_names`, and `arg_aliases`.

Every backend's renderer picks the new family up automatically
via the constraint-predicate dispatch on the family's torch
`.support`. No per-backend edit is needed unless the family
requires a backend-specific arithmetic transform or wrapper
shape.

## 8. Adding a new backend

1. Choose the target tree-sitter grammar (`stan`, `python`,
   `julia`, `scheme`, `javascript`, `bugs`, `jags`).
2. Implement a
   [`RendererBase`][quivers.transpile.renderers._base.RendererBase]
   subclass under `src/quivers/transpile/renderers/<backend>.py`.
   Override `declare`, `sample`, `marginalize`, and `broadcast`.
3. Add a `target_names[<backend>] = ...` entry to every
   `FamilyMeta` in `FAMILY_META` for the families the backend
   supports. Omit the entry for unsupported families; the
   renderer's call-site lookup raises
   [`UnsupportedConstruct`][quivers.transpile.UnsupportedConstruct]
   with a precise kind.
4. Register the renderer in `src/quivers/transpile/__init__.py`'s
   `_RENDERERS` table, with the appropriate grammar string.

The IR walk, `FAMILY_META` consultation, and constraint-
predicate dispatch are inherited from `RendererBase`. A typical
backend implementation is one file, 700 to 1400 lines, with no
imports from any other backend's renderer.

## 9. The five rules

The architecture enforces five structural invariants:

1. **Single source of truth per concept.** Family metadata
   (event rank, support, argument constraints, per-target
   distribution name, argument aliases) lives in one place.
   Walkers query it; they never duplicate or override.
2. **No `if family == "X"` in any renderer.** Renderer behaviour
   dispatches on the support predicates of §2.2 and on
   `FAMILY_META.target_names[backend]`.
3. **No silent drops of AST fields.** Every `AxisSpec.over`,
   `AxisSpec.iid_over`, `ObserveStep.via`,
   `MarginalizeStep.reduction`, and `MarginalizeStep.scope` is
   consumed by Lower or raised on by a renderer with a precise
   `UnsupportedConstruct` kind.
4. **Backend-symmetric abstractions.** Each renderer reads from
   the same `FAMILY_META`, the same `Lower` output, and the
   same `RendererBase` helpers. No backend is more privileged
   than another.
5. **No fallbacks, no placeholders.** When a renderer cannot
   lower a construct, it raises `UnsupportedConstruct` with a
   precise kind. Never emits `__placeholder__` or "tracked
   later" or broken code.

The IR shape, the `FamilyMeta` schema, and the `Renderer`
Protocol jointly make these invariants structural: a renderer
that violates one of them produces a schema that fails the
structural matrix test, or fails to compile against the
`Renderer` Protocol, or raises a typed `UnsupportedConstruct`
rather than emitting wrong bytes.

## References

* [Transpilation correctness](transpile-correctness.md). The
  per-arrow lemma chain that lifts to the natural isomorphism
  $\eta_{\mathsf{T}}: \mathsf{S}_{\mathrm{QVR}} \xRightarrow{\cong}
  \mathsf{S}_{\mathsf{T}} \circ \mathsf{T}_{\mathsf{T}}$ in
  $\mathbf{Kern}$.
* [QVR programs](programs.md). The source-language `Program`
  structure that `Lower` consumes.
* [Continuous families](../guides/continuous-families.md). The
  catalogue of
  [`ContinuousMorphism`][quivers.continuous.morphisms.ContinuousMorphism]
  subclasses
  ([`ConditionalNormal`][quivers.continuous.families.ConditionalNormal],
  [`ConditionalDirichlet`][quivers.continuous.families.ConditionalDirichlet],
  [`ConditionalBetaBinomial`][quivers.continuous.families.ConditionalBetaBinomial],
  ...) the inference layer instantiates at run time.
* Bob Carpenter, Andrew Gelman, Matthew D. Hoffman, Daniel Lee,
  Ben Goodrich, Michael Betancourt, Marcus Brubaker, Jiqiang Guo,
  Peter Li, and Allen Riddell. 2017. Stan: A probabilistic
  programming language. *Journal of Statistical Software*,
  76(1):1-32.
  [https://doi.org/10.18637/jss.v076.i01](https://doi.org/10.18637/jss.v076.i01)
* Du Phan, Neeraj Pradhan, and Martin Jankowiak. 2019.
  Composable effects for flexible and accelerated probabilistic
  programming in NumPyro. *arXiv preprint arXiv:1912.11554*.
  [https://doi.org/10.48550/arXiv.1912.11554](https://doi.org/10.48550/arXiv.1912.11554)
* Eli Bingham, Jonathan P. Chen, Martin Jankowiak, Fritz
  Obermeyer, Neeraj Pradhan, Theofanis Karaletsos, Rohit Singh,
  Paul Szerlip, Paul Horsfall, and Noah D. Goodman. 2019. Pyro:
  Deep universal probabilistic programming. *Journal of Machine
  Learning Research*, 20(28):1-6.
  [http://jmlr.org/papers/v20/18-403.html](http://jmlr.org/papers/v20/18-403.html)
* Hong Ge, Kai Xu, and Zoubin Ghahramani. 2018. Turing: A
  language for flexible probabilistic inference. In
  *International Conference on Artificial Intelligence and
  Statistics*, pages 1682-1690.
  [https://proceedings.mlr.press/v84/ge18b.html](https://proceedings.mlr.press/v84/ge18b.html)
* Marco F. Cusumano-Towner, Feras A. Saad, Alexander K. Lew, and
  Vikash K. Mansinghka. 2019. Gen: A general-purpose
  probabilistic programming system with programmable inference.
  In *Proceedings of the 40th ACM SIGPLAN Conference on
  Programming Language Design and Implementation*, pages
  221-236.
  [https://doi.org/10.1145/3314221.3314642](https://doi.org/10.1145/3314221.3314642)
* Noah D. Goodman, Vikash K. Mansinghka, Daniel M. Roy, Keith
  Bonawitz, and Joshua B. Tenenbaum. 2008. Church: A language for
  generative models. In *Proceedings of the Twenty-Fourth
  Conference on Uncertainty in Artificial Intelligence (UAI)*,
  pages 220-229.
  [https://arxiv.org/abs/1206.3255](https://arxiv.org/abs/1206.3255)
* Noah D. Goodman and Andreas Stuhlmüller. 2014. *The Design and
  Implementation of Probabilistic Programming Languages*. Online
  textbook. [http://dippl.org](http://dippl.org)
* John K. Kruschke. 2014. *Doing Bayesian Data Analysis: A
  Tutorial with R, JAGS, and Stan*. Second edition. Academic
  Press.
  [https://doi.org/10.1016/C2012-0-00477-2](https://doi.org/10.1016/C2012-0-00477-2)
