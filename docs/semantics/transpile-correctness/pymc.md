# PyMC

Per-target obligations of [Transpilation correctness](index.md)
for $\mathsf{T} = \mathrm{PyMC}$.

## Semantics

PyMC's denotational semantics is the factor-graph semantics of
[Koller and Friedman 2009](https://mitpress.mit.edu/9780262013192/probabilistic-graphical-models/)
Chapter 4 implemented via the
[`pymc.Model`](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.Model.html)
context: each `pymc.<Distribution>(name, ..., dims=..., observed=...)`
constructor inside the `with pymc.Model()` scope registers a
random variable; the joint log-density is computed by
[`pymc.Model.compile_logp`](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.Model.html#pymc.Model.compile_logp).
The reference is Salvatier, Wiecki, and Fonnesbeck
([2016](https://doi.org/10.7717/peerj-cs.55)) for PyMC3 and the
[PyMC v5 documentation](https://www.pymc.io/projects/docs/en/stable/)
for the current API.

## Unconstrained-space change of variables

PyMC applies per-distribution unconstrained-space transforms
during inference (e.g.
[`pymc.distributions.transforms.log`](https://www.pymc.io/projects/docs/en/stable/api/distributions/transforms.html)
for `<lower=0>`-style constraints) and adds the Jacobian
automatically via the same change-of-variables principle as Stan.
The constrained-space density that `compile_logp` returns at a
constrained-space point is the QVR reference; no model-side
emit-time Jacobian is needed. $\Psi_{\mathsf{PyMC}} = \mathrm{id}$
at the constrained-space level.

## Family parameterizations

PyMC families use the
[`pymc.distributions.*`](https://www.pymc.io/projects/docs/en/stable/api/distributions.html)
classes with PyMC-specific keyword arguments. The QVR ↔ PyMC arg
mapping requires non-trivial aliases:

| QVR arg | PyMC arg |
|---|---|
| `loc` | `mu` |
| `scale` | `sigma` |
| `concentration` (Dirichlet) | `a` |
| `concentration1` (Beta) | `alpha` |
| `concentration0` (Beta) | `beta` |
| `probs` | `p` |
| `total_count` (Binomial) | `n` |
| `df` (StudentT) | `nu` |

These live in
[`FAMILY_META`](../../api/transpile/family_meta.md)`[F].arg_aliases["pymc"]`
per the SSoT rule. The renamings are syntactic; the density
$f_{\mathsf{PyMC}}(v \mid \text{renamed args})$ equals
$f_{\mathrm{QVR}}(v \mid \text{original args})$ term-by-term, so
$c_{F, \mathsf{PyMC}} = 0$ for every family with native PyMC
support.

## Per-construct emit

**Sample / observe.** `<name> = pymc.<Family>(<name>, **args,
dims=(<plate axes>), observed=<obs>)` inside the
`with pymc.Model() as model:` scope. The `dims=` declaration
carries the batch-axis names; PyMC's `coords` (in the
`pymc.Model(coords=...)` constructor) declares each axis with its
size.

**Plate.** PyMC handles plates via `dims=` declarations on the
random-variable constructor. The semantics of `dims=("doc",)` on
a length-20 axis is the product measure of 20 independent draws
(PyMC v5 documentation, "Coordinates and dimensions").

**Marginalize.** Explicit-latent rewrite. PyMC's
[`pm.Mixture`](https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.Mixture.html)
also supports analytic marginalization for finite mixtures; the
QVR renderer chooses the explicit rewrite for uniformity across
backends.

**Score / let / return.**
[`pymc.Potential("name", expr)`](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.Potential.html)
for score (Plummer 2003 zero-trick is not needed; PyMC has
`Potential` as a first-class log-density contribution);
deterministic assignment for let; the function returns the
`pymc.Model` instance.

## Acceptance

* **Tier 1 structural.** Every emit has a `def build_model(...)`
  with a `with pymc.Model(coords=...) as model:` body containing
  one `pymc.<Family>(...)` call per IR sample / observe step plus
  `return model`.
* **Tier 2 lens-laws.** Composition law holds.
* **Tier 3 external syntax.** `python -m ast` accepts every emit.
* **Tier 4 numeric equivalence.**
  `pymc.Model.compile_logp()(point_dict)` evaluated at 256 grid
  points + corners agrees with the QVR reference within $10^{-6}$.
