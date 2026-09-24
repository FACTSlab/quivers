# Probabilistic Resolution Semantics

## Overview

This example implements the finite coffee-price analysis in Sections 6.2 and
6.3 of Pavlo Kuchmiichuk's
[“Restricting resolutions: effect-driven semantics of numerical approximation”](https://chive.pub/eprints/at%3A%2F%2Fdid%3Aplc%3Aodsg5cj2o5k2pu5hzcksnztg%2Fpub.chive.eprint.submission%2F3mvl7ruyc5f2i).
The analysis treats a numeral as uncertain between resolution descriptions.
An approximator conditions that distribution on the non-exact descriptions,
and assertion updates a prior over quantities by marginalizing over the
remaining resolutions.

We call the QVR realization the **resolution-update computation (RUC)**. It
implements four parts of the paper's worked example: (i) the exact and loose
descriptions for *five dollars*, (ii) the resolution prior in equation 49,
(iii) conditioning on non-exact descriptions in equation 51, and (iv) the
common-ground update in equation 53. The source uses integer cents rather than
binary floating-point dollars, so the ten-cent and twenty-cent boundaries are
exact.

[Open the complete QVR source](qiec/probabilistic-resolution-semantics.qvr).

## Indexed resolution descriptions

The precision of a description is part of its type:

<!-- compile: qiec -->
```qvr
index Precision = Exact | Loose

family Resolution[A : Type](p : Precision) : Type
    constructor ExactAt : A -> Resolution[A](Exact)
    constructor Around : A * Int -> Resolution[A](Loose)

define supported[p : Precision](description : Resolution[Int](p), quantity : Int) : Real !{} =
    case description motive (q : Precision) => Real
        ExactAt(center) =>
            if quantity == center then
                return 1.0
            else
                return 0.0
        Around(center, radius) =>
            if abs(quantity - center) <= radius then
                return 1.0
            else
                return 0.0
```

`ExactAt(500)` realizes the singleton description \(\{5.00\}\).
`Around(500, 10)` and `Around(500, 20)` realize the two loose descriptions
\(\{4.90,5.00,5.10\}\) and \(\{4.80,\ldots,5.20\}\). Constructor refinement
ensures that the exact constructor cannot be supplied where the semantics asks
for a loose description, while the indexed case checks both forms in one total
definition.

## Resolution choice as an effect

The paper replaces an unweighted indeterminacy effect with a probability
effect. The RUC specializes that effect to resolution descriptions:

<!-- compile: qiec -->
```qvr
index Precision = Exact | Loose

family Resolution[A : Type](p : Precision) : Type
    constructor ExactAt : A -> Resolution[A](Exact)
    constructor Around : A * Int -> Resolution[A](Loose)

effect Resolve[p : Precision]
    choose : Int -> Resolution[Int](p)

handler condition_nonexact for Resolve[Loose] : Real -> Real [coverage=total, forwards=none, implementation=authored]
    return probability =>
        return probability
    choose(center : Int) resumes omega =>
        let within_ten <- resume(construct Around[Int](center, 10) as Resolution[Int](Loose))
        let within_twenty <- resume(construct Around[Int](center, 20) as Resolution[Int](Loose))
        return 0.75 * within_ten + 0.25 * within_twenty
```

The two resumptions evaluate the continuation at the descriptions that remain
after conditioning. Their weights are the paper's equation 51: removing the
exact description from \((0.6,0.3,0.1)\) and renormalizing gives
\((0.75,0.25)\). The `omega` grade is substantive here because probabilistic
marginalization must run the continuation once for each supported alternative.

The exact handler instead has grade `1` and resumes with the unique exact
description. Each likelihood computation allocates a scoped `Resolve`
instance, so two numerals introduce distinct resolution choices even when
they use the same interface and handler.

The source also exposes `request_loose` with the row
`!{unresolved | rho lacks unresolved}`. This is the effect-polymorphic form of
the lexical request: it preserves a caller's other effects while proving that
the tail cannot duplicate the resolution instance. The exported update uses
closed, locally handled requests; `request_loose` separately records the open
library boundary a larger discourse computation would call.

## Updating the common ground

For a candidate quantity \(m\), `bare_likelihood` computes the semantic factor
in equation 53. The exact description receives weight 0.6; the non-exact
handler supplies the conditional distribution, whose total prior mass is 0.4:

<!-- compile: false -->
```qvr
define bare_likelihood(quantity : Int, numeral : Int) : Real !{} =
    let exact <- exact_likelihood(quantity, numeral)
    let loose <- approximate_likelihood(quantity, numeral)
    return 0.6 * exact + 0.4 * loose
```

Thus the factors at prices 470 through 530 cents are
`[0.0, 0.1, 0.4, 1.0, 0.4, 0.1, 0.0]`, exactly the values described after
equation 53. `approximate_five` uses the conditioned resolution distribution
instead, multiplies those likelihoods by the supplied quantity prior, and
normalizes. This is the paper's common-ground update rather than a flat
interval around five.

With a uniform quantity prior, the two interpretations are:

| price | $4.70 | $4.80 | $4.90 | $5.00 | $5.10 | $5.20 | $5.30 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bare *five* | 0 | 0.05 | 0.20 | 0.50 | 0.20 | 0.05 | 0 |
| *approximately five* | 0 | 1/14 | 2/7 | 2/7 | 2/7 | 1/14 | 0 |

The bare numeral peaks at five because every resolution supports the central
value. Conditioning away the exact description produces a three-way plateau
from $4.90 through $5.10, with the widest description contributing the smaller
outer mass.

## Try it

Check the complete module and run both updates under the QIEC reference
machine:

```python
from quivers.dsl import load

model = load("docs/examples/qiec/probabilistic-resolution-semantics.qvr")
uniform = (1.0 / 7.0,) * 7

approximate = model.run("hear_approximately_five", uniform)
bare = model.run("bare_five", uniform)

assert approximate.value == (
    0.0,
    1.0 / 14.0,
    2.0 / 7.0,
    2.0 / 7.0,
    2.0 / 7.0,
    1.0 / 14.0,
    0.0,
)
assert bare.value == (0.0, 0.05, 0.20, 0.50, 0.20, 0.05, 0.0)
```

```bash
qvr check docs/examples/qiec/probabilistic-resolution-semantics.qvr --target pyro
qvr run docs/examples/qiec/probabilistic-resolution-semantics.qvr approximate_five '[0.14285714285714285, 0.14285714285714285, 0.14285714285714285, 0.14285714285714285, 0.14285714285714285, 0.14285714285714285, 0.14285714285714285]' --json
qvr run docs/examples/qiec/probabilistic-resolution-semantics.qvr bare_five '[0.14285714285714285, 0.14285714285714285, 0.14285714285714285, 0.14285714285714285, 0.14285714285714285, 0.14285714285714285, 0.14285714285714285]' --json
```

The exported `hear_approximately_five` program accepts `price_prior` as its
observed input and returns the updated seven-point distribution. The Pyro and
NumPyro emitters carry the same indexed constructors, scoped instances,
handlers, and resumption grades across the QIEC boundary.

This example does not implement the paper's Section 5 account of
discourse-anaphoric rescue for mixed-precision stacks. Kuchmiichuk explicitly
abstracts away from that discourse state when deriving equations 52 and 53;
the RUC adopts the same scope. A full rescue model would additionally require
the paper's input and output effects for storing and retrieving resolution
referents.
