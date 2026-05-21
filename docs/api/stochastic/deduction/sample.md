# Forward Sampling

`sample_corpus` draws length-fixed yields from the chart's
length-conditional distribution
$p(s \mid \text{length} = L, \mathbf{w}) =
Z(s; \mathbf{w}) / \sum_{s' \text{ of length } L} Z(s'; \mathbf{w})$.
The implementation enumerates every length-$L$ sequence over the
deduction's surface vocabulary, evaluates $\log Z$ exactly via
the chart, softmaxes the log-weights, and draws a multinomial.
The procedure is exact (no MCMC over derivations); the $|V|^L$
enumeration cost is the fundamental cost of forward sampling
from a globally-normalised chart-defined distribution.

::: quivers.stochastic.deduction.sample
