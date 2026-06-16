# Beta-Binomial A/B Test

## Overview

A two-arm [A/B test](https://en.wikipedia.org/wiki/A/B_testing) on a binary conversion outcome with per-arm [Beta-Binomial](https://en.wikipedia.org/wiki/Beta-binomial_distribution) likelihood. Each arm carries an independent positive concentration pair $(c^{(1)}_a, c^{(0)}_a)$ under a heavy-tailed [HalfCauchy](https://en.wikipedia.org/wiki/Cauchy_distribution#Truncation_and_folding) prior; the observed successes per arm are Beta-Binomial draws with shared trial count $N$ and per-arm concentrations.

Generative structure:

$$
\begin{aligned}
c^{(1)}_a &\sim \mathrm{HalfCauchy}(2),\\
c^{(0)}_a &\sim \mathrm{HalfCauchy}(2),\\
y_a \mid c^{(1)}_a, c^{(0)}_a &\sim \mathrm{BetaBinomial}\!\bigl(N,\, c^{(1)}_a,\, c^{(0)}_a\bigr).
\end{aligned}
$$

The Beta-Binomial integrates the latent per-arm conversion rate $p_a$ analytically: under the prior $p_a \sim \mathrm{Beta}(c^{(1)}_a, c^{(0)}_a)$ the marginal $\int \mathrm{Binomial}(y_a; N, p_a)\,\mathrm{Beta}(p_a; c^{(1)}_a, c^{(0)}_a)\,dp_a$ is closed-form, so the observed counts carry direct evidence about the concentration parameters without introducing an intermediate rate variable.

## QVR Source

```qvr
object Arm : FinSet 2

program ab_test : Arm -> Arm
    sample conc1 : Arm <- HalfCauchy(2.0)
    sample conc0 : Arm <- HalfCauchy(2.0)

    observe y : Arm <- BetaBinomial(N, conc1, conc0)
    return conc1

export ab_test
```

## Walkthrough

`object Arm : FinSet 2` declares the two-arm plate. The per-arm concentration parameters `conc1` and `conc0` are drawn under independent HalfCauchy priors with scale $2$, which place most of their mass near zero (uninformative-leaning) while admitting heavy upper tails for arms whose data demand a sharply-peaked likelihood. The `observe` step scores the observed success counts $y$ under the per-arm Beta-Binomial likelihood with shared trial count $N$ supplied at fit time from the host-data channel. `return conc1` projects onto the success-concentration field; the per-arm posterior contrast is the [shrinkage estimator](https://en.wikipedia.org/wiki/Shrinkage_estimator) for the A/B test.

## Use Cases

Conjugate A/B testing with closed-form rate marginals: the model is appropriate whenever counts of binary successes are summarised per group and the analyst wants posterior inference on per-group concentration rather than per-trial sampling of an explicit rate. Typical applications: clinical trial endpoints, click-through rate comparisons, conversion-funnel A/B tests with heterogeneous arms. The HalfCauchy prior on each concentration is standard for weakly-informative scale parameters and shrinks small-sample arms toward the population mean.

## References

- John G. Skellam. 1948. A probability distribution derived from the binomial distribution by regarding the probability of success as variable between the sets of trials. *Journal of the Royal Statistical Society. Series B (Methodological)*, 10(2):257-261.
