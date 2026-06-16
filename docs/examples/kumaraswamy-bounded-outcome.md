# Bounded-Outcome Regression via Kumaraswamy

## Overview

A regression on a $(0, 1)$-bounded response with the [Kumaraswamy distribution](https://en.wikipedia.org/wiki/Kumaraswamy_distribution) as the likelihood. The Kumaraswamy is a close cousin of the Beta distribution with the same support and similar shape flexibility, but unlike Beta its CDF and quantile function are closed-form polynomials in elementary functions, which makes it preferable in models that need cheap inverse-CDF sampling, quantile-based inference, or copula constructions.

Generative structure:

$$
\begin{aligned}
\alpha &\sim \mathrm{HalfNormal}(2),\\
\beta &\sim \mathrm{HalfNormal}(2),\\
y_n \mid \alpha, \beta &\sim \mathrm{Kumaraswamy}(\alpha, \beta).
\end{aligned}
$$

The Kumaraswamy density is $f(y; \alpha, \beta) = \alpha \beta\, y^{\alpha - 1} (1 - y^\alpha)^{\beta - 1}$ on $y \in (0, 1)$.

## QVR Source

```qvr
object Resp : FinSet 1

program kumaraswamy_regression : Resp -> Resp
    sample alpha <- HalfNormal(2.0)
    sample beta <- HalfNormal(2.0)

    observe y : Resp <- Kumaraswamy(alpha, beta)
    return alpha

export kumaraswamy_regression
```

## Walkthrough

Both shape parameters get `HalfNormal(2)` priors that concentrate mass near the near-uniform region (small $\alpha$, $\beta$). The Kumaraswamy is then directly observed on the response. `return alpha` exposes the first shape parameter for posterior inspection; the analyst typically reports the implied mean $E[Y] = \beta B(1 + 1/\alpha, \beta)$ and concentration $1/(1 + 1/\beta)$ as derived quantities.

## Use Cases

Any bounded-outcome regression where the analyst needs efficient quantile evaluation: financial fraction-of-volume forecasting, biology fraction-of-population modelling, fraction-positive A/B testing where the closed-form quantile speeds Bayesian decision-making. Drop-in alternative to Beta regression when downstream operations are quantile-based rather than moment-based.

## References

- Ponnambalam Kumaraswamy. 1980. A generalized probability density function for double-bounded random processes. *Journal of Hydrology*, 46(1-2):79-88.
