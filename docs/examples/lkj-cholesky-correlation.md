# Correlation-Matrix Prior via LKJCholesky

## Overview

A prior over $K \times K$ correlation matrices through their Cholesky factor. The [LKJ family](https://en.wikipedia.org/wiki/Lewandowski-Kurowicka-Joe_distribution) (named after Lewandowski, Kurowicka, and Joe) is the standard parametric prior on correlation matrices: a single shape parameter $\eta > 0$ controls how concentrated the prior is around the identity ($\eta > 1$ favours diagonals; $\eta = 1$ is uniform over correlation matrices; $\eta < 1$ favours off-diagonal mass). The Cholesky-factor form is numerically preferable to the dense correlation matrix because the unit-lower-triangular factor admits a smooth bijection to unconstrained reals.

Generative structure:

$$
\begin{aligned}
\eta &\sim \mathrm{HalfNormal}(2),\\
L &\sim \mathrm{LKJCholesky}(\eta),
\end{aligned}
$$

where $L L^\top$ is the implied correlation matrix.

## QVR Source

```qvr
object Dim : FinSet 4

program correlation_model : Dim -> Dim
    sample eta <- HalfNormal(2.0)
    sample chol : Dim <- LKJCholesky(eta)
    return chol

export correlation_model
```

## Walkthrough

`object Dim : FinSet 4` declares the matrix dimension; the `sample chol : Dim <- ...` line carries that as the codomain, so `LKJCholesky(eta)` produces a $4 \times 4$ lower-triangular Cholesky factor whose rows have unit Euclidean norm. The shape parameter $\eta$ is drawn from a half-Normal, giving a weakly-informative prior over how strongly the correlation matrix concentrates around the identity. Downstream consumers reconstruct the correlation matrix as $\Sigma = L L^\top$ when they need the dense form.

## Use Cases

Multivariate Gaussian models with correlated dimensions: factor analysis, multivariate hierarchical regression, copula constructions where the marginals are modelled separately. The Cholesky form is the recommended parameterisation in Stan (`cholesky_factor_corr`), NumPyro (`LKJCholesky`), and PyMC (`LKJCholeskyCov`), since constraint-respecting samplers (HMC, NUTS) work in the unconstrained tangent space.

## References

- Daniel Lewandowski, Dorota Kurowicka, and Harry Joe. 2009. Generating random correlation matrices based on vines and extended onion method. *Journal of Multivariate Analysis*, 100(9):1989-2001.
