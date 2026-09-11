# Correlation-Matrix Prior via LKJCholesky

## Overview

A prior over $K \times K$ correlation matrices through their Cholesky factor. The [LKJ family](https://en.wikipedia.org/wiki/Lewandowski-Kurowicka-Joe_distribution) uses a shape parameter $\eta > 0$ to control concentration around the identity ($\eta > 1$ favours weaker correlations; $\eta = 1$ is uniform over correlation matrices; $\eta < 1$ favours stronger correlations). Its Cholesky factor is lower triangular with positive diagonal and unit row norms; it is not unit-lower-triangular.

Generative structure:

$$
\begin{aligned}
\eta &\sim \mathrm{HalfNormal}(2),\\
L &\sim \mathrm{LKJCholesky}(\eta),
\end{aligned}
$$

where $L L^\top$ is the implied correlation matrix.

## QVR source

```qvr
# Correlation-Matrix Prior via LKJCholesky
#
# A prior over correlation matrices, given in its Cholesky-factor
# form (named after Lewandowski-Kurowicka-Joe). The Cholesky
# factorisation is numerically preferable to the dense matrix when
# downstream inference samples in unconstrained space, since the
# factor admits a smooth bijection onto an unconstrained vector
# while the dense correlation matrix does not.
#
# Generative structure:
#
#   eta     ~ HalfNormal(2)                          shape parameter
#   chol    ~ LKJCholesky(eta)                       Cholesky factor
#
# The program carries no observe step: it is the prior surface a
# multivariate model composes with, and Sigma = chol * chol^T is
# the correlation matrix a downstream likelihood would consume.
#
# Dim is the matrix dimension the LKJCholesky family reads off its
# index annotation, and Factor is the value space of one row of
# the returned factor, a point of R^K.
#
# Reference: [Lewandowski et al. 2009](https://doi.org/10.1016/j.jmva.2009.04.008).

object Dim : FinSet 4
object Factor : Real 4

program correlation_model : Dim -> Factor
    sample eta <- HalfNormal(2.0)
    sample chol : Dim <- LKJCholesky(eta)
    return chol

export correlation_model
```

## Walkthrough

`object Dim : FinSet 4` declares the matrix dimension; the `sample chol : Dim <- ...` line carries that as the family's matrix axis, so `LKJCholesky(eta)` produces a $4 \times 4$ lower-triangular Cholesky factor whose rows have unit Euclidean norm. `object Factor : Real 4` is the value space of one row of that factor, and the program's signature `Dim -> Factor` names it as the codomain, since the codomain of a program is the space its returned value lives in rather than an index set. The shape parameter $\eta$ is drawn from a half-Normal, giving a weakly-informative prior over how strongly the correlation matrix concentrates around the identity. Downstream consumers reconstruct the correlation matrix as $\Sigma = L L^\top$ when they need the dense form.

## Working with the factor

The program declares a prior and no likelihood, so there is nothing to fit until a downstream model consumes the factor. What a consumer needs from it is the dense correlation matrix, recovered by one multiplication.

```python
import torch

torch.manual_seed(0)

n_dim = 4
eta = 2.0
chol = torch.distributions.LKJCholesky(n_dim, eta).sample()
correlation = chol @ chol.T

print("row norms:", chol.norm(dim=-1))
print("diagonal: ", correlation.diagonal())
```

Both printed lines are vectors of ones: the factor's rows are unit-norm by construction, so the implied $\Sigma = L L^\top$ is a correlation matrix rather than a general covariance. A model with per-dimension scales $\tau$ turns it into a covariance as $\mathrm{diag}(\tau)\,\Sigma\,\mathrm{diag}(\tau)$, which is the form a multivariate-Normal likelihood consumes.

## Use cases

Multivariate Gaussian models with correlated dimensions: factor analysis, multivariate hierarchical regression, copula constructions where the marginals are modelled separately. The Cholesky form is the recommended parameterisation in Stan (`cholesky_factor_corr`), NumPyro (`LKJCholesky`), and PyMC (`LKJCholeskyCov`), since constraint-respecting samplers (HMC, NUTS) work in the unconstrained tangent space.

## References

- Daniel Lewandowski, Dorota Kurowicka, and Harry Joe. 2009. Generating random correlation matrices based on vines and extended onion method. *Journal of Multivariate Analysis*, 100(9):1989-2001.
