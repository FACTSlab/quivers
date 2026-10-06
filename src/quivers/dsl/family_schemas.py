"""Keyword schemas for QVR distribution-family construction.

A family call matches one complete schema. Most families have one canonical
schema supplied by their registry record; families with mutually exclusive
parameterizations declare each alternative here.
"""

from __future__ import annotations


_FAMILY_PARAMETERS: dict[str, tuple[str, ...]] = {
    "Normal": ("loc", "scale"),
    "LogitNormal": ("mu", "sigma"),
    "Beta": ("concentration1", "concentration0"),
    "TruncatedNormal": ("mu", "sigma", "low", "high"),
    "Dirichlet": ("concentration",),
    "Cauchy": ("loc", "scale"),
    "Laplace": ("loc", "scale"),
    "Gumbel": ("loc", "scale"),
    "LogNormal": ("loc", "scale"),
    "StudentT": ("df", "loc", "scale"),
    "Exponential": ("rate",),
    "Gamma": ("concentration", "rate"),
    "Chi2": ("df",),
    "HalfCauchy": ("scale",),
    "HalfNormal": ("scale",),
    "InverseGamma": ("concentration", "rate"),
    "Weibull": ("scale", "concentration"),
    "Pareto": ("scale", "alpha"),
    "Kumaraswamy": ("concentration1", "concentration0"),
    "ContinuousBernoulli": ("probs",),
    "FisherSnedecor": ("df1", "df2"),
    "Uniform": ("low", "high"),
    "MultivariateNormal": ("loc", "scale_tril"),
    "LowRankMVN": ("loc", "cov_factor", "cov_diag"),
    "RelaxedBernoulli": ("temperature", "probs"),
    "RelaxedOneHotCategorical": ("temperature", "probs"),
    "Wishart": ("df", "scale_tril"),
    "InverseWishart": ("df", "scale_tril"),
    "MatrixNormal": ("loc", "row_covariance", "col_covariance"),
    "GP": ("length_scale", "amplitude"),
    "Horseshoe": ("scale",),
    "Bernoulli": ("probs",),
    "Categorical": ("probs",),
    "Poisson": ("rate",),
    "NegativeBinomial": ("total_count", "probs"),
    "Geometric": ("probs",),
    "Binomial": ("total_count", "probs"),
    "VonMises": ("loc", "concentration"),
    "LogisticNormal": ("loc", "scale"),
    "OneHotCategorical": ("probs",),
    "LKJCholesky": ("concentration",),
    "Mixture": ("weights", "components"),
    "MixtureNormal": ("weights", "loc", "scale"),
    "Independent": ("base", "reinterpreted_batch_ndims"),
    "Transformed": ("base", "transforms"),
    "Truncated": ("base", "low", "high"),
    "ZeroInflatedPoisson": ("zero_prob", "rate"),
    "HurdlePoisson": ("zero_prob", "rate"),
    "ZeroOneInflatedBeta": ("mu", "phi", "zoi", "coi"),
    "Restrict": ("base", "low", "high"),
    "Normalize": ("base",),
    "PointMass": ("value",),
    "Pushforward": ("base", "bijector"),
    "LKJCorrelationFactor": ("concentration",),
    "BetaBinomial": ("total_count", "concentration1", "concentration0"),
    "OrderedLogistic": ("predictor", "cutpoints"),
    "OrderedProbit": ("eta", "cutpoints"),
    "Logistic": ("loc", "scale"),
    "GeneralizedPareto": ("loc", "scale", "concentration"),
    "HalfStudentT": ("df", "scale"),
}


_ALTERNATIVE_SCHEMAS: dict[str, tuple[tuple[str, ...], ...]] = {
    "Categorical": (("probs",), ("logits",)),
    "Restrict": (
        ("base", "low", "high"),
        ("base", "low"),
        ("base", "high"),
    ),
    "Truncate": (
        ("base", "low", "high"),
        ("base", "low"),
        ("base", "high"),
    ),
}


FAMILY_ALIASES: dict[str, str] = {
    "Pushforward": "Transformed",
    "Truncate": "Restrict",
}


def family_parameterizations(
    family: str, canonical: tuple[str, ...]
) -> tuple[tuple[str, ...], ...]:
    """Return every complete keyword schema accepted by ``family``."""
    alternatives = _ALTERNATIVE_SCHEMAS.get(family)
    if alternatives is not None:
        return alternatives
    parameters = _FAMILY_PARAMETERS.get(family, canonical)
    return (parameters,) if parameters else ()


def family_parameter_names(family: str) -> tuple[str, ...] | None:
    """Return the canonical QVR keyword order for ``family``."""
    return _FAMILY_PARAMETERS.get(family)


def render_parameterizations(schemas: tuple[tuple[str, ...], ...]) -> str:
    """Render schemas compactly for diagnostics."""
    return " or ".join("(" + ", ".join(schema) + ")" for schema in schemas)


__all__ = [
    "family_parameter_names",
    "family_parameterizations",
    "FAMILY_ALIASES",
    "render_parameterizations",
]
