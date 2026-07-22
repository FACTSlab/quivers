"""Response families and link functions for the formula frontend.

Each `Family` knows how to emit a QVR ``observe`` step plus
the inverse-link expression that wraps a linear predictor.  The
registry maps brms-style family names to a typed family record;
the compiler dispatches on the family name to slot the right
inverse-link / observe lines into the emitted ``.qvr`` source.

Adding a new family is one entry in `families`; the
implementation is uniform across all of GLM / GLMM / GAMLSS-style
distributional regression because each family declares its
location parameter (mandatory) and any auxiliary parameters
(``sigma``, ``phi``, ``disp``, ``zi``, ``alpha``) with their
priors and links separately.
"""

from __future__ import annotations

from typing import Mapping

import didactic.api as dx


class Link(dx.Model):
    """Inverse-link function bridging a linear predictor on
    :math:`\\mathbb{R}` to the family's natural parameter space.

    Attributes
    ----------
    name : str
        Canonical name (e.g. ``"identity"``, ``"logit"``, ``"log"``,
        ``"probit"``, ``"cloglog"``).
    inverse_expr : str
        QVR let-expression template applied to a linear predictor
        ``eta`` to produce the family parameter.  Substitution
        token ``{eta}`` is replaced by the linear predictor's
        variable name.  ``"{eta}"`` itself is the identity link.
    """

    name: str
    inverse_expr: str


class AuxParam(dx.Model):
    """An auxiliary family parameter (scale, dispersion, etc.).

    Attributes
    ----------
    name : str
        Variable name to emit in the ``.qvr`` source.
    prior : str
        Prior distribution expression (e.g. ``"HalfCauchy(2.0)"``).
    link : Link
        Link applied to a per-row linear predictor if the parameter
        is itself distributionally regressed (``y ~ x``, ``sigma ~
        x``, etc.); otherwise unused.
    """

    name: str
    prior: str
    link: Link


class Family(dx.Model):
    """Response family: observation kernel plus its link and any
    auxiliary parameters.

    The compiler emits, for each family-keyed regression:

    1. The auxiliary-parameter latent draws (intercept-only by
       default; one per parameter named in `aux_params`).
    2. A ``let eta = <linear predictor>`` binding the location's
       linear predictor.
    3. A ``let mu = <link.inverse_expr>`` binding the natural
       parameter.
    4. A single ``observe y : Resp <- <observe_family>(mu, ...)``
       step parameterised by ``mu`` and the auxiliary parameters.
    """

    name: str
    location_link: Link
    observe_family: str
    extra_observe_args: tuple[str, ...] = ()
    aux_params: tuple[AuxParam, ...] = ()


_IDENTITY = Link(name="identity", inverse_expr="{eta}")
_LOGIT = Link(name="logit", inverse_expr="sigmoid({eta})")
_LOG = Link(name="log", inverse_expr="exp({eta})")
_SOFTMAX = Link(name="softmax", inverse_expr="softmax({eta})")
_INVERSE = Link(name="inverse", inverse_expr="1.0 / {eta}")


#: Built-in inverse-link registry.
links: Mapping[str, Link] = {
    "identity": _IDENTITY,
    "logit": _LOGIT,
    "log": _LOG,
    "softmax": _SOFTMAX,
    "inverse": _INVERSE,
}


#: Built-in family registry.  Keyed by brms-style family name; the
#: compiler dispatches on family name and slots each family's
#: observe step + link into the emitted ``.qvr`` source.
families: Mapping[str, Family] = {
    "gaussian": Family(
        name="gaussian",
        location_link=_IDENTITY,
        observe_family="Normal",
        extra_observe_args=("sigma",),
        aux_params=(
            AuxParam(
                name="sigma",
                prior="HalfCauchy(2.0)",
                link=_LOG,
            ),
        ),
    ),
    "bernoulli": Family(
        name="bernoulli",
        location_link=_LOGIT,
        observe_family="Bernoulli",
    ),
    "binomial": Family(
        name="binomial",
        location_link=_LOGIT,
        observe_family="Bernoulli",
    ),
    "categorical": Family(
        name="categorical",
        location_link=_SOFTMAX,
        observe_family="Categorical",
    ),
    "poisson": Family(
        name="poisson",
        location_link=_LOG,
        observe_family="Poisson",
    ),
    "negative_binomial": Family(
        name="negative_binomial",
        location_link=_LOG,
        observe_family="NegativeBinomial",
        extra_observe_args=("disp",),
        aux_params=(
            AuxParam(
                name="disp",
                prior="Gamma(2.0, 2.0)",
                link=_LOG,
            ),
        ),
    ),
    "gamma": Family(
        name="gamma",
        location_link=_LOG,
        observe_family="Gamma",
        extra_observe_args=("shape",),
        aux_params=(
            AuxParam(
                name="shape",
                prior="Gamma(2.0, 2.0)",
                link=_LOG,
            ),
        ),
    ),
    "beta": Family(
        name="beta",
        location_link=_LOGIT,
        observe_family="Beta",
        extra_observe_args=("phi",),
        aux_params=(
            AuxParam(
                name="phi",
                prior="HalfCauchy(2.0)",
                link=_LOG,
            ),
        ),
    ),
    "student_t": Family(
        name="student_t",
        location_link=_IDENTITY,
        observe_family="StudentT",
        extra_observe_args=("nu", "sigma"),
        aux_params=(
            AuxParam(
                name="nu",
                prior="Gamma(2.0, 0.1)",
                link=_LOG,
            ),
            AuxParam(
                name="sigma",
                prior="HalfCauchy(2.0)",
                link=_LOG,
            ),
        ),
    ),
    "cumulative": Family(
        name="cumulative",
        location_link=_IDENTITY,
        observe_family="OrderedLogistic",
    ),
    "zero_inflated_poisson": Family(
        name="zero_inflated_poisson",
        location_link=_LOG,
        observe_family="ZeroInflatedPoisson",
        extra_observe_args=("zi",),
        aux_params=(
            AuxParam(
                name="zi",
                prior="Beta(2.0, 2.0)",
                link=_LOGIT,
            ),
        ),
    ),
    "hurdle_poisson": Family(
        name="hurdle_poisson",
        location_link=_LOG,
        observe_family="HurdlePoisson",
        extra_observe_args=("zi",),
        aux_params=(
            AuxParam(
                name="zi",
                prior="Beta(2.0, 2.0)",
                link=_LOGIT,
            ),
        ),
    ),
    "zero_one_inflated_beta": Family(
        name="zero_one_inflated_beta",
        location_link=_LOGIT,
        observe_family="ZeroOneInflatedBeta",
        extra_observe_args=("phi", "zoi", "coi"),
        aux_params=(
            AuxParam(
                name="phi",
                prior="HalfCauchy(2.0)",
                link=_LOG,
            ),
            AuxParam(
                name="zoi",
                prior="Beta(2.0, 2.0)",
                link=_LOGIT,
            ),
            AuxParam(
                name="coi",
                prior="Beta(2.0, 2.0)",
                link=_LOGIT,
            ),
        ),
    ),
    "mixture": Family(
        name="mixture",
        location_link=_IDENTITY,
        observe_family="MixtureNormal",
        extra_observe_args=("loc", "scale"),
        aux_params=(
            AuxParam(
                name="loc",
                prior="Normal(0.0, 5.0)",
                link=_IDENTITY,
            ),
            AuxParam(
                name="scale",
                prior="HalfCauchy(2.0)",
                link=_LOG,
            ),
        ),
    ),
}
