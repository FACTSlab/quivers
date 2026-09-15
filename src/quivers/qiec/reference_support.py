"""Shared helpers of the reference distribution backend.

The argument readers, scalar special functions, and the sampler and
density signatures live here so the family tables can be split across
modules without any of them importing the backend that installs them.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import math
import random


class DistributionError(ValueError):
    """A distribution could not be constructed, sampled, or scored."""


def real_parameter(arguments: Mapping[str, object], name: str, family: str) -> float:
    """Read one real-valued parameter.

    Parameters
    ----------
    arguments : Mapping[str, object]
        The named parameters.
    name : str
        The parameter wanted.
    family : str
        The family, for the diagnostic.

    Returns
    -------
    float
        The parameter as a float.

    Raises
    ------
    DistributionError
        If the parameter is absent or not a number.
    """
    try:
        value = arguments[name]
    except KeyError as error:
        raise DistributionError(
            f"reference backend needs parameter {name!r} of {family}"
        ) from error
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise DistributionError(f"parameter {name!r} of {family} is not a number")
    return float(value)


def vector_parameter(
    arguments: Mapping[str, object], name: str, family: str
) -> tuple[float, ...]:
    """Read one vector-valued parameter.

    Parameters
    ----------
    arguments : Mapping[str, object]
        The named parameters.
    name : str
        The parameter wanted.
    family : str
        The family, for the diagnostic.

    Returns
    -------
    tuple[float, ...]
        The parameter's components.

    Raises
    ------
    DistributionError
        If the parameter is absent or not a sequence of numbers.
    """
    try:
        value = arguments[name]
    except KeyError as error:
        raise DistributionError(
            f"reference backend needs parameter {name!r} of {family}"
        ) from error
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise DistributionError(f"parameter {name!r} of {family} is not a vector")
    return tuple(float(item) for item in value)


def probability(arguments: Mapping[str, object], family: str) -> float:
    """The success probability of a Bernoulli-like family.

    Parameters
    ----------
    arguments : Mapping[str, object]
        The named parameters, holding ``probs`` or ``logits``.
    family : str
        The family, for the diagnostic.

    Returns
    -------
    float
        The probability, from ``probs`` directly or ``logits`` through the
        logistic function.

    Raises
    ------
    DistributionError
        If neither parameterization is supplied.
    """
    if "probs" in arguments:
        return real_parameter(arguments, "probs", family)
    if "logits" in arguments:
        return 1.0 / (1.0 + math.exp(-real_parameter(arguments, "logits", family)))
    raise DistributionError(f"{family} needs probs or logits")


def simplex(arguments: Mapping[str, object], family: str) -> tuple[float, ...]:
    """The category probabilities of a categorical-like family.

    Parameters
    ----------
    arguments : Mapping[str, object]
        The named parameters, holding ``probs`` or ``logits``.
    family : str
        The family, for the diagnostic.

    Returns
    -------
    tuple[float, ...]
        Probabilities summing to one, from ``probs`` normalized or
        ``logits`` through the softmax.

    Raises
    ------
    DistributionError
        If neither parameterization is supplied.
    """
    if "probs" in arguments:
        probs = vector_parameter(arguments, "probs", family)
        total = sum(probs)
        return tuple(item / total for item in probs)
    if "logits" in arguments:
        logits = vector_parameter(arguments, "logits", family)
        peak = max(logits)
        weights = [math.exp(item - peak) for item in logits]
        total = sum(weights)
        return tuple(item / total for item in weights)
    raise DistributionError(f"{family} needs probs or logits")


def log_choose(n: float, k: float) -> float:
    """``log C(n, k)`` through the log-gamma function.

    Parameters
    ----------
    n : float
        The number of trials.
    k : float
        The number of successes.

    Returns
    -------
    float
        The log binomial coefficient.
    """
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def normal_log_prob(value: float, loc: float, scale: float) -> float:
    """The log density of a normal distribution.

    Parameters
    ----------
    value : float
        The point.
    loc : float
        The mean.
    scale : float
        The standard deviation.

    Returns
    -------
    float
        The log density.
    """
    return (
        -0.5 * ((value - loc) / scale) ** 2
        - math.log(scale)
        - 0.5 * math.log(2 * math.pi)
    )


type Sampler = Callable[[Mapping[str, object], random.Random], object]
type Density = Callable[[Mapping[str, object], object], float]


def finite_value(value: object, family: str) -> float:
    """Read the point a scalar family is scored at.

    Parameters
    ----------
    value : object
        The point.
    family : str
        The family, for the diagnostic.

    Returns
    -------
    float
        The point as a float.

    Raises
    ------
    DistributionError
        If the point is not a number.
    """
    if isinstance(value, bool):
        return float(value)
    if not isinstance(value, int | float):
        raise DistributionError(f"{family} is scored at a non-numeric value {value!r}")
    return float(value)


__all__ = [
    "Density",
    "DistributionError",
    "Sampler",
    "finite_value",
    "log_choose",
    "normal_log_prob",
    "probability",
    "real_parameter",
    "simplex",
    "vector_parameter",
]
