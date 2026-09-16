"""The step records a `MonadicProgram` is a sequence of.

A draw step names the morphism it draws from and the variables it binds,
a let step a deterministic binding, and a score step a callable whose
result joins the log joint. The records carry no behaviour: the reference
machine runs them through the program's kernel encoding.

A step callable reads the environment by name. One that declares the
names it reads, through [`reading`][quivers.continuous.program_steps.reading],
is encoded as a step over those values alone; one that declares nothing
is encoded over every value bound before it, which is correct but makes
a long program's encoding grow with its length.
"""

from __future__ import annotations

import collections.abc
from typing import TypeVar

import torch

StepCallable = TypeVar("StepCallable", bound=collections.abc.Callable[..., object])

#: The attribute a step callable declares the names it reads under.
READS = "reads"


def reading(
    function: StepCallable, names: collections.abc.Iterable[str]
) -> StepCallable:
    """Declare the environment names a step callable reads.

    Parameters
    ----------
    function : StepCallable
        The callable, a let or score step's value.
    names : Iterable[str]
        The names it reads from the environment it is called with.

    Returns
    -------
    StepCallable
        The same callable, carrying the names under ``reads``.
    """
    setattr(function, READS, frozenset(names))
    return function


def reads_of(function: object) -> frozenset[str] | None:
    """The environment names a step callable declares it reads.

    Parameters
    ----------
    function : object
        A let or score step's value.

    Returns
    -------
    frozenset[str] | None
        The declared names, or ``None`` when the callable declares none
        and may read any bound value.
    """
    names = getattr(function, READS, None)
    if names is None:
        return None
    assert isinstance(names, frozenset)
    return names


class _StepSpec:
    """Metadata record for a single draw step.

    Parameters
    ----------
    vars : tuple[str, ...]
        Bound variable name(s). Single-element for simple binding,
        multi-element for destructuring.
    morphism_name : str
        Key into the program's morphism module dict.
    args : tuple[str, ...] or None
        Names of bound variables to use as input (stacked along
        feature dim), or None for the program input.
    """

    __slots__ = ("vars", "morphism_name", "args", "is_observed", "is_marginalized")

    def __init__(
        self,
        vars: tuple[str, ...],
        morphism_name: str,
        args: tuple[str, ...] | None,
        is_observed: bool = False,
        is_marginalized: bool = False,
    ) -> None:
        self.vars: tuple[str, ...] = vars
        self.morphism_name: str = morphism_name
        self.args: tuple[str, ...] | None = args
        self.is_observed: bool = is_observed
        # When True, the variable bound by this step is fully
        # integrated out by a subsequent `_ScoreSpec`
        # (a marginalize block's runtime callable). It must NOT be
        # surfaced as a latent to inference algorithms: the guide
        # cannot reparameterize it (the support is typically
        # discrete) and the marginalize already accounts for its
        # density contribution in the score step.
        self.is_marginalized: bool = is_marginalized


class _LetSpec:
    """Metadata for a deterministic let binding (no morphism).

    Parameters
    ----------
    var : str
        Variable name to bind.
    value : float, str, or callable
        Constant literal (float), name of a bound variable to
        alias (str), or a callable that computes the value from
        the environment dict.
    """

    __slots__ = ("var", "value")

    def __init__(
        self, var: str, value: float | str | collections.abc.Callable[..., torch.Tensor]
    ) -> None:
        self.var = var
        self.value = value


class _ScoreSpec:
    """Metadata for a step whose callable contributes to log_joint.

    Used for marginalize blocks: the callable computes a log-density
    contribution (the marginal log-likelihood obtained by summing the
    body's per-latent-value scores against the categorical prior).
    The callable's return value is both stored in ``env[var]`` for
    later reference and added to ``total`` in ``log_joint``. Under
    ``rsample`` it behaves like a let binding (its result is stored;
    nothing is scored).

    Parameters
    ----------
    var : str
        Variable name to bind the callable's return value to.
    score : callable
        ``score(env) -> torch.Tensor`` of shape ``(batch,)``: the
        per-sample log-density contribution.
    """

    __slots__ = ("var", "score")

    def __init__(
        self,
        var: str,
        score: collections.abc.Callable[..., torch.Tensor],
    ) -> None:
        self.var = var
        self.score = score


__all__ = ["READS", "_LetSpec", "_ScoreSpec", "_StepSpec", "reading", "reads_of"]
