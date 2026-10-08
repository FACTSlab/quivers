"""The steps a `MonadicProgram` is a sequence of.

A program is built from four kinds of step, each a record of the
[`Step`][quivers.continuous.program_steps.Step] union:

- [`Draw`][quivers.continuous.program_steps.Draw] binds names to a
  sample from a morphism, applied to the program input or to earlier
  bindings;
- [`Observe`][quivers.continuous.program_steps.Observe] is a draw whose
  value is supplied as data and scored rather than sampled;
- [`Let`][quivers.continuous.program_steps.Let] binds a name
  deterministically, to a constant, an alias, or a function of earlier
  bindings;
- [`Score`][quivers.continuous.program_steps.Score] binds a name to a
  function of earlier bindings whose value also joins the log joint.

These are the records the DSL compiler emits, so a program assembled
from them in Python is the program the compiler would build from the
equivalent source.

A step function reads the environment by name. One that declares the
names it reads, through [`reading`][quivers.continuous.program_steps.reading],
is encoded as a step over those values alone; one that declares nothing
is encoded over every value bound before it, which is correct but makes
a long program's encoding grow with its length.
"""

from __future__ import annotations

import collections.abc
from typing import Literal, TypeVar

import didactic.api as dx
import torch

from quivers.continuous.morphisms import ContinuousMorphism
from quivers.core.morphisms import Morphism

StepCallable = TypeVar("StepCallable", bound=collections.abc.Callable[..., object])

type Environment = dict[str, torch.Tensor]
"""The bindings a step function reads, by name."""

type StepFunction = collections.abc.Callable[[Environment], torch.Tensor]
"""A let or score step's function of the environment."""


def _check_indices(indices: tuple[str, ...]) -> tuple[str, ...]:
    if not indices:
        raise ValueError("an Indexed argument needs at least one index")
    return tuple(indices)


class Indexed(dx.Model):
    """A binding gathered at index bindings, as ``mu[group]`` in source.

    With ``mu`` of shape ``(K, ...)`` and an integer ``group`` of shape
    ``(N,)``, the argument has shape ``(N, ...)``. Several indices
    gather left to right.

    Attributes
    ----------
    name : str
        The binding gathered from.
    indices : tuple[str, ...]
        The integer bindings it is gathered at, at least one.
    """

    name: str
    indices: tuple[str, ...] = dx.field(converter=_check_indices)


type StepArgument = str | Indexed
"""One argument of a draw or observe step: a binding's name, or an
[`Indexed`][quivers.continuous.program_steps.Indexed] gather of one."""

READS = "reads"
"""The attribute a step function declares the names it reads under."""


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


def reads_of(function: collections.abc.Callable[..., object]) -> frozenset[str] | None:
    """The environment names a step callable declares it reads.

    Parameters
    ----------
    function : Callable
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


class Step(dx.TaggedUnion, discriminator="kind"):
    """One step of a [`MonadicProgram`][quivers.continuous.MonadicProgram].

    The variants are [`Draw`][quivers.continuous.program_steps.Draw],
    [`Observe`][quivers.continuous.program_steps.Observe],
    [`Let`][quivers.continuous.program_steps.Let], and
    [`Score`][quivers.continuous.program_steps.Score].
    """


class Draw(Step):
    """Bind names to a sample from a morphism.

    In source, ``sample x <- f(a, b)``. With no arguments the morphism
    reads the program input; with one, it reads that binding; with
    several, it reads their concatenation along the feature axis, each
    placed by the declared parameter shape when the morphism is an
    inline distribution.

    Attributes
    ----------
    names : tuple[str, ...]
        The names bound. One name binds the sample; several destructure
        a tuple-returning sub-program, or split a product-codomain
        sample along its components.
    morphism : ContinuousMorphism | Morphism
        The kernel sampled from. A [`Morphism`][quivers.core.Morphism]
        is applied as a deterministic step through its tensor.
    args : tuple[StepArgument, ...] | None
        The bindings the morphism reads, or ``None`` for the program
        input.
    kind : Literal["draw"]
        The variant tag.
    """

    names: tuple[str, ...]
    morphism: ContinuousMorphism | Morphism = dx.field(opaque=True)
    args: tuple[StepArgument, ...] | None = dx.field(default=None, opaque=True)
    kind: Literal["draw"] = "draw"


class Observe(Step):
    """Score supplied data under a morphism.

    In source, ``observe y ~ f(a, b)``. The step reads its arguments as
    [`Draw`][quivers.continuous.program_steps.Draw] does; its value is
    supplied under its name through ``observations`` when sampling and
    through the intermediates when scoring, and its log density joins
    the log joint.

    Attributes
    ----------
    names : tuple[str, ...]
        The names bound to the observed value.
    morphism : ContinuousMorphism | Morphism
        The kernel the data are scored under.
    args : tuple[StepArgument, ...] | None
        The bindings the morphism reads, or ``None`` for the program
        input.
    kind : Literal["observe"]
        The variant tag.
    """

    names: tuple[str, ...]
    morphism: ContinuousMorphism | Morphism = dx.field(opaque=True)
    args: tuple[StepArgument, ...] | None = dx.field(default=None, opaque=True)
    kind: Literal["observe"] = "observe"


class Let(Step):
    """Bind a name deterministically.

    In source, ``let w = expression``.

    Attributes
    ----------
    name : str
        The name bound.
    value : float | str | StepFunction
        A constant, broadcast over the batch; the name of an earlier
        binding, which the step aliases; or a function of the
        environment, which may declare what it reads through
        [`reading`][quivers.continuous.program_steps.reading].
    kind : Literal["let"]
        The variant tag.
    """

    name: str
    value: float | str | StepFunction = dx.field(opaque=True)
    kind: Literal["let"] = "let"


class Score(Step):
    """Bind a name to a log-density contribution.

    The function's value is bound, as a let step's would be, and added
    to the log joint. The DSL compiler emits one for a ``marginalize``
    block, whose function sums the block's scores over the
    marginalized variable.

    Attributes
    ----------
    name : str
        The name bound.
    score : StepFunction
        The contribution, of shape ``(batch,)``.
    kind : Literal["score"]
        The variant tag.
    """

    name: str
    score: StepFunction = dx.field(opaque=True)
    kind: Literal["score"] = "score"


__all__ = [
    "READS",
    "Draw",
    "Environment",
    "Indexed",
    "Let",
    "Observe",
    "Score",
    "Step",
    "StepArgument",
    "StepFunction",
    "reading",
    "reads_of",
]
