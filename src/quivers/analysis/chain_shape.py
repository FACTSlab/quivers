"""ChainShape: per-step metadata derived from a compiled QVR program.

A :class:`ChainShape` walks a :class:`quivers.dsl.ast_nodes.Module`
AST once and records, for every ``let`` / ``latent`` / ``observe`` /
``marginalize`` step inside the module's program block:

* the bound variable name,
* the source line / column where the step was declared,
* the program's governing algebra (looked up from the top-level
  ``algebra <name>`` declaration via the algebra registry),
* the step's *depth* in the chain of stochastic / scoring binds,
  numbered from 1 at the first ``latent`` / ``observe`` and
  incremented at every subsequent stochastic step. ``let`` steps
  carry the depth of the most recent preceding stochastic step
  (``0`` if there isn't one yet),
* the step's intermediate axis size where it can be derived from
  the program's object declarations (``Resp`` cardinality for
  per-row lets / observes; the plate object cardinality for
  per-level latents; the morphism's codomain otherwise). When the
  size cannot be statically inferred we record ``None``.

The result is purely metadata; ChainShape never rewrites or
recompiles the program. The fields are what downstream tooling
(saturation-free init, saturation warnings, hook-based telemetry)
reads.
"""

from __future__ import annotations

from typing import Literal, Mapping

import didactic.api as dx

from quivers.core.algebras import Algebra
from quivers.dsl.ast_nodes import (
    CompositionDecl,
    LetStep,
    MarginalizeStep,
    Module,
    ObserveStep,
    ProgramDecl,
    ProgramStep,
    SampleStep,
    TypeDecl,
    TypeExpr,
    TypeFromExpr,
    TypeName,
)
from quivers.dsl.compiler._prelude import _ALGEBRA_REGISTRY

StepKind = Literal["latent", "observe", "marginalize", "let"]


class StepShape(dx.Model):
    """Per-step metadata derived by :class:`ChainShape`.

    Attributes
    ----------
    name : str
        Bound variable name (the LHS of the step).
    kind : str
        One of ``"latent"`` / ``"observe"`` / ``"marginalize"``
        (stochastic-bind steps) or ``"let"`` (deterministic).
    source_line, source_col : int
        Source position where the step was declared.
    depth : int
        The 1-indexed position of the step in the chain of
        stochastic binds (``latent`` / ``observe`` /
        ``marginalize``). Let steps inherit the depth of the most
        recent stochastic predecessor; a let before any stochastic
        step has depth ``0``.
    algebra_name : str
        Name of the program's governing algebra.
    intermediate_size : int | None
        Cardinality of the bound coordinate's intermediate axis,
        when statically inferable from the AST. ``None`` if not.
    """

    name: str
    kind: StepKind
    source_line: int = 0
    source_col: int = 0
    depth: int = 0
    algebra_name: str = ""
    intermediate_size: int | None = None


class ChainShape(dx.Model):
    """Sequence of :class:`StepShape` records for a program.

    Attributes
    ----------
    algebra_name : str
        Name of the program's governing algebra (the top-level
        ``algebra <name>`` declaration).
    object_cardinalities : Mapping[str, int]
        Cardinality of every ``object X : N`` declaration preceding
        the program. Used to resolve plate sizes referenced by
        ``: T`` annotations on latents / observes.
    steps : tuple[StepShape, ...]
        Per-step metadata in source order.
    """

    algebra_name: str = ""
    object_cardinalities: Mapping[str, int] = dx.field(
        default_factory=dict, opaque=True
    )
    steps: tuple[StepShape, ...] = ()

    @property
    def stochastic_depth(self) -> int:
        """Number of stochastic-bind steps (``latent``, ``observe``,
        or ``marginalize``)."""
        return sum(1 for s in self.steps if s.kind != "let")

    @property
    def algebra(self) -> Algebra | None:
        """Resolve :attr:`algebra_name` against the algebra
        registry. Returns ``None`` if the name is unknown (e.g. a
        user-defined inline composition rule not registered)."""
        rule = _ALGEBRA_REGISTRY.get(self.algebra_name)
        if isinstance(rule, Algebra):
            return rule
        return None

    def latents(self) -> tuple[StepShape, ...]:
        """All ``latent`` (sample-mode) steps."""
        return tuple(s for s in self.steps if s.kind == "latent")

    def observes(self) -> tuple[StepShape, ...]:
        """All ``observe`` (score-mode) steps."""
        return tuple(s for s in self.steps if s.kind == "observe")

    @classmethod
    def from_module(cls, module: Module) -> "ChainShape":
        """Build a :class:`ChainShape` from a compiled
        :class:`Module` AST.

        Walks the module's statement list, captures the algebra
        name from any top-level :class:`CompositionDecl`, captures
        every :class:`TypeDecl`'s numeric cardinality, then walks
        the unique :class:`ProgramDecl`'s steps in source order.
        :class:`MarginalizeStep` bodies are walked recursively;
        their inner steps are recorded after the enclosing
        :class:`MarginalizeStep`.
        """
        algebra_name = "product_fuzzy"
        cardinalities: dict[str, int] = {}
        program: ProgramDecl | None = None
        for stmt in module.statements:
            if isinstance(stmt, CompositionDecl):
                algebra_name = stmt.name
            elif isinstance(stmt, TypeDecl):
                cardinality = _type_decl_cardinality(stmt)
                if cardinality is not None:
                    cardinalities[stmt.name] = cardinality
            elif isinstance(stmt, ProgramDecl) and program is None:
                program = stmt

        steps: list[StepShape] = []
        depth = 0

        def record(
            name: str,
            kind: StepKind,
            line: int,
            col: int,
            intermediate: int | None,
        ) -> None:
            nonlocal depth
            depth += 1
            steps.append(
                StepShape(
                    name=name,
                    kind=kind,
                    source_line=line,
                    source_col=col,
                    depth=depth,
                    algebra_name=algebra_name,
                    intermediate_size=intermediate,
                )
            )

        def walk(program_steps: tuple[ProgramStep, ...]) -> None:
            for step in program_steps:
                if isinstance(step, SampleStep):
                    record(
                        step.vars[0] if step.vars else "",
                        "latent",
                        step.line,
                        step.col,
                        _index_size(step.index, cardinalities),
                    )
                elif isinstance(step, ObserveStep):
                    record(
                        step.var,
                        "observe",
                        step.line,
                        step.col,
                        _index_size(step.index, cardinalities),
                    )
                elif isinstance(step, MarginalizeStep):
                    record(
                        step.var,
                        "marginalize",
                        step.line,
                        step.col,
                        _index_size(step.index, cardinalities),
                    )
                    walk(step.scope)
                elif isinstance(step, LetStep):
                    steps.append(
                        StepShape(
                            name=step.name,
                            kind="let",
                            source_line=step.line,
                            source_col=step.col,
                            depth=depth,
                            algebra_name=algebra_name,
                            intermediate_size=None,
                        )
                    )

        if program is not None:
            walk(program.draws)

        return cls(
            algebra_name=algebra_name,
            object_cardinalities=dict(cardinalities),
            steps=tuple(steps),
        )


def _type_decl_cardinality(decl: TypeDecl) -> int | None:
    """Read a numeric cardinality off a ``type X : N`` decl.

    Returns ``None`` for non-numeric type forms (products, coproducts,
    free monoids, residuated patterns, continuous spaces).
    """
    init = decl.init
    if not isinstance(init, TypeFromExpr):
        return None
    expr = init.expr
    if not isinstance(expr, TypeName):
        return None
    try:
        return int(expr.name)
    except ValueError:
        return None


def _index_size(
    index: TypeExpr | None, cardinalities: dict[str, int],
) -> int | None:
    """Best-effort cardinality of a step's plate index.

    Returns ``1`` for unindexed scalar steps (no ``: T``), the integer
    cardinality when the index is a numeric literal or registered
    named object, and ``None`` for product / coproduct / free-monoid
    indices whose runtime size is opaque to the static analyser.
    """
    if index is None:
        return 1
    if isinstance(index, TypeName):
        try:
            return int(index.name)
        except ValueError:
            return cardinalities.get(index.name)
    return None
