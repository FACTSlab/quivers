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
    AlgebraDecl,
    BindStep,
    LetStep,
    Module,
    ObjectDecl,
    ProgramDecl,
    ProgramStep,
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
        name from any top-level :class:`AlgebraDecl`, captures every
        ``object`` cardinality, then walks the unique
        :class:`ProgramDecl`'s steps in source order. ``marginalize``
        bodies are walked recursively; their inner steps are
        recorded after the enclosing ``marginalize`` step.
        """
        algebra_name = "product_fuzzy"
        cardinalities: dict[str, int] = {}
        program: ProgramDecl | None = None
        for stmt in module.statements:
            if isinstance(stmt, AlgebraDecl):
                algebra_name = stmt.name
            elif isinstance(stmt, ObjectDecl):
                cardinality = _object_cardinality(stmt)
                if cardinality is not None:
                    cardinalities[stmt.name] = cardinality
            elif isinstance(stmt, ProgramDecl) and program is None:
                program = stmt

        steps: list[StepShape] = []
        depth = 0

        def walk(program_steps: tuple[ProgramStep, ...]) -> None:
            nonlocal depth
            for step in program_steps:
                if isinstance(step, BindStep):
                    if step.mode == "sample":
                        depth += 1
                        kind: StepKind = "latent"
                    elif step.mode == "score":
                        depth += 1
                        kind = "observe"
                    else:
                        depth += 1
                        kind = "marginalize"
                    intermediate = _bind_step_size(step, cardinalities)
                    steps.append(
                        StepShape(
                            name=step.vars[0] if step.vars else "",
                            kind=kind,
                            source_line=step.line,
                            source_col=step.col,
                            depth=depth,
                            algebra_name=algebra_name,
                            intermediate_size=intermediate,
                        )
                    )
                    if step.scope is not None:
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


def _object_cardinality(decl: ObjectDecl) -> int | None:
    """Read a numeric cardinality off an ``object X : N`` decl.

    Returns ``None`` for the non-numeric type forms (product,
    coproduct, free monoid, type aliases).
    """
    type_expr = decl.type_expr
    if not isinstance(type_expr, TypeName):
        return None
    try:
        return int(type_expr.name)
    except ValueError:
        return None


def _bind_step_size(step: BindStep, cardinalities: dict[str, int]) -> int | None:
    """Best-effort cardinality of the value bound by a :class:`BindStep`.

    For plate steps (``: T``), this is the cardinality of the plate
    type ``T`` when ``T`` is a numeric or registered object. For
    unindexed scalar binds (no ``: T``), the value is scalar; we
    return ``1``. Returns ``None`` when the plate type is a
    non-numeric TypeExpr (product / coproduct / free monoid) since
    its cardinality may not be known until runtime.
    """
    if step.index is None:
        return 1
    if isinstance(step.index, TypeName):
        try:
            return int(step.index.name)
        except ValueError:
            return cardinalities.get(step.index.name)
    return None
