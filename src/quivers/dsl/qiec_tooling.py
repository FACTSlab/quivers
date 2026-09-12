"""Shared QIEC analysis and symbol metadata for interactive tooling.

The runtime compiler and the QIEC lowerer are deliberately separate pipelines.
This module is the small adapter used by ``qvr check``, the REPL, and the LSP:
it lowers and validates QIEC declarations, while continuing to compile the
other QVR declarations in the same source module with their compiler.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from quivers.dsl.ast_nodes import Module, Statement
from quivers.dsl.ast_nodes.qiec import (
    QiecComputationDecl,
    QiecEffectDecl,
    QiecEffectInstanceDecl,
    QiecFamilyDecl,
    QiecHandlerDecl,
    QiecIndexDecl,
)
from quivers.dsl.compiler import Compiler, CompileError
from quivers.dsl.qiec_lowering import (
    QIEC_STATEMENT_TYPES,
    has_qiec_surface,
    non_qiec_projection,
)
from quivers.qiec import QiecModule


type QiecBindingKind = Literal[
    "index",
    "index-constructor",
    "family",
    "constructor",
    "effect",
    "operation",
    "instance",
    "handler",
    "computation",
]


QIEC_DIAGNOSTIC_CODES = frozenset(
    {
        "qiec-route",
        "qiec-kind",
        "qiec-index",
        "qiec-row",
        "qiec-coverage",
        "qiec-skolem-escape",
        "qiec-handler",
        "qiec-unhandled-effect",
        "qiec-backend",
    }
)


@dataclass(frozen=True, slots=True)
class QiecToolingDiagnostic:
    """A frontend-neutral QIEC diagnostic."""

    message: str
    code: str
    line: int = 0
    col: int = 0


@dataclass(frozen=True, slots=True)
class QiecBinding:
    """One source-level name exposed to completion and navigation."""

    name: str
    kind: QiecBindingKind
    declaration: object
    owner: Statement

    @property
    def qualified_name(self) -> str:
        """Return the declaration-qualified identity of this binding.

        Signature members are scoped by their owning index, family, or effect.
        Top-level declarations already have a module-level identity, so their
        qualified spelling is just their authored name.
        """

        if self.declaration is self.owner:
            return self.name
        owner_name = getattr(self.owner, "name", None)
        if isinstance(owner_name, str):
            return f"{owner_name}.{self.name}"
        return self.name

    @property
    def semantic_kind(self) -> str:
        if self.kind in {"index", "family", "effect"}:
            return "type"
        if self.kind == "instance":
            return "variable"
        if self.kind in {"index-constructor", "constructor", "operation"}:
            return "function"
        if self.kind in {"handler", "computation"}:
            return "function"
        return "variable"


@dataclass(slots=True)
class ToolingAnalysis:
    """The two elaboration products needed by user-facing tools."""

    compiler: Compiler
    env: dict[str, Any]
    non_qiec_module: Module
    qiec_module: QiecModule | None
    qiec_diagnostics: tuple[QiecToolingDiagnostic, ...]
    compile_error: CompileError | None


def is_qiec_statement(statement: Statement) -> bool:
    """Return whether ``statement`` belongs to the v0.19 QIEC surface."""

    return isinstance(statement, QIEC_STATEMENT_TYPES)


def analyze_module(
    module: Module,
    *,
    module_name: str,
    file_path: str | None = None,
) -> ToolingAnalysis:
    """Run the QIEC lowerer and existing compiler without conflating them.

    QIEC failures are returned as stable diagnostics. Existing compiler errors
    remain separately typed so callers can retain their established diagnostic
    code and presentation.
    """

    qiec_module: QiecModule | None = None
    qiec_diags: list[QiecToolingDiagnostic] = []
    non_qiec_module = non_qiec_projection(module)
    try:
        compiler = Compiler(
            module,
            module_name=module_name,
            file_path=file_path or "<source>",
        )
        qiec_module = compiler.qiec_module
    except Exception as error:
        if not has_qiec_surface(module):
            raise
        qiec_diags.append(qiec_diagnostic(error))
        compiler = Compiler(non_qiec_module)
    compile_error: CompileError | None = None
    try:
        env = compiler.compile_env()
    except CompileError as error:
        compile_error = error
        env = {}
    for name, declaration in qiec_surface_values(module).items():
        env.setdefault(name, declaration)
    return ToolingAnalysis(
        compiler=compiler,
        env=env,
        non_qiec_module=non_qiec_module,
        qiec_module=qiec_module,
        qiec_diagnostics=tuple(qiec_diags),
        compile_error=compile_error,
    )


def qiec_diagnostic(error: Exception) -> QiecToolingDiagnostic:
    """Normalize lowerer/kernel exceptions into the public QIEC code set."""

    raw_code = str(getattr(error, "code", "") or "")
    code = raw_code if raw_code in QIEC_DIAGNOSTIC_CODES else _infer_code(error)
    return QiecToolingDiagnostic(
        message=str(getattr(error, "message", "") or error),
        code=code,
        line=max(0, int(getattr(error, "line", 0) or 0)),
        col=max(
            0,
            int(getattr(error, "col", None) or getattr(error, "column", 0) or 0),
        ),
    )


def _infer_code(error: Exception) -> str:
    text = f"{type(error).__name__} {error}".lower()
    if "route" in text or "protocol" in text or "lowering" in text:
        return "qiec-route"
    if "coverage" in text or "exhaust" in text or "unreachable" in text:
        return "qiec-coverage"
    if "skolem" in text or "escape" in text:
        return "qiec-skolem-escape"
    if "handler" in text or "resumption" in text or "resume" in text:
        return "qiec-handler"
    if "unhandled" in text:
        return "qiec-unhandled-effect"
    if "row" in text or "effect instance" in text or "lacks" in text:
        return "qiec-row"
    if "index" in text or "constructor" in text or "family" in text:
        return "qiec-index"
    return "qiec-kind"


def qiec_bindings(module: Module) -> tuple[QiecBinding, ...]:
    """Return top-level and signature-member names in source order."""

    out: list[QiecBinding] = []
    for statement in module.statements:
        if isinstance(statement, QiecIndexDecl):
            out.append(QiecBinding(statement.name, "index", statement, statement))
            out.extend(
                QiecBinding(
                    constructor.name, "index-constructor", constructor, statement
                )
                for constructor in statement.constructors
            )
        elif isinstance(statement, QiecFamilyDecl):
            out.append(QiecBinding(statement.name, "family", statement, statement))
            out.extend(
                QiecBinding(constructor.name, "constructor", constructor, statement)
                for constructor in statement.constructors
            )
        elif isinstance(statement, QiecEffectDecl):
            out.append(QiecBinding(statement.name, "effect", statement, statement))
            out.extend(
                QiecBinding(operation.name, "operation", operation, statement)
                for operation in statement.operations
            )
        elif isinstance(statement, QiecEffectInstanceDecl):
            out.append(QiecBinding(statement.name, "instance", statement, statement))
        elif isinstance(statement, QiecHandlerDecl):
            out.append(QiecBinding(statement.name, "handler", statement, statement))
        elif isinstance(statement, QiecComputationDecl):
            out.append(QiecBinding(statement.name, "computation", statement, statement))
    return tuple(out)


def qiec_binding_map(module: Module) -> dict[str, QiecBinding]:
    """Return unambiguous source identities for QIEC bindings.

    Every signature member is available through ``Owner.member``. A bare
    member spelling remains available when exactly one declaration owns that
    spelling and it does not collide with a top-level declaration. This keeps
    the convenient ``:type get`` form for a single effect while preventing a
    later ``Writer.get`` from silently replacing ``Reader.get``.
    """

    bindings = qiec_bindings(module)
    out = {
        binding.name: binding
        for binding in bindings
        if binding.declaration is binding.owner
    }
    members_by_name: dict[str, list[QiecBinding]] = {}
    for binding in bindings:
        if binding.declaration is binding.owner:
            continue
        out[binding.qualified_name] = binding
        members_by_name.setdefault(binding.name, []).append(binding)
    for name, members in members_by_name.items():
        if name not in out and len(members) == 1:
            out[name] = members[0]
    return out


def qiec_binding_candidates(module: Module, name: str) -> tuple[QiecBinding, ...]:
    """Return every binding matching an authored or qualified spelling."""

    if "." in name:
        exact = qiec_binding_map(module).get(name)
        return (exact,) if exact is not None else ()
    return tuple(binding for binding in qiec_bindings(module) if binding.name == name)


def qiec_env_kinds(module: Module) -> dict[str, str]:
    """Return semantic-token classifications for QIEC names."""

    return {
        name: binding.semantic_kind
        for name, binding in qiec_binding_map(module).items()
    }


def qiec_surface_values(module: Module) -> dict[str, object]:
    """Expose authored QIEC declarations in the shared inspection environment."""

    return {
        name: binding.declaration for name, binding in qiec_binding_map(module).items()
    }


def qiec_module_name(file_path: str | Path | None) -> str:
    """Derive a deterministic source module name for editor/CLI entry points."""

    if file_path is None:
        return "repl"
    raw = str(file_path)
    if raw.startswith("file://"):
        raw = raw.removeprefix("file://")
    stem = Path(raw).stem
    return stem or "module"


__all__ = [
    "QIEC_DIAGNOSTIC_CODES",
    "QIEC_STATEMENT_TYPES",
    "QiecBinding",
    "QiecToolingDiagnostic",
    "ToolingAnalysis",
    "analyze_module",
    "has_qiec_surface",
    "is_qiec_statement",
    "qiec_binding_map",
    "qiec_binding_candidates",
    "qiec_bindings",
    "qiec_diagnostic",
    "qiec_env_kinds",
    "qiec_module_name",
    "qiec_surface_values",
]
