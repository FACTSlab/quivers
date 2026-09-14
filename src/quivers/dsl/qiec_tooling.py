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
    QiecSequenceComputation,
    QiecPureBinding,
    QiecInstanceComputation,
    QiecHandlerReturnClause,
    QiecHandlerOperationClause,
    QiecHandleComputation,
    QiecCaseComputation,
    QiecIfComputation,
    QiecBindComputation,
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
    "local",
    "parameter",
    "scoped-instance",
    "branch-binder",
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
        "qiec-handler-body",
        "qiec-unhandled-effect",
        "qiec-backend",
        "qiec-call",
        "qiec-call-arity",
        "qiec-recursion",
        "qiec-resumption",
        "qiec-instance-escape",
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
    scope: tuple[int, int] | None = None
    """The 1-based line range the binding is visible over.

    ``None`` for a module-level declaration, which is visible throughout.
    A local binder carries the span of the region that may refer to it,
    so a tool can offer it at a position inside that region and nowhere
    else.
    """

    @property
    def is_local(self) -> bool:
        """Whether this binding is confined to part of one declaration.

        Returns
        -------
        bool
            True when the binding carries a lexical scope, which is how
            a caller distinguishes a name that may be offered anywhere in
            the module from one that may not.
        """
        return self.scope is not None

    def covers(self, line: int) -> bool:
        """Whether this binding is visible at a source line.

        Parameters
        ----------
        line : int
            1-based line number.

        Returns
        -------
        bool
            True for a module-level binding at any line, and for a local
            binding only inside its own span. A tool that offers a local
            outside its span is offering a name that does not resolve.
        """
        if self.scope is None:
            return True
        start, end = self.scope
        return start <= line <= end

    @property
    def qualified_name(self) -> str:
        """Return the declaration-qualified identity of this binding.

        Signature members are scoped by their owning index, family, or effect.
        Top-level declarations already have a module-level identity, so their
        qualified spelling is just their authored name.

        Returns
        -------
        str
            ``Owner.member`` for a signature member, and the bare name
            for a top-level declaration.
        """

        if self.declaration is self.owner:
            return self.name
        owner_name = getattr(self.owner, "name", None)
        if isinstance(owner_name, str):
            return f"{owner_name}.{self.name}"
        return self.name

    @property
    def semantic_kind(self) -> str:
        """The token class a highlighter should give this binding.

        Returns
        -------
        str
            One of ``"type"``, ``"function"``, or ``"variable"``. The
            mapping is coarser than `kind` on purpose: an editor theme
            colours by role, and an index, a family, and an effect all
            read as types at a use site.
        """
        if self.kind in {"index", "family", "effect"}:
            return "type"
        if self.kind == "instance":
            return "variable"
        if self.kind in {"index-constructor", "constructor", "operation"}:
            return "function"
        if self.kind in {"handler", "computation"}:
            return "function"
        if self.kind == "branch-binder":
            return "variable"
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
    """Return whether ``statement`` belongs to the v0.19 QIEC surface.

    Parameters
    ----------
    statement : Statement
        The parsed statement to classify.

    Returns
    -------
    bool
        True for a QIEC declaration. This is the split the tooling layer
        works from: QIEC statements go to the lowerer and everything else
        to the existing compiler.
    """

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

    Parameters
    ----------
    module : Module
        The parsed module, carrying both QIEC and ordinary declarations.
    module_name : str
        Name the QIEC lowerer derives stable identities from. Two modules
        of the same name produce the same identities, so this must match
        what the rest of the pipeline uses.
    file_path : str or None
        Path recorded on diagnostics, or None when the source has no file.

    Returns
    -------
    ToolingAnalysis
        The compiler, its environment, the non-QIEC projection, the
        lowered QIEC module when lowering succeeded, the QIEC
        diagnostics, and the compiler error when one occurred. Nothing
        raises: a tool needs whichever half succeeded even when the other
        failed.
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
    """Normalize lowerer/kernel exceptions into the public QIEC code set.

    Parameters
    ----------
    error : Exception
        The exception raised by the lowerer or the kernel.

    Returns
    -------
    QiecToolingDiagnostic
        A diagnostic whose code is drawn from the published set. An
        exception carrying a recognised code keeps it; anything else is
        classified by its text, so a caller always receives a stable code
        rather than an implementation detail.
    """

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
    """Classify an exception that carries no stable code of its own.

    Parameters
    ----------
    error : Exception
        The exception to classify.

    Returns
    -------
    str
        A member of the published code set. The tests are ordered from
        most specific to least, and ``"qiec-kind"`` is the fallback,
        since a failure that matches nothing more specific is a kinding
        failure more often than anything else.
    """
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
    """Return top-level and signature-member names in source order.

    Parameters
    ----------
    module : Module
        The parsed module to read.

    Returns
    -------
    tuple[QiecBinding, ...]
        Every module-level QIEC name, and the members each declaration
        owns. These carry no scope, being visible throughout. Names local
        to a body come from
        [`qiec_local_bindings`][quivers.dsl.qiec_tooling.qiec_local_bindings]
        instead.
    """

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


def _computation_span(node: object) -> tuple[int, int]:
    """The 1-based line range a computation subtree occupies.

    Parameters
    ----------
    node : object
        A computation, clause, or any node carrying `line`, together with
        whatever it nests.

    Returns
    -------
    tuple[int, int]
        The smallest and largest line found. A node whose subtree records
        no position yields ``(0, 0)``, which `QiecBinding.covers` then
        treats as covering nothing rather than everything.
    """
    lines: list[int] = []

    def visit(value: object) -> None:
        """Record a node's line and descend into its fields.

        Parameters
        ----------
        value : object
            A node, a sequence of nodes, or a leaf. Anything without a
            position and without fields contributes nothing.
        """
        if isinstance(value, tuple | list):
            for item in value:
                visit(item)
            return
        line = getattr(value, "line", None)
        if isinstance(line, int) and line > 0:
            lines.append(line)
        dump = getattr(value, "model_dump", None)
        if dump is None:
            return
        # `model_dump` gives the field names; the values are read off the
        # node itself so nested models are visited rather than the plain
        # dictionaries a dump would produce.
        for name in dump():
            if name in ("line", "col", "kind"):
                continue
            visit(getattr(value, name, None))

    visit(node)
    if not lines:
        return (0, 0)
    return (min(lines), max(lines))


def _walk_computation_bindings(
    computation: object,
    owner: Statement,
    out: list[QiecBinding],
) -> None:
    """Collect the binders a computation introduces, with their scopes.

    A binder's scope is the span of the region that may refer to it, not
    the span of the construct that introduces it. A `let` is visible in
    its continuation, a scoped instance and a case branch's names in
    their bodies, so each scope is taken from the sub-node the name
    reaches rather than from the binding form itself.

    Parameters
    ----------
    computation : object
        The computation to walk. Anything that is not a recognised
        computation form is ignored, so this is safe to call on a clause
        body that may be absent.
    owner : Statement
        The declaration the binders belong to, recorded on each binding.
    out : list[QiecBinding]
        Accumulator, appended to in source order.
    """
    if isinstance(computation, QiecPureBinding | QiecBindComputation):
        out.append(
            QiecBinding(
                computation.binder.name,
                "local",
                computation.binder,
                owner,
                _computation_span(computation.then),
            )
        )
        if isinstance(computation, QiecBindComputation):
            _walk_computation_bindings(computation.first, owner, out)
        _walk_computation_bindings(computation.then, owner, out)
        return
    if isinstance(computation, QiecInstanceComputation):
        out.append(
            QiecBinding(
                computation.name,
                "scoped-instance",
                computation,
                owner,
                _computation_span(computation.body),
            )
        )
        _walk_computation_bindings(computation.body, owner, out)
        return
    if isinstance(computation, QiecSequenceComputation):
        _walk_computation_bindings(computation.first, owner, out)
        _walk_computation_bindings(computation.then, owner, out)
        return
    if isinstance(computation, QiecHandleComputation):
        _walk_computation_bindings(computation.body, owner, out)
        return
    if isinstance(computation, QiecIfComputation):
        _walk_computation_bindings(computation.then, owner, out)
        _walk_computation_bindings(computation.otherwise, owner, out)
        return
    if isinstance(computation, QiecCaseComputation):
        for branch in computation.branches:
            span = _computation_span(branch.body)
            out.extend(
                QiecBinding(field.name, "branch-binder", field, owner, span)
                for field in branch.fields
            )
            _walk_computation_bindings(branch.body, owner, out)
        return


def qiec_local_bindings(module: Module) -> tuple[QiecBinding, ...]:
    """Every binder local to a declaration body, in source order.

    These are the names a tool may offer only inside part of a file:
    `let` bindings, a handler clause's parameters, a scoped instance, and
    the names a case branch binds. Each carries the line range it is
    visible over, so a caller filters with
    [`QiecBinding.covers`][tests-free reference].

    Parameters
    ----------
    module : Module
        The parsed module to traverse.

    Returns
    -------
    tuple[QiecBinding, ...]
        The local bindings. A module with no QIEC computation or handler
        declarations yields nothing.
    """
    out: list[QiecBinding] = []
    for statement in module.statements:
        if isinstance(statement, QiecComputationDecl):
            out.extend(
                QiecBinding(
                    parameter.name,
                    "parameter",
                    parameter,
                    statement,
                    _computation_span(statement.body),
                )
                for parameter in statement.parameters
            )
            _walk_computation_bindings(statement.body, statement, out)
        elif isinstance(statement, QiecHandlerDecl):
            for clause in statement.clauses:
                if isinstance(clause, QiecHandlerReturnClause):
                    span = _computation_span(clause.body)
                    out.append(
                        QiecBinding(
                            clause.binder.name,
                            "parameter",
                            clause.binder,
                            statement,
                            span,
                        )
                    )
                    _walk_computation_bindings(clause.body, statement, out)
                    continue
                if not isinstance(clause, QiecHandlerOperationClause):
                    continue
                if clause.body is None:
                    continue
                span = _computation_span(clause.body)
                out.extend(
                    QiecBinding(parameter.name, "parameter", parameter, statement, span)
                    for parameter in clause.parameters
                )
                _walk_computation_bindings(clause.body, statement, out)
    return tuple(out)


def qiec_binding_map(module: Module) -> dict[str, QiecBinding]:
    """Return unambiguous source identities for QIEC bindings.

    Every signature member is available through ``Owner.member``. A bare
    member spelling remains available when exactly one declaration owns that
    spelling and it does not collide with a top-level declaration. This keeps
    the convenient ``:type get`` form for a single effect while preventing a
    later ``Writer.get`` from silently replacing ``Reader.get``.

    Parameters
    ----------
    module : Module
        The parsed module to read.

    Returns
    -------
    dict[str, QiecBinding]
        Spelling to binding. A member owned by two declarations appears
        only under its qualified spellings, so a lookup never silently
        resolves to one of two equally good answers.
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
    """Return every binding matching an authored or qualified spelling.

    Parameters
    ----------
    module : Module
        The parsed module to search.
    name : str
        A bare member name or a qualified ``Owner.member`` spelling.

    Returns
    -------
    tuple[QiecBinding, ...]
        Matching bindings. A qualified spelling yields at most one. A
        bare spelling yields every owner that declares it, which is what
        lets a caller report an ambiguity rather than resolve it
        arbitrarily.
    """

    if "." in name:
        exact = qiec_binding_map(module).get(name)
        return (exact,) if exact is not None else ()
    return tuple(binding for binding in qiec_bindings(module) if binding.name == name)


def qiec_env_kinds(module: Module) -> dict[str, str]:
    """Return semantic-token classifications for QIEC names.

    Parameters
    ----------
    module : Module
        The parsed module to classify.

    Returns
    -------
    dict[str, str]
        Spelling to token class, for the language server's semantic
        token legend. Only unambiguous spellings appear, since the map it
        is built from omits the rest.
    """

    return {
        name: binding.semantic_kind
        for name, binding in qiec_binding_map(module).items()
    }


def qiec_surface_values(module: Module) -> dict[str, object]:
    """Expose authored QIEC declarations in the shared inspection environment.

    Parameters
    ----------
    module : Module
        The parsed module to expose.

    Returns
    -------
    dict[str, object]
        Spelling to the authored declaration node. These are source AST
        nodes rather than lowered or compiled objects, so inspecting a
        name in the REPL shows what the user wrote.
    """

    return {
        name: binding.declaration for name, binding in qiec_binding_map(module).items()
    }


def qiec_module_name(file_path: str | Path | None) -> str:
    """Derive a deterministic source module name for editor/CLI entry points.

    The name enters every stable identity the lowerer derives, so it has
    to depend on the path alone and not on how the path was spelled.

    Parameters
    ----------
    file_path : str or pathlib.Path or None
        The source path, possibly a ``file://`` URI, or None for input
        that has no file, such as a REPL line.

    Returns
    -------
    str
        The path's stem, ``"repl"`` when there is no path, and
        ``"module"`` when the path has an empty stem.
    """

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
