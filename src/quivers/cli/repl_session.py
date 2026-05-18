"""UI-agnostic REPL engine.

A :class:`ReplSession` owns the in-memory environment that ``qvr repl``,
the Textual TUI, the prompt_toolkit fallback, and the Jupyter kernel
all drive. Every meta-command dispatches to a method on this class and
returns a :class:`ReplResponse`; the frontends decide how to render it.

The session never imports any UI library and never reads from stdin or
writes to stdout. That keeps it fully testable from pytest.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

import didactic.api as dx

from quivers.dsl import Compiler, CompileError, ParseError, parse
from quivers.dsl.ast_nodes import (
    ExprCompose,
    ExprIdent,
    ExprTensorProduct,
    Module,
    Statement,
    TypeDecl,
    TypeExpr,
    TypeFromExpr,
)
from quivers.dsl.constraints import Violation, check_constraints
from quivers.dsl.emit import module_to_source


Severity = Literal["error", "warning", "info", "ok"]


class Diagnostic(dx.Model):
    """One structured diagnostic surfaced by the REPL or LSP."""

    message: str
    severity: Severity
    line: int = 0
    col: int = 0
    end_line: int = 0
    end_col: int = 0
    code: str = ""


class ReplResponse(dx.Model):
    """A single command's result, ready for any frontend to render.

    ``body`` is the primary payload, which a frontend renders by
    pairing it with ``body_kind`` (``text`` / ``qvr`` / ``json`` /
    ``markdown``); each frontend is responsible for choosing the
    concrete renderer (Rich syntax block, prompt_toolkit ANSI, etc.)
    from ``body_kind``. ``diagnostics`` carry structured errors and
    warnings.
    """

    body: str = ""
    diagnostics: tuple[Diagnostic, ...] = ()
    body_kind: Literal["text", "qvr", "json", "markdown"] = "text"

    @property
    def ok(self) -> bool:
        return not any(d.severity == "error" for d in self.diagnostics)


class SessionOptions(dx.Model):
    """User-tunable knobs exposed via ``:set``.

    ``theme`` selects a Rich syntax theme for QVR bodies rendered
    via ``:info`` / ``:dump`` etc. Common values: ``ansi_dark``,
    ``ansi_light``, ``monokai``, ``nord``, ``solarized-dark``,
    ``solarized-light``, ``github-dark``.
    """

    highlight: bool = True
    unicode: bool = True
    show_axes: bool = True
    paranoid: bool = False
    autoload_on_save: bool = True
    theme: str = "ansi_dark"


class ReplSession:
    """Stateful evaluator for a single REPL session."""

    def __init__(self) -> None:
        self._loaded_path: Path | None = None
        self._loaded_source: str = ""
        self._module: Module = Module(statements=())
        self._compiler: Compiler | None = None
        self._env: dict[str, Any] = {}
        self._last_diags: tuple[Diagnostic, ...] = ()
        self.options = SessionOptions()
        # Track when the loaded file was last read so :reload can be
        # auto-fired on a modified mtime.
        self._loaded_mtime: float | None = None
        # Pinned `:watch EXPR` re-evaluations. The TUI renders these in
        # a dedicated panel; each install + reload re-runs every entry.
        self._watches: list[str] = []
        self._watch_results: dict[str, str] = {}

    # ----- introspection ------------------------------------------------

    @property
    def loaded_path(self) -> Path | None:
        return self._loaded_path

    @property
    def module(self) -> Module:
        return self._module

    @property
    def env(self) -> dict[str, Any]:
        """Live environment dict (objects + spaces + morphisms + rules)."""
        return dict(self._env)

    @property
    def diagnostics(self) -> tuple[Diagnostic, ...]:
        return self._last_diags

    def watch_results(self) -> dict[str, str]:
        """Return the current ``expr -> rendered`` map for pinned watches."""
        return dict(self._watch_results)

    def env_kinds(self) -> dict[str, str]:
        """Return a name -> semantic-token-type map for the live env.

        Frontends pass this to the highlighter so an identifier renders
        in its env-known colour everywhere it appears, regardless of
        whether the surrounding grammar context parses cleanly.
        """
        if self._compiler is None:
            return {}
        kinds: dict[str, str] = {}
        for name in self._compiler.objects:
            kinds[name] = "type"
        for name in self._compiler.spaces:
            kinds[name] = "type"
        for name in self._compiler.morphisms:
            kinds[name] = "function"
        for name in self._compiler.rules:
            kinds[name] = "namespace"
        return kinds

    # ----- entry points -------------------------------------------------

    def dispatch(self, line: str) -> ReplResponse:
        """Route a raw input line to the right handler."""
        stripped = line.strip()
        if not stripped:
            return ReplResponse()
        if stripped.startswith(":"):
            cmd, _, rest = stripped[1:].partition(" ")
            return self._dispatch_meta(cmd, rest.strip())
        # Bare line: try statement first, then expression.
        return self._eval_source(stripped)

    def _dispatch_meta(self, cmd: str, arg: str) -> ReplResponse:
        method = _META_COMMANDS.get(cmd)
        if method is None:
            return _err(f"unknown command :{cmd}; try :help")
        return method(self, arg)

    # ----- :load / :reload ----------------------------------------------

    def load_file(self, path: str | Path) -> ReplResponse:
        p = Path(path).expanduser()
        if not p.exists():
            return _err(f"file not found: {p}")
        try:
            source = p.read_bytes()
            module = parse(source, file_path=str(p))
        except ParseError as e:
            self._last_diags = (
                Diagnostic(message=str(e), severity="error", code="parse"),
            )
            return _resp("", self._last_diags)
        self._loaded_source = source.decode("utf-8", errors="replace")
        return self._install_module(module, source_path=p)

    def reload(self) -> ReplResponse:
        if self._loaded_path is None:
            return _err("no file loaded; use :load <FILE>")
        prior = set(self._env)
        result = self.load_file(self._loaded_path)
        if not result.ok:
            return result
        new = set(self._env) - prior
        removed = prior - set(self._env)
        body_lines = [f"reloaded {self._loaded_path}"]
        if new:
            body_lines.append("  added: " + ", ".join(sorted(new)))
        if removed:
            body_lines.append("  removed: " + ", ".join(sorted(removed)))
        return _resp("\n".join(body_lines), result.diagnostics)

    def _install_module(
        self, module: Module, *, source_path: Path | None = None
    ) -> ReplResponse:
        diags: list[Diagnostic] = [
            _violation_to_diag(v) for v in check_constraints(module)
        ]
        compiler = Compiler(module)
        env: dict[str, Any] = {}
        try:
            env = compiler.compile_env()
        except CompileError as e:
            diags.append(
                Diagnostic(
                    message=str(e),
                    severity="error",
                    line=getattr(e, "line", 0),
                    col=getattr(e, "col", 0),
                    code="compile",
                )
            )
        self._module = module
        self._compiler = compiler
        self._env = env
        self._last_diags = tuple(diags)
        if source_path is not None:
            self._loaded_path = source_path
            try:
                self._loaded_mtime = source_path.stat().st_mtime
            except OSError:
                self._loaded_mtime = None
        self._refresh_watches()
        body = (
            f"loaded {source_path}: {_env_counts(env)}"
            if source_path is not None
            else f"installed module: {_env_counts(env)}"
        )
        return _resp(body, tuple(diags))

    # ----- :type / :kind ------------------------------------------------

    def type_of(self, expr_source: str) -> ReplResponse:
        """Resolve `expr_source` as a TypeExpr or morphism and print its type.

        Strategy: try to parse `type __probe__ : <expr_source>` so the
        existing parser handles whatever surface form the user typed.
        If that succeeds, walk the resulting :class:`TypeDecl`'s
        :class:`TypeFromExpr` initializer through the compiler's
        resolution mixin.

        Failing that, try parsing `let __probe__ = <expr_source>` to
        catch let-expression syntax, then `output <expr_source>` so
        morphism references work; report dom -> cod.
        """
        if not self._compiler:
            return _err("no environment loaded; use :load <FILE> first")

        # Fast path: bare identifier resolves directly out of the env.
        # GHCi-shaped :type queries are dominated by name lookups, so we
        # handle them without round-tripping through the parser.
        bare = expr_source.strip()
        if bare.isidentifier():
            if bare in self._compiler.morphisms:
                m = self._compiler.morphisms[bare]
                return _resp(
                    self._type_line_for_morphism(bare, m),
                    body_kind="qvr",
                )
            if bare in self._compiler.objects:
                obj = self._compiler.objects[bare]
                return _resp(
                    f"object {bare} : {_pretty_object(obj)}",
                    body_kind="qvr",
                )
            if bare in self._compiler.spaces:
                sp = self._compiler.spaces[bare]
                return _resp(
                    f"space {bare} : {_pretty_object(sp)}",
                    body_kind="qvr",
                )

        probe = self._scratch_compiler()

        # Path 1: type-level. Probe via a ``type __probe__ : <expr>``
        # declaration, then re-resolve the inner expression through
        # the scratch compiler so identifiers in scope land on the
        # right object/space.
        try:
            mod = parse(f"type __probe__ : {expr_source}", file_path="<type>")
        except ParseError:
            mod = None
        if mod is not None and mod.statements:
            stmt = mod.statements[0]
            if isinstance(stmt, TypeDecl) and isinstance(stmt.init, TypeFromExpr):
                try:
                    obj = probe._resolve_any_space(stmt.init.expr)
                    return _resp(
                        f"{expr_source} :: {_pretty_object(obj)}",
                        body_kind="qvr",
                    )
                except CompileError as e:
                    return _err(f"type error: {e}")
                except Exception as e:
                    return _err(f"type error: {e}")

        # Path 2: morphism-level via output binding
        try:
            mod = parse(f"export {expr_source}", file_path="<expr>")
        except ParseError as e:
            return _err(f"parse error: {e}")
        # Re-run the full compile so morphism algebra is wired.
        scratch = Compiler(_extend_module(self._module, mod.statements))
        try:
            scratch.compile()
        except CompileError as e:
            return _err(f"compile error: {e}")
        program = getattr(scratch, "_output_expr", None)
        # Resolve the output expression to a morphism and pretty-print.
        try:
            morph = scratch._compile_expr(program) if program is not None else None
        except CompileError as e:
            return _err(f"compile error: {e}")
        if morph is None:
            return _err("expression did not resolve to a morphism")
        return _resp(
            self._type_line_for_morphism(expr_source, morph),
            body_kind="qvr",
        )

    def _type_line_for_morphism(self, name: str, morph: Any) -> str:
        """Render a morphism's signature in valid-QVR notation.

        The unified ``morphism NAME : DOM -> COD [role=latent]`` form
        is the only signature line that round-trips cleanly through
        the QVR grammar without a body block, so the TUI's tokenizer
        sees the dom and cod identifiers as type positions. The
        binding's true role is surfaced by ``:info``.
        """
        del self
        return f"morphism {name} : {_pretty_morphism(morph)} [role=latent]"

    def kind_of(self, expr_source: str) -> ReplResponse:
        try:
            mod = parse(
                f"type __probe__ : {expr_source}", file_path="<kind>"
            )
        except ParseError as e:
            return _err(f"parse error: {e}")
        if (
            not mod.statements
            or not isinstance(mod.statements[0], TypeDecl)
            or not isinstance(mod.statements[0].init, TypeFromExpr)
        ):
            return _err("expected a type expression")
        texpr: TypeExpr = mod.statements[0].init.expr
        klass = type(texpr).__name__
        variants = sorted(cls.__name__ for cls in TypeExpr.__variants__.values())
        return _resp(
            f"{expr_source} : {klass}\n  TypeExpr variants: {', '.join(variants)}",
            body_kind="qvr",
        )

    # ----- :info / :doc -------------------------------------------------

    def info(self, name: str, *, python: bool = False) -> ReplResponse:
        decl = self._find_decl(name)
        if decl is None:
            if name in self._env:
                value = self._env[name]
                return _resp(
                    f"{name} :: {_pretty_runtime_value(value)}\n"
                    f"  (bound, no source declaration)"
                )
            return _err(f"unknown name: {name}")
        line = getattr(decl, "line", 0)
        col = getattr(decl, "col", 0)
        loc = (
            f"{self._loaded_path}:{line}:{col}"
            if self._loaded_path is not None and line
            else type(decl).__name__
        )
        if python:
            rendered = repr(decl)
            body_kind: Literal["text", "qvr", "json", "markdown"] = "text"
        else:
            rendered = self._render_decl_qvr(decl)
            body_kind = "qvr"
        docs = getattr(decl, "docs", ())
        doc_block = "\n".join(f"-- {d}" for d in docs)
        body = f"{rendered}\n-- declared at {loc}"
        if doc_block:
            body = f"{doc_block}\n{body}"
        return _resp(body, body_kind=body_kind)

    def _render_decl_qvr(self, decl: Statement) -> str:
        """Return the declaration as QVR source.

        Order of preference:
        1. Verbatim slice of the original source between this decl's
           start byte and the next decl's start byte (preserves comments
           and formatting).
        2. ``quivers.dsl.emit.module_to_source`` (canonical re-emission).
        3. ``repr(decl)`` as a last-ditch fallback for variants neither
           the slicer nor the emitter handles.
        """
        sliced = self._slice_source_for(decl)
        if sliced is not None:
            return sliced.rstrip() + "\n"
        return _render_decl(decl)

    def _slice_source_for(self, decl: Statement) -> str | None:
        """Return the original source lines that produced ``decl``."""
        if not self._loaded_source:
            return None
        start_line = getattr(decl, "line", 0)
        if not start_line:
            return None
        lines = self._loaded_source.splitlines()
        if start_line - 1 >= len(lines):
            return None
        # Find the next declaration's start line; everything between
        # `start_line` and that line belongs to this declaration.
        end_line = len(lines) + 1
        for other in self._module.statements:
            if other is decl:
                continue
            other_line = getattr(other, "line", 0)
            if other_line > start_line and other_line < end_line:
                end_line = other_line
        # Walk forward past trailing blank lines so the slice is tight.
        while end_line - 1 > start_line and not lines[end_line - 2].strip():
            end_line -= 1
        return "\n".join(lines[start_line - 1 : end_line - 1])

    def doc(self, name: str) -> ReplResponse:
        decl = self._find_decl(name)
        if decl is None:
            return _err(f"unknown name: {name}")
        docs = getattr(decl, "docs", ())
        if not docs:
            return _resp(f"{name}: (no doc comment)")
        return _resp("\n".join(docs), body_kind="markdown")

    # ----- :browse ------------------------------------------------------

    def browse(self, namespace: str = "") -> ReplResponse:
        groups: dict[str, list[str]] = {
            "objects": [],
            "spaces": [],
            "morphisms": [],
            "rules": [],
        }
        compiler = self._compiler
        if compiler is None:
            return _err("no environment loaded; use :load <FILE> first")
        groups["objects"] = sorted(compiler.objects)
        groups["spaces"] = sorted(compiler.spaces)
        groups["morphisms"] = sorted(compiler.morphisms)
        groups["rules"] = sorted(compiler.rules)
        if namespace:
            ns = namespace.rstrip("s") + "s"
            if ns not in groups:
                return _err(f"unknown namespace: {namespace}")
            groups = {ns: groups[ns]}
        lines: list[str] = []
        for ns, names in groups.items():
            if not names:
                continue
            lines.append(f"{ns}:")
            for n in names:
                lines.append(f"  {n}")
        if not lines:
            return _resp("(empty environment)")
        return _resp("\n".join(lines))

    # ----- :dump --------------------------------------------------------

    def dump(self, name: str, *, as_json: bool = False) -> ReplResponse:
        decl = self._find_decl(name)
        if decl is None:
            return _err(f"unknown name: {name}")
        if as_json:
            return _resp(decl.model_dump_json(indent=2), body_kind="json")
        return _resp(repr(decl))

    # ----- :edit --------------------------------------------------------

    def edit(self, name: str, *, editor: str | None = None) -> ReplResponse:
        decl = self._find_decl(name)
        if decl is None:
            return _err(f"unknown name: {name}")
        text = _render_decl(decl)
        editor_cmd = editor or os.environ.get("EDITOR") or "vi"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".qvr", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(text + "\n")
            tmp_path = Path(tmp.name)
        try:
            subprocess.run([editor_cmd, str(tmp_path)], check=True)
            edited = tmp_path.read_text(encoding="utf-8")
        except subprocess.CalledProcessError as e:
            return _err(f"editor exited non-zero ({e.returncode})")
        finally:
            try:
                tmp_path.unlink()
            except OSError:
                pass
        try:
            new_mod = parse(edited, file_path=f"<edit:{name}>")
        except ParseError as e:
            return _err(f"parse error in edited source: {e}")
        if not new_mod.statements:
            return _err("edited source contains no statements; aborting")
        replaced = _replace_decl_by_name(self._module, name, new_mod.statements)
        if replaced is None:
            return _err(f"could not locate {name} in current module")
        return self._install_module(replaced, source_path=self._loaded_path)

    # ----- :trace -------------------------------------------------------

    def trace(self, expr_source: str) -> ReplResponse:
        """Step through elaboration of a morphism expression."""
        if self._compiler is None:
            return _err("no environment loaded")
        try:
            mod = parse(f"export {expr_source}", file_path="<trace>")
        except ParseError as e:
            return _err(f"parse error: {e}")
        scratch = Compiler(_extend_module(self._module, mod.statements))
        # Re-run statement compilation so the env is fresh.
        try:
            scratch.compile()
        except CompileError as e:
            return _err(f"compile error: {e}")
        # Walk the expression once and emit shape info as we go.
        lines: list[str] = []
        try:
            steps = _trace_expr(scratch, scratch._output_expr)
        except Exception as e:
            return _err(f"trace failed: {e}")
        for step in steps:
            lines.append(step)
        return _resp("\n".join(lines))

    # ----- :save / :watch / :unwatch -----------------------------------

    def save(self, path: str = "") -> ReplResponse:
        """Write the live module back to a ``.qvr`` file."""
        target = Path(path).expanduser() if path else self._loaded_path
        if target is None:
            return _err("usage: :save <FILE> (no file currently loaded)")
        try:
            source = module_to_source(self._module)
        except NotImplementedError as e:
            return _err(
                f"cannot emit: {e}. (the canonical emitter does not yet "
                "support every statement variant; edit the source file "
                "directly instead)"
            )
        try:
            target.write_text(source, encoding="utf-8")
        except OSError as e:
            return _err(f"write failed: {e}")
        return _resp(f"saved {target}")

    def watch(self, expr: str) -> ReplResponse:
        """Pin an expression for re-evaluation on every recompile."""
        expr = expr.strip()
        if not expr:
            return _err("usage: :watch <EXPR>")
        if expr not in self._watches:
            self._watches.append(expr)
        self._refresh_watches()
        rendered = self._watch_results.get(expr, "(unresolved)")
        return _resp(f"watch {expr} => {rendered}", body_kind="qvr")

    def unwatch(self, expr: str) -> ReplResponse:
        """Remove a previously-pinned watch expression."""
        expr = expr.strip()
        if not expr:
            # No argument: clear every watch.
            self._watches.clear()
            self._watch_results.clear()
            return _resp("cleared all watches")
        if expr not in self._watches:
            return _err(f"not watching: {expr}")
        self._watches.remove(expr)
        self._watch_results.pop(expr, None)
        return _resp(f"unwatched {expr}")

    def _refresh_watches(self) -> None:
        """Re-evaluate every pinned watch against the current env."""
        self._watch_results = {}
        for expr in self._watches:
            try:
                response = self.type_of(expr)
                self._watch_results[expr] = response.body if response.ok else "(error)"
            except Exception:
                self._watch_results[expr] = "(error)"

    # ----- :set ---------------------------------------------------------

    def set_option(self, raw: str) -> ReplResponse:
        if "=" not in raw:
            return _err("usage: :set option=value")
        key, _, val = raw.partition("=")
        key = key.strip().replace("-", "_")
        if not hasattr(self.options, key):
            return _err(f"unknown option: {key}")
        current = getattr(self.options, key)
        v: bool | str
        if isinstance(current, bool):
            v = val.strip().lower() in ("1", "true", "yes", "on")
        else:
            v = val.strip()
        setattr(self.options, key, v)
        return _resp(f"{key} = {v}")

    # ----- :help --------------------------------------------------------

    def help(self, arg: str = "") -> ReplResponse:
        if arg:
            entry = _HELP.get(arg)
            if entry is None:
                return _err(f"no help for :{arg}")
            return _resp(entry)
        body = "\n".join(
            f":{name:<10} {summary}" for name, summary in _HELP_SUMMARIES.items()
        )
        return _resp(body)

    # ----- bare-line evaluation ----------------------------------------

    def _eval_source(self, src: str) -> ReplResponse:
        """Treat `src` first as statements appended to the current module."""
        try:
            new_mod = parse(src, file_path="<repl>")
            extended = _extend_module(self._module, new_mod.statements)
            return self._install_module(extended, source_path=self._loaded_path)
        except ParseError:
            pass
        # Fall back to expression: print its type.
        return self.type_of(src)

    # ----- helpers ------------------------------------------------------

    def _find_decl(self, name: str) -> Statement | None:
        for stmt in self._module.statements:
            n = getattr(stmt, "name", None)
            if n == name:
                return stmt
        return None

    def _scratch_compiler(self) -> Compiler:
        """A fresh Compiler with the current module already elaborated."""
        c = Compiler(self._module)
        try:
            c.compile_env()
        except CompileError:
            pass
        return c

    def autoreload_if_stale(self) -> ReplResponse | None:
        """Re-run :reload if the loaded file's mtime advanced."""
        if not self.options.autoload_on_save:
            return None
        if self._loaded_path is None or self._loaded_mtime is None:
            return None
        try:
            current = self._loaded_path.stat().st_mtime
        except OSError:
            return None
        if current > self._loaded_mtime:
            return self.reload()
        return None


# ---------------------------------------------------------------------------
# meta-command dispatch table
# ---------------------------------------------------------------------------


def _cmd_load(s: ReplSession, arg: str) -> ReplResponse:
    if not arg:
        return _err("usage: :load <FILE>")
    return s.load_file(arg)


def _cmd_reload(s: ReplSession, arg: str) -> ReplResponse:
    del arg
    return s.reload()


def _cmd_type(s: ReplSession, arg: str) -> ReplResponse:
    if not arg:
        return _err("usage: :type <EXPR>")
    return s.type_of(arg)


def _cmd_kind(s: ReplSession, arg: str) -> ReplResponse:
    if not arg:
        return _err("usage: :kind <TYPE-EXPR>")
    return s.kind_of(arg)


def _cmd_info(s: ReplSession, arg: str) -> ReplResponse:
    parts = arg.split()
    if not parts:
        return _err("usage: :info <NAME> [--python]")
    name = parts[0]
    return s.info(name, python="--python" in parts[1:])


def _cmd_doc(s: ReplSession, arg: str) -> ReplResponse:
    if not arg:
        return _err("usage: :doc <NAME>")
    return s.doc(arg)


def _cmd_browse(s: ReplSession, arg: str) -> ReplResponse:
    return s.browse(arg)


def _cmd_dump(s: ReplSession, arg: str) -> ReplResponse:
    parts = arg.split()
    if not parts:
        return _err("usage: :dump <NAME> [--json]")
    name = parts[0]
    return s.dump(name, as_json=("--json" in parts[1:]))


def _cmd_edit(s: ReplSession, arg: str) -> ReplResponse:
    if not arg:
        return _err("usage: :edit <NAME>")
    return s.edit(arg)


def _cmd_trace(s: ReplSession, arg: str) -> ReplResponse:
    if not arg:
        return _err("usage: :trace <EXPR>")
    return s.trace(arg)


def _cmd_save(s: ReplSession, arg: str) -> ReplResponse:
    return s.save(arg)


def _cmd_watch(s: ReplSession, arg: str) -> ReplResponse:
    return s.watch(arg)


def _cmd_unwatch(s: ReplSession, arg: str) -> ReplResponse:
    return s.unwatch(arg)


def _cmd_set(s: ReplSession, arg: str) -> ReplResponse:
    return s.set_option(arg)


def _cmd_help(s: ReplSession, arg: str) -> ReplResponse:
    return s.help(arg)


def _cmd_quit(s: ReplSession, arg: str) -> ReplResponse:
    del s, arg
    return _resp("__quit__")


_META_COMMANDS = {
    "load": _cmd_load,
    "l": _cmd_load,
    "reload": _cmd_reload,
    "r": _cmd_reload,
    "type": _cmd_type,
    "t": _cmd_type,
    "kind": _cmd_kind,
    "k": _cmd_kind,
    "info": _cmd_info,
    "i": _cmd_info,
    "doc": _cmd_doc,
    "browse": _cmd_browse,
    "b": _cmd_browse,
    "dump": _cmd_dump,
    "edit": _cmd_edit,
    "trace": _cmd_trace,
    "save": _cmd_save,
    "s": _cmd_save,
    "watch": _cmd_watch,
    "w": _cmd_watch,
    "unwatch": _cmd_unwatch,
    "set": _cmd_set,
    "help": _cmd_help,
    "h": _cmd_help,
    "quit": _cmd_quit,
    "q": _cmd_quit,
    "exit": _cmd_quit,
}


_HELP_SUMMARIES = {
    "load FILE": "parse + compile a .qvr file into the session",
    "reload": "re-run :load on the last file, diffing the env",
    "type EXPR": "infer and print the type of EXPR",
    "kind TYPE": "report the AST kind of a type expression",
    "info NAME": "show the declaration and source location of NAME (--python for AST repr)",
    "doc NAME": "render the doc comment attached to NAME",
    "browse [NS]": "list bound names, optionally per namespace",
    "dump NAME": "show the AST node for NAME (add --json for didactic dump)",
    "edit NAME": "open $EDITOR on NAME's source, splice back on save",
    "trace EXPR": "step through elaboration of a morphism expression",
    "save FILE": "write the live module back to FILE via the canonical emitter",
    "watch EXPR": "pin EXPR for re-eval after every recompile (watch panel)",
    "unwatch EXPR": "remove EXPR from the watch list (no arg clears all)",
    "set k=v": "toggle session options (highlight, unicode, theme, ...)",
    "help [CMD]": "list commands or detail one",
    "quit": "exit the REPL",
}


_HELP: dict[str, str] = {
    "load": "Parse and compile <FILE>; the environment is rebound to the new module.",
    "reload": "Re-parse the most recently loaded file and show which names changed.",
    "type": "Resolve <EXPR> as either a type expression or a morphism reference; "
    "in the first case prints the underlying SetObject, in the second the "
    "domain -> codomain signature.",
    "kind": "Show the AST kind (didactic discriminator) of a type expression and "
    "enumerate the sibling variants.",
    "info": "Show NAME's declaration as verbatim .qvr source (sliced from the "
    "loaded file), plus the source location and any leading doc comment. "
    "Pass --python to see the didactic AST `repr()` instead.",
    "doc": "Render only the doc comment for NAME, useful for piping.",
    "browse": "List every bound name grouped by namespace. Pass a namespace name "
    "(objects/spaces/morphisms/rules) to restrict the listing.",
    "dump": "Pretty-print the AST node for NAME. Pass --json for didactic's "
    "model_dump_json output.",
    "edit": "Open $EDITOR on NAME's declaration; on save the edited text is "
    "spliced back into the module and recompiled.",
    "trace": "Step through morphism elaboration, surfacing each intermediate "
    "domain/codomain.",
    "set": "Toggle session options: highlight=true|false, unicode=true|false, "
    "show_axes=true|false, paranoid=true|false, autoload_on_save=true|false.",
    "help": "Without an argument, list every command. With one, print its help.",
    "quit": "Leave the REPL.",
}


# ---------------------------------------------------------------------------
# rendering helpers
# ---------------------------------------------------------------------------


def _resp(
    body: str,
    diagnostics: Iterable[Diagnostic] = (),
    *,
    body_kind: Literal["text", "qvr", "json", "markdown"] = "text",
) -> ReplResponse:
    return ReplResponse(body=body, diagnostics=tuple(diagnostics), body_kind=body_kind)


def _err(message: str) -> ReplResponse:
    return ReplResponse(
        diagnostics=(Diagnostic(message=message, severity="error", code="repl"),)
    )


def _env_counts(env: dict[str, Any]) -> str:
    items = sum(1 for k in env if not k.startswith("__"))
    return f"{items} binding(s)"


def _violation_to_diag(v: Violation) -> Diagnostic:
    return Diagnostic(
        message=v.message,
        severity="error",
        line=v.line,
        col=v.col,
        code=v.code,
    )


def _pretty_object(obj: Any) -> str:
    """Render a SetObject / ContinuousSpace in QVR-shaped notation.

    Examples:
        FinSet(name='X', cardinality=3)            -> "X"
        FinSet(name='', cardinality=3)              -> "FinSet(3)"
        ProductSet(components=(A, B))               -> "A * B"
        CoproductSet(components=(A, B))             -> "A + B"
        FreeMonoid(name='Words', alphabet=A)        -> "Words"
        EnumSet(name='Tags', members=('NP','S'))    -> "Tags"
        FreeResiduated(name='Cat', ...)             -> "Cat"
        ContinuousSpace subtypes                    -> their `name`
    """
    kind = type(obj).__name__
    name = getattr(obj, "name", "") or ""
    if kind in ("ProductSet", "ProductSpace"):
        comps = getattr(obj, "components", ())
        if comps:
            return " * ".join(_pretty_object(c) for c in comps)
    if kind == "CoproductSet":
        comps = getattr(obj, "components", ())
        if comps:
            return " + ".join(_pretty_object(c) for c in comps)
    if kind == "FinSet":
        if name:
            return name
        cardinality = getattr(obj, "cardinality", None)
        return f"FinSet({cardinality})" if cardinality is not None else "FinSet"
    if name:
        # The discrete-to-continuous embedding wraps a FinSet inside
        # an `Euclidean(name="idx(FinSet(name='Source', ...))", ...)`.
        # Strip the wrapper so users see the original object name.
        if (
            name.startswith("idx(FinSet(name='")
            and "'" in name[len("idx(FinSet(name='") :]
        ):
            inner = name[len("idx(FinSet(name='") :]
            return inner.split("'", 1)[0]
        return name
    # ContinuousSpace constructors keep their constructor + args
    # readable via repr, but we want to avoid raw didactic output.
    constructor = getattr(obj, "constructor", None)
    if constructor is not None:
        args = getattr(obj, "args", ()) or ()
        return f"{constructor}({', '.join(str(a) for a in args)})"
    return kind


def _pretty_runtime_value(value: Any) -> str:
    return _pretty_object(value)


def _pretty_morphism(m: Any) -> str:
    dom = getattr(m, "domain", getattr(m, "dom", None))
    cod = getattr(m, "codomain", getattr(m, "cod", None))
    if dom is None or cod is None:
        return f"{type(m).__name__}"
    return f"{_pretty_object(dom)} -> {_pretty_object(cod)}"


def _render_decl(decl: Statement) -> str:
    """Emit a declaration as canonical source, falling back to repr.

    `quivers.dsl.emit.module_to_source` covers the common declarations
    and raises NotImplementedError otherwise; we catch and fall back so
    :info / :edit never crash on a rare variant.
    """
    try:
        return module_to_source(Module(statements=(decl,))).rstrip("\n")
    except NotImplementedError:
        return repr(decl)


def _extend_module(base: Module, additions: Iterable[Statement]) -> Module:
    return Module(statements=tuple(base.statements) + tuple(additions))


def _replace_decl_by_name(
    base: Module, name: str, replacements: Iterable[Statement]
) -> Module | None:
    out: list[Statement] = []
    replaced = False
    repls = list(replacements)
    for stmt in base.statements:
        if not replaced and getattr(stmt, "name", None) == name:
            out.extend(repls)
            replaced = True
            continue
        out.append(stmt)
    if not replaced:
        return None
    return Module(statements=tuple(out))


def _trace_expr(compiler: Compiler, expr: Any) -> list[str]:
    """Yield human-readable lines describing each elaboration step."""
    out: list[str] = []
    seen: set[int] = set()

    def visit(e: Any, depth: int) -> None:
        if id(e) in seen:
            return
        seen.add(id(e))
        pad = "  " * depth
        if isinstance(e, ExprIdent):
            morph = compiler.morphisms.get(e.name)
            if morph is not None:
                out.append(f"{pad}{e.name} :: {_pretty_morphism(morph)}")
            else:
                out.append(f"{pad}{e.name} :: <unresolved>")
            return
        if isinstance(e, ExprCompose):
            out.append(f"{pad}compose {e.op}")
            visit(e.left, depth + 1)
            visit(e.right, depth + 1)
            try:
                morph = compiler._compile_expr(e)
                out.append(f"{pad}  => {_pretty_morphism(morph)}")
            except CompileError as ce:
                out.append(f"{pad}  => error: {ce}")
            return
        if isinstance(e, ExprTensorProduct):
            out.append(f"{pad}tensor")
            visit(e.left, depth + 1)
            visit(e.right, depth + 1)
            return
        try:
            morph = compiler._compile_expr(e)
            out.append(f"{pad}{type(e).__name__} :: {_pretty_morphism(morph)}")
        except Exception as ex:
            out.append(f"{pad}{type(e).__name__} :: <{ex}>")

    visit(expr, 0)
    return out


__all__ = [
    "Diagnostic",
    "ReplResponse",
    "ReplSession",
    "SessionOptions",
]
