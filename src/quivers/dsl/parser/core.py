"""Top-level parse entry points and doc-comment attachment."""

from __future__ import annotations

from pathlib import Path

from quivers.dsl.ast_nodes import Module, Statement
from quivers.dsl.parser._registry import ParseError, _Tree, _registry
from quivers.dsl.parser.statements import _walk_statement


def parse(source: str | bytes, file_path: str = "<source>") -> Module:
    """Parse `.qvr` source bytes into a `Module`."""
    if isinstance(source, str):
        source_bytes = source.encode("utf-8")
    else:
        source_bytes = source

    schema = _registry().parse_with_protocol("qvr", source_bytes, file_path)
    tree = _Tree(schema, source_bytes)

    root_id = next(
        (v.id for v in schema.vertices if v.kind == "source_file"),
        None,
    )
    if root_id is None:
        raise ParseError(f"panproto schema has no source_file vertex for {file_path}")

    statements: list[Statement] = []
    pending_docs: list[str] = []
    for child in tree.positional(root_id):
        ckind = tree.kind(child)
        if ckind == "line_comment":
            # plain `# ...` comments are dropped at parse time
            continue
        if ckind == "doc_comment":
            # `## ...` doc comments are accumulated; attached to the
            # next statement that carries a docs field.
            text = tree.text(child)
            stripped = text[2:].lstrip() if text.startswith("##") else text
            pending_docs.append(stripped.rstrip())
            continue
        result = _walk_statement(tree, child)
        results = result if isinstance(result, list) else [result]
        if pending_docs:
            docs = tuple(pending_docs)
            results = [_attach_docs(s, docs) for s in results]
            pending_docs = []
        statements.extend(results)
    return Module(statements=tuple(statements))


def _attach_docs(stmt: Statement, docs: tuple[str, ...]) -> Statement:
    """Attach accumulated ``##`` doc-comment lines to a Statement.

    Returns a copy of ``stmt`` with its ``docs`` field extended;
    Statement variants that lack a ``docs`` field are returned
    unchanged. didactic Models are immutable; `Model.with_` is
    the field-replacement constructor.
    """
    # `docs` is a declared field on every Statement variant that
    # accepts a leading doc comment (ObjectDecl, MorphismDecl,
    # SchemaDecl, ProgramDecl, BundleDecl, ...). Probe via the
    # class's field-spec registry rather than instance __getattr__,
    # since dx.Model's attribute fall-through raises AttributeError
    # on undeclared field accesses.
    fields = getattr(type(stmt), "__field_specs__", None)
    if fields is None or "docs" not in fields:
        return stmt
    existing = stmt.docs  # type: ignore[attr-defined]
    return stmt.with_(docs=tuple(existing) + docs)  # type: ignore[attr-defined]


def parse_file(path: str | Path) -> Module:
    """Parse a `.qvr` file at `path`."""
    p = Path(path)
    return parse(p.read_bytes(), str(p))


# ---------------------------------------------------------------------------
# structural-compression walkers
# ---------------------------------------------------------------------------
