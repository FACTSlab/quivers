\
"""Parse-time registry singleton, ParseError, and the _Tree view."""

from __future__ import annotations

import warnings

import panproto


class ParseError(Exception):
    """Raised when the .qvr source fails to parse or wrap into AST nodes."""


# ---------------------------------------------------------------------------
# panproto registry singleton
# ---------------------------------------------------------------------------

_REGISTRY: panproto.AstParserRegistry | None = None


def _registry() -> panproto.AstParserRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        from quivers.dsl import _dev_grammar

        if _dev_grammar.is_active():
            _REGISTRY = _dev_grammar.registry()  # type: ignore[assignment]
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _REGISTRY = panproto.AstParserRegistry()
        if "qvr" not in _REGISTRY.protocol_names():
            raise ParseError(
                "panproto registry has no `qvr` protocol; install "
                "`panproto-grammars-all` (or a pack containing qvr)"
            )
    return _REGISTRY


# ---------------------------------------------------------------------------
# tree-walking helpers
# ---------------------------------------------------------------------------


class _Tree:
    """Indexed view of a parsed panproto schema."""

    def __init__(self, schema, source: bytes) -> None:
        self.schema = schema
        self.source = source
        self.vertices = {v.id: v for v in schema.vertices}
        self.children: dict[str, list] = {}
        for e in schema.edges:
            self.children.setdefault(e.src, []).append(e)
        self._consts: dict[str, dict[str, str]] = {}

    def consts(self, vid: str) -> dict[str, str]:
        c = self._consts.get(vid)
        if c is None:
            c = {item.sort: item.value for item in self.schema.constraints_for(vid)}
            self._consts[vid] = c
        return c

    def kind(self, vid: str) -> str:
        return self.vertices[vid].kind

    def text(self, vid: str) -> str:
        c = self.consts(vid)
        lit = c.get("literal-value")
        if lit is not None:
            return lit
        sb = c.get("start-byte")
        eb = c.get("end-byte")
        if sb is not None and eb is not None:
            return self.source[int(sb) : int(eb)].decode("utf-8")
        return ""

    def line_col(self, vid: str) -> tuple[int, int]:
        c = self.consts(vid)
        sb = c.get("start-byte")
        if sb is None:
            return 0, 0
        prefix = self.source[: int(sb)]
        line = prefix.count(b"\n") + 1
        last_nl = prefix.rfind(b"\n")
        col = (int(sb) - last_nl - 1) if last_nl >= 0 else int(sb)
        return line, col

    def _sort_key(self, vid: str) -> int:
        sb = self.consts(vid).get("start-byte")
        return int(sb) if sb is not None else 0

    def positional(self, parent_id: str) -> list[str]:
        kids = [e.tgt for e in self.children.get(parent_id, []) if e.kind == "child_of"]
        kids.sort(key=self._sort_key)
        return kids

    def field(self, parent_id: str, name: str) -> str | None:
        for e in self.children.get(parent_id, []):
            if e.kind == name:
                return e.tgt
        return None

    def fields(self, parent_id: str, name: str) -> list[str]:
        kids = [e.tgt for e in self.children.get(parent_id, []) if e.kind == name]
        kids.sort(key=self._sort_key)
        return kids
