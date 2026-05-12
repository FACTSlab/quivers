"""Signatures: typed multi-sorted algebras with binders.

A :class:`Signature` is the runtime form of a ``signature { … }`` DSL
declaration. It carries the sort table, constructor table, binder
table, and (for graph-shaped signatures) vertex / edge kind tables.

Terms over a signature are first-class Python values
(:class:`Term` instances): each carries the constructor / binder
name and its positional arguments. The framework's encoder and
decoder runtimes walk these records uniformly.

The de-Bruijn discipline is enforced structurally: a ``BoundVar``
term carries an integer index; binders push a fresh entry onto an
implicit context Γ tracked by the encoder / decoder runtime.

All structured value types in this module are didactic
:class:`~didactic.api.Model` records, matching the quivers
convention for schema-bearing data; the only fields held opaque
are runtime-only artefacts that don't round-trip (torch tensors,
arbitrary Python data leaves).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import didactic.api as dx
import torch

# A raw data-leaf value: anything a user can pass at a data-sorted
# argument position. The union enumerates the cases we accept;
# anything outside this list is rejected at runtime with a typed
# error rather than silently coerced.
type DataLeaf = str | int | float | bytes | bool

# A positional argument inside a Term. Object positions carry a
# Term; data positions carry a DataLeaf; index positions carry an
# int. (Term is a forward reference here because it's defined
# below in this module.)
type TermArg = Term | DataLeaf | int

# The closed enumeration of sort kinds a signature can declare.
type SortKind = Literal["object", "index", "data"]


# ---------------------------------------------------------------------------
# Sort, constructor, binder, edge specs (all schema-bearing)
# ---------------------------------------------------------------------------


class SortVocabEntry(dx.Model):
    """One closed-vocabulary entry on a data sort.

    Each entry is a tagged Python value: ``kind`` is the leaf
    type (``"string" | "integer" | "float"``), ``value`` is the
    decoded Python value (``str``, ``int``, or ``float``).
    """

    kind: Literal["string", "integer", "float"]
    value: DataLeaf


class Sort(dx.Model):
    """A sort declaration: name, kind, optional dim, optional
    closed vocabulary.

    A data sort may carry a ``vocab`` tuple of
    :class:`SortVocabEntry` records. Object / index sorts must
    have an empty vocab (the framework checks at runtime).
    """

    name: str
    kind: SortKind
    dim: int | None = None
    vocab: tuple[SortVocabEntry, ...] = ()

    @property
    def vocab_values(self) -> tuple[DataLeaf, ...]:
        """The decoded Python values in declaration order."""
        return tuple(e.value for e in self.vocab)


class Constructor(dx.Model):
    """A typed constructor symbol."""

    name: str
    domain: tuple[str, ...] = ()
    codomain: str = ""

    @property
    def arity(self) -> int:
        return len(self.domain)


class BinderVarSpec(dx.Model):
    """A scoped variable introduced by a binder.

    ``sort`` is the variable's own sort. ``annot_sort`` is the sort
    of an optional *type annotation* — a sibling argument in the
    enclosing Term that travels alongside the variable's embedding
    in Γ. Used to track per-variable type information.
    """

    var: str
    sort: str
    annot_sort: str | None = None


class BinderArgSpec(dx.Model):
    """A scoped argument of a binder."""

    arg: str
    sort: str


class Binder(dx.Model):
    """A binder constructor: introduces variables of given sorts into
    the scope of given arguments."""

    name: str
    binds: tuple[BinderVarSpec, ...] = ()
    scoped: tuple[BinderArgSpec, ...] = ()
    codomain: str = ""

    @property
    def arity(self) -> int:
        n_annots = sum(1 for b in self.binds if b.annot_sort is not None)
        return n_annots + len(self.scoped)

    def domain(self) -> tuple[str, ...]:
        """Positional sort sequence of the binder's arguments in the
        enclosing :class:`Term`: type-annotations for each annotated
        bound variable (in declaration order, outer-context-evaluated),
        followed by the scoped arguments (extended-context-evaluated).
        """
        return tuple(
            b.annot_sort for b in self.binds if b.annot_sort is not None
        ) + tuple(a.sort for a in self.scoped)


class VertexKind(dx.Model):
    """A vertex kind in a graph-shaped signature."""

    name: str
    kind: SortKind
    dim: int | None = None


class EdgeKind(dx.Model):
    """An edge kind in a graph-shaped signature."""

    name: str
    src: str
    tgt: str
    directed: bool = True


# ---------------------------------------------------------------------------
# Signature (whole-algebra record)
# ---------------------------------------------------------------------------


class Signature(dx.Model):
    """A multi-sorted algebra signature with optional binders / graph
    structure.

    Sort, constructor, binder, vertex_kind and edge_kind tables are
    represented as tuples of records (rather than dicts) so the
    enclosing :class:`dx.Model` keeps a schema-bearing layout. The
    runtime exposes O(1) name-keyed lookup methods (``sort(name)``,
    ``constructor(name)``, etc.) on top of those tuples.
    """

    name: str
    params: tuple[str, ...] = ()
    sorts_t: tuple[Sort, ...] = ()
    constructors_t: tuple[Constructor, ...] = ()
    binders_t: tuple[Binder, ...] = ()
    vertex_kinds_t: tuple[VertexKind, ...] = ()
    edge_kinds_t: tuple[EdgeKind, ...] = ()

    # ---- dict-like lookups ----

    @property
    def sorts(self) -> dict[str, Sort]:
        return {s.name: s for s in self.sorts_t}

    @property
    def constructors(self) -> dict[str, Constructor]:
        return {c.name: c for c in self.constructors_t}

    @property
    def binders(self) -> dict[str, Binder]:
        return {b.name: b for b in self.binders_t}

    @property
    def vertex_kinds(self) -> dict[str, VertexKind]:
        return {v.name: v for v in self.vertex_kinds_t}

    @property
    def edge_kinds(self) -> dict[str, EdgeKind]:
        return {e.name: e for e in self.edge_kinds_t}

    # ---- shape queries ----

    def is_inductive(self) -> bool:
        return bool(self.constructors_t) or bool(self.binders_t)

    def is_graph(self) -> bool:
        return bool(self.vertex_kinds_t) and bool(self.edge_kinds_t)

    def all_ops(self) -> Iterable[str]:
        for c in self.constructors_t:
            yield c.name
        for b in self.binders_t:
            yield b.name

    def codomain_of(self, op: str) -> str:
        for c in self.constructors_t:
            if c.name == op:
                return c.codomain
        for b in self.binders_t:
            if b.name == op:
                return b.codomain
        raise KeyError(
            f"signature {self.name!r}: op {op!r} not a constructor or binder"
        )

    def domain_of(self, op: str) -> tuple[str, ...]:
        for c in self.constructors_t:
            if c.name == op:
                return c.domain
        for b in self.binders_t:
            if b.name == op:
                return b.domain()
        raise KeyError(
            f"signature {self.name!r}: op {op!r} not a constructor or binder"
        )

    def is_binder(self, op: str) -> bool:
        return any(b.name == op for b in self.binders_t)

    def sort_dim(self, sort: str) -> int | None:
        for s in self.sorts_t:
            if s.name == sort:
                return s.dim
        for v in self.vertex_kinds_t:
            if v.name == sort:
                return v.dim
        return None


# ---------------------------------------------------------------------------
# Term representation
# ---------------------------------------------------------------------------


class Term(dx.Model):
    """A closed term over a signature.

    ``op`` is the constructor name, the binder name, or the
    framework-reserved ``"BoundVar"`` for a de-Bruijn variable
    reference.

    ``args`` are positional children. Object positions carry a
    :class:`Term`; data positions carry a :data:`DataLeaf` raw
    value; index positions carry a non-negative ``int``. ``args``
    is held opaque (its element type is the open union
    :data:`TermArg`); the encoder / decoder enforce sort
    agreement structurally at use time.

    No wrapping such as ``Term("Data", …)`` is used at any
    position; raw values appear directly at data positions.
    """

    op: str
    args: tuple = dx.field(default=(), opaque=True)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        if not self.args:
            return self.op
        inside = ", ".join(repr(a) for a in self.args)
        return f"{self.op}({inside})"

    def to_tuple(self) -> tuple:
        """A serialisable form matching the agenda's item-tuple
        convention: ``(op, *args_serialised)``. Children that are
        :class:`Term`s are recursively serialised; data leaves and
        indices pass through."""
        out: list = [self.op]
        for a in self.args:
            if isinstance(a, Term):
                out.append(a.to_tuple())
            else:
                out.append(a)
        return tuple(out)


def bound_var(index: int) -> Term:
    """A de-Bruijn variable reference."""
    if not isinstance(index, int) or index < 0:
        raise TypeError(f"bound_var requires a non-negative int, got {index!r}")
    return Term(op="BoundVar", args=(index,))


def make_term(op: str, *args) -> Term:
    """Construct a term over a signature.

    Each ``arg`` must be a :class:`Term`, a :data:`DataLeaf` raw
    value, or a non-negative int (for index-sorted positions). The
    encoder / decoder runtime validate sort agreement at use
    time.
    """
    if not isinstance(op, str):
        raise TypeError(f"make_term requires a string op, got {type(op).__name__}")
    return Term(op=op, args=tuple(args))


# ---------------------------------------------------------------------------
# de-Bruijn context
# ---------------------------------------------------------------------------


# `ContextEntry` and `Context` are intentionally plain dataclasses
# rather than didactic models: they hold raw torch tensors at every
# scope position, and didactic's tuple-of-model encoding strips
# per-instance opaque storage when entries are placed inside a
# parent model's tuple field. They are runtime-only structures
# (never serialised through panproto), so the schema-bearing
# benefit of dx.Model doesn't apply here.


@dataclass(frozen=True)
class ContextEntry:
    """One scope entry on the binder stack.

    ``embedding`` is a runtime tensor; ``type_term`` is the binder's
    annotation term captured at scope-extension time.
    """

    var_sort: str
    embedding: torch.Tensor
    type_term: Term | None = None


@dataclass(frozen=True)
class Context:
    """The de-Bruijn scope context threaded through encoder and
    decoder runtimes.

    Position 0 is the most recently bound variable; lookup ``var(i)``
    returns the i-th entry from the top.
    """

    entries: tuple[ContextEntry, ...] = ()

    def push(
        self,
        var_sort: str,
        embedding: torch.Tensor,
        type_term: Term | None = None,
    ) -> "Context":
        return Context(
            entries=(
                ContextEntry(
                    var_sort=var_sort,
                    embedding=embedding,
                    type_term=type_term,
                ),
            )
            + self.entries,
        )

    def var(self, index: int) -> torch.Tensor:
        if index < 0 or index >= len(self.entries):
            raise IndexError(
                f"Context.var({index}): out of range "
                f"(context depth {len(self.entries)})"
            )
        return self.entries[index].embedding

    def type_of(self, index: int) -> Term | None:
        return self.entries[index].type_term

    def depth(self) -> int:
        return len(self.entries)

    def by_sort(self, sort: str) -> list[tuple[int, ContextEntry]]:
        """All entries whose ``var_sort`` matches; returned as
        ``(depth-index, entry)`` pairs for the decoder's
        categorical-over-in-scope-variables."""
        return [(i, e) for i, e in enumerate(self.entries) if e.var_sort == sort]


EMPTY_CONTEXT = Context()
