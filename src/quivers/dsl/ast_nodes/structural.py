"""Structural-compression helper AST nodes.

These are the non-Statement support models used by
`SignatureDecl`, `EncoderDecl`, `DecoderDecl`, and
`LossDecl`. The four Statement subclasses themselves live in
``declarations.py`` because they participate in the top-level
`Statement` tagged union.
"""

from typing import Literal

import didactic.api as dx

from quivers.dsl.ast_nodes._shared import OptionEntry
from quivers.dsl.ast_nodes.let_expressions import LetExprNode

class SortVocabLiteral(dx.Model):
    """One entry of a data sort's closed vocabulary.

    The literal carries its surface text plus a tag so the compiler
    can decode each entry into the canonical Python value the
    runtime stores in `Sort.vocab` (``str``, ``int``, or
    ``float``).
    """

    kind: Literal["string", "integer", "float"]
    text: str

class SortDecl(dx.Model):
    """One sort within a signature.

    `kind` is one of ``"object"``, ``"index"``, ``"data"``. The
    surface form moves ``dim`` and ``vocab`` into the unified
    option block: ``s : data [dim=16, vocab=["a", "b"]]``. The
    compiler reads them back out of ``options`` at elaboration time.
    """

    name: str
    kind: Literal["object", "index", "data"]
    options: tuple[OptionEntry, ...] = ()
    line: int = 0
    col: int = 0

class ConstructorDecl(dx.Model):
    """A typed operation `name : s_1, ..., s_n -> s` in a signature."""

    name: str
    domain: tuple[str, ...]
    codomain: str
    line: int = 0
    col: int = 0

class BinderVar(dx.Model):
    """A variable introduced by a binder.

    ``var`` and ``annot`` are names used for diagnostics only;
    references are by de-Bruijn index inside the scope. ``sort`` is
    the sort of the variable itself; ``annot_sort`` is the sort of
    the variable's type annotation, if one is supplied. When
    ``annot_sort`` is set, the binder constructor takes one
    additional positional argument (immediately preceding the
    bound variable's role in the scope) of that sort, which the
    encoder / decoder thread into Γ alongside the variable's
    embedding.
    """

    var: str
    sort: str
    annot: str | None = None
    annot_sort: str | None = None

class BinderArg(dx.Model):
    """An argument of a binder constructor; ``scoped`` arguments live
    in the extended context."""

    arg: str
    sort: str

class BinderDecl(dx.Model):
    """A binder constructor introducing new scoped variables."""

    name: str
    binds: tuple[BinderVar, ...]
    scoped: tuple[BinderArg, ...]
    codomain: str
    line: int = 0
    col: int = 0

class VertexKindDecl(dx.Model):
    """A vertex kind in a graph-shaped signature.

    Like `SortDecl`, the surface threads ``dim`` and any
    other modifier through the unified option block.
    """

    name: str
    kind: Literal["object", "index", "data"]
    options: tuple[OptionEntry, ...] = ()
    line: int = 0
    col: int = 0

class EdgeKindDecl(dx.Model):
    """An edge kind in a graph-shaped signature.

    ``directed`` is True for ``src -> tgt``, False for ``src -- tgt``.
    """

    name: str
    src: str
    tgt: str
    directed: bool = True
    line: int = 0
    col: int = 0

class SortDim(dx.Model):
    """A `(sort, dim)` association declared in a encoder/decoder."""

    sort: str
    dim: int

class EncoderVarInit(dx.Model):
    """One `var_init <var_sort> [from <annot_sort> [as <name>]]` rule.

    ``annot_sort=None`` is the unannotated-binder case (no type
    annotation; the body sees no extra arg).  When ``annot_sort`` is
    set, ``ty`` is the body's parameter name bound to the annotation
    embedding.
    """

    var_sort: str
    annot_sort: str | None = None
    ty: str | None = None
    body: LetExprNode
    line: int = 0
    col: int = 0

class EncoderRule(dx.Model):
    """A per-operation encoder function.

    The body is a let-expression evaluated in an environment where the
    constructor arguments (as named in ``args``) are bound to child
    vectors, plus framework-supplied helpers (``ctx`` for binder
    contexts, ``state``/``prefix`` for recurrent / attention shapes).

    ``mode`` selects sequence sugar:

    * ``"plain"`` for direct algebra-hom rule (default).
    * ``"recurrent"`` for left-fold, ``state`` carries the accumulator.
    * ``"attention"`` for ``prefix`` carries the running list of prior
      compressed children.
    """

    op: str
    args: tuple[str, ...]
    body: LetExprNode
    mode: Literal["plain", "recurrent", "attention"] = "plain"
    state_var: str | None = None
    prefix_var: str | None = None
    line: int = 0
    col: int = 0

class EncoderInitRule(dx.Model):
    """Graph-signature initializer: maps vertex `data` payloads to
    initial vertex embeddings before message passing."""

    kind: str
    arg: str
    body: LetExprNode
    line: int = 0
    col: int = 0

class EncoderMessageRule(dx.Model):
    """Graph-signature message: maps a `(src, tgt)` pair on an edge
    kind to a message vector."""

    edge_kind: str
    src: str
    tgt: str
    body: LetExprNode
    line: int = 0
    col: int = 0

class EncoderUpdateRule(dx.Model):
    """Graph-signature update: maps `(self_embed, aggregated_msgs)`
    to the next vertex embedding, per vertex kind."""

    vertex_kind: str
    self_var: str
    msgs_var: str
    body: LetExprNode
    line: int = 0
    col: int = 0

__all__ = [
    "SortVocabLiteral",
    "SortDecl",
    "ConstructorDecl",
    "BinderVar",
    "BinderArg",
    "BinderDecl",
    "VertexKindDecl",
    "EdgeKindDecl",
    "SortDim",
    "EncoderVarInit",
    "EncoderRule",
    "EncoderInitRule",
    "EncoderMessageRule",
    "EncoderUpdateRule",
]
