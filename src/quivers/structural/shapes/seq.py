"""Sequence-shaped signatures and canonical encoders / decoders.

Surface algebra::

    Seq[A] = Nil | Cons(A, Seq[A])

The principal shape behind autoregressive LMs (the "compress a
prefix to a vector, decode next-token from the vector" loop). All
encoders and decoders below are realised on top of the generic
[`quivers.structural.Encoder`][quivers.structural.Encoder] and
[`quivers.structural.Decoder`][quivers.structural.Decoder] runtimes, with sequence-
specific per-op functions.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

from ..encoder import Encoder, _PerOpFn
from ..decoder import Decoder
from ..signature import (
    DataLeaf,
    Constructor,
    Signature,
    Sort,
    Term,
)


def seq_signature(name: str = "Seq", dim: int = 64) -> Signature:
    """Return a sequence signature `Seq` with element sort `A` (data)."""
    sorts = {
        "Seq": Sort(name="Seq", kind="object", dim=dim),
        "A": Sort(name="A", kind="data", dim=dim),
    }
    constructors = {
        "Nil": Constructor(name="Nil", domain=(), codomain="Seq"),
        "Cons": Constructor(
            name="Cons",
            domain=("A", "Seq"),
            codomain="Seq",
        ),
    }
    return Signature(
        name=name,
        sorts_t=tuple(sorts.values()),
        constructors_t=tuple(constructors.values()),
    )


def _data_embedder(
    dim: int,
) -> tuple[nn.ParameterDict, Callable[[DataLeaf], torch.Tensor]]:
    table = nn.ParameterDict()

    def embed(key: DataLeaf, table=table, dim=dim) -> torch.Tensor:
        skey = str(key).replace(".", "_")
        if skey not in table:
            table[skey] = nn.Parameter(torch.randn(dim) * 0.1)
        return table[skey]

    return table, embed


def _learnable_nil(dim: int) -> tuple[nn.Module, torch.Tensor]:
    """A learnable zero-state vector for the `Nil` constructor.

    Returned as a `(module, parameter)` pair: ``module`` owns the
    parameter so that the enclosing `Encoder` picks it up
    through ``modules_owned`` and exposes it to optimizers via
    ``.parameters()``.
    """
    holder = nn.Module()
    p = nn.Parameter(torch.zeros(dim))
    holder.register_parameter("nil", p)
    return holder, p


def rnn_encoder(sig: Signature | None = None, dim: int = 64) -> Encoder:
    """A GRU-cell encoder: ``Cons(head, tail)`` updates a hidden
    state from the tail's compressed vector and the head's element
    embedding."""
    sig = sig or seq_signature(dim=dim)
    cell = nn.GRUCell(dim, dim)
    nil_mod, nil_const = _learnable_nil(dim)
    table, embedder = _data_embedder(dim)

    def cons_fn(a: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        return cell(a.reshape(1, dim), tail.reshape(1, dim)).reshape(dim)

    def nil_fn() -> torch.Tensor:
        return nil_const

    op_fns = {
        "Nil": _PerOpFn("Nil", "plain", (), nil_fn),
        "Cons": _PerOpFn("Cons", "plain", ("head", "tail"), cons_fn),
    }
    return Encoder(
        name="RNN",
        signature=sig,
        sort_dims={"Seq": dim, "A": dim},
        op_fns=op_fns,
        var_init_fns={},
        data_embedders={"A": embedder},
        modules_owned=[cell, table, nil_mod],
    )


def transformer_encoder(sig: Signature | None = None, dim: int = 64) -> Encoder:
    """A transformer-style encoder: each ``Cons`` step combines
    the head's element embedding with the tail's running compressed
    vector through a learned MLP over the concatenation."""
    sig = sig or seq_signature(dim=dim)
    head_proj = nn.Linear(dim, dim)
    tail_proj = nn.Linear(dim, dim)
    fuse = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, dim))
    nil_mod, nil_const = _learnable_nil(dim)
    table, embedder = _data_embedder(dim)

    def cons_fn(a: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        h = head_proj(a)
        t = tail_proj(tail)
        cat = torch.cat([h, t], dim=-1)
        return fuse(cat)

    def nil_fn() -> torch.Tensor:
        return nil_const

    op_fns = {
        "Nil": _PerOpFn("Nil", "plain", (), nil_fn),
        "Cons": _PerOpFn("Cons", "plain", ("head", "tail"), cons_fn),
    }
    return Encoder(
        name="Tfm",
        signature=sig,
        sort_dims={"Seq": dim, "A": dim},
        op_fns=op_fns,
        var_init_fns={},
        data_embedders={"A": embedder},
        modules_owned=[head_proj, tail_proj, fuse, table, nil_mod],
    )


def bow_encoder(sig: Signature | None = None, dim: int = 64) -> Encoder:
    """An order-independent sum-of-element-embeddings encoder.

    Each ``Cons(head, tail)`` step adds the head's element embedding
    to the running tail embedding; ``Nil`` is a learnable zero.
    Equivalent to a bag-of-tokens fixed-length representation.
    """
    sig = sig or seq_signature(dim=dim)
    nil_mod, nil_const = _learnable_nil(dim)
    table, embedder = _data_embedder(dim)

    def cons_fn(a: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        return a + tail

    def nil_fn() -> torch.Tensor:
        return nil_const

    op_fns = {
        "Nil": _PerOpFn("Nil", "plain", (), nil_fn),
        "Cons": _PerOpFn("Cons", "plain", ("head", "tail"), cons_fn),
    }
    return Encoder(
        name="BoW",
        signature=sig,
        sort_dims={"Seq": dim, "A": dim},
        op_fns=op_fns,
        var_init_fns={},
        data_embedders={"A": embedder},
        modules_owned=[table, nil_mod],
    )


def ar_decoder(
    sig: Signature | None = None,
    dim: int = 64,
    vocab: list[DataLeaf] | None = None,
    depth: int = 64,
) -> Decoder:
    """An autoregressive decoder over a sequence signature.

    ``vocab`` is the closed token set the primitive head selects
    from. Required; the runtime raises on an empty vocabulary.
    """
    if not vocab:
        raise ValueError("ar_decoder requires a non-empty vocabulary")
    sig = sig or seq_signature(dim=dim)
    # Two structural choices: Nil, Cons. (Seq has no binders and so
    # no BoundVar candidate is ever activated.)
    structure_head = nn.Linear(dim, 2)
    primitive = nn.Linear(dim, len(vocab))
    # Factor for Cons (arity 2 over sort Seq): split into head + tail.
    factor_lin = nn.Linear(dim, 2 * dim)
    # Bilinear binder-select. Seq itself never invokes this, but
    # signatures that compose Seq into a larger algebra (with a
    # binder reaching into Seq scope) do; the scorer is a real
    # learned bilinear key/query pair.
    bs_q = nn.Linear(dim, dim)
    bs_k = nn.Linear(dim, dim)

    def structure_fn(v: torch.Tensor) -> torch.Tensor:
        return structure_head(v.reshape(-1))

    def primitive_fn(v: torch.Tensor) -> torch.Tensor:
        return primitive(v.reshape(-1))

    def factor_2(v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = factor_lin(v.reshape(-1))
        return out[:dim], out[dim:]

    def binder_select_fn(v: torch.Tensor, embeds: list[torch.Tensor]) -> torch.Tensor:
        qv = bs_q(v.reshape(-1))
        keys = torch.stack([bs_k(e.reshape(-1)) for e in embeds], dim=0)
        return keys @ qv

    return Decoder(
        name="AR",
        signature=sig,
        sort_dims={"Seq": dim, "A": dim},
        depth=depth,
        structure_fns={"Seq": structure_fn},
        primitive_fns={"A": primitive_fn},
        factor_fns={"Seq": {2: factor_2}},
        binder_select_fn=binder_select_fn,
        data_vocab={"A": vocab},
        modules_owned=[structure_head, primitive, factor_lin, bs_q, bs_k],
    )


def list_to_term(elements: list[DataLeaf]) -> Term:
    """Convert a Python list to a right-nested ``Cons(..., Nil)`` term."""
    term = Term(op="Nil", args=())
    for e in reversed(elements):
        term = Term(op="Cons", args=(e, term))
    return term


def term_to_list(term: Term) -> list[DataLeaf]:
    """Inverse of `list_to_term`."""
    out: list[DataLeaf] = []
    cur = term
    while isinstance(cur, Term) and cur.op == "Cons":
        head, tail = cur.args
        out.append(head)
        cur = tail
    return out
