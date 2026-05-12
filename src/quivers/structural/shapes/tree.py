"""Tree-shaped signatures and canonical encoders / decoders.

Surface algebra::

    Tree[L, B] = Leaf(L) | Node(B, Tree, Tree)
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

from ..encoder import Encoder, _PerOpFn
from ..decoder import Decoder
from ..signature import DataLeaf, Constructor, Signature, Sort, Term


def tree_signature(name: str = "Tree", dim: int = 64) -> Signature:
    sorts = {
        "Tree": Sort(name="Tree", kind="object", dim=dim),
        "L": Sort(name="L", kind="data", dim=dim),
        "B": Sort(name="B", kind="data", dim=dim),
    }
    constructors = {
        "Leaf": Constructor(name="Leaf", domain=("L",), codomain="Tree"),
        "Node": Constructor(
            name="Node", domain=("B", "Tree", "Tree"), codomain="Tree",
        ),
    }
    return Signature(name=name, sorts_t=tuple(sorts.values()), constructors_t=tuple(constructors.values()))


def _data_embedder(dim: int) -> tuple[nn.ParameterDict, Callable[[DataLeaf], torch.Tensor]]:
    table = nn.ParameterDict()

    def embed(key: DataLeaf) -> torch.Tensor:
        skey = str(key).replace(".", "_")
        if skey not in table:
            table[skey] = nn.Parameter(torch.randn(dim) * 0.1)
        return table[skey]

    return table, embed


def tree_lstm_encoder(sig: Signature | None = None, dim: int = 64) -> Encoder:
    """A binary-tree LSTM encoder (Tai et al. 2015 child-sum
    variant)."""
    sig = sig or tree_signature(dim=dim)

    leaf_proj = nn.Linear(dim, dim)
    label_proj = nn.Linear(dim, dim)
    gate_mlp = nn.Sequential(
        nn.Linear(3 * dim, dim),
        nn.Tanh(),
        nn.Linear(dim, dim),
    )
    out_mlp = nn.Sequential(
        nn.Linear(3 * dim, dim),
        nn.Tanh(),
        nn.Linear(dim, dim),
    )
    leaf_tbl, leaf_embed = _data_embedder(dim)
    label_tbl, label_embed = _data_embedder(dim)

    def leaf_fn(token: torch.Tensor) -> torch.Tensor:
        return leaf_proj(token)

    def node_fn(label: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        b = label_proj(label)
        cat = torch.cat([b, left, right], dim=-1)
        gate = torch.sigmoid(gate_mlp(cat))
        h = torch.tanh(out_mlp(cat))
        return gate * h + (1 - gate) * (0.5 * (left + right))

    op_fns = {
        "Leaf": _PerOpFn("Leaf", "plain", ("t",), leaf_fn),
        "Node": _PerOpFn("Node", "plain", ("b", "l", "r"), node_fn),
    }
    return Encoder(
        name="TreeLSTM",
        signature=sig,
        sort_dims={"Tree": dim, "L": dim, "B": dim},
        op_fns=op_fns,
        var_init_fns={},
        data_embedders={"L": leaf_embed, "B": label_embed},
        modules_owned=[leaf_proj, label_proj, gate_mlp, out_mlp, leaf_tbl, label_tbl],
    )


def tree_decoder(
    sig: Signature | None = None,
    dim: int = 64,
    leaf_vocab: list[DataLeaf] | None = None,
    label_vocab: list[DataLeaf] | None = None,
    depth: int = 12,
) -> Decoder:
    """Top-down structural decoder over a tree signature."""
    if not leaf_vocab or not label_vocab:
        raise ValueError(
            "tree_decoder requires non-empty leaf_vocab and label_vocab"
        )
    sig = sig or tree_signature(dim=dim)
    structure = nn.Linear(dim, 3)  # Leaf / Node / BoundVar (unused)
    leaf_head = nn.Linear(dim, len(leaf_vocab))
    label_head = nn.Linear(dim, len(label_vocab))
    factor_1 = nn.Linear(dim, dim)
    factor_3 = nn.Linear(dim, 3 * dim)
    bs_q = nn.Linear(dim, dim)
    bs_k = nn.Linear(dim, dim)

    def structure_fn(v: torch.Tensor) -> torch.Tensor:
        return structure(v.reshape(-1))

    def f1(v: torch.Tensor) -> tuple[torch.Tensor]:
        return (factor_1(v.reshape(-1)),)

    def f3(v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = factor_3(v.reshape(-1))
        return out[:dim], out[dim : 2 * dim], out[2 * dim :]

    def binder_select_fn(v: torch.Tensor, embeds: list[torch.Tensor]) -> torch.Tensor:
        qv = bs_q(v.reshape(-1))
        keys = torch.stack([bs_k(e.reshape(-1)) for e in embeds], dim=0)
        return keys @ qv

    return Decoder(
        name="TreeAR",
        signature=sig,
        sort_dims={"Tree": dim, "L": dim, "B": dim},
        depth=depth,
        structure_fns={"Tree": structure_fn},
        primitive_fns={
            "L": lambda v, h=leaf_head: h(v.reshape(-1)),
            "B": lambda v, h=label_head: h(v.reshape(-1)),
        },
        factor_fns={"Tree": {1: f1, 3: f3}},
        binder_select_fn=binder_select_fn,
        data_vocab={"L": leaf_vocab, "B": label_vocab},
        modules_owned=[structure, leaf_head, label_head, factor_1, factor_3, bs_q, bs_k],
    )
