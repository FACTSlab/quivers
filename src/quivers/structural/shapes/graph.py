"""Graph-shaped signatures and canonical encoder.

A graph signature is declared with vertex kinds and edge kinds; the
canonical `GNN` encoder runs ``iterations`` rounds of
message passing with per-edge-kind message functions and
per-vertex-kind update functions.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

from ..encoder import Encoder
from ..signature import DataLeaf, EdgeKind, Signature, VertexKind


def graph_signature(
    name: str,
    vertex_kinds: dict[str, int],
    edge_kinds: dict[str, tuple[str, str, bool]],
) -> Signature:
    """Construct a graph signature.

    ``vertex_kinds`` maps name -> embedding dim.
    ``edge_kinds`` maps name -> (src_kind, tgt_kind, directed).
    """
    vk = {n: VertexKind(name=n, kind="data", dim=d) for n, d in vertex_kinds.items()}
    ek = {
        n: EdgeKind(name=n, src=s, tgt=t, directed=directed)
        for n, (s, t, directed) in edge_kinds.items()
    }
    return Signature(
        name=name, vertex_kinds_t=tuple(vk.values()), edge_kinds_t=tuple(ek.values())
    )


def gnn_encoder(
    sig: Signature,
    iterations: int = 4,
    dim: int = 64,
    readout: str = "mean",
) -> Encoder:
    """A GNN encoder: per-edge-kind message MLP, per-vertex-kind
    GRU update, mean / sum / max readout.
    """
    init_fns: dict[str, Callable[[DataLeaf], torch.Tensor]] = {}
    message_fns: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {}
    update_fns: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {}
    modules_owned: list[nn.Module] = []

    for vname, v in sig.vertex_kinds.items():
        vdim = v.dim or dim
        table = nn.ParameterDict()
        modules_owned.append(table)

        def make_init(table=table, vdim=vdim):
            def init(payload):
                key = str(payload).replace(".", "_")
                if key not in table:
                    table[key] = nn.Parameter(torch.randn(vdim) * 0.1)
                return table[key]

            return init

        init_fns[vname] = make_init()

        gru = nn.GRUCell(vdim, vdim)
        modules_owned.append(gru)

        def make_update(gru=gru, vdim=vdim):
            def upd(self_e, msg):
                return gru(msg.reshape(1, vdim), self_e.reshape(1, vdim)).reshape(vdim)

            return upd

        update_fns[vname] = make_update()

    for ename, e in sig.edge_kinds.items():
        src_dim = sig.vertex_kinds[e.src].dim or dim
        tgt_dim = sig.vertex_kinds[e.tgt].dim or dim
        out_dim = tgt_dim
        mlp = nn.Sequential(
            nn.Linear(src_dim + tgt_dim, max(out_dim, 64)),
            nn.Tanh(),
            nn.Linear(max(out_dim, 64), out_dim),
        )
        modules_owned.append(mlp)

        def make_msg(mlp=mlp, src_dim=src_dim, tgt_dim=tgt_dim):
            def msg(s, t):
                cat = torch.cat([s.reshape(-1), t.reshape(-1)], dim=0)
                return mlp(cat)

            return msg

        message_fns[ename] = make_msg()

    def readout_fn(embeds, mode=readout):
        stack = torch.stack(embeds, dim=0)
        if mode == "mean":
            return stack.mean(dim=0)
        if mode == "sum":
            return stack.sum(dim=0)
        if mode == "max":
            return stack.max(dim=0).values
        raise ValueError(f"unknown readout {mode!r}")

    sort_dims = {n: (v.dim or dim) for n, v in sig.vertex_kinds.items()}

    return Encoder(
        name="GNN",
        signature=sig,
        sort_dims=sort_dims,
        op_fns={},
        var_init_fns={},
        data_embedders={},
        modules_owned=modules_owned,
        iterations=iterations,
        init_fns=init_fns,
        message_fns=message_fns,
        update_fns=update_fns,
        readout=readout_fn,
    )
