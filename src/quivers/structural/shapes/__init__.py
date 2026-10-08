"""Canonical signatures and encoder/decoder factories for the
three principal compressible shapes: sequences, trees, and graphs.

The stdlib shapes are defined directly in Python on top of the
generic [`quivers.structural.Encoder`][quivers.structural.Encoder] and
[`quivers.structural.Decoder`][quivers.structural.Decoder] runtimes so users can both
import them as objects and inspect them. QVR's surface form for
declaring these is unchanged: users write ``signature``,
``encoder``, ``decoder`` blocks; the shapes module provides
ready-made building blocks for the common cases.
"""

from __future__ import annotations

from quivers.structural.shapes.seq import (
    seq_signature,
    rnn_encoder,
    transformer_encoder,
    bow_encoder,
    ar_decoder,
    list_to_term,
    term_to_list,
)
from quivers.structural.shapes.tree import (
    tree_signature,
    tree_lstm_encoder,
    tree_decoder,
)
from quivers.structural.shapes.graph import (
    graph_signature,
    gnn_encoder,
)

__all__ = [
    # seq
    "seq_signature",
    "rnn_encoder",
    "transformer_encoder",
    "bow_encoder",
    "ar_decoder",
    "list_to_term",
    "term_to_list",
    # tree
    "tree_signature",
    "tree_lstm_encoder",
    "tree_decoder",
    # graph
    "graph_signature",
    "gnn_encoder",
]
