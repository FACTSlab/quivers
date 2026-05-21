"""Canonical signatures and encoder/decoder factories for the
three principal compressible shapes — sequences, trees, graphs.

The stdlib shapes are defined directly in Python on top of the
generic [`quivers.structural.Encoder`][quivers.structural.Encoder] and
[`quivers.structural.Decoder`][quivers.structural.Decoder] runtimes so users can both
import them as objects and inspect them. QVR's surface form for
declaring these is unchanged — users write ``signature``,
``encoder``, ``decoder`` blocks; the shapes module provides
ready-made building blocks for the common cases.
"""

from .seq import (
    seq_signature,
    rnn_encoder,
    transformer_encoder,
    bow_encoder,
    ar_decoder,
)
from .tree import (
    tree_signature,
    tree_lstm_encoder,
    tree_decoder,
)
from .graph import (
    graph_signature,
    gnn_encoder,
)

__all__ = [
    "seq_signature",
    "rnn_encoder",
    "transformer_encoder",
    "bow_encoder",
    "ar_decoder",
    "tree_signature",
    "tree_lstm_encoder",
    "tree_decoder",
    "graph_signature",
    "gnn_encoder",
]
