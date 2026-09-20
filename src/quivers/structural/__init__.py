"""Structural compression: signatures, encoders, decoders, losses.

Runtime substrate for QVR signatures, encoders, decoders, and attached losses:
a uniform algebraic interface for compressing structured objects to
fixed-length vectors and decoding them under a learned distribution.
"""

from .encoder import Encoder
from .decoder import Decoder
from .losses import LossEntry, LossRegistry
from .signature import (
    Binder,
    BinderArgSpec,
    BinderVarSpec,
    Constructor,
    Context,
    DataLeaf,
    EMPTY_CONTEXT,
    EdgeKind,
    Signature,
    Sort,
    Term,
    VertexKind,
    bound_var,
    make_term,
)

__all__ = [
    "Binder",
    "BinderArgSpec",
    "BinderVarSpec",
    "Encoder",
    "Constructor",
    "Context",
    "DataLeaf",
    "Decoder",
    "EMPTY_CONTEXT",
    "EdgeKind",
    "LossEntry",
    "LossRegistry",
    "Signature",
    "Sort",
    "Term",
    "VertexKind",
    "bound_var",
    "make_term",
]
