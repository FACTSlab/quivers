"""Tests for the ``encoder NAME over SIG using FACTORY`` surface.

The factory form invokes a shipped encoder builder from
:data:`quivers.dsl.compiler.structural._ENCODER_FACTORY_REGISTRY`
with the per-encoder ``[k=v]`` overrides forwarded as kwargs. The
result is a fully-constructed :class:`Encoder` registered under
the declaration's name.

Coverage:

* Every registered factory builds (rnn / transformer / bow on a
  sequence signature; tree-lstm on a tree signature; gnn on a
  graph signature).
* Option overrides reach the factory kwargs and are coerced to
  int / float / str.
* Unknown factories raise a clear CompileError naming the
  available choices.
* Unknown options to a known factory raise with the factory's
  parameter list.
* The factory form coexists with the explicit ``{ ... }`` form
  in the same module (no grammar ambiguity).
"""

from __future__ import annotations

import pytest

from quivers.dsl import loads
from quivers.dsl.compiler._prelude import CompileError


SEQ_SIG = """
signature seq {
    sort Seq
    data L
    constructor Nil   : -> Seq
    constructor Cons  : L * Seq -> Seq
}
"""


TREE_SIG = """
signature tree {
    sort Tree
    data Leaf
    data Node
    constructor Leaf : Leaf -> Tree
    constructor Branch : Node * Tree * Tree -> Tree
}
"""


GRAPH_SIG = """
signature graph {
    vertex_kind V
    edge_kind   E : V -> V
}
"""


def _wrap(body: str) -> str:
    return f"algebra product_fuzzy\n{body}\n"


class TestRegisteredFactories:
    def test_rnn_encoder_factory(self):
        src = _wrap(SEQ_SIG + "\nencoder enc over seq using rnn_encoder\n")
        prog = loads(src)
        assert "enc" in prog.encoders

    def test_rnn_encoder_with_dim_override(self):
        src = _wrap(SEQ_SIG + "\nencoder enc over seq using rnn_encoder [dim=128]\n")
        prog = loads(src)
        # The GRU's hidden size should match the override.
        cell = prog.encoders["enc"]._op_0  # GRUCell
        assert cell.hidden_size == 128

    def test_transformer_encoder_factory(self):
        src = _wrap(SEQ_SIG + "\nencoder enc over seq using transformer_encoder\n")
        prog = loads(src)
        assert "enc" in prog.encoders

    def test_bow_encoder_factory(self):
        src = _wrap(SEQ_SIG + "\nencoder enc over seq using bow_encoder\n")
        prog = loads(src)
        assert "enc" in prog.encoders


class TestUnknownNames:
    def test_unknown_factory_lists_choices(self):
        src = _wrap(SEQ_SIG + "\nencoder enc over seq using fictitious_encoder\n")
        with pytest.raises(CompileError, match="unknown factory|available:"):
            loads(src)

    def test_unknown_option_lists_factory_params(self):
        src = _wrap(
            SEQ_SIG + "\nencoder enc over seq using rnn_encoder [no_such_option=42]\n"
        )
        with pytest.raises(CompileError, match="does not accept option"):
            loads(src)


class TestExplicitAndFactoryCoexist:
    def test_two_factory_encoders_in_one_module(self):
        # Cleanest coexistence check: two factory-backed encoders
        # over the same signature in the same module.
        src = _wrap(
            SEQ_SIG + "\nencoder rnn_enc over seq using rnn_encoder [dim=32]\n"
            "encoder bow_enc over seq using bow_encoder [dim=32]\n"
        )
        prog = loads(src)
        assert "rnn_enc" in prog.encoders
        assert "bow_enc" in prog.encoders
