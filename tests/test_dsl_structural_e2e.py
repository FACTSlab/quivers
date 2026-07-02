"""End-to-end tests for the shipped term-autoencoder example.

Compiles ``docs/examples/source/term_autoencoder.qvr``, the gallery
example for the combined signature / encoder / decoder / loss
surface, and exercises the full round trip: fold a term into a code
vector, sample and score terms from the code, and evaluate the
attached reconstruction loss. Negative cases pin the structural
diagnostics for unknown option keys, ops outside the signature, and
unresolved sort dims.

Requires ``QVR_USE_LOCAL_GRAMMAR=1`` (set by ``tests/conftest.py``,
and defaulted again here for direct invocation) so parsing picks up
the in-tree grammar at ``grammars/qvr/``.
"""

from __future__ import annotations

import os
import textwrap
from collections.abc import Mapping
from pathlib import Path
from typing import cast

os.environ.setdefault("QVR_USE_LOCAL_GRAMMAR", "1")

import pytest
import torch

from quivers.dsl import CompileError, load, loads, parse
from quivers.dsl.ast_nodes import (
    DecoderDecl,
    EncoderDecl,
    LossDecl,
    SignatureDecl,
)
from quivers.program import Program
from quivers.structural import Term, bound_var, make_term
from quivers.structural.decoder import Decoder
from quivers.structural.encoder import Encoder
from quivers.structural.losses import LossRegistry, TrainEnv
from quivers.structural.signature import Signature

_EXAMPLE = (
    Path(__file__).resolve().parent.parent
    / "docs"
    / "examples"
    / "source"
    / "term_autoencoder.qvr"
)


@pytest.fixture(scope="module")
def program() -> Program:
    return load(_EXAMPLE)


def _artifacts(program: Program) -> tuple[Encoder, Decoder, LossRegistry]:
    """Pull the example's compiled artifacts off the Program container
    with runtime type verification (the container exposes them as
    dynamically attached attributes)."""
    encoders = program.encoders
    decoders = program.decoders
    losses = program.losses
    assert isinstance(encoders, dict)
    assert isinstance(decoders, dict)
    assert isinstance(losses, LossRegistry)
    enc = cast("dict[str, Encoder]", encoders)["Enc"]
    dec = cast("dict[str, Decoder]", decoders)["Dec"]
    assert isinstance(enc, Encoder)
    assert isinstance(dec, Decoder)
    return enc, dec, losses


def _loss_env(term: Term) -> Mapping[str, TrainEnv]:
    """Wrap a term observation as a loss-evaluation environment."""
    return cast("Mapping[str, TrainEnv]", {"term": term})


def _example_term() -> Term:
    # \x : base. plus x
    return make_term(
        "Lam",
        make_term("Base", "base"),
        make_term("App", make_term("Const", "plus"), bound_var(0)),
    )


# ---------------------------------------------------------------------------
# Compilation and registries
# ---------------------------------------------------------------------------


def test_example_compiles_and_registers_artifacts(program: Program) -> None:
    signatures = program.signatures
    assert isinstance(signatures, dict)
    assert set(signatures) == {"STLC"}
    sig = cast("dict[str, Signature]", signatures)["STLC"]
    assert isinstance(sig, Signature)
    assert set(sig.sorts) == {"Term", "Type", "Name"}
    assert sig.sorts["Name"].kind == "data"
    assert sig.sorts["Name"].vocab_values == ("f", "x", "plus", "base")
    assert sig.constructors["App"].domain == ("Term", "Term")
    assert sig.binders["Lam"].binds[0].annot_sort == "Type"

    enc, dec, losses = _artifacts(program)
    assert enc.name == "Enc"
    assert dec.name == "Dec"
    assert len(losses.entries) == 1
    entry = losses.entries[0]
    assert entry.name == "reconstruct"
    assert entry.attachment_kind == "decoder"
    assert entry.target == "Dec"
    assert entry.weight is not None


def test_doc_comments_attach_to_every_declaration() -> None:
    module = parse(_EXAMPLE.read_text())
    structural = [
        stmt
        for stmt in module.statements
        if isinstance(stmt, (SignatureDecl, EncoderDecl, DecoderDecl, LossDecl))
    ]
    assert len(structural) == 4
    for decl in structural:
        assert decl.docs, f"{decl.kind} carries no #! doc comment"


# ---------------------------------------------------------------------------
# Encoder / decoder round trip
# ---------------------------------------------------------------------------


def test_encoder_compresses_a_term(program: Program) -> None:
    enc, _, _ = _artifacts(program)
    # The explicit ``op`` rule is installed for App; scaffolded
    # defaults cover the remaining operators.
    rule = enc.op_fns["App"]
    assert rule.mode == "plain"
    assert rule.args == ("fun", "arg")
    assert set(enc.op_fns) == {"Const", "App", "Base", "Lam"}
    # The var_init override is keyed by the (variable, annotation)
    # sort pair the Lam binder declares.
    assert ("Term", "Type") in enc.var_init_fns

    code = enc(_example_term())
    assert code.shape == (24,)
    assert code.requires_grad


def test_decoder_samples_and_scores(program: Program) -> None:
    enc, dec, _ = _artifacts(program)
    torch.manual_seed(0)
    sample = dec(torch.randn(24))
    assert isinstance(sample, Term)

    term = _example_term()
    lp = dec.log_prob(term, enc(term))
    assert torch.is_tensor(lp)
    assert lp.shape == ()
    assert torch.isfinite(lp)
    lp.backward()


def test_decoder_samples_data_leaves_from_declared_vocab(
    program: Program,
) -> None:
    _, dec, _ = _artifacts(program)
    vocab = {"f", "x", "plus", "base"}

    def leaves(term: Term) -> list[str]:
        if term.op == "Data":
            return [str(term.args[0])]
        out: list[str] = []
        for arg in term.args:
            if isinstance(arg, Term):
                out.extend(leaves(arg))
            elif isinstance(arg, str):
                out.append(arg)
        return out

    torch.manual_seed(1)
    for _ in range(5):
        sample = dec(torch.randn(24))
        assert set(leaves(sample)) <= vocab


# ---------------------------------------------------------------------------
# Loss evaluation
# ---------------------------------------------------------------------------


def test_loss_evaluates_reconstruction_nll(program: Program) -> None:
    enc, dec, losses = _artifacts(program)
    term = _example_term()

    total = losses.evaluate(_loss_env(term))
    assert torch.is_tensor(total)
    assert total.requires_grad
    # weight=1.0, so the registry total is exactly the negative
    # round-trip log-likelihood.
    expected = -dec.log_prob(term, enc(term))
    torch.testing.assert_close(total, expected)

    attached = losses.evaluate_on("decoder", "Dec", _loss_env(term))
    torch.testing.assert_close(attached, expected)
    total.backward()


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

_BASE = """
signature S
    sorts
        Term : object [dim=8]
    constructors
        A : -> Term
        B : Term -> Term
"""


def test_unknown_encoder_option_key_is_rejected() -> None:
    src = _BASE + """
encoder E : S [facotry=bow_encoder]
    dim Term = 8
"""
    with pytest.raises(
        CompileError,
        match=r"encoder 'E': unknown option 'facotry'; did you mean 'factory'\?",
    ):
        loads(textwrap.dedent(src))


def test_unknown_decoder_option_key_is_rejected() -> None:
    src = _BASE + """
decoder D : S [depht=4]
    body |-> recursive
"""
    with pytest.raises(
        CompileError,
        match=r"decoder 'D': unknown option 'depht'; did you mean 'depth'\?",
    ):
        loads(textwrap.dedent(src))


def test_encoder_op_outside_signature_is_rejected() -> None:
    src = _BASE + """
encoder E : S
    op Zap(x) |-> x
"""
    with pytest.raises(
        CompileError,
        match=r"encoder 'E': op 'Zap' is not in signature 'S'",
    ):
        loads(textwrap.dedent(src))


def test_unresolved_sort_dim_is_rejected() -> None:
    # A sort with no dim on the signature and no ``dim`` override in
    # the decoder block has no resolvable embedding width.
    src = """
signature S2
    sorts
        Term : object
    constructors
        A : -> Term

decoder D : S2 [depth=3]
    body |-> recursive
"""
    with pytest.raises(CompileError, match=r"decoder 'D': sort 'Term' has no dim"):
        loads(textwrap.dedent(src))
