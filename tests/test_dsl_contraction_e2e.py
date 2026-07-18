"""End-to-end DSL tests for ``contraction`` declarations.

The fixture is the bilinear-scoring example at
``docs/examples/source/tensor_contraction.qvr``: three latent arrows
(two embeddings and a third-order interaction tensor) combined by a
ternary ``contraction`` under ``rule=real`` and bound at a ``define``
site. The suite checks four contracts:

1. The example compiles through both :func:`quivers.dsl.load` and
   :func:`quivers.dsl.loads`.
2. The compiled contraction carries the expected runtime wiring: the
   einsum spec inferred from the typed signature, the input arity and
   parameter names, and the registered composition rule.
3. A forward pass materializes the bilinear score, both through the
   exported Program and through the wiring applied to tiny hand-built
   tensors, matching ``torch.einsum`` exactly under the sum-product
   rule.
4. The error surface: an unknown ``rule=`` name reports the available
   registry, and an unknown option key gets the did-you-mean
   diagnostic.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch


from quivers.core.morphisms import ObservedMorphism
from quivers.core.wiring import EinsumWiring
from quivers.dsl import CompileError, load, loads
from quivers.dsl.compiler import Compiler
from quivers.dsl.parser import parse_file

_EXAMPLE = (
    Path(__file__).resolve().parent.parent
    / "docs"
    / "examples"
    / "source"
    / "tensor_contraction.qvr"
)


@pytest.fixture(scope="module")
def compiler() -> Compiler:
    """One compiled instance of the example, shared across the
    inspection tests below."""
    c = Compiler(parse_file(_EXAMPLE))
    c.compile()
    return c


# ---------------------------------------------------------------------------
# 1. the example compiles end-to-end
# ---------------------------------------------------------------------------


class TestExampleCompiles:
    def test_load_compiles_the_example_file(self) -> None:
        prog = load(_EXAMPLE)
        assert prog.morphism is not None

    def test_loads_compiles_the_example_source(self) -> None:
        prog = loads(_EXAMPLE.read_text())
        assert prog.morphism is not None

    def test_exported_arrow_is_an_observed_morphism(self) -> None:
        # The define site runs the contraction eagerly and returns
        # the result as an ObservedMorphism with the contraction's
        # declared typing.
        prog = load(_EXAMPLE)
        morph = prog.morphism
        assert isinstance(morph, ObservedMorphism)
        assert tuple(morph.domain.shape) == (4,)
        assert tuple(morph.codomain.shape) == (3,)


# ---------------------------------------------------------------------------
# 2. the compiled contraction carries the expected runtime wiring
# ---------------------------------------------------------------------------


class TestRuntimeWiring:
    def test_contraction_is_registered(self, compiler: Compiler) -> None:
        assert "bilinear_score" in compiler.contractions

    def test_wiring_spec_is_inferred_from_signature(self, compiler: Compiler) -> None:
        # p : Item -> PredDim, a : Item -> ArgDim,
        # w : (PredDim * ArgDim) -> Judgment, output Item -> Judgment.
        # PredDim and ArgDim appear in two inputs each and not in the
        # output (contracted); Item and Judgment propagate.
        wiring = compiler.contractions["bilinear_score"].wiring
        assert isinstance(wiring, EinsumWiring)
        assert wiring.spec == "ab, ac, bcd -> ad"
        assert wiring.input_specs == ("ab", "ac", "bcd")
        assert wiring.output_spec == "ad"
        assert wiring.input_arity == 3

    def test_input_wires_keep_declared_names_and_shapes(
        self, compiler: Compiler
    ) -> None:
        contraction = compiler.contractions["bilinear_score"]
        names = [name for name, _, _ in contraction.input_types]
        assert names == ["p", "a", "w"]
        shapes = [
            (tuple(dom.shape), tuple(cod.shape))
            for _, dom, cod in contraction.input_types
        ]
        assert shapes == [((4,), (2,)), ((4,), (2,)), ((2, 2), (3,))]

    def test_contraction_uses_the_named_rule(self, compiler: Compiler) -> None:
        contraction = compiler.contractions["bilinear_score"]
        assert contraction.algebra.name == "Real"
        assert contraction.wiring.composition_rule is contraction.algebra

    def test_output_typing_matches_declaration(self, compiler: Compiler) -> None:
        contraction = compiler.contractions["bilinear_score"]
        assert tuple(contraction.domain.shape) == (4,)
        assert tuple(contraction.codomain.shape) == (3,)


# ---------------------------------------------------------------------------
# 3. forward pass
# ---------------------------------------------------------------------------


class TestForwardPass:
    def test_program_forward_matches_einsum_of_latents(self) -> None:
        # The contraction result is materialized from the declared
        # latents at the define site; recompute it independently from
        # the same compiled environment.
        c = Compiler(parse_file(_EXAMPLE))
        program = c.compile()
        out = program()
        assert tuple(out.shape) == (4, 3)
        p = c.morphisms["pred_embed"].tensor
        a = c.morphisms["arg_embed"].tensor
        w = c.morphisms["interaction"].tensor
        expected = torch.einsum("ib,ic,bcd->id", p, a, w)
        assert torch.allclose(out, expected)

    def test_wiring_forward_on_tiny_tensors(self, compiler: Compiler) -> None:
        wiring = compiler.contractions["bilinear_score"].wiring
        p = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        a = 0.5 * torch.arange(8, dtype=torch.float32).reshape(4, 2)
        w = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
        out = wiring.apply(p, a, w)
        assert tuple(out.shape) == (4, 3)
        assert torch.allclose(out, torch.einsum("ib,ic,bcd->id", p, a, w))

    def test_gradients_flow_through_the_wiring(self, compiler: Compiler) -> None:
        wiring = compiler.contractions["bilinear_score"].wiring
        p = torch.rand(4, 2, requires_grad=True)
        a = torch.rand(4, 2)
        w = torch.rand(2, 2, 3)
        wiring.apply(p, a, w).sum().backward()
        assert p.grad is not None
        assert bool((p.grad != 0).any())


# ---------------------------------------------------------------------------
# 4. error surface
# ---------------------------------------------------------------------------


class TestContractionErrors:
    def test_unknown_rule_reports_available_registry(self) -> None:
        src = _EXAMPLE.read_text().replace("[rule=real]", "[rule=nosuchrule]")
        with pytest.raises(CompileError, match="unknown rule 'nosuchrule'") as exc:
            loads(src)
        assert "available:" in str(exc.value)
        assert "real" in str(exc.value)

    def test_unknown_option_key_gets_did_you_mean(self) -> None:
        src = _EXAMPLE.read_text().replace("[rule=real]", "[rule=real, sharing=[Item]]")
        with pytest.raises(CompileError, match="unknown option 'sharing'") as exc:
            loads(src)
        assert "did you mean 'share'" in str(exc.value)
        assert "valid options" in str(exc.value)
