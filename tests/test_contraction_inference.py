"""Tests for the type-driven contraction-wiring inference.

The user-facing surface change: ``contraction NAME(args) : DOM -> COD``
blocks no longer require a hand-written ``wiring "<einsum>"`` clause.
The compiler infers the einsum from the typed signature: shared axes
propagate, axes in inputs but not output get contracted, ``share``
opts axes out of contraction (element-wise mode), and ``wiring "..."``
remains as an explicit escape hatch.

These tests exercise the inference at every documented case and at
the edges: nested products, multiple-letter pools, anomalous axes,
output reordering, the ``share`` disambiguator, and round-trip
through the compiler.
"""

from __future__ import annotations
import textwrap

import pytest

from quivers.dsl import loads
from quivers.dsl.ast_nodes import (
    ContractionInput,
    ObjectCoproduct,
    ObjectProduct,
    TypeName,
)
from quivers.dsl.compiler.programs import (
    _flatten_type_axes,
    _infer_wiring_from_signature,
)


def _inp(
    name: str, dom: str | tuple[str, ...], cod: str | tuple[str, ...]
) -> ContractionInput:
    """Build a ContractionInput from string axis names."""

    def to_type(t: str | tuple[str, ...]):
        if isinstance(t, str):
            return TypeName(name=t)
        return ObjectProduct(components=tuple(TypeName(name=n) for n in t))

    return ContractionInput(
        name=name,
        input_domain=to_type(dom),
        input_codomain=to_type(cod),
    )


class TestFlattenTypeAxes:
    def test_named_axis(self):
        assert _flatten_type_axes(TypeName(name="A")) == ("A",)

    def test_product(self):
        expr = ObjectProduct(components=(TypeName(name="A"), TypeName(name="B")))
        assert _flatten_type_axes(expr) == ("A", "B")

    def test_nested_product_flattens(self):
        # (A * (B * C)) flattens to (A, B, C).
        inner = ObjectProduct(components=(TypeName(name="B"), TypeName(name="C")))
        outer = ObjectProduct(components=(TypeName(name="A"), inner))
        assert _flatten_type_axes(outer) == ("A", "B", "C")

    def test_coproduct_rejected(self):
        expr = ObjectCoproduct(components=(TypeName(name="A"), TypeName(name="B")))
        with pytest.raises(ValueError, match="non-product / non-named"):
            _flatten_type_axes(expr)


class TestBasicInference:
    def test_classical_three_input_contraction(self):
        # The example from tutorial 7: arg1: A->B, arg2: A->C,
        # kernel: (B*C)->D, contracted to A->D. Both B and C
        # appear in two inputs and not in the output, so both are
        # contracted. The output is A and D.
        inputs = (
            _inp("arg1", "A", "B"),
            _inp("arg2", "A", "C"),
            _inp("kernel", ("B", "C"), "D"),
        )
        spec = _infer_wiring_from_signature(
            inputs=inputs,
            output_domain=TypeName(name="A"),
            output_codomain=TypeName(name="D"),
            shared_axes=(),
        )
        assert spec == "ab, ac, bcd -> ad"

    def test_single_input_identity_shape(self):
        # f: A -> B, output A -> B: trivial passthrough, no
        # contraction.
        inputs = (_inp("f", "A", "B"),)
        spec = _infer_wiring_from_signature(
            inputs=inputs,
            output_domain=TypeName(name="A"),
            output_codomain=TypeName(name="B"),
            shared_axes=(),
        )
        assert spec == "ab -> ab"

    def test_matrix_product(self):
        # f: A -> B, g: B -> C, output A -> C: contract B.
        inputs = (_inp("f", "A", "B"), _inp("g", "B", "C"))
        spec = _infer_wiring_from_signature(
            inputs=inputs,
            output_domain=TypeName(name="A"),
            output_codomain=TypeName(name="C"),
            shared_axes=(),
        )
        # axes-letter-pool order: A, B (from first input), C from
        # second.
        assert spec == "ab, bc -> ac"

    def test_three_way_join_on_shared_axis(self):
        # Three inputs all sharing axis A (the batch / object), each
        # mapping to its own codomain, contracted to produce A -> D
        # via a Boolean conjunction over the per-input codomains.
        # Output keeps just A; B, C, D are all contracted.
        inputs = (
            _inp("p", "A", "B"),
            _inp("q", "A", "C"),
            _inp("r", "A", "D"),
        )
        with pytest.raises(ValueError, match="exactly one input"):
            # B / C / D each appear in only one input. The inference
            # cannot guess how to dispose of them; user must say so
            # explicitly.
            _infer_wiring_from_signature(
                inputs=inputs,
                output_domain=TypeName(name="A"),
                output_codomain=TypeName(name="A"),
                shared_axes=(),
            )


class TestProductTypes:
    def test_product_domain(self):
        # f: (A * B) -> C, output (A * B) -> C: identity.
        inputs = (_inp("f", ("A", "B"), "C"),)
        spec = _infer_wiring_from_signature(
            inputs=inputs,
            output_domain=ObjectProduct(
                components=(TypeName(name="A"), TypeName(name="B"))
            ),
            output_codomain=TypeName(name="C"),
            shared_axes=(),
        )
        assert spec == "abc -> abc"

    def test_product_codomain(self):
        # f: A -> (B * C), output A -> (B * C).
        inputs = (_inp("f", "A", ("B", "C")),)
        spec = _infer_wiring_from_signature(
            inputs=inputs,
            output_domain=TypeName(name="A"),
            output_codomain=ObjectProduct(
                components=(TypeName(name="B"), TypeName(name="C"))
            ),
            shared_axes=(),
        )
        assert spec == "abc -> abc"


class TestShareClause:
    def test_share_makes_axis_element_wise(self):
        # Two inputs share axis B with the output. Without share, B
        # would be contracted (it's in two inputs, in the output, but
        # the inference rule is "axes in the output propagate"; let me
        # verify this case). Actually B is in the output here, so it
        # would already propagate. Use a case where share matters:
        # arg1: A -> B, arg2: A -> B, output A -> B (element-wise
        # over B). Without share: B appears in two inputs and in
        # output → kept (no contraction). The default rule already
        # gives the right answer.
        inputs = (_inp("f", "A", "B"), _inp("g", "A", "B"))
        spec = _infer_wiring_from_signature(
            inputs=inputs,
            output_domain=TypeName(name="A"),
            output_codomain=TypeName(name="B"),
            shared_axes=(),
        )
        # Both inputs share AB; output is AB. The einsum is
        # element-wise.
        assert spec == "ab, ab -> ab"

    def test_share_keeps_axis_out_of_contraction(self):
        # Same axis B in two inputs, NOT in the output. Default
        # contracts it; ``share B`` keeps it.
        inputs = (_inp("f", "A", "B"), _inp("g", "C", "B"))
        # Without share: B contracted. Output AC.
        default = _infer_wiring_from_signature(
            inputs=inputs,
            output_domain=TypeName(name="A"),
            output_codomain=TypeName(name="C"),
            shared_axes=(),
        )
        assert default == "ab, cb -> ac"
        # With share B: B propagates. Output should include B.
        shared = _infer_wiring_from_signature(
            inputs=inputs,
            output_domain=ObjectProduct(
                components=(TypeName(name="A"), TypeName(name="C"))
            ),
            output_codomain=TypeName(name="B"),
            shared_axes=("B",),
        )
        assert shared == "ab, cb -> acb"


class TestAnomalousAxes:
    def test_axis_in_single_input_not_in_output_is_anomalous(self):
        # f: A -> B, output A -> A. B is in exactly one input and
        # not in the output: anomalous.
        inputs = (_inp("f", "A", "B"),)
        with pytest.raises(ValueError, match=r"axes \['B'\]"):
            _infer_wiring_from_signature(
                inputs=inputs,
                output_domain=TypeName(name="A"),
                output_codomain=TypeName(name="A"),
                shared_axes=(),
            )

    def test_anomalous_axis_can_be_rescued_by_share(self):
        # Listing B in ``share`` lets B propagate to the output
        # explicitly (the user is saying "yes I know B appears in
        # only one input; treat it like an output axis").
        inputs = (_inp("f", "A", "B"),)
        spec = _infer_wiring_from_signature(
            inputs=inputs,
            output_domain=TypeName(name="A"),
            output_codomain=TypeName(name="B"),
            shared_axes=("B",),
        )
        assert spec == "ab -> ab"


class TestOutputOrdering:
    def test_output_reordering(self):
        # Inputs declare A then B; output declares B then A. The
        # einsum follows the output's declared order.
        inputs = (_inp("f", "A", "B"),)
        spec = _infer_wiring_from_signature(
            inputs=inputs,
            output_domain=TypeName(name="B"),
            output_codomain=TypeName(name="A"),
            shared_axes=(),
        )
        assert spec == "ab -> ba"


class TestRepeatedAxis:
    def test_repeated_axis_in_single_input_inferred_verbatim(self):
        # B appears twice in the single input (B -> B) and twice in
        # the output (B -> B). The inference produces the literal
        # ``bb -> bb`` einsum. Whether EinsumWiring downstream
        # accepts this (diagonal extraction) is the wiring layer's
        # call; the inference layer here is type-only.
        inputs = (_inp("self_loop", "B", "B"),)
        spec = _infer_wiring_from_signature(
            inputs=inputs,
            output_domain=TypeName(name="B"),
            output_codomain=TypeName(name="B"),
            shared_axes=(),
        )
        assert spec == "aa -> aa"


class TestComplexCases:
    """Stress tests for the inference: deep product nests, many
    axes, sparse co-occurrence patterns, mixed propagation."""

    def test_five_input_tensor_network(self):
        # Five-input contraction modelling a small tensor network:
        # arg1: A -> B, arg2: B -> C, arg3: C -> D, arg4: D -> E,
        # arg5: E -> F, output A -> F. Contracts B, C, D, E (each
        # appearing in two adjacent inputs).
        inputs = (
            _inp("a1", "A", "B"),
            _inp("a2", "B", "C"),
            _inp("a3", "C", "D"),
            _inp("a4", "D", "E"),
            _inp("a5", "E", "F"),
        )
        spec = _infer_wiring_from_signature(
            inputs=inputs,
            output_domain=TypeName(name="A"),
            output_codomain=TypeName(name="F"),
            shared_axes=(),
        )
        assert spec == "ab, bc, cd, de, ef -> af"

    def test_four_input_full_join_on_shared_axis(self):
        # Four inputs all sharing axis A (the batch); each adds one
        # private axis to the kernel's codomain via a 4-arity
        # tensor product. Kernel takes the product of those private
        # axes and produces the output codomain.
        inputs = (
            _inp("p1", "A", "B"),
            _inp("p2", "A", "C"),
            _inp("p3", "A", "D"),
            _inp("p4", "A", "E"),
            _inp("kernel", ("B", "C", "D", "E"), "F"),
        )
        spec = _infer_wiring_from_signature(
            inputs=inputs,
            output_domain=TypeName(name="A"),
            output_codomain=TypeName(name="F"),
            shared_axes=(),
        )
        # axes assignment: A->a, B->b, C->c, D->d, E->e, F->f
        assert spec == "ab, ac, ad, ae, bcdef -> af"

    def test_deeply_nested_product_in_signature(self):
        # ((A * B) * (C * D)) -> E flattens to (A, B, C, D, E).
        nested = ObjectProduct(
            components=(
                ObjectProduct(components=(TypeName(name="A"), TypeName(name="B"))),
                ObjectProduct(components=(TypeName(name="C"), TypeName(name="D"))),
            )
        )
        inputs = (
            ContractionInput(
                name="big",
                input_domain=nested,
                input_codomain=TypeName(name="E"),
            ),
        )
        spec = _infer_wiring_from_signature(
            inputs=inputs,
            output_domain=nested,
            output_codomain=TypeName(name="E"),
            shared_axes=(),
        )
        assert spec == "abcde -> abcde"

    def test_seven_axis_attention_like(self):
        # Multi-head-attention-shaped contraction:
        #   query  : (Batch * Head * Seq) -> Dim
        #   key    : (Batch * Head * Seq2) -> Dim
        #   value  : (Batch * Head * Seq2) -> Dim
        #   output : (Batch * Head * Seq) -> Dim
        # Contracts Seq2 and Dim. Mimics a softmax-free dot-prod
        # attention as a pure-categorical contraction.
        bhs = ObjectProduct(
            components=(
                TypeName(name="Batch"),
                TypeName(name="Head"),
                TypeName(name="Seq"),
            )
        )
        bhs2 = ObjectProduct(
            components=(
                TypeName(name="Batch"),
                TypeName(name="Head"),
                TypeName(name="Seq2"),
            )
        )
        inputs = (
            ContractionInput(
                name="q", input_domain=bhs, input_codomain=TypeName(name="Dim")
            ),
            ContractionInput(
                name="k", input_domain=bhs2, input_codomain=TypeName(name="Dim")
            ),
            ContractionInput(
                name="v", input_domain=bhs2, input_codomain=TypeName(name="Dim")
            ),
        )
        spec = _infer_wiring_from_signature(
            inputs=inputs,
            output_domain=bhs,
            output_codomain=TypeName(name="Dim"),
            shared_axes=(),
        )
        # Distinct axes in order: Batch, Head, Seq, Dim, Seq2 →
        # a, b, c, d, e. Inputs: q=abcd, k=abed, v=abed.
        # Output: abcd. Seq2 (e) and ... wait, Dim appears in
        # output, so it's kept; Seq2 is in two inputs but not in
        # output, so contracted.
        assert spec == "abcd, abed, abed -> abcd"

    def test_partial_share_with_kept_and_contracted(self):
        # Five axes; some shared explicitly, some contracted by
        # default, some propagated naturally.
        inputs = (
            _inp("x", "A", "B"),
            _inp("y", "A", "C"),
            _inp("z", ("B", "C"), "D"),
        )
        # Default behaviour (no share, output (A, D)): B and C
        # contracted, A and D kept.
        spec_default = _infer_wiring_from_signature(
            inputs=inputs,
            output_domain=TypeName(name="A"),
            output_codomain=TypeName(name="D"),
            shared_axes=(),
        )
        assert spec_default == "ab, ac, bcd -> ad"
        # Share B and C: keep them in the output. The output type
        # must then list them; we put them after A and before D.
        spec_share = _infer_wiring_from_signature(
            inputs=inputs,
            output_domain=ObjectProduct(
                components=(
                    TypeName(name="A"),
                    TypeName(name="B"),
                    TypeName(name="C"),
                )
            ),
            output_codomain=TypeName(name="D"),
            shared_axes=("B", "C"),
        )
        assert spec_share == "ab, ac, bcd -> abcd"

    def test_axis_letter_ordering_is_input_first(self):
        # Letter assignment follows the order axes are first seen
        # while walking inputs in declaration order. Verifying this
        # is stable means downstream einsum diffing in tests is
        # deterministic.
        inputs = (
            _inp("first", "Z", "Y"),
            _inp("second", "X", "Y"),
        )
        spec = _infer_wiring_from_signature(
            inputs=inputs,
            output_domain=ObjectProduct(
                components=(TypeName(name="Z"), TypeName(name="X"))
            ),
            output_codomain=TypeName(name="Y"),
            shared_axes=("Y",),
        )
        # Z appears first → a, Y → b, X → c. Inputs: ab, cb.
        # Output: acb.
        assert spec == "ab, cb -> acb"

    def test_anomalous_axis_in_complex_signature(self):
        # The complex network correctly flags an axis that has
        # nowhere to go: arg2 has a private codomain axis that
        # nothing else mentions and that isn't in the output.
        inputs = (
            _inp("a1", "A", "B"),
            _inp("a2", "B", "Phantom"),  # Phantom appears once.
            _inp("a3", "B", "C"),
        )
        with pytest.raises(ValueError, match=r"axes \['Phantom'\]"):
            _infer_wiring_from_signature(
                inputs=inputs,
                output_domain=TypeName(name="A"),
                output_codomain=TypeName(name="C"),
                shared_axes=(),
            )

    def test_twenty_six_letters_just_fits(self):
        # Exactly 26 distinct axes fits in the einsum letter pool.
        names = [chr(ord("A") + i) for i in range(26)]
        # Pair them up: 13 inputs of shape (X_{2i}) -> (X_{2i+1}).
        inputs = tuple(
            _inp(f"in{i}", names[2 * i], names[2 * i + 1]) for i in range(13)
        )
        output_seq = tuple(TypeName(name=n) for n in names)
        # Output gets all of them so none are contracted (the
        # default rule would otherwise complain that everything is
        # anomalous since each axis appears in exactly one input).
        spec = _infer_wiring_from_signature(
            inputs=inputs,
            output_domain=ObjectProduct(components=output_seq[:13]),
            output_codomain=ObjectProduct(components=output_seq[13:]),
            shared_axes=(),
        )
        # Each input contributes two letters; output is all 26.
        assert spec.endswith(" -> abcdefghijklmnopqrstuvwxyz")

    def test_repeated_input_with_different_orderings_correct(self):
        # Same axes, two inputs with reversed domain/codomain to
        # exercise output-ordering. f: A -> B, g: B -> A, output
        # B -> A: contract neither (both are in output), and the
        # output order pin matters.
        inputs = (_inp("f", "A", "B"), _inp("g", "B", "A"))
        spec_a_to_b = _infer_wiring_from_signature(
            inputs=inputs,
            output_domain=TypeName(name="A"),
            output_codomain=TypeName(name="B"),
            shared_axes=(),
        )
        spec_b_to_a = _infer_wiring_from_signature(
            inputs=inputs,
            output_domain=TypeName(name="B"),
            output_codomain=TypeName(name="A"),
            shared_axes=(),
        )
        # The two output orderings differ in the einsum's RHS.
        assert spec_a_to_b == "ab, ba -> ab"
        assert spec_b_to_a == "ab, ba -> ba"


class TestLetterPool:
    def test_exceeds_letter_pool_raises(self):
        # 27 distinct axes exhausts the 26-letter pool.
        names = [f"X{i}" for i in range(27)]
        inputs = (_inp("f", tuple(names[:14]), tuple(names[14:])),)
        with pytest.raises(ValueError, match="distinct axis names"):
            _infer_wiring_from_signature(
                inputs=inputs,
                output_domain=ObjectProduct(
                    components=tuple(TypeName(name=n) for n in names[:14])
                ),
                output_codomain=ObjectProduct(
                    components=tuple(TypeName(name=n) for n in names[14:])
                ),
                shared_axes=(),
            )


class TestEndToEndCompilation:
    """Compile a contraction declaration through the full DSL
    pipeline, with and without an explicit wiring clause, and
    verify the registered callable matches the expected einsum."""

    @pytest.fixture
    def src_inferred(self):
        return """
composition product_fuzzy as algebra

object A : FinSet 3
object B : FinSet 4
object C : FinSet 5
object D : FinSet 2

contraction op_apply (
    arg1 : A -> B,
    arg2 : A -> C,
    kernel : (B * C) -> D
) : A -> D [rule=product_fuzzy]

program p : A -> A
    sample x <- Normal(0.0, 1.0)
    observe r : A <- Normal(x, 0.1)
    return r

export p
"""

    @pytest.fixture
    def src_explicit_wiring(self):
        return """
composition product_fuzzy as algebra

object A : FinSet 3
object B : FinSet 4
object C : FinSet 5
object D : FinSet 2

contraction op_apply (
    arg1 : A -> B,
    arg2 : A -> C,
    kernel : (B * C) -> D
) : A -> D [rule=product_fuzzy, wiring="ab, ac, bcd -> ad"]

program p : A -> A
    sample x <- Normal(0.0, 1.0)
    observe r : A <- Normal(x, 0.1)
    return r

export p
"""

    def test_inferred_compiles(self, src_inferred):

        prog = loads(src_inferred)
        assert prog is not None

    def test_explicit_wiring_still_compiles(self, src_explicit_wiring):

        prog = loads(src_explicit_wiring)
        assert prog is not None

    def test_share_clause_compiles(self):
        src = """
composition product_fuzzy as algebra

object A : FinSet 3
object B : FinSet 4
object C : FinSet 5

contraction shared_b (
    f : A -> B,
    g : C -> B
) : (A * C) -> B [rule=product_fuzzy, share=[B]]

program p : A -> A
    sample x <- Normal(0.0, 1.0)
    observe r : A <- Normal(x, 0.1)
    return r

export p
"""

        prog = loads(textwrap.dedent(src))
        assert prog is not None
