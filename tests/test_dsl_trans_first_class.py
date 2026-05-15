"""First-class transformations in the DSL.

A transformation (a :class:`AlgebraHomomorphism` or
:class:`MorphismTransformation`) is a value in the DSL: it can be
let-bound, composed with ``>>>``, and passed to ``change_base``.
This file exercises:

* Bare-name singleton references (``expectation``, ``log_prob``,
  ``boolean_embedding``, …).
* Constructor calls (``softmax(B)``, ``l1_normalize(B)``,
  ``l2_normalize(B)``, ``bayes_invert(prior)``).
* Let-binding to user-chosen names.
* Composition ``t1 >>> t2`` with source/target compatibility
  checks at compile time.
* Application inside ``change_base``, including chains.
"""

from __future__ import annotations

import os

import pytest
import torch

from quivers.dsl import loads
from quivers.dsl.compiler import CompileError


_LOCAL_GRAMMAR = pytest.mark.skipif(
    os.environ.get("QVR_USE_LOCAL_GRAMMAR", "") not in ("1", "true", "True"),
    reason="needs QVR_USE_LOCAL_GRAMMAR=1 to pick up the in-tree grammar",
)


# ---------------------------------------------------------------------------
# Let-binding a transformation
# ---------------------------------------------------------------------------


@_LOCAL_GRAMMAR
def test_let_binds_singleton_transformation() -> None:
    src = """
    algebra real
    object A : 3
    object B : 4
    latent f : A -> B
    let phi = boolean_embedding
    let g = f.change_base(phi)
    export g
    """
    # phi has source=Boolean, target=ProductFuzzyAlgebra; f is over Real.
    # The change_base will raise at runtime because source doesn't
    # match.  Just confirm the parse + let-binding compiles up to
    # the point of trying to apply it.
    with pytest.raises(CompileError, match="algebra"):
        loads(src)


@_LOCAL_GRAMMAR
def test_let_binds_constructor_result() -> None:
    src = """
    algebra product_fuzzy
    object A : 3
    object B : 4
    latent f : A -> B
    let normalize = softmax(B)
    let g = f.change_base(normalize)
    export g
    """
    program = loads(src)
    tensor = program.morphism.tensor
    assert torch.allclose(tensor.sum(dim=-1), torch.ones(3), atol=1e-5)


@_LOCAL_GRAMMAR
def test_let_binds_then_reuses_transformation() -> None:
    """A single trans can be applied to two different morphisms."""
    src = """
    algebra product_fuzzy
    object A : 3
    object B : 4
    latent f : A -> B
    latent h : A -> B
    let normalize = softmax(B)
    let f_norm = f.change_base(normalize)
    let h_norm = h.change_base(normalize)
    export f_norm
    """
    program = loads(src)
    assert program.morphism is not None


@_LOCAL_GRAMMAR
def test_let_trans_name_disjoint_from_morphism_name() -> None:
    """Trans and morphism namespaces are disjoint."""
    src = """
    algebra product_fuzzy
    object A : 3
    object B : 4
    latent f : A -> B
    let t = softmax(B)
    let t = f
    export f
    """
    with pytest.raises(CompileError, match="already bound"):
        loads(src)


# ---------------------------------------------------------------------------
# Composition ``>>>``
# ---------------------------------------------------------------------------


@_LOCAL_GRAMMAR
def test_trans_compose_chains_two_compatible_steps() -> None:
    """``t1 >>> t2`` applies t1 then t2; the morphism's algebra
    travels from t1.source through t1.target == t2.source to
    t2.target."""
    src = """
    algebra product_fuzzy
    object A : 3
    object B : 4
    latent f : A -> B
    let s = softmax(B)
    let pipe = s >>> expectation
    let g = f.change_base(pipe)
    export g
    """
    # softmax : ProductFuzzyAlgebra -> Markov ; expectation : Markov ->
    # ProductFuzzyAlgebra.  The chain takes f from ProductFuzzyAlgebra through
    # Markov and back to ProductFuzzyAlgebra.
    program = loads(src)
    assert program.morphism is not None
    tensor = program.morphism.tensor
    assert tensor.shape == (3, 4)


@_LOCAL_GRAMMAR
def test_trans_compose_inline_in_change_base() -> None:
    """``f.change_base(t1 >>> t2)`` works without an intermediate
    let."""
    src = """
    algebra product_fuzzy
    object A : 3
    object B : 4
    latent f : A -> B
    let g = f.change_base(softmax(B) >>> expectation)
    export g
    """
    program = loads(src)
    assert program.morphism is not None


@_LOCAL_GRAMMAR
def test_trans_compose_three_steps_flattens() -> None:
    """``t1 >>> t2 >>> t3`` flattens into a single 3-step
    sequence; intermediate boundaries are typed."""
    src = """
    algebra real
    object A : 3
    object B : 4
    latent f : A -> B
    let pipe = probability_clamp >>> probability_to_real >>> counting_from_real
    let g = f.change_base(pipe)
    export g
    """
    # Real -> Probability -> Real -> Counting
    program = loads(src)
    assert program.morphism is not None


@_LOCAL_GRAMMAR
def test_trans_compose_source_target_mismatch_errors() -> None:
    """``boolean_embedding >>> expectation`` is a type error:
    boolean_embedding's target is ProductFuzzyAlgebra, but expectation
    expects Markov."""
    src = """
    algebra boolean
    object A : 3
    object B : 4
    latent f : A -> B
    let bad = boolean_embedding >>> expectation
    let g = f.change_base(bad)
    export g
    """
    with pytest.raises(CompileError, match="does not match"):
        loads(src)


# ---------------------------------------------------------------------------
# Constructor-call resolution
# ---------------------------------------------------------------------------


@_LOCAL_GRAMMAR
def test_softmax_constructor_accepts_object() -> None:
    src = """
    algebra product_fuzzy
    object A : 3
    object B : 4
    latent f : A -> B
    let g = f.change_base(softmax(B))
    export g
    """
    program = loads(src)
    assert program.morphism is not None


@_LOCAL_GRAMMAR
def test_constructor_args_resolve_against_object_scope() -> None:
    src = """
    algebra real
    object A : 3
    object B : 4
    object NotUsed : 2
    latent f : A -> B
    let g = f.change_base(l1_normalize(B))
    export g
    """
    program = loads(src)
    tensor = program.morphism.tensor
    assert (tensor >= 0).all()


# ---------------------------------------------------------------------------
# Singleton-vs-constructor resolution
# ---------------------------------------------------------------------------


@_LOCAL_GRAMMAR
def test_bare_singleton_resolves() -> None:
    src = """
    algebra boolean
    object A : 3
    object B : 4
    latent f : A -> B
    let g = f.change_base(boolean_embedding)
    export g
    """
    program = loads(src)
    assert program.morphism is not None


@_LOCAL_GRAMMAR
def test_constructor_referenced_bare_errors_helpfully() -> None:
    """Using a constructor's name without parentheses surfaces a
    pointed error."""
    src = """
    algebra real
    object A : 3
    object B : 4
    latent f : A -> B
    let g = f.change_base(softmax)
    export g
    """
    with pytest.raises(CompileError, match="needs arguments"):
        loads(src)


@_LOCAL_GRAMMAR
def test_unknown_name_in_change_base_errors() -> None:
    src = """
    algebra real
    object A : 3
    object B : 4
    latent f : A -> B
    let g = f.change_base(no_such_thing)
    export g
    """
    with pytest.raises(CompileError, match="undefined transformation"):
        loads(src)


@_LOCAL_GRAMMAR
def test_unknown_constructor_errors() -> None:
    src = """
    algebra real
    object A : 3
    object B : 4
    latent f : A -> B
    let g = f.change_base(no_such_constructor(B))
    export g
    """
    with pytest.raises(CompileError, match="undefined"):
        loads(src)
