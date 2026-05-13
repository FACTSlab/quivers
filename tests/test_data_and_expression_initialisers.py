"""Data-derived, expression-derived, and hyperparameter-dependent
initializers for ``observed`` / ``latent`` morphism declarations.

Three kinds of initializers are tested end-to-end:

* ``observed f : A -> B = from_data("KEY")`` — runtime-supplied
  frozen tensor. The compiler resolves the string key against the
  ``data=`` dictionary passed to :func:`quivers.dsl.loads` and
  binds the resulting tensor as the morphism's data buffer.
* ``observed f : A -> B = (g >> h).freeze`` — expression-derived
  initializer whose constituent morphisms' parameters are pinned;
  gradient flow stops at the freeze.
* ``latent f : A -> B [scale=0.1]`` — hyperparameter-dependent
  initialisation already supported via the option block; tested
  for regression.
"""

from __future__ import annotations

import os

import pytest
import torch

from quivers.dsl import loads


_LOCAL_GRAMMAR = pytest.mark.skipif(
    os.environ.get("QVR_USE_LOCAL_GRAMMAR", "") not in ("1", "true", "True"),
    reason="needs QVR_USE_LOCAL_GRAMMAR=1 to pick up the in-tree grammar",
)


# ---------------------------------------------------------------------------
# from_data initializer
# ---------------------------------------------------------------------------


@_LOCAL_GRAMMAR
def test_from_data_binds_supplied_tensor() -> None:
    """``observed f : A -> B = from_data("KEY")`` binds the
    supplied tensor as the morphism's data buffer."""
    src = """
    quantale product_fuzzy
    object A : 3
    object B : 4

    observed h : A -> B = from_data("H")
    export h
    """
    data_tensor = torch.tensor(
        [[0.1, 0.2, 0.3, 0.4],
         [0.5, 0.6, 0.7, 0.8],
         [0.9, 0.1, 0.2, 0.3]]
    )
    m = loads(src, data={"H": data_tensor})
    assert m.morphism.tensor.shape == (3, 4)
    assert torch.allclose(m.morphism.tensor, data_tensor)
    assert m.morphism.domain.name == "A"
    assert m.morphism.codomain.name == "B"


@_LOCAL_GRAMMAR
def test_from_data_unknown_key_errors() -> None:
    from quivers.dsl.compiler import CompileError

    src = """
    quantale product_fuzzy
    object A : 2
    object B : 2
    observed h : A -> B = from_data("MISSING_KEY")
    export h
    """
    with pytest.raises(CompileError, match="unknown data key"):
        loads(src, data={"OTHER_KEY": torch.zeros(2, 2)})


@_LOCAL_GRAMMAR
def test_from_data_without_data_dict_errors() -> None:
    from quivers.dsl.compiler import CompileError

    src = """
    quantale product_fuzzy
    object A : 2
    object B : 2
    observed h : A -> B = from_data("KEY")
    export h
    """
    with pytest.raises(CompileError, match="unknown data key"):
        loads(src)


@_LOCAL_GRAMMAR
def test_from_data_shape_mismatch_with_declared_types_errors() -> None:
    """A data tensor whose shape doesn't match the declared
    domain/codomain shapes must be rejected with a clear error."""
    from quivers.dsl.compiler import CompileError

    src = """
    quantale product_fuzzy
    object A : 3
    object B : 4
    observed h : A -> B = from_data("H")
    export h
    """
    # Wrong shape: (2, 4) instead of (3, 4).
    with pytest.raises(CompileError):
        loads(src, data={"H": torch.zeros(2, 4)})


@_LOCAL_GRAMMAR
def test_from_data_does_not_register_parameters() -> None:
    """A ``from_data``-initialized morphism is structural / frozen;
    its tensor is a buffer, not an ``nn.Parameter``."""
    src = """
    quantale product_fuzzy
    object A : 2
    object B : 2
    observed h : A -> B = from_data("H")
    export h
    """
    m = loads(src, data={"H": torch.tensor([[0.1, 0.2], [0.3, 0.4]])})
    params = list(m.parameters())
    assert len(params) == 0


# ---------------------------------------------------------------------------
# .freeze postfix
# ---------------------------------------------------------------------------


@_LOCAL_GRAMMAR
def test_freeze_materialises_composition_tensor() -> None:
    """``(f >> g).freeze`` produces an :class:`ObservedMorphism`
    whose tensor equals the composition's materialised tensor."""
    src = """
    quantale product_fuzzy
    object A : 2
    object B : 2
    object C : 2

    latent f : A -> B
    latent g : B -> C

    let chain = f >> g
    let frozen = chain.freeze
    export frozen
    """
    m = loads(src)
    # Frozen morphism has no learnable parameters.
    params = list(m.parameters())
    assert len(params) == 0
    # Tensor shape matches the composition.
    assert m.morphism.tensor.shape == (2, 2)


@_LOCAL_GRAMMAR
def test_freeze_detaches_gradients() -> None:
    """Gradients do not propagate through a freeze boundary."""
    src = """
    quantale product_fuzzy
    object A : 2
    object B : 2
    latent f : A -> B
    let frozen = f.freeze
    export frozen
    """
    m = loads(src)
    loss = m.morphism.tensor.sum()
    # ``frozen`` carries a detached tensor; ``loss.backward()``
    # has no grad-requiring leaves, so the call raises rather
    # than mutating anyone's .grad.
    with pytest.raises(RuntimeError, match="does not require grad"):
        loss.backward()


# ---------------------------------------------------------------------------
# Expression-derived initializer (no freeze): parameters propagate
# ---------------------------------------------------------------------------


@_LOCAL_GRAMMAR
def test_expression_initializer_propagates_parameters() -> None:
    """``observed h : A -> B = f`` (or any other expression) is an
    alias bound through the compiler; the underlying morphism's
    parameters are reachable via the alias. Without the
    ``.freeze`` modifier the binding does NOT detach."""
    src = """
    quantale product_fuzzy
    object A : 3
    object B : 3
    latent f : A -> B
    observed h : A -> B = f
    export h
    """
    m = loads(src)
    # ``h`` is bound to ``f``'s underlying LatentMorphism; its
    # tensor has gradient lineage.
    assert m.morphism.tensor.requires_grad


# ---------------------------------------------------------------------------
# Hyperparameter-dependent initialisation (option block)
# ---------------------------------------------------------------------------


@_LOCAL_GRAMMAR
def test_latent_morphism_accepts_scale_option() -> None:
    """The ``[scale=...]`` option block already supports
    hyperparameter-dependent init for ``latent`` morphisms; this
    is a regression test for the surface."""
    src = """
    quantale product_fuzzy
    object A : 5
    object B : 5
    latent f : A -> B [scale=0.01]
    export f
    """
    m = loads(src)
    # With a tiny scale the raw parameter starts near zero, so
    # sigmoid(raw) starts near 0.5.
    t = m.morphism.tensor
    assert t.shape == (5, 5)
    # All entries should be in [0, 1] (sigmoid output).
    assert torch.all((t >= 0) & (t <= 1))
    # Mean should be close to 0.5 because scale is tiny.
    assert abs(float(t.mean()) - 0.5) < 0.1


# ---------------------------------------------------------------------------
# Combination: from_data with a downstream learnable morphism
# ---------------------------------------------------------------------------


@_LOCAL_GRAMMAR
def test_from_data_composed_with_latent_yields_learnable_chain() -> None:
    """A frozen data-derived morphism composed with a learnable
    one yields a chain whose only learnable parameters come from
    the learnable side."""
    src = """
    quantale product_fuzzy
    object A : 3
    object B : 4
    object C : 5

    observed h : A -> B = from_data("H")
    latent g : B -> C
    let chain = h >> g
    export chain
    """
    H = torch.rand(3, 4)
    m = loads(src, data={"H": H})
    # The chain's parameters come exclusively from ``g`` (since
    # ``h`` is a frozen data-derived morphism).
    params = list(m.parameters())
    assert len(params) == 1
