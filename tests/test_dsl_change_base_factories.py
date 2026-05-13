"""DSL surface for parametric morphism transformations.

Exercises the constructor-call form of :class:`MorphismTransformation`
values inside ``change_base``:

* ``f.change_base(softmax(B))`` — Tier 1: constructor with an
  object argument resolved at compile time.
* ``f.change_base(l1_normalize(B))`` — same shape.
* ``f.change_base(l2_normalize(B))`` — same shape, identity
  target quantale.
* ``f.change_base(bayes_invert(prior))`` — Tier 2: constructor
  with a morphism argument; the prior's tensor is read when the
  constructor runs to produce the :class:`BayesInvert`.
* Bare-name lookup paths still resolve singletons.
* Helpful error messages for misuse (bare constructor without
  args, unknown constructor, unknown argument).
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
# Tier 1 — object-argument constructors
# ---------------------------------------------------------------------------


@_LOCAL_GRAMMAR
def test_softmax_compiles() -> None:
    src = """
    quantale product_fuzzy
    object A : 3
    object B : 4
    latent f : A -> B
    let g = f.change_base(softmax(B))
    export g
    """
    program = loads(src)
    assert program.morphism is not None
    # The result tensor should be row-stochastic along B.
    tensor = program.morphism.tensor
    assert tensor.shape == (3, 4)
    assert torch.allclose(tensor.sum(dim=-1), torch.ones(3), atol=1e-5)


@_LOCAL_GRAMMAR
def test_l1_normalize_compiles() -> None:
    src = """
    quantale real
    object A : 3
    object B : 4
    latent f : A -> B
    let g = f.change_base(l1_normalize(B))
    export g
    """
    program = loads(src)
    tensor = program.morphism.tensor
    # All entries non-negative and row-summing to 1.
    assert (tensor >= 0).all()
    assert torch.allclose(tensor.sum(dim=-1), torch.ones(3), atol=1e-5)


@_LOCAL_GRAMMAR
def test_l2_normalize_compiles() -> None:
    src = """
    quantale real
    object A : 3
    object B : 4
    latent f : A -> B
    let g = f.change_base(l2_normalize(B))
    export g
    """
    program = loads(src)
    tensor = program.morphism.tensor
    norms = tensor.pow(2).sum(dim=-1).sqrt()
    assert torch.allclose(norms, torch.ones(3), atol=1e-5)


# ---------------------------------------------------------------------------
# Tier 2 — morphism-argument constructor
# ---------------------------------------------------------------------------


@_LOCAL_GRAMMAR
def test_bayes_invert_with_morphism_prior() -> None:
    src = """
    quantale markov
    object Unit : 1
    object A : 3
    observed prior : Unit -> A = from_data("PRIOR")
    latent f : A -> A
    let g = f.change_base(bayes_invert(prior))
    export g
    """
    prior = torch.tensor([[0.5, 0.3, 0.2]])
    program = loads(src, data={"PRIOR": prior})
    assert program.morphism is not None
    out = program.morphism.tensor
    # BayesInvert returns a Markov kernel (rows sum to 1).
    assert torch.allclose(out.sum(dim=-1), torch.ones(3), atol=1e-5)


# ---------------------------------------------------------------------------
# Bare-name lookups still work
# ---------------------------------------------------------------------------


@_LOCAL_GRAMMAR
def test_bare_name_homomorphism_still_resolves() -> None:
    src = """
    quantale boolean
    object A : 3
    object B : 4
    latent f : A -> B
    let g = f.change_base(boolean_embedding)
    export g
    """
    program = loads(src)
    assert program.morphism is not None


# ---------------------------------------------------------------------------
# Error messages
# ---------------------------------------------------------------------------


@_LOCAL_GRAMMAR
def test_bare_constructor_without_args_errors() -> None:
    src = """
    quantale real
    object A : 3
    object B : 4
    latent f : A -> B
    let g = f.change_base(softmax)
    export g
    """
    with pytest.raises(CompileError, match="needs arguments"):
        loads(src)


@_LOCAL_GRAMMAR
def test_unknown_constructor_errors() -> None:
    src = """
    quantale real
    object A : 3
    object B : 4
    latent f : A -> B
    let g = f.change_base(not_a_real_constructor(B))
    export g
    """
    with pytest.raises(CompileError, match="undefined"):
        loads(src)


@_LOCAL_GRAMMAR
def test_unknown_constructor_arg_errors() -> None:
    src = """
    quantale real
    object A : 3
    object B : 4
    latent f : A -> B
    let g = f.change_base(softmax(NotAnObject))
    export g
    """
    with pytest.raises(CompileError, match="unresolved"):
        loads(src)
