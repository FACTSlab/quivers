"""DSL surface for parametric morphism transformations.

Exercises the factory-call form of ``change_base``:

* ``f.change_base(softmax_over(B))`` — Tier 1: factory with an
  object argument resolved at compile time.
* ``f.change_base(l1_normalize_over(B))`` — same shape.
* ``f.change_base(l2_normalize_over(B))`` — same shape, identity
  target quantale.
* ``f.change_base(bayes_invert(prior))`` — Tier 2: factory with
  a morphism argument; the prior's tensor is read when the
  factory runs to produce the :class:`BayesInvert`.
* Bare-name lookup paths still resolve singletons.
* Helpful error messages for misuse (bare factory without args,
  unknown factory, unknown argument).
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
# Tier 1 — object-argument factories
# ---------------------------------------------------------------------------


@_LOCAL_GRAMMAR
def test_softmax_over_compiles() -> None:
    src = """
    quantale product_fuzzy
    object A : 3
    object B : 4
    latent f : A -> B
    let g = f.change_base(softmax_over(B))
    export g
    """
    program = loads(src)
    assert program.morphism is not None
    # The result tensor should be row-stochastic along B.
    tensor = program.morphism.tensor
    assert tensor.shape == (3, 4)
    assert torch.allclose(tensor.sum(dim=-1), torch.ones(3), atol=1e-5)


@_LOCAL_GRAMMAR
def test_l1_normalize_over_compiles() -> None:
    src = """
    quantale real
    object A : 3
    object B : 4
    latent f : A -> B
    let g = f.change_base(l1_normalize_over(B))
    export g
    """
    program = loads(src)
    tensor = program.morphism.tensor
    # All entries non-negative and row-summing to 1.
    assert (tensor >= 0).all()
    assert torch.allclose(tensor.sum(dim=-1), torch.ones(3), atol=1e-5)


@_LOCAL_GRAMMAR
def test_l2_normalize_over_compiles() -> None:
    src = """
    quantale real
    object A : 3
    object B : 4
    latent f : A -> B
    let g = f.change_base(l2_normalize_over(B))
    export g
    """
    program = loads(src)
    tensor = program.morphism.tensor
    norms = tensor.pow(2).sum(dim=-1).sqrt()
    assert torch.allclose(norms, torch.ones(3), atol=1e-5)


# ---------------------------------------------------------------------------
# Tier 2 — morphism-argument factory
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
    observed f : A -> B = identity_like(A, B)
    let g = f.change_base(boolean_embedding)
    export g
    """
    # ``identity_like`` doesn't exist; this is structural only —
    # we just want the parse + change_base resolution path to work.
    # Use a latent f instead.
    src2 = """
    quantale boolean
    object A : 3
    object B : 4
    latent f : A -> B
    let g = f.change_base(boolean_embedding)
    export g
    """
    program = loads(src2)
    assert program.morphism is not None


# ---------------------------------------------------------------------------
# Error messages
# ---------------------------------------------------------------------------


@_LOCAL_GRAMMAR
def test_bare_factory_without_args_errors() -> None:
    src = """
    quantale real
    object A : 3
    object B : 4
    latent f : A -> B
    let g = f.change_base(softmax_over)
    export g
    """
    with pytest.raises(CompileError, match="factory"):
        loads(src)


@_LOCAL_GRAMMAR
def test_unknown_factory_errors() -> None:
    src = """
    quantale real
    object A : 3
    object B : 4
    latent f : A -> B
    let g = f.change_base(not_a_real_factory(B))
    export g
    """
    with pytest.raises(CompileError, match="undefined"):
        loads(src)


@_LOCAL_GRAMMAR
def test_unknown_factory_arg_errors() -> None:
    src = """
    quantale real
    object A : 3
    object B : 4
    latent f : A -> B
    let g = f.change_base(softmax_over(NotAnObject))
    export g
    """
    with pytest.raises(CompileError, match="unresolved"):
        loads(src)
