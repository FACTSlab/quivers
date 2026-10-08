"""Runtime checks of the functor and monad laws.

Each function takes a typeclass instance and a small set of
representative objects or morphisms, then asserts that the relevant
laws hold to within numerical tolerance by comparing the tensors of
the two sides of each equation.

The same laws are recorded as equations in the theories of
[`quivers.monadic.theories`][quivers.monadic.theories], so a law is
stated once as a Python predicate and once as a theory equation.

References
----------
- Hughes, J. (2000). *Generalising monads to arrows*. Science of
  Computer Programming, 37(1-3), 67-111.
  [doi:10.1016/S0167-6423(99)00023-4](https://doi.org/10.1016/S0167-6423(99)00023-4)
"""

from __future__ import annotations

import torch

from quivers.core.morphisms import Morphism
from quivers.core.morphisms import identity as id_
from quivers.core.objects import SetObject
from quivers.monadic.typeclasses import Functor, Monad


_TOL = 1e-5


def _close(t1: torch.Tensor, t2: torch.Tensor) -> bool:
    """Tensor approximate equality within `_TOL`."""
    if t1.shape != t2.shape:
        return False
    return bool(torch.allclose(t1, t2, atol=_TOL, rtol=_TOL))


def check_functor_laws(
    inst: Functor, A: SetObject, B: SetObject, C: SetObject, f: Morphism, g: Morphism
) -> None:
    """Assert the two functor laws hold for ``inst`` on the given data.

    Laws:

    - identity:    ``F(id_A) = id_{F(A)}``
    - composition: ``F(g ∘ f) = F(g) ∘ F(f)``

    Parameters
    ----------
    inst : Functor
        The functor under test.
    A, B, C : SetObject
        The objects ``f`` and ``g`` connect.
    f : Morphism
        A morphism ``A → B``.
    g : Morphism
        A morphism ``B → C``.

    Raises
    ------
    AssertionError
        If either law fails.
    """
    fA = inst.fmap_obj(A)
    fid_lhs = inst.fmap(A, A, id_(A)).tensor
    fid_rhs = id_(fA).tensor
    assert _close(fid_lhs, fid_rhs), "Functor identity law violated"

    F_composed = inst.fmap(A, C, f >> g).tensor
    F_then = (inst.fmap(A, B, f) >> inst.fmap(B, C, g)).tensor
    assert _close(F_composed, F_then), "Functor composition law violated"


def check_monad_laws(inst: Monad, A: SetObject) -> None:
    """Assert the three monad laws hold for ``inst`` at ``A``.

    Laws, in terms of ``pure`` and ``join``:

    - left unit:     ``join_A ∘ pure_{F(A)} = id_{F(A)}``
    - right unit:    ``join_A ∘ F(pure_A) = id_{F(A)}``
    - associativity: ``join_A ∘ F(join_A) = join_A ∘ join_{F(A)}``

    The check materializes ``F(F(F(A)))``, so ``A`` should be small for
    instances whose carrier grows quickly, such as function spaces.

    Parameters
    ----------
    inst : Monad
        The monad under test.
    A : SetObject
        The object at which the laws are checked.

    Raises
    ------
    AssertionError
        If any law fails.
    """
    fA = inst.fmap_obj(A)
    ffA = inst.fmap_obj(fA)
    join_A = inst.join(A)
    id_fA = id_(fA).tensor

    left = (inst.pure(fA) >> join_A).tensor
    assert _close(left, id_fA), "Monad left unit law violated"

    right = (inst.fmap(A, fA, inst.pure(A)) >> join_A).tensor
    assert _close(right, id_fA), "Monad right unit law violated"

    inner_first = (inst.fmap(ffA, fA, join_A) >> join_A).tensor
    outer_first = (inst.join(fA) >> join_A).tensor
    assert _close(inner_first, outer_first), "Monad associativity law violated"


__all__ = [
    "check_functor_laws",
    "check_monad_laws",
]
