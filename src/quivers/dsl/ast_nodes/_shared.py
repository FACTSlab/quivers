"""Helper models shared by multiple ast_nodes submodules."""

from typing import Literal

import didactic.api as dx


# ---------------------------------------------------------------------------
# axis-role surface: per-distribution event / batch axis specification
# ---------------------------------------------------------------------------


class AxisSpec(dx.Model):
    """Axis-role specification on a distribution clause.

    Surface form: ``over <axes> [iid over <axes>]``.

    ``over`` names the event axes, the axes on which the family's
    joint structure (an MVN covariance, a MatrixNormal Kronecker pair,
    a GP kernel) lives.  The axis count must match the family's
    declared event rank; the positional ordering corresponds
    positionally to the family's event-axis ordering.

    ``iid_over`` is an optional readability assertion naming the batch
    axes (the complement of ``over`` in the surrounding morphism's
    type signature).  Inconsistency with the type signature or with
    ``over`` is a compile-time error.

    Axis names resolve against the named factors of the surrounding
    morphism's dom/cod.  The reserved tokens ``dom`` and ``cod`` are
    legal shortcuts only when the corresponding side is a single
    unfactored object.
    """

    over: tuple[str, ...]
    iid_over: tuple[str, ...] = ()
    line: int = 0
    col: int = 0


class MorphismPrior(dx.Model):
    """Parameter prior on a ``latent`` morphism's representing tensor.

    Surface form: ``~ Family(args) [options] [axis_role_clause]``.

    Promotes the declared morphism from a free-parameter point
    estimate to a random variable whose representing tensor is drawn
    from the named family at the requested axis-role configuration.
    Categorically: the morphism becomes the deterministic wrap of a
    sample from ``family(args)``, with the family's event/batch
    structure controlled by ``axes``.
    """

    family: str
    args: tuple[str | float, ...] = ()
    options: dict[str, str] = dx.field(default_factory=dict)
    axes: AxisSpec | None = None
    line: int = 0
    col: int = 0


# ---------------------------------------------------------------------------
# composition-level type alias used by AlgebraDecl
# ---------------------------------------------------------------------------


type CompositionLevel = Literal[
    "algebra", "semigroupoid", "bilinear_form", "composition_rule"
]
"""Algebraic level the file declares for its composition rule.

The four levels correspond to the
:class:`~quivers.core.algebras.CompositionRule`-hierarchy:

* ``"algebra"`` requires a full :class:`Algebra` (unit, zero,
  meet, negate, identity, dagger, cup/cap).
* ``"semigroupoid"`` requires a :class:`Semigroupoid`
  (associative `tensor_op`, no identity required).
* ``"bilinear_form"`` requires a :class:`BilinearForm`
  (no associativity promise).
* ``"composition_rule"`` is permissive: any
  :class:`CompositionRule` is accepted.
"""


__all__ = ["AxisSpec", "MorphismPrior", "CompositionLevel"]
