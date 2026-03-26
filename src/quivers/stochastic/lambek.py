"""Weighted type-logical grammar (Lambek calculus) parser.

Implements a differentiable chart parser for the non-commutative
Lambek calculus (L), supporting:

- **Right application** (modus ponens right):  A/B, B  ⊢  A
- **Left application** (modus ponens left):    B, A\\\\B  ⊢  A
  (equivalently written B, B\\\\A ⊢ A with the convention A\\\\B = A
  seeks B on the left)
- **Right lifting** (/L):  A  ⊢  B/(A\\\\B)
- **Left lifting** (\\\\L):  A  ⊢  (B/A)\\\\B
- **Product introduction** (⊗I):  A, B  ⊢  A⊗B
- **Product elimination** (⊗E):  A⊗B  ⊢  A, B (projected)

The Lambek calculus is the internal logic of a residuated
monoidal (bi-closed) category:

    A ⊗ B ⊢ C   iff   A ⊢ C/B   iff   B ⊢ A\\\\C

Unlike CCG, the Lambek calculus does NOT permit:
- Composition as a primitive rule (it's derivable)
- Crossed composition
- Permuting rules

This makes the Lambek calculus resource-sensitive and
order-preserving, modeling word order constraints naturally.

Extended calculi
----------------
This module also supports:

- **NL (Non-associative Lambek)**: If ``associative=False``, the
  parser uses tree-structured proof search, disallowing reassociation
  of the product.
- **LP (Lambek with permutation)**: If ``commutative=True``, the
  product is commutative, relaxing word order constraints.
- **Displacement calculus**: Not yet supported; would require
  additional modalities.

Categorical perspective
-----------------------
The Lambek calculus is the internal logic of a residuated
monoidal category (a biclosed monoidal category). The types
are objects; the proofs are morphisms. A valid derivation
A_1 ⊗ ... ⊗ A_n ⊢ B corresponds to a morphism in the category.

The parsing algorithm searches for such morphisms (proofs),
computing the total weight (probability) of all derivations.

Examples
--------
>>> from quivers.stochastic.categories import CategorySystem
>>> from quivers.stochastic.lambek import LambekParser
>>> cs = CategorySystem.from_atoms(["S", "NP", "N"])
>>> cs.add_slash(cs["S"], cs["NP"], "\\\\")   # S\\NP
>>> cs.add_slash(cs["S\\\\NP"], cs["NP"], "/")  # (S\\NP)/NP
>>> cs.add_slash(cs["NP"], cs["N"], "/")      # NP/N
>>> parser = LambekParser(cs, n_terminals=100, start="S")
>>> tokens = torch.randint(0, 100, (4, 5))
>>> log_probs = parser(tokens)
"""

from __future__ import annotations


from quivers.stochastic.categories import (
    AtomicCategory,
    Category,
    CategorySystem,
)
from quivers.stochastic.parsers import ChartParser
from quivers.stochastic.rules import lambek_rules


class LambekParser(ChartParser):
    """Weighted chart parser for type-logical grammar (Lambek calculus).

    A convenience subclass of ``ChartParser`` that builds Lambek
    calculus rules from a ``CategorySystem`` and grammar options.

    Parameters
    ----------
    category_system : CategorySystem
        The finite category inventory.
    n_terminals : int
        Vocabulary size.
    start : str or Category
        The start category (default "S").
    associative : bool
        Use associative Lambek calculus (default True).
    commutative : bool
        Use commutative product / LP calculus (default False).
    enable_lifting : bool
        Enable Lambek lifting rules (default True).
    enable_product : bool
        Enable product introduction (default True).
    """

    def __init__(
        self,
        category_system: CategorySystem,
        n_terminals: int,
        start: str | Category = "S",
        associative: bool = True,
        commutative: bool = False,
        enable_lifting: bool = True,
        enable_product: bool = True,
    ) -> None:
        # resolve start category
        if isinstance(start, str):
            start_cat = AtomicCategory(start)

        else:
            start_cat = start

        if start_cat not in category_system:
            raise ValueError(f"start category {start_cat!r} not in category system")

        # build Lambek rule system
        rules = lambek_rules(
            category_system,
            associative=associative,
            commutative=commutative,
            enable_lifting=enable_lifting,
            enable_product=enable_product,
        )

        super().__init__(
            rule_system=rules,
            n_terminals=n_terminals,
            start_idx=category_system.index(start_cat),
            category_system=category_system,
        )

        self._associative = associative
        self._commutative = commutative

    def __repr__(self) -> str:
        variants = []

        if not self._associative:
            variants.append("non-associative")

        if self._commutative:
            variants.append("commutative")

        variant_str = f", variants=[{', '.join(variants)}]" if variants else ""

        return (
            f"LambekParser(categories={self._n_cat}, "
            f"terminals={self._n_term}, "
            f"binary_rules={self._n_rules}, "
            f"unary_rules={self._n_unary}"
            f"{variant_str})"
        )
