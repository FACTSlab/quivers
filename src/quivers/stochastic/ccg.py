"""Weighted Combinatory Categorial Grammar (CCG) parser.

Implements a differentiable CKY-style chart parser for weighted CCG,
supporting the full set of standard combinatory rules:

- **Forward application** (>):  X/Y  Y  →  X
- **Backward application** (<):  Y  X\\Y  →  X
- **Forward composition** (>B):  X/Y  Y/Z  →  X/Z
- **Backward composition** (<B):  Y\\Z  X\\Y  →  X\\Z
- **Forward crossed composition** (>Bx):  X/Y  Y\\Z  →  X\\Z
- **Backward crossed composition** (<Bx):  Y/Z  X\\Y  →  X/Z
- **Generalized forward composition** (>B^n):  X/Y  Y|_1 Z_1 ... |_n Z_n  →  X|_1 Z_1 ... |_n Z_n
- **Forward type raising** (>T):  X  →  T/(T\\X)
- **Backward type raising** (<T):  X  →  T\\(T/X)

All computation is in log-space for numerical stability. The lexicon
(word → category distribution) is a learnable stochastic morphism,
while the combinatory rules are structural (fixed by the grammar).

Categorical perspective
-----------------------
CCG is the internal language of a closed monoidal category. The
slash categories X/Y and X\\Y are left and right internal homs,
and the application rules are evaluation morphisms (counits of
the hom-tensor adjunction). Composition rules correspond to
composition of internal hom morphisms.

Examples
--------
>>> from quivers.stochastic.categories import CategorySystem
>>> from quivers.stochastic.ccg import CCGParser
>>> cs = CategorySystem.from_atoms(["S", "NP", "N"])
>>> cs.add_slash(cs["S"], cs["NP"], "\\\\")   # S\\NP
>>> cs.add_slash(cs["S\\\\NP"], cs["NP"], "/")  # (S\\NP)/NP
>>> cs.add_slash(cs["NP"], cs["N"], "/")      # NP/N
>>> parser = CCGParser(cs, n_terminals=100, start="S")
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
from quivers.stochastic.rules import ccg_rules


class CCGParser(ChartParser):
    """Weighted CKY chart parser for Combinatory Categorial Grammar.

    A convenience subclass of ``ChartParser`` that builds CCG-specific
    rules from a ``CategorySystem`` and grammar options.

    Parameters
    ----------
    category_system : CategorySystem
        The finite inventory of grammatical categories.
    n_terminals : int
        Vocabulary size.
    start : str or Category
        The start (sentence) category.
    enable_composition : bool
        Enable harmonic composition (>B, <B). Default True.
    enable_crossed_composition : bool
        Enable crossed composition (>Bx, <Bx). Default True.
    enable_type_raising : bool
        Enable type raising (>T, <T). Default False.
    generalized_composition_depth : int
        Maximum depth for generalized composition. Default 1.
    """

    def __init__(
        self,
        category_system: CategorySystem,
        n_terminals: int,
        start: str | Category = "S",
        enable_composition: bool = True,
        enable_crossed_composition: bool = True,
        enable_type_raising: bool = False,
        generalized_composition_depth: int = 1,
    ) -> None:
        # resolve start category
        if isinstance(start, str):
            start_cat = AtomicCategory(start)

        else:
            start_cat = start

        if start_cat not in category_system:
            raise ValueError(f"start category {start_cat!r} not in category system")

        # build CCG rule system
        rules = ccg_rules(
            category_system,
            enable_composition=enable_composition,
            enable_crossed_composition=enable_crossed_composition,
            enable_type_raising=enable_type_raising,
            generalized_composition_depth=generalized_composition_depth,
        )

        super().__init__(
            rule_system=rules,
            n_terminals=n_terminals,
            start_idx=category_system.index(start_cat),
            category_system=category_system,
        )

    def __repr__(self) -> str:
        return (
            f"CCGParser(categories={self._n_cat}, "
            f"terminals={self._n_term}, "
            f"binary_rules={self._n_rules}, "
            f"unary_rules={self._n_unary})"
        )
