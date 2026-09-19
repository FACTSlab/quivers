"""Kernel-generated equality evidence.

The first QIEC kernel has no arbitrary user proof language.  Evidence is
either reflexivity or a rigid branch given introduced by GADT elimination.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from quivers.qiec.identifiers import EqualityId
from quivers.qiec.types import EqualityType


@dataclass(frozen=True, slots=True)
class Reflexivity:
    """Evidence that an equality holds because both sides are the same term.

    Parameters
    ----------
    equality
        The proposition witnessed, whose sides must be equal.
    tag
        The serialization discriminator; always ``"reflexivity"``.
    """

    equality: EqualityType
    tag: Literal["reflexivity"] = "reflexivity"


@dataclass(frozen=True, slots=True)
class BranchGiven:
    """Evidence granted by a case branch's constructor refinement.

    Parameters
    ----------
    id
        The stable identity of the given, derived from the branch scope.
    equality
        The proposition the branch's constructor makes available.
    tag
        The serialization discriminator; always ``"branch_given"``.
    """

    id: EqualityId
    equality: EqualityType
    tag: Literal["branch_given"] = "branch_given"


type EqualityEvidence = Reflexivity | BranchGiven


__all__ = ["BranchGiven", "EqualityEvidence", "Reflexivity"]
