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
    equality: EqualityType
    tag: Literal["reflexivity"] = "reflexivity"


@dataclass(frozen=True, slots=True)
class BranchGiven:
    id: EqualityId
    equality: EqualityType
    tag: Literal["branch_given"] = "branch_given"


type EqualityEvidence = Reflexivity | BranchGiven


__all__ = ["BranchGiven", "EqualityEvidence", "Reflexivity"]
