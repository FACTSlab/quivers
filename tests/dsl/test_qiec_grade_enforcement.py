"""Resumption grades enforced on handlers written in QVR source.

The kernel's grade analysis is tested directly elsewhere. What matters
here is that a grade written in a source file actually binds: a clause
declaring `resumes 0` and then resuming must be rejected when the module
is lowered, not merely when a kernel record is built by hand.
"""

from __future__ import annotations

import pytest

from quivers.dsl import parse
from quivers.dsl.qiec_lowering import QiecDiagnosticError, lower_qvr_to_qiec


_TEMPLATE = """effect State[S : Type]
    get : Unit -> S

handler h for State[Int] : Int -> Int [coverage=total, implementation=authored]
    return x =>
        return x
    get() resumes {grade} =>
{body}
"""


def _lower(grade: str, body: str) -> None:
    """Lower a handler with the given grade and clause body.

    Parameters
    ----------
    grade : str
        The declared resumption grade.
    body : str
        The clause body, already indented to its block.

    Raises
    ------
    QiecDiagnosticError
        If the module does not lower and check.
    """
    source = _TEMPLATE.format(grade=grade, body=body)
    lower_qvr_to_qiec(parse(source), module_name="grades")


_RESUMES = "        resume(0)"
_RETURNS = "        return 0"


@pytest.mark.parametrize(
    ("grade", "body"),
    [
        ("0", _RETURNS),
        ("aff", _RETURNS),
        ("aff", _RESUMES),
        ("1", _RESUMES),
        ("omega", _RESUMES),
        ("omega", _RETURNS),
    ],
    ids=lambda value: value.strip(),
)
def test_a_clause_within_its_grade_lowers(grade: str, body: str) -> None:
    """A body that keeps its promise is accepted."""
    _lower(grade, body)


@pytest.mark.parametrize(
    ("grade", "body", "reason"),
    [
        ("0", _RESUMES, "declares grade 0"),
        ("1", _RETURNS, "does not resume at all"),
    ],
    ids=["zero-grade-resumes", "linear-grade-never-resumes"],
)
def test_a_clause_breaking_its_grade_is_rejected(
    grade: str, body: str, reason: str
) -> None:
    """A body that breaks its promise fails at the source position.

    The zero case is the one with teeth: a handler declared `0` may be
    compiled without keeping the continuation alive, so resuming anyway
    would be undefined at runtime rather than merely surprising.
    """
    with pytest.raises(QiecDiagnosticError, match=reason):
        _lower(grade, body)
