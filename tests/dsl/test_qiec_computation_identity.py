"""What a computation's stable identity does and does not depend on.

A `ComputationId` is what a call points at, so it has to survive edits
that do not change which declaration is meant. Moving a declaration down
the file, or reformatting above it, must leave the identity alone, or
every call in a serialized module would break on a whitespace change.

Equally it has to separate declarations that really are different, since
two computations sharing an identity would make a call ambiguous and a
module's own signature table lossy.
"""

from __future__ import annotations

from quivers.dsl import parse
from quivers.dsl.qiec_lowering import lower_qvr_to_qiec


_SOURCE = """define answer() : Int !{} =
    return 0
"""


def _identity(source: str, *, module_name: str = "example") -> str:
    """The stable identity of the source's first computation.

    Parameters
    ----------
    source : str
        A module declaring at least one computation.
    module_name : str
        Module name the lowerer derives identities from.

    Returns
    -------
    str
        The wire form of the computation's identifier.
    """
    lowered = lower_qvr_to_qiec(parse(source), module_name=module_name)
    return str(lowered.computations[0].id)


def test_moving_a_declaration_does_not_change_its_identity() -> None:
    """Position is diagnostic metadata, not identity.

    Two blank lines above the declaration change its reported line and
    nothing else. If this failed, adding an import or a comment would
    invalidate every call to it.
    """
    assert _identity(_SOURCE) == _identity("\n\n" + _SOURCE)


def test_position_is_still_recorded_even_though_it_is_not_identity() -> None:
    """The origin keeps the line, so diagnostics can still point at it.

    This is the other half of the previous test: the position must be
    carried, just not hashed.
    """
    here = lower_qvr_to_qiec(parse(_SOURCE), module_name="example")
    moved = lower_qvr_to_qiec(parse("\n\n" + _SOURCE), module_name="example")
    assert here.computations[0].origin.line != moved.computations[0].origin.line


def test_the_module_name_separates_identities() -> None:
    """Two modules may each declare `answer` without colliding."""
    assert _identity(_SOURCE, module_name="one") != _identity(
        _SOURCE, module_name="two"
    )


def test_the_declaration_name_separates_identities() -> None:
    """Renaming a computation gives a different declaration.

    A rename is a different declaration by design: the identity is
    nominal, so a rename is recorded in a migration rather than being
    invisible.
    """
    renamed = _SOURCE.replace("answer", "different")
    assert _identity(_SOURCE) != _identity(renamed)


def test_the_body_does_not_change_the_identity() -> None:
    """Editing a body does not make it a different declaration.

    Identity tracks which declaration is meant, not what it currently
    does, so a call keeps resolving across an edit to the callee.
    """
    edited = _SOURCE.replace("return 0", "return 1")
    assert _identity(_SOURCE) == _identity(edited)
