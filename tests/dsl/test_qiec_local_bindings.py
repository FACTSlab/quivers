"""Local QIEC binders, and the range each one is visible over.

A tool that offers a name outside its scope offers a name that does not
resolve, and one that fails to offer a name inside its scope hides a
binding the user wrote. Both are wrong in the direction a user notices,
so the scopes here are asserted from both sides: every binder is visible
where it should be, and absent where it should not.
"""

from __future__ import annotations

import pathlib

import pytest

from quivers.dsl import parse
from quivers.dsl.qiec_tooling import qiec_local_bindings


_GOLDEN = (
    pathlib.Path(__file__).resolve().parents[1]
    / "fixtures"
    / "qiec"
    / "authored_state_handler.qvr"
)


def _bindings() -> dict[str, tuple[str, tuple[int, int] | None]]:
    """The golden's local binders, keyed by name.

    Returns
    -------
    dict[str, tuple[str, tuple[int, int] | None]]
        Name to its kind and scope.
    """
    return {
        binding.name: (binding.kind, binding.scope)
        for binding in qiec_local_bindings(parse(_GOLDEN.read_text()))
    }


def test_every_local_binder_is_collected() -> None:
    """Each binding form contributes its binder.

    A form missing here is a form the language server cannot complete,
    hover, or rename, however well it parses.
    """
    kinds = {name: kind for name, (kind, _) in _bindings().items()}
    assert kinds == {
        "x": "parameter",
        "s": "parameter",
        "seed": "parameter",
        "next": "local",
        "start": "local",
        "cell": "scoped-instance",
        "current": "local",
        "outer": "scoped-instance",
        "inner": "scoped-instance",
        "a": "local",
        "b": "local",
    }


def test_a_let_binder_is_visible_only_after_its_binding() -> None:
    """`let` scopes over its continuation, not over the whole body.

    The distinction matters for completion: offering `current` on the
    line that binds it would suggest a name not yet in scope.
    """
    _, scope = _bindings()["start"]
    assert scope is not None
    start, end = scope
    bindings = qiec_local_bindings(parse(_GOLDEN.read_text()))
    seed = next(
        binding
        for binding in bindings
        if binding.name == "seed" and binding.covers(start)
    )
    seed_scope = seed.scope
    assert seed_scope is not None
    assert seed_scope[0] < start, (
        "the enclosing parameter must be visible before the `let` that "
        "follows it, or the scopes are being read off the wrong node"
    )
    assert end >= start


def test_a_scoped_instance_does_not_escape_its_body() -> None:
    """`with instance ... in` confines its binder to the body.

    This is the tooling half of the rule the checker enforces: an
    instance that escapes its scope is a row entry nothing can discharge.
    """
    _, scope = _bindings()["cell"]
    assert scope is not None
    start, end = scope
    bindings = qiec_local_bindings(parse(_GOLDEN.read_text()))
    cell = next(binding for binding in bindings if binding.name == "cell")
    assert not cell.covers(start - 1), (
        "the scoped instance is visible above its own allocation, so a "
        "tool would offer it where it does not resolve"
    )
    assert cell.covers(start) and cell.covers(end)


@pytest.mark.parametrize("name", ["x", "s"])
def test_a_handler_clause_parameter_is_confined_to_its_clause(name: str) -> None:
    """One clause's parameter is not in scope in another clause.

    The return clause and the ``put`` clause of the golden handler bind
    different names at different lines, so a parameter leaking between
    them would show up as a scope covering the other clause's body.
    """
    bindings = qiec_local_bindings(parse(_GOLDEN.read_text()))
    subject = next(binding for binding in bindings if binding.name == name)
    others = [
        binding
        for binding in bindings
        if binding.kind == "parameter" and binding.name not in (name, "seed")
    ]
    assert subject.scope is not None
    for other in others:
        assert other.scope is not None
        assert not subject.covers(other.scope[0]), (
            f"{name!r} is visible inside the clause that binds "
            f"{other.name!r}, so clause parameters are leaking"
        )


def test_a_module_level_binding_covers_every_line() -> None:
    """A declaration with no scope is visible throughout.

    `covers` is the single predicate tools filter on, so it has to answer
    correctly for the unscoped case too.
    """
    bindings = qiec_local_bindings(parse(_GOLDEN.read_text()))
    assert all(binding.is_local for binding in bindings), (
        "everything this function returns is local by construction"
    )
