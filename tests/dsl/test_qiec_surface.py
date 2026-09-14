"""Focused parse and canonical-emission tests for the QIEC v0.19 surface."""

from __future__ import annotations

import importlib

import pytest

from quivers.dsl.ast_nodes import (
    ObjectEffectApply,
    QiecBindComputation,
    QiecCaseComputation,
    QiecComputationDecl,
    QiecConstructorValue,
    QiecEffectBinder,
    QiecEffectDecl,
    QiecEffectInstanceDecl,
    QiecFamilyDecl,
    QiecHandleComputation,
    QiecHandlerDecl,
    QiecIndexApplication,
    QiecIndexBinder,
    QiecIndexDecl,
    QiecNatSort,
    QiecPerformComputation,
    QiecSequenceComputation,
    QiecTypeApplication,
    QiecTypeBinder,
)
from quivers.dsl.emit import module_to_source
from quivers.dsl.parser import parse
from quivers.dsl.parser import ParseError


FAMILIES = """\
#! A closed unary natural-number index.
index Nat = Z | S(Nat)

#! A length-indexed vector.
family Vec[A : Type](n : Nat) : Type
    constructor Nil : Vec[A](Z)
    constructor Cons[m : Nat] : A * Vec[A](m) -> Vec[A](S(m))
"""


EFFECTS = """\
effect State[S : Type]
    get : Unit -> S
    put : S -> Unit

instance cell : State[Int]

handler run_state[S : Type, A : Type] for State[S] : A -> A [introduces=!{cell | rho lacks cell}, coverage=partial, forwards=unknown]
    get resumes 1
    put resumes 0
"""


COMPUTATIONS = """\
define swap[cell_effect : Effect](next : Int) : Int !{cell | rho lacks cell} =
    let prior : Int <- perform cell.get()
    perform cell.put(next)
    return prior

define singleton[A : Type](value : A) : Vec[A](S(Z)) !{} =
    return construct Cons[A, Z](value, construct Nil[A]() as Vec[A](Z)) as Vec[A](S(Z))

define head[A : Type, n : Nat](xs : Vec[A](S(n))) : A !{} =
    case xs motive (m : Nat) => A
        Cons[n](head, tail) =>
            return head

define handled(next : Int) : Int !{} =
    handle cell with run_state[Int, Int] in
        return next
"""


def _assert_fixed_point(source: str) -> None:
    canonical = module_to_source(parse(source))
    assert module_to_source(parse(canonical)) == canonical


def test_indexed_family_surface_is_distinct_and_located() -> None:
    module = parse(FAMILIES)
    index, family = module.statements
    assert isinstance(index, QiecIndexDecl)
    assert index.docs == ("A closed unary natural-number index.",)
    # Declaration spans include their attached doc-comment group.
    assert index.line == 1 and index.col == 0
    assert isinstance(index.constructors[1].arguments[0], QiecNatSort)

    assert isinstance(family, QiecFamilyDecl)
    assert family.docs == ("A length-indexed vector.",)
    assert isinstance(family.parameters[0], QiecTypeBinder)
    assert isinstance(family.indices[0], QiecIndexBinder)
    assert family.constructors[1].name == "Cons"
    assert len(family.constructors[1].arguments) == 2
    result = family.constructors[1].result
    assert isinstance(result, QiecTypeApplication)
    assert isinstance(result.indices[0], QiecIndexApplication)
    assert result.line > 0 and result.col > 0
    assert not isinstance(result, ObjectEffectApply)


def test_parameterized_effect_instances_and_handler_signature() -> None:
    module = parse(EFFECTS)
    effect, instance, handler = module.statements
    assert isinstance(effect, QiecEffectDecl)
    assert [operation.name for operation in effect.operations] == ["get", "put"]
    assert all(operation.arguments for operation in effect.operations)

    assert isinstance(instance, QiecEffectInstanceDecl)
    assert instance.effect.name == "State"
    assert len(instance.effect.arguments) == 1

    assert isinstance(handler, QiecHandlerDecl)
    assert handler.coverage == "partial"
    assert handler.forwards_unknown
    assert handler.introduced.tail == "rho"
    assert handler.introduced.lacks == ("cell",)
    assert [clause.grade for clause in handler.clauses] == ["1", "0"]
    assert all(clause.line > 0 for clause in handler.clauses)


def test_computation_surface_covers_stable_qiec_terms() -> None:
    swap, singleton, head, handled = parse(COMPUTATIONS).statements
    assert isinstance(swap, QiecComputationDecl)
    assert isinstance(swap.binders[0], QiecEffectBinder)
    assert swap.effects.tail == "rho"
    assert isinstance(swap.body, QiecBindComputation)
    assert isinstance(swap.body.first, QiecPerformComputation)
    assert isinstance(swap.body.then, QiecSequenceComputation)

    assert isinstance(singleton, QiecComputationDecl)
    assert isinstance(singleton.body.value, QiecConstructorValue)
    assert isinstance(singleton.body.value.fields[1], QiecConstructorValue)

    assert isinstance(head, QiecComputationDecl)
    assert isinstance(head.body, QiecCaseComputation)
    branch = head.body.branches[0]
    assert branch.constructor == "Cons"
    assert len(branch.static_arguments) == 1
    assert [field.name for field in branch.fields] == ["head", "tail"]
    assert branch.line > 0 and branch.col > 0

    assert isinstance(handled, QiecComputationDecl)
    assert isinstance(handled.body, QiecHandleComputation)
    assert handled.body.handler.name == "run_state"
    assert len(handled.body.handler.static_arguments) == 2


def test_qiec_surface_emit_is_a_canonical_fixed_point() -> None:
    _assert_fixed_point(FAMILIES)
    _assert_fixed_point(EFFECTS)
    _assert_fixed_point(COMPUTATIONS)


def test_effect_row_can_have_a_constrained_tail_without_entries() -> None:
    source = """\
define open_row() : Unit !{| rho lacks cell} =
    return unit
"""
    declaration = parse(source).statements[0]
    assert isinstance(declaration, QiecComputationDecl)
    assert declaration.effects.entries == ()
    assert declaration.effects.tail == "rho"
    assert declaration.effects.lacks == ("cell",)
    _assert_fixed_point(source)


def test_current_grammar_loader_fails_closed_without_verified_artifacts(
    monkeypatch,
) -> None:
    registry_module = importlib.import_module("quivers.dsl.parser._registry")
    previous_registry = registry_module._REGISTRY
    previous_library = registry_module._GRAMMAR_LIBRARY
    monkeypatch.setattr(registry_module, "_REGISTRY", None)
    monkeypatch.setattr(registry_module, "_GRAMMAR_LIBRARY", None)

    def unavailable():
        raise FileNotFoundError("deliberately unavailable")

    monkeypatch.setattr(registry_module, "_parser_artifacts", unavailable)
    try:
        with pytest.raises(ParseError, match="verified current QVR parser"):
            registry_module._registry()
    finally:
        registry_module._REGISTRY = previous_registry
        registry_module._GRAMMAR_LIBRARY = previous_library
