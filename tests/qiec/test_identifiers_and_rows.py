import json

import pytest

from quivers.qiec import (
    BOOL,
    INT,
    NAT,
    AttachmentId,
    DynamicAddressFrame,
    EffectId,
    EffectBinder,
    EffectInstanceId,
    EffectRef,
    EffectRow,
    EffectVariable,
    IndexBinder,
    IndexConstructor,
    IndexLiteral,
    OperationId,
    RowEntry,
    RowVariableId,
    RowVariable,
    SiteProvenance,
    SourceOrigin,
    TypeBinder,
    TypeConstructorRef,
    TypeId,
    TypeVariable,
    UserIndexSort,
    instantiate_telescope,
    unify_effect_rows,
)


def test_identifiers_are_namespaced_stable_and_round_trip() -> None:
    first = EffectId.derive("example", "Reader")
    second = EffectId.derive("example", "Reader")
    operation = OperationId.derive("example", "Reader", 1)

    assert first == second
    assert str(first) != str(operation)
    assert EffectId.parse(first.to_data()) == first
    assert AttachmentId.parse(str(AttachmentId.derive("host", "dist")))


def test_site_identity_uses_structural_origin_not_diagnostic_coordinates() -> None:
    before = SourceOrigin(
        "model",
        ("declarations", 2, "body"),
        "sample",
        "qvr-source/v0.19",
        file="model.qvr",
        line=10,
        column=4,
    )
    after = SourceOrigin(
        "model",
        ("declarations", 2, "body"),
        "sample",
        "qvr-source/v0.19",
        file="renamed.qvr",
        line=90,
        column=1,
    )
    provenance = SiteProvenance(
        before,
        (DynamicAddressFrame("plate", 3),),
        (1, 0),
    )

    assert before.site_id() == after.site_id()
    assert provenance.dynamic_key()[1] == (("plate", 3),)
    json.dumps(provenance.to_data())


def test_effect_rows_are_finite_maps_over_lexical_instances() -> None:
    effect_id = EffectId.derive("example", "State")
    effect = EffectRef(effect_id, "State")
    left_id = EffectInstanceId.derive("scope", "left")
    right_id = EffectInstanceId.derive("scope", "right")
    tail_id = RowVariableId.derive("tests", "rho")
    tail = RowVariable("rho", tail_id, (left_id, right_id))

    row = EffectRow(
        (RowEntry(right_id, effect), RowEntry(left_id, effect)),
        tail,
    )
    assert row == EffectRow(tuple(reversed(row.entries)), tail)
    assert row.lookup(left_id) == effect
    assert row.without(left_id).contains(right_id)
    assert len(row.entries) == 2
    assert EffectRow((), tail).add(RowEntry(left_id, effect)).contains(left_id)

    addable = EffectRow(
        (),
        RowVariable("addable", RowVariableId.derive("tests", "addable")),
    )
    added = addable.add(RowEntry(left_id, effect))
    assert added.tail is not None
    assert added.tail.proves_lacks(left_id)
    assert added.add(RowEntry(left_id, effect)) is added
    conflicting = EffectRef(EffectId.derive("example", "Other"), "Other")
    with pytest.raises(ValueError, match="two interfaces"):
        added.add(RowEntry(left_id, conflicting))

    with pytest.raises(ValueError, match="cannot contain"):
        EffectRow((RowEntry(left_id, effect), RowEntry(left_id, effect)))

    with pytest.raises(ValueError, match="must prove"):
        EffectRow(
            (RowEntry(left_id, effect),),
            RowVariable("open", RowVariableId.derive("tests", "open")),
        )

    shared = RowVariableId.derive("tests", "shared")
    left_open = EffectRow((), RowVariable("before", shared, (left_id,)))
    right_open = EffectRow((), RowVariable("after", shared, (right_id,)))
    merged = left_open.union(right_open)
    assert merged.tail is not None
    assert merged.tail.proves_lacks(left_id)
    assert merged.tail.proves_lacks(right_id)

    with pytest.raises(ValueError, match="distinct row variables"):
        left_open.union(
            EffectRow(
                (),
                RowVariable("other", RowVariableId.derive("tests", "other")),
            )
        )


def test_open_row_unification_produces_stable_capture_free_substitutions() -> None:
    effect = EffectRef(EffectId.derive("tests", "Unified"), "Unified")
    left_instance = EffectInstanceId.derive("tests", "left-entry")
    right_instance = EffectInstanceId.derive("tests", "right-entry")
    left_tail = RowVariable(
        "rho",
        RowVariableId.derive("tests", "left-tail"),
        (left_instance,),
    )
    right_tail = RowVariable(
        "rho",
        RowVariableId.derive("tests", "right-tail"),
        (right_instance,),
    )
    left = EffectRow((), left_tail).add(RowEntry(left_instance, effect))
    right = EffectRow((), right_tail).add(RowEntry(right_instance, effect))

    solution = unify_effect_rows(left, right)
    assert solution.substitution.apply(left) == solution.row
    assert solution.substitution.apply(right) == solution.row
    assert solution.row.tail is not None
    assert solution.row.tail.identity == RowVariableId.derive(
        "unify",
        *sorted((left_tail.identity, right_tail.identity), key=str),
        *sorted((left_instance, right_instance), key=str),
    )

    closed = EffectRow((RowEntry(left_instance, effect),))
    open_empty = EffectRow(
        (),
        RowVariable("tail", RowVariableId.derive("tests", "closed-solution")),
    )
    closed_solution = unify_effect_rows(open_empty, closed)
    assert closed_solution.substitution.apply(open_empty) == closed


def test_stable_references_ignore_diagnostic_names() -> None:
    type_id = TypeId.derive("tests", "semantic-type")
    left_type = TypeConstructorRef(type_id, "Before")
    right_type = TypeConstructorRef(type_id, "After", (TypeBinder("ignored"),))
    assert left_type == right_type
    assert hash(left_type) == hash(right_type)

    effect_id = EffectId.derive("tests", "semantic-effect")
    left_effect = EffectRef(effect_id, "Before", (BOOL,))
    right_effect = EffectRef(effect_id, "After", (BOOL,))
    assert left_effect == right_effect
    assert hash(left_effect) == hash(right_effect)
    # Identity is the declaration plus its static arguments, so the
    # display name is ignored on both sides while a different argument
    # separates two applications of the same interface.
    assert left_effect != EffectRef(effect_id, "After", (INT,))


def test_user_index_constructors_validate_membership_and_arity() -> None:
    unary = UserIndexSort("Unary", ("zero", "succ"), (0, 1))
    zero = IndexLiteral("zero", unary)
    assert IndexConstructor("succ", (zero,), unary).arguments == (zero,)

    with pytest.raises(ValueError, match="expects 1 arguments"):
        IndexConstructor("succ", (), unary)
    with pytest.raises(ValueError, match="not a constructor"):
        IndexConstructor("other", (), unary)
    with pytest.raises(ValueError, match="not nullary"):
        IndexLiteral("succ", unary)


def test_telescope_instantiation_preserves_static_namespaces() -> None:
    effect = EffectRef(EffectId.derive("example", "Reader"), "Reader")
    telescope = (
        TypeBinder("a"),
        IndexBinder("n", NAT),
        # Effect variables are kinded independently of value types.
        EffectBinder("e"),
    )
    substitution = instantiate_telescope(
        telescope,
        (BOOL, IndexLiteral(2, NAT), effect),
    )

    assert substitution.type("a") == BOOL
    assert substitution.index("n") == IndexLiteral(2, NAT)
    assert substitution.effect("e") == effect
    assert TypeVariable("a") != EffectVariable("a")

    with pytest.raises(TypeError, match="index argument"):
        instantiate_telescope((IndexBinder("n", NAT),), (BOOL,))
