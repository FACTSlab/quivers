import json

import pytest

from quivers.qiec import (
    BOOL,
    INT,
    NAT,
    STRING,
    TYPE,
    ArrowKind,
    AttachmentId,
    AttachmentRef,
    Case,
    CaseBranch,
    CaseMotive,
    ConstructorDecl,
    ConstructorId,
    ConstructorValue,
    EffectDef,
    EffectId,
    EffectRef,
    EffectRequest,
    EffectRow,
    FamilyDecl,
    FamilyId,
    Handle,
    HandlerClauseDef,
    HandlerDef,
    HandlerId,
    IndexLiteral,
    LiteralValue,
    OperationDef,
    OperationId,
    Perform,
    QIEC_ABI,
    QIEC_WIRE_FORMAT,
    ResumptionGrade,
    Return,
    ShapeIndex,
    SiteProvenance,
    SourceOrigin,
    StaticScopeId,
    StaticVariableId,
    SerializationError,
    TypeApplication,
    TypeBinder,
    TypeVariable,
    UserIndexSort,
    dumps,
    from_data,
    instantiate_effect,
    loads,
    to_data,
)
from quivers.qiec.evaluator import RuntimeAttachments


def _representative_graph() -> tuple[object, ...]:
    state_id = EffectId.derive("serialization", "State")
    get_id = OperationId.derive(state_id, "get")
    state_variable = TypeVariable("state")
    get = OperationDef(get_id, "get", (), (), state_variable)
    state = EffectDef(
        EffectRef(state_id, "State"),
        (TypeBinder("state"),),
        (get,),
    )
    int_state = state.apply((INT,))
    entry = instantiate_effect(
        int_state,
        module="serialization",
        lexical_path=("main", "state"),
    )
    handler = HandlerDef(
        HandlerId.derive("serialization", "run-state-int"),
        "run-state-int",
        int_state,
        (HandlerClauseDef(get_id, ResumptionGrade.LINEAR),),
        INT,
        INT,
    )
    provenance = SiteProvenance(
        SourceOrigin(
            "serialization",
            ("body", 0),
            "State.get",
            "qvr-source/v0.19",
            "serialization.qvr",
            4,
            8,
        )
    )
    request = EffectRequest(
        entry.instance,
        int_state,
        get_id,
        (),
        (),
        INT,
        provenance,
    )
    handled = Handle(entry.instance, handler.id, Perform(request))

    family_id = FamilyId.derive("serialization", "Expr")
    constructor_id = ConstructorId.derive(family_id, "Bool")
    family = FamilyDecl(
        family_id,
        "Expr",
        (),
        (TypeBinder("index", TYPE, refinable=True),),
        (constructor_id,),
    )
    constructor = ConstructorDecl(
        constructor_id,
        family_id,
        "Bool",
        (),
        (),
        (BOOL,),
    )
    expr_bool = TypeApplication(family.type_constructor, (BOOL,))
    value = ConstructorValue(constructor_id, (), (), expr_bool)
    case = Case(
        value,
        CaseMotive(
            (TypeBinder("index", TYPE, refinable=True),),
            STRING,
        ),
        (
            CaseBranch(
                constructor_id,
                (),
                (),
                Return(LiteralValue("true", STRING)),
                StaticScopeId.derive("tests", "serialization-branch"),
            ),
        ),
    )
    attachment = AttachmentRef(
        AttachmentId.derive("serialization", "host-tensor"),
        INT,
    )
    color = UserIndexSort("Color", ("red", "blue"))
    return (
        ArrowKind(TYPE, TYPE),
        IndexLiteral(2, NAT),
        ShapeIndex((IndexLiteral(2, NAT), IndexLiteral(3, NAT))),
        color,
        TypeVariable(
            "scoped",
            TYPE,
            StaticVariableId.derive("tests", "serialization-variable"),
        ),
        state,
        int_state,
        EffectRow((entry,)),
        handler,
        family,
        constructor,
        value,
        case,
        request,
        handled,
        attachment,
        b"\x00qiec\xff",
    )


def test_representative_qiec_graph_round_trips_deterministically() -> None:
    graph = _representative_graph()
    encoded = dumps(graph)

    assert encoded == dumps(graph)
    assert loads(encoded) == graph
    assert json.loads(encoded)["$schema"] == QIEC_WIRE_FORMAT
    assert json.loads(encoded)["abi"] == QIEC_ABI
    decoded_ids = loads(dumps((EffectId.derive("concrete"),)))
    assert isinstance(decoded_ids, tuple)
    assert isinstance(decoded_ids[0], EffectId)


def test_data_api_preserves_tuples_and_has_an_explicit_envelope() -> None:
    graph = (INT, (LiteralValue((2, ("nested", None)), INT),))
    data = to_data(graph)

    assert set(data) == {"$schema", "abi", "root"}
    assert from_data(data) == graph


def test_codec_rejects_runtime_values_and_host_callables() -> None:
    with pytest.raises(SerializationError, match="runtime or unregistered"):
        dumps(RuntimeAttachments())
    with pytest.raises(SerializationError, match="host callables"):
        dumps(LiteralValue(lambda: 1, INT))  # type: ignore[arg-type]


def test_codec_rejects_unknown_tags_versions_and_malformed_tuples() -> None:
    envelope = {
        "$schema": QIEC_WIRE_FORMAT,
        "abi": QIEC_ABI,
        "root": {"$type": "terms.FutureNode", "fields": []},
    }
    with pytest.raises(SerializationError, match="unknown QIEC node tag"):
        from_data(envelope)

    wrong_abi = {**envelope, "abi": "qiec-core/v999"}
    with pytest.raises(SerializationError, match="ABI mismatch"):
        from_data(wrong_abi)

    bare_array = {**envelope, "root": [1, 2]}
    with pytest.raises(SerializationError, match="bare JSON arrays"):
        from_data(bare_array)


def test_codec_rejects_duplicate_json_object_keys() -> None:
    encoded = dumps(Return(LiteralValue(1, INT)))
    duplicate = encoded.replace(
        '"$schema":"qiec-json/v1"',
        '"$schema":"qiec-json/v1","$schema":"qiec-json/v1"',
        1,
    )

    with pytest.raises(SerializationError, match="duplicate JSON object key"):
        loads(duplicate)


@pytest.mark.parametrize(
    ("value", "wrong_tag"),
    [
        (Return(LiteralValue(1, INT)), "perform"),
        (_representative_graph()[13], "future_request"),
    ],
)
def test_codec_rejects_invalid_literal_tag_fields(
    value: object,
    wrong_tag: str,
) -> None:
    data = to_data(value)

    def replace_tag(node: object) -> bool:
        if isinstance(node, dict) and "$type" in node:
            fields = node.get("fields")
            if isinstance(fields, list):
                for field in fields:
                    if isinstance(field, dict) and field.get("name") == "tag":
                        field["value"] = wrong_tag
                        return True
                    if isinstance(field, dict) and replace_tag(field.get("value")):
                        return True
        if isinstance(node, dict):
            return any(replace_tag(item) for item in node.values())
        if isinstance(node, list):
            return any(replace_tag(item) for item in node)
        return False

    assert replace_tag(data["root"])
    with pytest.raises(SerializationError, match="invalid literal tag"):
        from_data(data)


def _encoded_field(node: object, name: str) -> dict[str, object]:
    assert isinstance(node, dict)
    encoded_fields = node["fields"]
    assert isinstance(encoded_fields, list)
    for field in encoded_fields:
        assert isinstance(field, dict)
        if field["name"] == name:
            return field
    raise AssertionError(f"missing encoded field {name!r}")


def test_codec_rejects_hostile_qiec_module_field_shapes() -> None:
    from quivers.qiec import QiecModule

    data = to_data(QiecModule("serialization", "qvr-source/v0.19"))
    _encoded_field(data["root"], "module")["value"] = 7

    with pytest.raises(SerializationError, match=r"QiecModule\.module: expected.*str"):
        from_data(data)


def test_codec_rejects_hostile_qiec_module_tuple_members() -> None:
    from quivers.qiec import QiecModule

    data = to_data(QiecModule("serialization", "qvr-source/v0.19"))
    families = _encoded_field(data["root"], "families")["value"]
    assert isinstance(families, dict)
    families["$tuple"] = [to_data(_representative_graph()[0])["root"]]

    with pytest.raises(
        SerializationError,
        match=r"QiecModule\.families\[0\]: expected.*FamilyDecl",
    ):
        from_data(data)


def test_codec_rejects_hostile_site_provenance_origin_shape() -> None:
    provenance = SiteProvenance(
        SourceOrigin(
            "serialization",
            ("request", 0),
            "perform",
            "qvr-source/v0.19",
        )
    )
    data = to_data(provenance)
    _encoded_field(data["root"], "origin")["value"] = "forged-origin"

    with pytest.raises(
        SerializationError,
        match=r"SiteProvenance\.origin: expected.*SourceOrigin",
    ):
        from_data(data)
