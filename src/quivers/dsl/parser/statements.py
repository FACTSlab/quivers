"""Top-level statement dispatcher and every per-declaration walker."""

from __future__ import annotations

from typing import Literal

from quivers.dsl.ast_nodes import (
    AlgebraDecl,
    AliasDecl,
    BinderArg,
    BinderDecl,
    BinderVar,
    BindStep,
    BundleDecl,
    CategoryDecl,
    CompositionRuleEntry,
    ConstructorDecl,
    ContractionDecl,
    ContractionInput,
    DecoderDecl,
    DeductionDecl,
    DiscretizeDecl,
    EdgeKindDecl,
    EmbedDecl,
    EncoderDecl,
    EncoderInitRule,
    EncoderMessageRule,
    EncoderRule,
    EncoderUpdateRule,
    EncoderVarInit,
    EnumSetLiteral,
    ExportDecl,
    FreeMonoidExpr,
    FreeResiduatedExpr,
    KernelDecl,
    LetDecl,
    LetExprNode,
    LexiconEntry,
    LossAttachment,
    LossDecl,
    MorphismDecl,
    ObjectDecl,
    ProgramDecl,
    ProgramParam,
    RuleDecl,
    SchemaDecl,
    SequentRule,
    SignatureDecl,
    SortDecl,
    SortDim,
    SortVocabLiteral,
    SpaceDecl,
    Statement,
    TypeAliasDecl,
    TypeExpr,
    VertexKindDecl,
)
from quivers.dsl.parser._helpers import (
    _required_text,
    _walk_options,
    _walk_return_pattern,
)
from quivers.dsl.parser._registry import ParseError, _Tree
from quivers.dsl.parser.axes import _walk_axis_role_clause, _walk_morphism_prior
from quivers.dsl.parser.expressions import (
    _walk_expr,
    _walk_let_arith,
    _walk_space,
    _walk_type,
)
from quivers.dsl.parser.program_steps import (
    _walk_program_param,
    _walk_program_step,
)


def _walk_statement(t: _Tree, vid: str) -> Statement | list[Statement]:
    k = t.kind(vid)
    line, col = t.line_col(vid)

    if k == "algebra_decl":
        nv = t.field(vid, "name")
        # Tree-sitter doesn't emit a named child for the literal
        # keyword token, so the ``level`` field doesn't appear in
        # the parse tree.  Recover the keyword by reading the
        # leading word from the declaration's source range.
        decl_text = t.text(vid).lstrip()
        level: str = "algebra"
        for kw in ("composition_rule", "bilinear_form", "semigroupoid", "algebra"):
            if decl_text.startswith(kw):
                level = kw
                break
        body_entries: list = []
        for child_vid in t.positional(vid):
            if t.kind(child_vid) == "composition_rule_block":
                for entry_vid in t.positional(child_vid):
                    if t.kind(entry_vid) != "composition_rule_entry":
                        continue
                    body_entries.append(_walk_composition_rule_entry(t, entry_vid))
        return AlgebraDecl(
            name=_required_text(t, nv, vid, "name"),
            declared_level=level,
            body=tuple(body_entries),
            line=line,
            col=col,
        )
    if k == "contraction_decl":
        return _walk_contraction_decl(t, vid, line, col)
    if k == "category_decl":
        out: list[Statement] = []
        for nv in t.fields(vid, "names"):
            ln, cl = t.line_col(nv)
            out.append(CategoryDecl(name=t.text(nv), line=ln, col=cl))
        if not out:
            return CategoryDecl(name="", line=line, col=col)
        return out if len(out) > 1 else out[0]
    if k == "rule_decl":
        return _walk_rule_decl(t, vid, line, col)
    if k == "schema_decl":
        return _walk_schema_decl(t, vid, line, col)
    if k == "object_decl":
        nv = t.field(vid, "name")
        tv = t.field(vid, "type")
        iv = t.field(vid, "init")
        if tv is not None:
            return ObjectDecl(
                name=_required_text(t, nv, vid, "name"),
                type_expr=_walk_type(t, tv),
                init=None,
                line=line,
                col=col,
            )
        if iv is not None:
            return ObjectDecl(
                name=_required_text(t, nv, vid, "name"),
                type_expr=None,
                init=_walk_object_initializer(t, iv),
                line=line,
                col=col,
            )
        raise ParseError(f"object_decl missing type/init at {vid}")
    if k == "morphism_decl":
        cs = t.consts(vid)
        prefix = t.source[int(cs["start-byte"]) : int(cs["start-byte"]) + 8].decode(
            "utf-8"
        )
        morph_kind = "observed" if prefix.startswith("observed") else "latent"
        opt_vid = t.field(vid, "options")
        options = _walk_options(t, opt_vid) if opt_vid else {}
        init_vid = t.field(vid, "init")
        init_expr = _walk_expr(t, init_vid) if init_vid else None
        nv = t.field(vid, "name")
        dv = t.field(vid, "domain")
        cv = t.field(vid, "codomain")
        if dv is None or cv is None:
            raise ParseError(f"morphism_decl missing domain/codomain at {vid}")
        prior_vid = t.field(vid, "prior")
        prior = _walk_morphism_prior(t, prior_vid) if prior_vid else None
        return MorphismDecl(
            morphism_kind=morph_kind,  # type: ignore[arg-type]
            name=_required_text(t, nv, vid, "name"),
            domain=_walk_type(t, dv),
            codomain=_walk_type(t, cv),
            init_expr=init_expr,
            options=options,
            prior=prior,
            line=line,
            col=col,
        )
    if k == "ERROR":
        raise ParseError(
            f"syntax error at line {line}, col {col}: "
            f"{t.source[int(t.consts(vid)['start-byte']) : int(t.consts(vid)['end-byte'])].decode('utf-8')!r}"
        )
    if k in ("space_decl", "type_alias_decl"):
        nv = t.field(vid, "name")
        vv = t.field(vid, "value")
        if vv is None:
            raise ParseError(f"{k} missing value at {vid}")
        return SpaceDecl(
            name=_required_text(t, nv, vid, "name"),
            space_expr=_walk_space(t, vv),
            line=line,
            col=col,
        )
    if k == "alias_decl":
        nv = t.field(vid, "name")
        vv = t.field(vid, "value")
        if vv is None:
            raise ParseError(f"alias_decl missing value at {vid}")
        return AliasDecl(
            name=_required_text(t, nv, vid, "name"),
            type_expr=_walk_type(t, vv),
            line=line,
            col=col,
        )
    if k == "bundle_decl":
        nv = t.field(vid, "name")
        rule_vids = t.fields(vid, "rules")
        return BundleDecl(
            name=_required_text(t, nv, vid, "name"),
            rules=tuple(t.text(r) for r in rule_vids),
            line=line,
            col=col,
        )
    if k == "kernel_decl":
        rep_vid = t.field(vid, "replicate")
        replicate = int(t.text(t.positional(rep_vid)[0])) if rep_vid else None
        opt_vid = t.field(vid, "options")
        options = _walk_options(t, opt_vid) if opt_vid else {}
        nv = t.field(vid, "name")
        dv = t.field(vid, "domain")
        cv = t.field(vid, "codomain")
        fv = t.field(vid, "family")
        axes_vid = t.field(vid, "axes")
        axes = _walk_axis_role_clause(t, axes_vid) if axes_vid else None
        if dv is None or cv is None:
            raise ParseError(f"kernel_decl missing domain/codomain at {vid}")
        family = t.text(fv) if fv is not None else None
        return KernelDecl(
            name=_required_text(t, nv, vid, "name"),
            domain=_walk_type(t, dv),
            codomain=_walk_type(t, cv),
            family=family,
            options=options,
            axes=axes,
            replicate=replicate,
            line=line,
            col=col,
        )
    if k == "discretize_decl":
        opt_vid = t.field(vid, "options")
        options = _walk_options(t, opt_vid) if opt_vid else {}
        nv = t.field(vid, "name")
        sv = t.field(vid, "space")
        bv = t.field(vid, "bins")
        if bv is None:
            raise ParseError(f"discretize_decl missing bins at {vid}")
        return DiscretizeDecl(
            name=_required_text(t, nv, vid, "name"),
            space_name=_required_text(t, sv, vid, "space"),
            n_bins=int(t.text(bv)),
            options=options,
            line=line,
            col=col,
        )
    if k == "embed_decl":
        rep_vid = t.field(vid, "replicate")
        replicate = int(t.text(t.positional(rep_vid)[0])) if rep_vid else None
        nv = t.field(vid, "name")
        dv = t.field(vid, "domain")
        cv = t.field(vid, "codomain")
        return EmbedDecl(
            name=_required_text(t, nv, vid, "name"),
            domain_name=_required_text(t, dv, vid, "domain"),
            codomain_name=_required_text(t, cv, vid, "codomain"),
            replicate=replicate,
            line=line,
            col=col,
        )
    if k == "program_decl":
        params_vids = t.fields(vid, "params")
        data_params: list[str] = []
        type_params_list: list[ProgramParam] = []
        for pv in params_vids:
            if t.kind(pv) == "typed_program_param":
                type_params_list.append(_walk_program_param(t, pv))
            else:
                data_params.append(t.text(pv))
        if data_params and type_params_list:
            raise ParseError(
                f"program {_required_text(t, t.field(vid, 'name'), vid, 'name')!r} "
                f"mixes bare data parameters and typed template parameters"
            )
        params: tuple[str, ...] | None = tuple(data_params) if data_params else None
        type_params: tuple[ProgramParam, ...] | None = (
            tuple(type_params_list) if type_params_list else None
        )
        effects_vids = t.fields(vid, "effects")
        effects_set: frozenset[str] | None = (
            frozenset(t.text(ev) for ev in effects_vids) if effects_vids else None
        )
        over_vid = t.field(vid, "over_model")
        over_model: str | None = t.text(over_vid) if over_vid is not None else None
        steps = tuple(_walk_program_step(t, sv) for sv in t.fields(vid, "steps"))
        ret_vid = t.field(vid, "return")
        if ret_vid is None:
            raise ParseError(f"program_decl missing return at {vid}")
        return_vars, return_labels = _walk_return_pattern(t, ret_vid)
        nv = t.field(vid, "name")
        dv = t.field(vid, "domain")
        cv = t.field(vid, "codomain")
        if dv is None or cv is None:
            raise ParseError(f"program_decl missing domain/codomain at {vid}")
        # Posterior-block constraint: when `over M` is set, the body
        # must be deterministic — no `sample` / `score` binds.
        if over_model is not None:
            for s in steps:
                if isinstance(s, BindStep) and s.mode in ("sample", "score"):
                    raise ParseError(
                        f"program with 'over {over_model}' is a posterior "
                        "block and may not contain sample / observe binds "
                        "(posterior runs after conditioning); use 'let' "
                        "and 'marginalize' only"
                    )
        return ProgramDecl(
            name=_required_text(t, nv, vid, "name"),
            params=params,
            domain=_walk_type(t, dv),
            codomain=_walk_type(t, cv),
            draws=steps,
            return_vars=return_vars,
            return_labels=return_labels,
            effects=effects_set,
            over_model=over_model,
            type_params=type_params,
            line=line,
            col=col,
        )
    if k == "let_decl":
        where_vids = t.fields(vid, "where")
        where: tuple[Statement, ...] | None = None
        if where_vids:
            wd: list[Statement] = []
            for wv in where_vids:
                result = _walk_statement(t, wv)
                if isinstance(result, list):
                    wd.extend(result)
                else:
                    wd.append(result)
            where = tuple(wd) if wd else None
        nv = t.field(vid, "name")
        vv = t.field(vid, "value")
        if vv is None:
            raise ParseError(f"let_decl missing value at {vid}")
        return LetDecl(
            name=_required_text(t, nv, vid, "name"),
            expr=_walk_expr(t, vv),
            where=where,
            line=line,
            col=col,
        )
    if k == "export_decl":
        vv = t.field(vid, "value")
        if vv is None:
            raise ParseError(f"export_decl missing value at {vid}")
        return ExportDecl(
            expr=_walk_expr(t, vv),
            line=line,
            col=col,
        )
    if k == "deduction_decl":
        return _walk_deduction_decl(t, vid, line, col)
    if k == "signature_decl":
        return _walk_signature_decl(t, vid, line, col)
    if k == "encoder_decl":
        return _walk_encoder_decl(t, vid, line, col)
    if k == "decoder_decl":
        return _walk_decoder_decl(t, vid, line, col)
    if k == "loss_decl":
        return _walk_loss_decl(t, vid, line, col)
    raise ParseError(f"unexpected statement kind: {k}")


def _walk_deduction_decl(t: _Tree, vid: str, line: int, col: int) -> DeductionDecl:
    """Walk a ``deduction NAME : dom -> cod { … }`` block.

    Collects the brace-delimited body's atoms, sequent rules,
    and field assignments (semiring, start, depth) into a
    :class:`DeductionDecl`.
    """
    nv = t.field(vid, "name")
    dv = t.field(vid, "domain")
    cv = t.field(vid, "codomain")
    if nv is None or dv is None or cv is None:
        raise ParseError(f"deduction_decl missing name/domain/codomain at {vid}")
    atoms: list[str] = []
    rules: list[SequentRule] = []
    semiring: str | None = None
    start: str | None = None
    depth: int | None = None
    lex_entries: list[LexiconEntry] = []
    lex_from_file: str | None = None
    lex_from_file_learnable: bool = False
    axioms_source: str | None = None
    item_signature: str | None = None
    item_encoder: str | None = None
    # Walk children in order.
    for child in t.positional(vid):
        kk = t.kind(child)
        if kk == "deduction_atoms":
            for av in t.fields(child, "atoms"):
                atoms.append(t.text(av))
        elif kk == "deduction_rule":
            r_name = _required_text(t, t.field(child, "name"), child, "name")
            premises = tuple(_walk_type(t, pv) for pv in t.fields(child, "premises"))
            conc_vid = t.field(child, "conclusion")
            if conc_vid is None:
                raise ParseError(f"deduction_rule missing conclusion at {child}")
            conclusion = _walk_type(t, conc_vid)
            rl, rc = t.line_col(child)
            rules.append(
                SequentRule(
                    name=r_name,
                    premises=premises,
                    conclusion=conclusion,
                    line=rl,
                    col=rc,
                )
            )
        elif kk == "deduction_semiring":
            semiring = _required_text(t, t.field(child, "semiring"), child, "semiring")
        elif kk == "deduction_start":
            start = _required_text(t, t.field(child, "start"), child, "start")
        elif kk == "deduction_depth":
            depth = int(_required_text(t, t.field(child, "depth"), child, "depth"))
        elif kk == "deduction_lexicon":
            for entry_vid in t.positional(child):
                if t.kind(entry_vid) != "lexicon_entry":
                    continue
                lex_entries.append(_walk_lexicon_entry(t, entry_vid))
        elif kk == "deduction_lexicon_from_file":
            path_raw = _required_text(t, t.field(child, "path"), child, "path")
            # Strip surrounding quotes from the string literal.
            if path_raw.startswith('"') and path_raw.endswith('"'):
                path_raw = path_raw[1:-1]
            lex_from_file = path_raw
            lex_from_file_learnable = t.field(child, "learnable") is not None
        elif kk == "deduction_axioms":
            axioms_source = _required_text(t, t.field(child, "source"), child, "source")
        elif kk == "deduction_signature":
            item_signature = _required_text(
                t, t.field(child, "signature"), child, "signature"
            )
        elif kk == "deduction_encoder_attach":
            item_encoder = _required_text(
                t, t.field(child, "encoder"), child, "encoder"
            )
    return DeductionDecl(
        name=_required_text(t, nv, vid, "name"),
        domain=_walk_type(t, dv),
        codomain=_walk_type(t, cv),
        atoms=tuple(atoms),
        rules=tuple(rules),
        semiring=semiring,
        start=start,
        depth=depth,
        lexicon=tuple(lex_entries),
        lexicon_from_file=lex_from_file,
        lexicon_from_file_learnable=lex_from_file_learnable,
        axioms_source=axioms_source,
        item_signature=item_signature,
        item_encoder=item_encoder,
        line=line,
        col=col,
    )


def _walk_lexicon_entry(t: _Tree, vid: str) -> LexiconEntry:
    """Walk a single `lexicon_entry` into a :class:`LexiconEntry`.

    Surface form: ``"word" : Category = lf_template @ learnable``.
    """
    wv = t.field(vid, "word")
    cv = t.field(vid, "category")
    lv = t.field(vid, "lf")
    if wv is None or cv is None or lv is None:
        raise ParseError(f"lexicon_entry malformed at {vid}")
    word_raw = t.text(wv)
    if word_raw.startswith('"') and word_raw.endswith('"'):
        word_raw = word_raw[1:-1]
    learnable_vid = t.field(vid, "learnable")
    learnable = learnable_vid is not None
    line, col = t.line_col(vid)
    return LexiconEntry(
        word=word_raw,
        category=_walk_type(t, cv),
        lf=_walk_let_arith(t, lv),
        learnable=learnable,
        line=line,
        col=col,
    )


def _walk_composition_rule_entry(t: _Tree, vid: str) -> CompositionRuleEntry:
    """Walk a single ``composition_rule_entry`` node.

    Two shapes:

    * ``key(p1, p2, …) = body`` — function-valued entry; the
      tuple of param names is non-empty.
    * ``key = body``           — value-valued entry; ``params``
      is empty and ``body`` is a literal-typed let-arith expression.
    """
    line, col = t.line_col(vid)
    key_vid = t.field(vid, "key")
    body_vid = t.field(vid, "body")
    if key_vid is None or body_vid is None:
        raise ParseError(f"composition_rule_entry missing key/body at {vid}")
    params = tuple(t.text(p_vid) for p_vid in t.fields(vid, "params"))
    return CompositionRuleEntry(
        key=t.text(key_vid),
        params=params,
        body=_walk_let_arith(t, body_vid),
        line=line,
        col=col,
    )


def _walk_contraction_decl(t: _Tree, vid: str, line: int, col: int) -> ContractionDecl:
    """Walk a ``contraction_decl`` node.

    Captures the input list (each entry a typed morphism arg),
    the output type ``domain -> codomain``, the rule name, and
    the wiring spec string.
    """
    name_vid = t.field(vid, "name")
    domain_vid = t.field(vid, "domain")
    codomain_vid = t.field(vid, "codomain")
    rule_vid = t.field(vid, "rule_name")
    if (
        name_vid is None
        or domain_vid is None
        or codomain_vid is None
        or rule_vid is None
    ):
        raise ParseError(f"contraction_decl missing required field at {vid}")
    inputs: list[ContractionInput] = []
    for inp_vid in t.fields(vid, "inputs"):
        in_name_vid = t.field(inp_vid, "name")
        in_dom_vid = t.field(inp_vid, "input_domain")
        in_cod_vid = t.field(inp_vid, "input_codomain")
        if in_name_vid is None or in_dom_vid is None or in_cod_vid is None:
            raise ParseError(f"contraction_input missing field at {inp_vid}")
        inp_line, inp_col = t.line_col(inp_vid)
        inputs.append(
            ContractionInput(
                name=t.text(in_name_vid),
                input_domain=_walk_type(t, in_dom_vid),
                input_codomain=_walk_type(t, in_cod_vid),
                line=inp_line,
                col=inp_col,
            )
        )
    # The optional wiring clause is one of two mutually-exclusive
    # forms: an einsum string literal (under
    # ``contraction_wiring_einsum``), or a ``share`` axis list
    # (under ``contraction_wiring_share``). The grammar emits at most
    # one as a child of the contraction_decl; both default to the
    # empty values when absent, leaving the type-driven inference
    # path to take over at compile time.
    wiring_text = ""
    shared_axes: list[str] = []
    wiring_vid = t.field(vid, "wiring")
    if wiring_vid is not None:
        wiring_kind = t.kind(wiring_vid)
        if wiring_kind == "contraction_wiring_einsum":
            spec_vid = t.field(wiring_vid, "wiring_spec")
            if spec_vid is not None:
                wiring_raw = t.text(spec_vid)
                if (
                    len(wiring_raw) >= 2
                    and wiring_raw[0] == '"'
                    and wiring_raw[-1] == '"'
                ):
                    wiring_text = wiring_raw[1:-1]
                else:
                    wiring_text = wiring_raw
        elif wiring_kind == "contraction_wiring_share":
            for axis_vid in t.fields(wiring_vid, "shared_axes"):
                shared_axes.append(t.text(axis_vid))
    return ContractionDecl(
        name=t.text(name_vid),
        inputs=tuple(inputs),
        domain=_walk_type(t, domain_vid),
        codomain=_walk_type(t, codomain_vid),
        rule_name=t.text(rule_vid),
        wiring_spec=wiring_text,
        shared_axes=tuple(shared_axes),
        line=line,
        col=col,
    )


def _walk_rule_decl(t: _Tree, vid: str, line: int, col: int) -> RuleDecl:
    nv = t.field(vid, "name")
    var_vids = t.fields(vid, "variables")
    prem_vids = t.fields(vid, "premises")
    concl_vid = t.field(vid, "conclusion")
    if concl_vid is None:
        raise ParseError(f"rule_decl missing conclusion at {vid}")
    return RuleDecl(
        name=_required_text(t, nv, vid, "name"),
        variables=tuple(t.text(v) for v in var_vids),
        premises=tuple(_walk_type(t, p) for p in prem_vids),
        conclusion=_walk_type(t, concl_vid),
        line=line,
        col=col,
    )


def _walk_object_initializer(t: _Tree, vid: str) -> EnumSetLiteral | FreeResiduatedExpr:
    k = t.kind(vid)
    line, col = t.line_col(vid)
    if k == "enum_set_literal":
        elem_vids = t.fields(vid, "elements")
        return EnumSetLiteral(
            elements=tuple(t.text(e) for e in elem_vids),
            line=line,
            col=col,
        )
    if k == "free_monoid_expr":
        gen_vid = t.field(vid, "generators")
        ml_vid = t.field(vid, "max_length")
        if gen_vid is None or ml_vid is None:
            raise ParseError(f"free_monoid_expr missing generators/max_length at {vid}")
        return FreeMonoidExpr(
            generators=t.text(gen_vid),
            max_length=int(t.text(ml_vid)),
            line=line,
            col=col,
        )
    if k == "free_residuated_expr":
        gen_vid = t.field(vid, "generators")
        if gen_vid is None:
            raise ParseError(f"free_residuated_expr missing generators at {vid}")
        depth = 1
        ops: list[str] = []
        # The grammar's free_residuated_arg variants carry one of two
        # field-tagged children: a depth integer or per-op identifier(s).
        for arg_vid in t.positional(vid):
            if t.kind(arg_vid) != "free_residuated_arg":
                continue
            d = t.field(arg_vid, "depth")
            if d is not None:
                depth = int(t.text(d))
                continue
            for op_vid in t.fields(arg_vid, "op"):
                ops.append(t.text(op_vid))
        if not ops:
            ops = ["slash"]
        return FreeResiduatedExpr(
            generators=t.text(gen_vid),
            depth=depth,
            ops=tuple(ops),
            line=line,
            col=col,
        )
    raise ParseError(f"unexpected object_initializer kind: {k}")


def _walk_schema_decl(t: _Tree, vid: str, line: int, col: int) -> SchemaDecl:
    nv = t.field(vid, "name")
    param_vids = t.fields(vid, "parameters")
    dom_vid = t.field(vid, "domain")
    cod_vid = t.field(vid, "codomain")
    if dom_vid is None or cod_vid is None:
        raise ParseError(f"schema_decl missing domain/codomain at {vid}")
    param_names: list[tuple[str, ...]] = []
    param_types: list[TypeExpr] = []
    for pv in param_vids:
        name_vids = t.fields(pv, "names")
        type_vid = t.field(pv, "type")
        if type_vid is None:
            raise ParseError(f"schema_parameter missing type at {pv}")
        param_names.append(tuple(t.text(n) for n in name_vids))
        param_types.append(_walk_type(t, type_vid))
    return SchemaDecl(
        name=_required_text(t, nv, vid, "name"),
        parameter_names=tuple(param_names),
        parameter_types=tuple(param_types),
        domain=_walk_type(t, dom_vid),
        codomain=_walk_type(t, cod_vid),
        line=line,
        col=col,
    )



def _walk_signature_decl(t: _Tree, vid: str, line: int, col: int) -> SignatureDecl:
    name_vid = t.field(vid, "name")
    if name_vid is None:
        raise ParseError(f"signature_decl missing name at {vid}")
    name = t.text(name_vid)
    params = tuple(t.text(c) for c in t.fields(vid, "params"))

    sorts: list[SortDecl] = []
    constructors: list[ConstructorDecl] = []
    binders: list[BinderDecl] = []
    vertex_kinds: list[VertexKindDecl] = []
    edge_kinds: list[EdgeKindDecl] = []

    for child in t.positional(vid):
        ck = t.kind(child)
        if ck == "signature_sorts":
            for s in t.fields(child, "sorts"):
                sorts.append(_walk_sort_decl(t, s))
        elif ck == "signature_constructors":
            for c in t.fields(child, "constructors"):
                constructors.append(_walk_constructor_decl(t, c))
        elif ck == "signature_binders":
            for b in t.fields(child, "binders"):
                binders.append(_walk_binder_decl(t, b))
        elif ck == "signature_vertex_kinds":
            for v in t.fields(child, "vertex_kinds"):
                vertex_kinds.append(_walk_vertex_kind_decl(t, v))
        elif ck == "signature_edge_kinds":
            for e in t.fields(child, "edge_kinds"):
                edge_kinds.append(_walk_edge_kind_decl(t, e))

    return SignatureDecl(
        name=name,
        params=params,
        sorts=tuple(sorts),
        constructors=tuple(constructors),
        binders=tuple(binders),
        vertex_kinds=tuple(vertex_kinds),
        edge_kinds=tuple(edge_kinds),
        line=line,
        col=col,
    )


def _walk_sort_decl(t: _Tree, vid: str) -> SortDecl:
    name = t.text(t.field(vid, "name"))
    kind_vid = t.field(vid, "kind")
    if kind_vid is None:
        raise ParseError(f"sort_decl missing kind at {vid}")
    kind_txt = t.text(kind_vid)
    dim_vid = t.field(vid, "dim")
    dim = int(t.text(dim_vid)) if dim_vid is not None else None
    vocab: list[SortVocabLiteral] = []
    for vlit_vid in t.fields(vid, "vocab"):
        # `vocab_literal` is a wrapper choice over string/integer/
        # float; descend to the inner concrete literal node.
        inner_vids = [
            c
            for c in t.positional(vlit_vid)
            if t.kind(c) in ("string", "integer", "float")
        ]
        if len(inner_vids) != 1:
            raise ParseError(
                f"sort_decl {name!r}: vocabulary literal at {vlit_vid} is "
                f"malformed (expected one of string/integer/float, got "
                f"{[t.kind(c) for c in inner_vids]!r})"
            )
        inner = inner_vids[0]
        lit_kind = t.kind(inner)
        vocab.append(SortVocabLiteral(kind=lit_kind, text=t.text(inner)))
    ln, cl = t.line_col(vid)
    return SortDecl(
        name=name,
        kind=kind_txt,
        dim=dim,
        vocab=tuple(vocab),
        line=ln,
        col=cl,
    )


def _walk_constructor_decl(t: _Tree, vid: str) -> ConstructorDecl:
    name = t.text(t.field(vid, "name"))
    domain_vids = t.fields(vid, "domain")
    domain = tuple(t.text(d) for d in domain_vids)
    codomain = t.text(t.field(vid, "codomain"))
    ln, cl = t.line_col(vid)
    return ConstructorDecl(
        name=name,
        domain=domain,
        codomain=codomain,
        line=ln,
        col=cl,
    )


def _walk_binder_decl(t: _Tree, vid: str) -> BinderDecl:
    name = t.text(t.field(vid, "name"))
    binds_list: list[BinderVar] = []
    for b in t.fields(vid, "binds"):
        annot_vid = t.field(b, "annot")
        annot_sort_vid = t.field(b, "annot_sort")
        binds_list.append(
            BinderVar(
                var=t.text(t.field(b, "var")),
                sort=t.text(t.field(b, "sort")),
                annot=t.text(annot_vid) if annot_vid is not None else None,
                annot_sort=(
                    t.text(annot_sort_vid) if annot_sort_vid is not None else None
                ),
            )
        )
    binds = tuple(binds_list)
    scoped = tuple(
        BinderArg(
            arg=t.text(t.field(a, "arg")),
            sort=t.text(t.field(a, "sort")),
        )
        for a in t.fields(vid, "scoped")
    )
    codomain = t.text(t.field(vid, "codomain"))
    ln, cl = t.line_col(vid)
    return BinderDecl(
        name=name,
        binds=binds,
        scoped=scoped,
        codomain=codomain,
        line=ln,
        col=cl,
    )


def _walk_vertex_kind_decl(t: _Tree, vid: str) -> VertexKindDecl:
    name = t.text(t.field(vid, "name"))
    kind_vid = t.field(vid, "kind")
    if kind_vid is None:
        raise ParseError(f"vertex_kind_decl missing kind at {vid}")
    kind_txt = t.text(kind_vid)
    dim_vid = t.field(vid, "dim")
    dim = int(t.text(dim_vid)) if dim_vid is not None else None
    ln, cl = t.line_col(vid)
    return VertexKindDecl(name=name, kind=kind_txt, dim=dim, line=ln, col=cl)


def _walk_edge_kind_decl(t: _Tree, vid: str) -> EdgeKindDecl:
    name = t.text(t.field(vid, "name"))
    src = t.text(t.field(vid, "src"))
    tgt = t.text(t.field(vid, "tgt"))
    arrow_vid = t.field(vid, "arrow")
    if arrow_vid is None:
        raise ParseError(f"edge_kind_decl missing arrow at {vid}")
    arrow_txt = t.text(arrow_vid)
    if arrow_txt == "->":
        directed = True
    elif arrow_txt == "--":
        directed = False
    else:
        raise ParseError(f"edge_kind_decl at {vid}: unknown arrow {arrow_txt!r}")
    ln, cl = t.line_col(vid)
    return EdgeKindDecl(
        name=name,
        src=src,
        tgt=tgt,
        directed=directed,
        line=ln,
        col=cl,
    )


def _walk_encoder_decl(t: _Tree, vid: str, line: int, col: int) -> EncoderDecl:
    name = t.text(t.field(vid, "name"))
    signature = t.text(t.field(vid, "signature"))
    sig_args = tuple(t.text(c) for c in t.fields(vid, "sig_args"))

    dims: list[SortDim] = []
    op_rules: list[EncoderRule] = []
    init_rules: list[EncoderInitRule] = []
    message_rules: list[EncoderMessageRule] = []
    update_rules: list[EncoderUpdateRule] = []
    iterations: int | None = None
    readout: LetExprNode | None = None
    var_inits: list[EncoderVarInit] = []

    for child in t.positional(vid):
        ck = t.kind(child)
        if ck == "encoder_dim":
            sort = t.text(t.field(child, "sort"))
            dim = int(t.text(t.field(child, "dim")))
            dims.append(SortDim(sort=sort, dim=dim))
        elif ck == "encoder_iterations":
            iterations = int(t.text(t.field(child, "iterations")))
        elif ck == "encoder_readout":
            readout = _walk_let_arith(t, t.field(child, "body"))
        elif ck == "encoder_op_rule":
            op = t.text(t.field(child, "op"))
            args = tuple(t.text(a) for a in t.fields(child, "args"))
            body = _walk_let_arith(t, t.field(child, "body"))
            state_v = t.field(child, "state")
            prefix_v = t.field(child, "prefix")
            mode = "plain"
            state_var = None
            prefix_var = None
            if state_v is not None:
                mode = "recurrent"
                state_var = t.text(state_v)
            elif prefix_v is not None:
                mode = "attention"
                prefix_var = t.text(prefix_v)
            cln, ccl = t.line_col(child)
            op_rules.append(
                EncoderRule(
                    op=op,
                    args=args,
                    body=body,
                    mode=mode,
                    state_var=state_var,
                    prefix_var=prefix_var,
                    line=cln,
                    col=ccl,
                )
            )
        elif ck == "encoder_init_rule":
            init_rules.append(
                EncoderInitRule(
                    kind=t.text(t.field(child, "kind")),
                    arg=t.text(t.field(child, "arg")),
                    body=_walk_let_arith(t, t.field(child, "body")),
                )
            )
        elif ck == "encoder_message_rule":
            message_rules.append(
                EncoderMessageRule(
                    edge_kind=t.text(t.field(child, "edge_kind")),
                    src=t.text(t.field(child, "src")),
                    tgt=t.text(t.field(child, "tgt")),
                    body=_walk_let_arith(t, t.field(child, "body")),
                )
            )
        elif ck == "encoder_update_rule":
            update_rules.append(
                EncoderUpdateRule(
                    vertex_kind=t.text(t.field(child, "vertex_kind")),
                    self_var=t.text(t.field(child, "self")),
                    msgs_var=t.text(t.field(child, "msgs")),
                    body=_walk_let_arith(t, t.field(child, "body")),
                )
            )
        elif ck == "encoder_var_init":
            vs_vid = t.field(child, "var_sort")
            if vs_vid is None:
                raise ParseError(f"encoder_var_init at {child} missing var_sort")
            annot_vid = t.field(child, "annot_sort")
            ty_vid = t.field(child, "ty")
            ln, cl = t.line_col(child)
            var_inits.append(
                EncoderVarInit(
                    var_sort=t.text(vs_vid),
                    annot_sort=(t.text(annot_vid) if annot_vid is not None else None),
                    ty=t.text(ty_vid) if ty_vid is not None else None,
                    body=_walk_let_arith(t, t.field(child, "body")),
                    line=ln,
                    col=cl,
                )
            )

    # Optional factory clause. ``using <factory> [k=v, ...]`` is an
    # alternative to the per-rule body; the parser emits a
    # ``factory`` field plus an optional ``factory_options`` block.
    factory = ""
    factory_options: dict[str, str] = {}
    factory_vid = t.field(vid, "factory")
    if factory_vid is not None:
        factory = t.text(factory_vid)
    fo_vid = t.field(vid, "factory_options")
    if fo_vid is not None:
        for entry_vid in t.positional(fo_vid):
            if t.kind(entry_vid) != "option_entry":
                continue
            k = t.text(t.field(entry_vid, "key"))
            v = t.text(t.field(entry_vid, "value"))
            factory_options[k] = v

    return EncoderDecl(
        name=name,
        signature=signature,
        sig_args=sig_args,
        dims=tuple(dims),
        op_rules=tuple(op_rules),
        init_rules=tuple(init_rules),
        message_rules=tuple(message_rules),
        update_rules=tuple(update_rules),
        iterations=iterations,
        readout=readout,
        var_inits=tuple(var_inits),
        factory=factory,
        factory_options=factory_options,
        line=line,
        col=col,
    )


def _walk_decoder_decl(t: _Tree, vid: str, line: int, col: int) -> DecoderDecl:
    name = t.text(t.field(vid, "name"))
    signature = t.text(t.field(vid, "signature"))
    sig_args = tuple(t.text(c) for c in t.fields(vid, "sig_args"))
    depth_vid = t.field(vid, "depth")
    depth = int(t.text(depth_vid)) if depth_vid is not None else 8

    dims: list[SortDim] = []
    structure_body: LetExprNode | None = None
    structure_arg: str | None = None
    primitive_body: LetExprNode | None = None
    primitive_arg: str | None = None
    factor_body: LetExprNode | None = None
    factor_arg: str | None = None
    binder_body: LetExprNode | None = None
    binder_arg: str | None = None
    recursive_default = False

    for child in t.positional(vid):
        ck = t.kind(child)
        if ck == "decoder_dim":
            sort = t.text(t.field(child, "sort"))
            dim = int(t.text(t.field(child, "dim")))
            dims.append(SortDim(sort=sort, dim=dim))
        elif ck == "decoder_structure":
            structure_arg = t.text(t.field(child, "arg"))
            structure_body = _walk_let_arith(t, t.field(child, "body"))
        elif ck == "decoder_primitive":
            primitive_arg = t.text(t.field(child, "arg"))
            primitive_body = _walk_let_arith(t, t.field(child, "body"))
        elif ck == "decoder_factor":
            factor_arg = t.text(t.field(child, "arg"))
            factor_body = _walk_let_arith(t, t.field(child, "body"))
        elif ck == "decoder_binder_select":
            binder_arg = t.text(t.field(child, "arg"))
            binder_body = _walk_let_arith(t, t.field(child, "body"))
        elif ck == "decoder_body_default":
            recursive_default = True

    return DecoderDecl(
        name=name,
        signature=signature,
        sig_args=sig_args,
        depth=depth,
        dims=tuple(dims),
        structure=structure_body,
        structure_arg=structure_arg,
        primitive=primitive_body,
        primitive_arg=primitive_arg,
        factor=factor_body,
        factor_arg=factor_arg,
        binder_select=binder_body,
        binder_select_arg=binder_arg,
        recursive_default=recursive_default,
        line=line,
        col=col,
    )


def _walk_loss_decl(t: _Tree, vid: str, line: int, col: int) -> LossDecl:
    name = t.text(t.field(vid, "name"))
    weight_vid = t.field(vid, "weight")
    weight = _walk_let_arith(t, weight_vid) if weight_vid is not None else None
    body = _walk_let_arith(t, t.field(vid, "body"))

    attachment = LossAttachment(attachment_kind="global")
    att_vid = t.field(vid, "attachment")
    if att_vid is not None:
        kind_vid = t.field(att_vid, "kind")
        tgt_vid = t.field(att_vid, "target")
        rule_vid = t.field(att_vid, "rule_name")
        ded_vid = t.field(att_vid, "deduction")
        chart_vid = t.field(att_vid, "chart_of")
        if rule_vid is not None and ded_vid is not None:
            attachment = LossAttachment(
                attachment_kind="rule",
                target=t.text(rule_vid),
                rule_deduction=t.text(ded_vid),
            )
        elif chart_vid is not None:
            attachment = LossAttachment(
                attachment_kind="chart",
                target=t.text(chart_vid),
            )
        elif tgt_vid is not None:
            if kind_vid is None:
                raise ParseError(f"loss_attachment at {att_vid}: missing kind")
            k = t.text(kind_vid)
            if k not in ("program", "deduction", "encoder", "decoder"):
                raise ParseError(f"loss_attachment at {att_vid}: unknown kind {k!r}")
            attachment = LossAttachment(
                attachment_kind=k,  # type: ignore[arg-type]
                target=t.text(tgt_vid),
            )

    return LossDecl(
        name=name,
        weight=weight,
        attachment=attachment,
        body=body,
        line=line,
        col=col,
    )
