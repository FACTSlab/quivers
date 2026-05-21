"""One-hop migrator: v0.9.0 source to HEAD source.

The homogenization hop. Per-declaration converters cover the rule
renames and surface-shape changes between v0.9.0 and HEAD:

* ``object_decl`` -> ``type X : FinSet N`` (or the unchanged
  expression body under the new keyword).
* ``type_alias_decl`` / ``space_decl`` / ``alias_decl`` -> ``type X : V``
  with ``Euclidean`` -> ``Real`` and bareword constructor canonicalised
  to parens.
* ``morphism_decl`` (latent/observed) / ``kernel_decl`` / ``embed_decl``
  / ``discretize_decl`` -> ``morphism X : A -> B [role=..., ...]``.
* ``algebra_decl`` / ``semigroupoid`` / ``bilinear_form`` /
  ``composition_rule`` -> ``composition X at <level>``.
* ``program_decl`` -> indented colon-block with effects hoisted into
  the option block; bare draw steps gain ``sample``.
* ``deduction_decl`` -> indented colon-block with brace bodies
  flattened into HEAD's atoms / rule / lexicon forms; metadata
  (semiring / start / depth / signature / encoder) hoists to the
  header option block.

Each converter returns target-revision source text for its
declaration; `migrate_source` validates the text by parsing
it through HEAD and concatenates. Grammar binding lives at the
per-decl validate step and at the final whole-file parse.
"""

from __future__ import annotations

import re

from quivers.cli.migrations._common import (
    DeclConverter,
    SchemaView,
    migrate_source,
)


_FLATTEN_NEWLINE_RE = re.compile(r"\n[ \t]*")


def _flatten_inline(text: str) -> str:
    return _FLATTEN_NEWLINE_RE.sub(" ", text).strip()


# ---------------------------------------------------------------------------
# Type-expression rendering
# ---------------------------------------------------------------------------


_SPACE_CTOR_RENAME: dict[str, str] = {"Euclidean": "Real"}


def _space_arg_text(view: SchemaView, arg_vid: str) -> str:
    if view.kind(arg_vid) == "space_kwarg":
        key_vid = view.field(arg_vid, "key")
        val_vid = view.field(arg_vid, "value")
        key = view.text(key_vid) if key_vid else ""
        val = view.text(val_vid) if val_vid else ""
        return f"{key}={val}"
    return view.text(arg_vid)


def _type_expr_text(view: SchemaView, type_vid: str) -> str:
    """Render a v0.9.0 ``_object_expr`` / ``_space_expr`` / object
    initializer subtree as HEAD-compatible source text."""
    kind = view.kind(type_vid)
    if kind == "type_atom":
        kids = view.positional(type_vid)
        if kids and view.kind(kids[0]) == "integer":
            return f"FinSet {view.text(kids[0])}"
        if kids:
            return view.text(kids[0])
        return view.text(type_vid)
    if kind == "space_atom":
        kids = view.positional(type_vid)
        return view.text(kids[0]) if kids else view.text(type_vid)
    if kind == "space_constructor_bare":
        ctor_vid = view.field(type_vid, "constructor")
        arg_vid = view.field(type_vid, "arg")
        ctor = view.text(ctor_vid) if ctor_vid else "Real"
        ctor = _SPACE_CTOR_RENAME.get(ctor, ctor)
        arg_text = view.text(arg_vid) if arg_vid else "0"
        return f"{ctor} {arg_text}"
    if kind == "space_constructor":
        ctor_vid = view.field(type_vid, "constructor")
        ctor = view.text(ctor_vid) if ctor_vid else "Real"
        ctor = _SPACE_CTOR_RENAME.get(ctor, ctor)
        positional: list[str] = []
        kwargs: list[str] = []
        for av in view.fields(type_vid, "args"):
            ak = view.kind(av)
            if ak == "space_kwarg":
                key_vid = view.field(av, "key")
                val_vid = view.field(av, "value")
                key = view.text(key_vid) if key_vid else ""
                val = view.text(val_vid) if val_vid else ""
                kwargs.append(f"{key}={val}")
            else:
                positional.append(view.text(av))
        head = ctor + (" " + " ".join(positional) if positional else "")
        if kwargs:
            head += f" [{', '.join(kwargs)}]"
        return head
    if kind in ("space_product", "type_product"):
        left = view.field(type_vid, "left")
        right = view.field(type_vid, "right")
        left_s = _type_expr_text(view, left) if left else "?"
        right_s = _type_expr_text(view, right) if right else "?"
        return f"{left_s} * {right_s}"
    if kind == "type_coproduct":
        left = view.field(type_vid, "left")
        right = view.field(type_vid, "right")
        left_s = _type_expr_text(view, left) if left else "?"
        right_s = _type_expr_text(view, right) if right else "?"
        return f"{left_s} + {right_s}"
    if kind == "type_paren":
        kids = view.positional(type_vid)
        return f"({_type_expr_text(view, kids[0])})" if kids else "()"
    if kind == "type_slash":
        result = view.field(type_vid, "result")
        arg = view.field(type_vid, "argument")
        direction = view.field(type_vid, "direction")
        result_s = _type_expr_text(view, result) if result else "?"
        arg_s = _type_expr_text(view, arg) if arg else "?"
        direction_s = view.text(direction) if direction else "/"
        return f"{result_s} {direction_s} {arg_s}"
    if kind == "type_effect_apply":
        eff_vid = view.field(type_vid, "effect")
        args = view.fields(type_vid, "args")
        eff = view.text(eff_vid) if eff_vid else "?"
        args_s = ", ".join(_type_expr_text(view, av) for av in args)
        return f"{eff}({args_s})"
    if kind == "enum_set_literal":
        elements = [view.text(av) for av in view.fields(type_vid, "elements")]
        return "{" + ", ".join(elements) + "}"
    if kind == "free_residuated_expr":
        gens = view.field(type_vid, "generators")
        gens_s = view.text(gens) if gens else ""
        parts: list[str] = [gens_s]
        for arg_vid in view.fields(type_vid, "child_of"):
            parts.append(view.text(arg_vid))
        return f"FreeResiduated({', '.join(parts)})"
    if kind == "free_monoid_expr":
        gens = view.field(type_vid, "generators")
        max_len = view.field(type_vid, "max_length")
        gens_s = view.text(gens) if gens else ""
        max_s = view.text(max_len) if max_len else ""
        return f"FreeMonoid({gens_s}, max_length={max_s})"
    return view.text(type_vid)


# ---------------------------------------------------------------------------
# Type-level decls
# ---------------------------------------------------------------------------


def _convert_object_decl(view: SchemaView, src_vid: str) -> str:
    name_vid = view.field(src_vid, "name")
    type_vid = view.field(src_vid, "type")
    init_vid = view.field(src_vid, "init")
    name = view.text(name_vid) if name_vid else ""
    if type_vid is not None:
        return f"object {name} : {_type_expr_text(view, type_vid)}\n"
    if init_vid is not None:
        return f"object {name} : {_type_expr_text(view, init_vid)}\n"
    return f"object {name} : FinSet 0\n"


def _convert_type_alias_decl(view: SchemaView, src_vid: str) -> str:
    name_vid = view.field(src_vid, "name")
    value_vid = view.field(src_vid, "value")
    name = view.text(name_vid) if name_vid else ""
    if value_vid is None:
        return f"object {name} : Real(1)\n"
    return f"object {name} : {_type_expr_text(view, value_vid)}\n"


def _convert_space_decl(view: SchemaView, src_vid: str) -> str:
    return _convert_type_alias_decl(view, src_vid)


def _convert_alias_decl(view: SchemaView, src_vid: str) -> str:
    return _convert_type_alias_decl(view, src_vid)


# ---------------------------------------------------------------------------
# Option-block / axis-role helpers
# ---------------------------------------------------------------------------


def _option_block_text(view: SchemaView, options_vid: str | None) -> str:
    if options_vid is None:
        return ""
    parts: list[str] = []
    for entry_vid in view.positional(options_vid):
        if view.kind(entry_vid) != "option_entry":
            continue
        key_vid = view.field(entry_vid, "key")
        val_vid = view.field(entry_vid, "value")
        if key_vid is None:
            continue
        key = view.text(key_vid)
        if val_vid is None:
            parts.append(key)
        else:
            parts.append(f"{key}={view.text(val_vid)}")
    return ", ".join(parts)


def _axis_list_text(view: SchemaView, axis_vid: str) -> str:
    if view.kind(axis_vid) == "axis_tuple":
        axes = [view.text(av) for av in view.fields(axis_vid, "axis")]
        return "[" + ", ".join(axes) + "]"
    return view.text(axis_vid)


def _axis_role_options(view: SchemaView, axes_vid: str | None) -> list[str]:
    if axes_vid is None:
        return []
    out: list[str] = []
    over = view.field(axes_vid, "over")
    if over is not None:
        out.append(f"over={_axis_list_text(view, over)}")
    iid_over = view.field(axes_vid, "iid_over")
    if iid_over is not None:
        out.append(f"iid_over={_axis_list_text(view, iid_over)}")
    return out


def _replicate_count_text(view: SchemaView, rep_vid: str | None) -> str | None:
    if rep_vid is None:
        return None
    for kid in view.positional(rep_vid):
        if view.kind(kid) == "integer":
            return view.text(kid)
    return None


# ---------------------------------------------------------------------------
# Morphism-family decls
# ---------------------------------------------------------------------------


def _morphism_text(
    view: SchemaView, src_vid: str, role: str,
) -> str:
    name_vid = view.field(src_vid, "name")
    domain_vid = view.field(src_vid, "domain")
    codomain_vid = view.field(src_vid, "codomain")
    options_vid = view.field(src_vid, "options")
    axes_vid = view.field(src_vid, "axes")
    family_vid = view.field(src_vid, "family")
    rep_vid = view.field(src_vid, "replicate")
    kind_vid = view.field(src_vid, "kind")

    if kind_vid is not None:
        kind_text = view.text(kind_vid)
        if kind_text in ("latent", "observed"):
            role = kind_text

    name = view.text(name_vid) if name_vid else ""
    domain = _type_expr_text(view, domain_vid) if domain_vid else ""
    codomain = _type_expr_text(view, codomain_vid) if codomain_vid else ""

    opts: list[str] = [f"role={role}"]
    rep = _replicate_count_text(view, rep_vid)
    if rep is not None:
        opts.append(f"repeat={rep}")
    existing = _option_block_text(view, options_vid)
    if existing:
        opts.append(existing)
    opts.extend(_axis_role_options(view, axes_vid))

    line = f"morphism {name} : {domain} -> {codomain} [{', '.join(opts)}]"
    if family_vid is not None:
        line += f" ~ {view.text(family_vid)}"
    return line + "\n"


def _convert_morphism_decl(view: SchemaView, src_vid: str) -> str:
    return _morphism_text(view, src_vid, "latent")


def _convert_kernel_decl(view: SchemaView, src_vid: str) -> str:
    return _morphism_text(view, src_vid, "kernel")


def _convert_embed_decl(view: SchemaView, src_vid: str) -> str:
    return _morphism_text(view, src_vid, "embed")


def _convert_discretize_decl(view: SchemaView, src_vid: str) -> str:
    name_vid = view.field(src_vid, "name")
    space_vid = view.field(src_vid, "space")
    bins_vid = view.field(src_vid, "bins")
    options_vid = view.field(src_vid, "options")
    name = view.text(name_vid) if name_vid else ""
    space = view.text(space_vid) if space_vid else ""
    bins = view.text(bins_vid) if bins_vid else "0"
    opts: list[str] = ["role=discretize", f"bins={bins}"]
    existing = _option_block_text(view, options_vid)
    if existing:
        opts.append(existing)
    return (
        f"morphism {name} : {space} -> _Bins_{bins} [{', '.join(opts)}]\n"
    )


# ---------------------------------------------------------------------------
# Algebra / composition
# ---------------------------------------------------------------------------


_ALGEBRA_LEVEL_MAP: dict[str, str] = {
    "algebra": "algebra",
    "semigroupoid": "semigroupoid",
    "bilinear_form": "bilinear_form",
    "composition_rule": "rule",
}


def _convert_algebra_decl(view: SchemaView, src_vid: str) -> str:
    name_vid = view.field(src_vid, "name")
    name = view.text(name_vid) if name_vid else ""
    raw = view.text(src_vid)
    keyword = raw.split(None, 1)[0] if raw else "algebra"
    level = _ALGEBRA_LEVEL_MAP.get(keyword, "algebra")
    return f"composition {name} at {level}\n"


# ---------------------------------------------------------------------------
# program_decl
# ---------------------------------------------------------------------------


def _draw_arg_text(view: SchemaView, arg_vid: str) -> str:
    return view.text(arg_vid).strip()


def _bind_step_text(view: SchemaView, step_vid: str) -> str:
    vars_vid = view.field(step_vid, "vars")
    index_vid = view.field(step_vid, "index")
    morphism_vid = view.field(step_vid, "morphism")
    args = view.fields(step_vid, "args")
    axes_vid = view.field(step_vid, "axes")

    vars_s = view.text(vars_vid) if vars_vid else "?"
    index_s = (
        f" : {_type_expr_text(view, index_vid)}"
        if index_vid is not None else ""
    )
    morphism = view.text(morphism_vid) if morphism_vid else "?"
    args_s = (
        "(" + ", ".join(_draw_arg_text(view, a) for a in args) + ")"
        if args else ""
    )
    out = f"sample {vars_s}{index_s} <- {morphism}{args_s}"
    axes_opts = _axis_role_options(view, axes_vid)
    if axes_opts:
        out += " [" + ", ".join(axes_opts) + "]"
    return out


def _observe_step_text(view: SchemaView, step_vid: str) -> str:
    var_vid = view.field(step_vid, "var")
    index_vid = view.field(step_vid, "index")
    via_vid = view.field(step_vid, "via")
    morphism_vid = view.field(step_vid, "morphism")
    args = view.fields(step_vid, "args")
    axes_vid = view.field(step_vid, "axes")

    var_s = view.text(var_vid) if var_vid else "?"
    index_s = (
        f" : {_type_expr_text(view, index_vid)}"
        if index_vid is not None else ""
    )
    morphism = view.text(morphism_vid) if morphism_vid else "?"
    args_s = (
        "(" + ", ".join(_draw_arg_text(view, a) for a in args) + ")"
        if args else ""
    )
    opts: list[str] = []
    if via_vid is not None:
        # HEAD's observe_step dropped the trailing ``via <spec>``
        # keyword and accepts the fibration as a named option
        # instead.
        opts.append(f"via={view.text(via_vid)}")
    opts.extend(_axis_role_options(view, axes_vid))
    opts_str = " [" + ", ".join(opts) + "]" if opts else ""
    return f"observe {var_s}{index_s} <- {morphism}{args_s}{opts_str}"


def _let_step_text(view: SchemaView, step_vid: str) -> str:
    name_vid = view.field(step_vid, "name")
    value_vid = view.field(step_vid, "value")
    name = view.text(name_vid) if name_vid else "?"
    raw = view.text(value_vid) if value_vid else ""
    return f"let {name} = {_flatten_inline(raw)}"


def _marginalize_step_text(
    view: SchemaView, step_vid: str, indent: str,
) -> str:
    var_vid = view.field(step_vid, "var")
    index_vid = view.field(step_vid, "index")
    morphism_vid = view.field(step_vid, "morphism")
    args = view.fields(step_vid, "args")
    over_vid = view.field(step_vid, "over")
    reduction_vid = view.field(step_vid, "reduction")

    var_s = view.text(var_vid) if var_vid else "?"
    index_s = (
        f" : {_type_expr_text(view, index_vid)}"
        if index_vid is not None else ""
    )
    morphism = view.text(morphism_vid) if morphism_vid else "?"
    args_s = (
        "(" + ", ".join(_draw_arg_text(view, a) for a in args) + ")"
        if args else ""
    )
    opts: list[str] = []
    if over_vid is not None:
        opts.append(f"over={_type_expr_text(view, over_vid)}")
    if reduction_vid is not None:
        opts.append(f"reduction={view.text(reduction_vid)}")
    opts_str = f" [{', '.join(opts)}]" if opts else ""

    header = (
        f"marginalize {var_s}{index_s} <- {morphism}{args_s}{opts_str}"
    )
    inner_indent = indent + "    "
    HEADER = frozenset({
        "var", "index", "morphism", "args", "over",
        "reduction", "options",
    })
    body_lines: list[str] = []
    for _edge_kind, child_vid in view.body_children(step_vid, HEADER):
        child_kind = view.kind(child_vid)
        if child_kind in ("line_comment", "doc_comment"):
            body_lines.append(inner_indent + view.text(child_vid).rstrip())
            continue
        body_lines.append(
            inner_indent + _program_step_text(view, child_vid, inner_indent),
        )
    body = "\n".join(body_lines)
    return header + ("\n" + body if body else "")


def _program_step_text(
    view: SchemaView, step_vid: str, indent: str = "",
) -> str:
    kind = view.kind(step_vid)
    if kind == "bind_step":
        return _bind_step_text(view, step_vid)
    if kind == "observe_step":
        return _observe_step_text(view, step_vid)
    if kind == "let_step":
        return _let_step_text(view, step_vid)
    if kind == "marginalize_step":
        return _marginalize_step_text(view, step_vid, indent)
    return _flatten_inline(view.text(step_vid))


def _convert_program_decl(view: SchemaView, src_vid: str) -> str:
    name_vid = view.field(src_vid, "name")
    domain_vid = view.field(src_vid, "domain")
    codomain_vid = view.field(src_vid, "codomain")
    over_model_vid = view.field(src_vid, "over_model")
    params = view.fields(src_vid, "params")
    effects = view.fields(src_vid, "effects")

    name = view.text(name_vid) if name_vid else ""
    params_str = (
        "(" + ", ".join(view.text(p) for p in params) + ")"
        if params else ""
    )
    domain = _type_expr_text(view, domain_vid) if domain_vid else ""
    codomain = _type_expr_text(view, codomain_vid) if codomain_vid else ""

    opts: list[str] = []
    if effects:
        items = [view.text(e) for e in effects]
        opts.append("effects=[" + ", ".join(items) + "]")
    if over_model_vid is not None:
        opts.append(f"over_model={view.text(over_model_vid)}")
    opts_str = f" [{', '.join(opts)}]" if opts else ""

    header = (
        f"program {name}{params_str} : {domain} -> {codomain}{opts_str}"
    )

    # Walk every non-header child in document order; this picks up
    # ``steps`` (program steps), ``return`` (the return step), and
    # any ``child_of`` extras (``line_comment``/``doc_comment``)
    # the parser attached to the program_decl vertex.
    HEADER = frozenset({
        "docs", "name", "params", "domain", "codomain",
        "options", "over_model",
    })
    body_lines: list[str] = []
    for edge_kind, child_vid in view.body_children(src_vid, HEADER):
        child_kind = view.kind(child_vid)
        if child_kind in ("line_comment", "doc_comment"):
            body_lines.append("    " + view.text(child_vid).rstrip())
            continue
        if edge_kind == "return":
            body_lines.append(
                "    return " + _flatten_inline(view.text(child_vid)),
            )
            continue
        body_lines.append(
            "    " + _program_step_text(view, child_vid, "    "),
        )
    return header + "\n" + "\n".join(body_lines) + "\n"


# ---------------------------------------------------------------------------
# deduction_decl
# ---------------------------------------------------------------------------


def _lexicon_entry_text(view: SchemaView, entry_vid: str) -> str:
    word_vid = view.field(entry_vid, "word")
    cat_vid = view.field(entry_vid, "category")
    lf_vid = view.field(entry_vid, "lf")
    learnable_vid = view.field(entry_vid, "learnable")
    word = view.text(word_vid) if word_vid else "\"\""
    cat = _type_expr_text(view, cat_vid) if cat_vid else ""
    lf = _flatten_inline(view.text(lf_vid)) if lf_vid else ""
    out = f"{word} : {cat} = {lf}"
    if learnable_vid is not None:
        out += " [learnable]"
    return out


def _convert_deduction_decl(view: SchemaView, src_vid: str) -> str:
    name_vid = view.field(src_vid, "name")
    domain_vid = view.field(src_vid, "domain")
    codomain_vid = view.field(src_vid, "codomain")
    name = view.text(name_vid) if name_vid else ""
    domain = _type_expr_text(view, domain_vid) if domain_vid else ""
    codomain = _type_expr_text(view, codomain_vid) if codomain_vid else ""

    opts: list[str] = []
    body_lines: list[str] = []
    HEADER = frozenset({"docs", "name", "domain", "codomain", "options"})
    for _edge_kind, child_vid in view.body_children(src_vid, HEADER):
        ck = view.kind(child_vid)
        if ck in ("line_comment", "doc_comment"):
            body_lines.append("    " + view.text(child_vid).rstrip())
            continue
        if ck == "deduction_atoms":
            atoms = [view.text(av) for av in view.fields(child_vid, "atoms")]
            body_lines.append("    atoms " + ", ".join(atoms))
        elif ck == "deduction_rule":
            rname_vid = view.field(child_vid, "name")
            prems = view.fields(child_vid, "premises")
            concl_vid = view.field(child_vid, "conclusion")
            rname = view.text(rname_vid) if rname_vid else "?"
            prems_s = ", ".join(_type_expr_text(view, p) for p in prems)
            concl_s = (
                _type_expr_text(view, concl_vid) if concl_vid else "?"
            )
            body_lines.append(f"    rule {rname} : {prems_s} |- {concl_s}")
        elif ck == "deduction_semiring":
            v = view.field(child_vid, "semiring")
            if v is not None:
                opts.append(f"semiring={view.text(v)}")
        elif ck == "deduction_start":
            v = view.field(child_vid, "start")
            if v is not None:
                opts.append(f"start={view.text(v)}")
        elif ck == "deduction_depth":
            v = view.field(child_vid, "depth")
            if v is not None:
                opts.append(f"depth={view.text(v)}")
        elif ck == "deduction_signature":
            v = view.field(child_vid, "signature")
            if v is not None:
                opts.append(f"signature={view.text(v)}")
        elif ck == "deduction_encoder_attach":
            v = view.field(child_vid, "encoder")
            if v is not None:
                opts.append(f"encoder={view.text(v)}")
        elif ck == "deduction_lexicon":
            body_lines.append("    lexicon")
            LEX_HEADER: frozenset[str] = frozenset()
            for _e, entry_vid in view.body_children(child_vid, LEX_HEADER):
                ek = view.kind(entry_vid)
                if ek in ("line_comment", "doc_comment"):
                    body_lines.append(
                        "        " + view.text(entry_vid).rstrip(),
                    )
                    continue
                if ek == "lexicon_entry":
                    body_lines.append(
                        "        " + _lexicon_entry_text(view, entry_vid),
                    )
        elif ck == "deduction_lexicon_from_file":
            body_lines.append("    " + view.text(child_vid).strip())

    opts_str = f" [{', '.join(opts)}]" if opts else ""
    header = f"deduction {name} : {domain} -> {codomain}{opts_str}"
    return header + "\n" + "\n".join(body_lines) + "\n"


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


_DECL_CONVERTERS: dict[str, DeclConverter] = {
    "object_decl": _convert_object_decl,
    "type_alias_decl": _convert_type_alias_decl,
    "space_decl": _convert_space_decl,
    "alias_decl": _convert_alias_decl,
    "morphism_decl": _convert_morphism_decl,
    "kernel_decl": _convert_kernel_decl,
    "embed_decl": _convert_embed_decl,
    "discretize_decl": _convert_discretize_decl,
    "algebra_decl": _convert_algebra_decl,
    "program_decl": _convert_program_decl,
    "deduction_decl": _convert_deduction_decl,
}

# Source-side rule names this hop semantically translates. Includes
# top-level decl kinds dispatched via ``_DECL_CONVERTERS`` AND
# sub-rules consumed internally by those converters (e.g. an
# ``axis_role_clause`` doesn't appear as a top-level dispatch entry
# but is read by the morphism / sample / observe converters and
# rewritten as ``[over=..., iid_over=...]`` option entries).
#
# Consumed by [`quivers.cli.migrations._vcs.check_chain_coverage`][quivers.cli.migrations._vcs.check_chain_coverage]
# to verify that every rule removed in the panproto schema diff
# between this hop's source and target has a corresponding handler
# here.
SOURCE_RULE_COVERAGE: frozenset[str] = frozenset(_DECL_CONVERTERS.keys()) | frozenset({
    # Type-expression subtree: the migrator's ``_type_expr_text``
    # walks and re-emits every shape under the homogenized names.
    "type_atom", "type_paren", "type_product", "type_coproduct",
    "type_slash", "type_effect_apply",
    "space_atom", "space_constructor", "space_constructor_bare",
    "space_product", "space_kwarg", "space_decl",
    # Object-initializer subtree (used inside ``object_decl`` body).
    "_object_initializer",
    # Type-alias decl is handled via `_convert_type_alias_decl`.
    "type_alias_decl", "alias_decl",
    # Morphism family: latent/observed/kernel/embed/discretize
    # collapse onto ``morphism_decl [role=...]``.
    "kernel_decl", "embed_decl", "discretize_decl",
    # Algebra/semigroupoid/bilinear_form/composition_rule -> ``composition``.
    "algebra_decl",
    # Axis-role clause: hoisted into the option block as
    # ``[over=..., iid_over=...]`` by the bind/observe/marginalize
    # / morphism emitters.
    "axis_role_clause", "axis_tuple", "_axis_list",
    # bind_step gains a ``sample`` keyword (becomes ``sample_step``
    # in HEAD vocabulary).
    "bind_step",
    # Via-spec on observe_step encoded as ``[via=...]`` option.
    "_via_spec", "via_product",
    # Replicate count is hoisted to ``[repeat=N]``.
    "replicate_count",
    # Deduction body metadata hoisted into the header option block.
    "deduction_semiring", "deduction_start", "deduction_depth",
    "deduction_signature", "deduction_encoder_attach",
    "deduction_axioms",
    # ``@ learnable`` lexicon marker -> ``[learnable]`` option.
    "learnable_marker",
    # Composition-rule body (inline ``{tensor_op = ..., join = ..., unit = ...}``)
    # preserved verbatim in the algebra_decl converter.
    "composition_rule_block",
    # Contraction wiring annotations (rare). Currently passed through
    # via the option block.
    "contraction_wiring_einsum", "contraction_wiring_share",
    # Morphism prior clause (``~ Family(args) [options]``) absorbed
    # into morphism_decl's ``~`` initializer.
    "morphism_prior",
    # Loss attachment metadata (rare).
    "loss_attachment", "loss_attachment_kind",
    # Helper anonymous types referenced only inside their parent.
    "_type_expr", "_space_expr", "_space_arg",
})


def migrate(source: bytes) -> bytes:
    # v0.10.0's tree-sitter grammar.js is byte-identical to v0.9.0's,
    # so we parse with the v0.9.0 parser. v0.11.0 is the upcoming
    # tag for the homogenized HEAD grammar; until tagged it lives at
    # the HEAD parser, so we emit through that.
    return migrate_source(source, "v0.9.0", "HEAD", _DECL_CONVERTERS)
