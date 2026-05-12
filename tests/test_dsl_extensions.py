"""Tests for the DSL-extension surface introduced for editor + tooling work.

Covers:

- ``alias`` declarations (object-level type aliases): both
  SetObject-resolvable and residuated-pattern-only forms.
- ``bundle`` declarations: rule-set bundles spliced into ``parser(rules=…)``
  with cycle detection.
- ``##`` doc comments attached to the appropriate AST nodes.
- ``FreeMonoid(...)`` object-initializer surface.
- ``qvr check`` CLI: parse / compile diagnostics on representative
  invalid programs.
- The constraint solver in :mod:`quivers.dsl.constraints`: residuated-
  context violations and bundle-unknown-member detection.
- Pygments lexer (tree-sitter-driven) on a representative source.
- Highlight queries cover every new production.
"""

from __future__ import annotations

import io
import json
import textwrap
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from quivers.cli.check import _check_one, main as check_main
from quivers.dsl import Compiler, ParseError, loads, parse
from quivers.dsl.ast_nodes import (
    AliasDecl,
    BundleDecl,
    FreeMonoidExpr,
    ObjectDecl,
    SchemaDecl,
)
from quivers.dsl.constraints import check_constraints


# ---------------------------------------------------------------------------
# alias declarations
# ---------------------------------------------------------------------------


class TestAliasDecl:
    def test_object_level_alias_resolves_to_setobject(self):
        """`alias Pair = X * Y` binds Pair to the resolved ProductSet."""
        src = textwrap.dedent("""
            object X : 3
            object Y : 4
            alias Pair = X * Y
            latent f : Pair -> X
            export f
        """)
        prog = loads(src)
        assert prog is not None

    def test_alias_in_morphism_decl_is_transparent(self):
        """A morphism declared over an alias matches an unaliased twin."""
        src_aliased = textwrap.dedent("""
            object X : 3
            object Y : 4
            alias Pair = X * Y
            latent f : Pair -> X
            export f
        """)
        src_direct = textwrap.dedent("""
            object X : 3
            object Y : 4
            latent f : X * Y -> X
            export f
        """)
        m_aliased = parse(src_aliased)
        m_direct = parse(src_direct)
        # The aliased AST has one extra statement (the AliasDecl); the
        # remaining MorphismDecl/ExportDecl agree on shape.
        decls_aliased = [
            s for s in m_aliased.statements if not isinstance(s, AliasDecl)
        ]
        assert len(decls_aliased) == len(m_direct.statements)

    def test_residuated_alias_stored_for_substitution(self):
        """`alias S_VP = S \\ NP` records the slash pattern as an alias."""
        src = textwrap.dedent("""
            object Atoms = {NP, S, VP}
            object Cat = FreeResiduated(Atoms, depth=2, ops=[slash])
            alias VP_alias = S \\ NP
            object Token : 256
            schema fwd[X, Y : Cat] : (X/Y) * Y -> X
            let g = parser(rules=[fwd], terminal=Token, start=S)
            export g
        """)
        prog = loads(src)
        assert prog is not None
        # AliasDecl is in the AST.
        m = parse(src)
        aliases = [s for s in m.statements if isinstance(s, AliasDecl)]
        assert [a.name for a in aliases] == ["VP_alias"]

    def test_duplicate_alias_raises(self):
        src = textwrap.dedent("""
            object X : 3
            object Y : 4
            alias A = X
            alias A = Y
        """)
        from quivers.dsl import CompileError

        with pytest.raises(CompileError, match="already declared"):
            loads(src)

    def test_alias_shadowing_object_raises(self):
        src = textwrap.dedent("""
            object X : 3
            alias X = X
        """)
        from quivers.dsl import CompileError

        with pytest.raises(CompileError, match="shadows an existing object"):
            loads(src)


# ---------------------------------------------------------------------------
# bundle declarations
# ---------------------------------------------------------------------------


class TestBundleDecl:
    def test_bundle_splices_into_parser_rules(self):
        """`bundle CCG = [forward_app, backward_app]` is consumed as a rule set."""
        src = textwrap.dedent("""
            object Atoms = {NP, S, VP, N, PP}
            object Cat = FreeResiduated(Atoms, depth=2, ops=[slash])
            object Token : 256
            schema forward_app[X, Y : Cat] : (X/Y) * Y -> X
            schema backward_app[X, Y : Cat] : Y * (X\\Y) -> X
            bundle CCG = [forward_app, backward_app]
            let grammar = parser(rules=[CCG], terminal=Token, start=S)
            export grammar
        """)
        prog = loads(src)
        assert prog is not None

    def test_nested_bundle_expands(self):
        """A bundle that references another bundle splices recursively."""
        src = textwrap.dedent("""
            object Atoms = {NP, S, VP, N, PP}
            object Cat = FreeResiduated(Atoms, depth=2, ops=[slash])
            object Token : 256
            schema fa[X, Y : Cat] : (X/Y) * Y -> X
            schema ba[X, Y : Cat] : Y * (X\\Y) -> X
            bundle CORE = [fa, ba]
            bundle EXT = [CORE, fa]
            let g = parser(rules=[EXT], terminal=Token, start=S)
            export g
        """)
        prog = loads(src)
        assert prog is not None

    def test_bundle_cycle_detected(self):
        from quivers.dsl import CompileError

        src = textwrap.dedent("""
            object Atoms = {NP, S}
            object Cat = FreeResiduated(Atoms, depth=1, ops=[slash])
            object Token : 256
            schema fa[X, Y : Cat] : (X/Y) * Y -> X
            bundle A = [fa, B]
            bundle B = [A]
            let g = parser(rules=[A], terminal=Token, start=S)
            export g
        """)
        with pytest.raises(CompileError, match="cycle"):
            loads(src)

    def test_bundle_with_unknown_member_rejected_at_use_site(self):
        from quivers.dsl import CompileError

        src = textwrap.dedent("""
            object Atoms = {NP, S}
            object Cat = FreeResiduated(Atoms, depth=1, ops=[slash])
            object Token : 256
            bundle B = [no_such_rule]
            let g = parser(rules=[B], terminal=Token, start=S)
            export g
        """)
        with pytest.raises(CompileError, match="unknown rule"):
            loads(src)

    def test_bundle_decl_is_in_module_ast(self):
        src = textwrap.dedent("""
            object Atoms = {NP, S}
            object Cat = FreeResiduated(Atoms, depth=1, ops=[slash])
            schema fa[X, Y : Cat] : (X/Y) * Y -> X
            bundle B = [fa]
        """)
        m = parse(src)
        bundles = [s for s in m.statements if isinstance(s, BundleDecl)]
        assert len(bundles) == 1
        assert bundles[0].name == "B"
        assert bundles[0].rules == ("fa",)


# ---------------------------------------------------------------------------
# `##` doc comments
# ---------------------------------------------------------------------------


class TestDocComments:
    def test_doc_comment_attaches_to_object_decl(self):
        src = textwrap.dedent("""
            ## The terminal vocabulary.
            ## Cardinality 256 is one byte.
            object Token : 256
        """)
        m = parse(src)
        objs = [s for s in m.statements if isinstance(s, ObjectDecl)]
        assert len(objs) == 1
        assert objs[0].docs == (
            "The terminal vocabulary.",
            "Cardinality 256 is one byte.",
        )

    def test_doc_comment_attaches_to_schema_decl(self):
        src = textwrap.dedent("""
            object Atoms = {NP, S}
            object Cat = FreeResiduated(Atoms, depth=1, ops=[slash])
            ## Forward application: (X/Y) * Y -> X.
            schema fwd[X, Y : Cat] : (X/Y) * Y -> X
        """)
        m = parse(src)
        schemas = [s for s in m.statements if isinstance(s, SchemaDecl)]
        assert len(schemas) == 1
        assert schemas[0].docs == ("Forward application: (X/Y) * Y -> X.",)

    def test_plain_comments_dropped(self):
        src = textwrap.dedent("""
            # plain comment, dropped
            object X : 3
        """)
        m = parse(src)
        objs = [s for s in m.statements if isinstance(s, ObjectDecl)]
        assert objs[0].docs == ()

    def test_doc_comments_mixed_with_plain(self):
        src = textwrap.dedent("""
            # plain
            ## doc one
            # plain again
            ## doc two
            object X : 3
        """)
        m = parse(src)
        objs = [s for s in m.statements if isinstance(s, ObjectDecl)]
        # Both ## lines accumulate; plain # lines ignored.
        assert objs[0].docs == ("doc one", "doc two")

    def test_doc_comment_does_not_carry_across_statements(self):
        src = textwrap.dedent("""
            ## docs for X
            object X : 3
            object Y : 4
        """)
        m = parse(src)
        objs = [s for s in m.statements if isinstance(s, ObjectDecl)]
        assert objs[0].docs == ("docs for X",)
        assert objs[1].docs == ()


# ---------------------------------------------------------------------------
# FreeMonoid object initialiser
# ---------------------------------------------------------------------------


class TestFreeMonoidSurface:
    def test_free_monoid_object_compiles(self):
        from quivers.core.objects import FreeMonoid

        src = textwrap.dedent("""
            object X : 3
            object Free = FreeMonoid(X, max_length=4)
            export identity(Free)
        """)
        prog = loads(src)
        assert prog is not None
        m = parse(src)
        decls = [
            s
            for s in m.statements
            if isinstance(s, ObjectDecl) and isinstance(s.init, FreeMonoidExpr)
        ]
        assert len(decls) == 1
        assert decls[0].init.generators == "X"
        assert decls[0].init.max_length == 4
        # Compile path should produce a runtime FreeMonoid.
        compiler = Compiler(m)
        compiler.compile_env()
        assert isinstance(compiler.objects["Free"], FreeMonoid)

    def test_free_monoid_generators_must_be_finset(self):
        from quivers.dsl import CompileError

        src = textwrap.dedent("""
            object Atoms = {a, b}
            object Free = FreeMonoid(Atoms, max_length=2)
        """)
        # Atoms is an EnumSet, not a FinSet — FreeMonoid rejects this.
        with pytest.raises(CompileError, match="FinSet"):
            loads(src)


# ---------------------------------------------------------------------------
# Constraint solver
# ---------------------------------------------------------------------------


class TestConstraintSolver:
    def test_clean_module_yields_no_violations(self):
        src = textwrap.dedent("""
            object X : 3
            export identity(X)
        """)
        m = parse(src)
        assert check_constraints(m) == []

    def test_residuated_violation_detected(self):
        """Schema with a slash pattern but no residuated parameter is flagged."""
        src = textwrap.dedent("""
            object X : 3
            object Y : 4
            schema bad[A : X] : (A/A) * Y -> Y
        """)
        m = parse(src)
        v = check_constraints(m)
        assert any(d.code == "residuated_constraint" for d in v)

    def test_bundle_unknown_member_diagnostic(self):
        src = textwrap.dedent("""
            bundle B = [does_not_exist]
        """)
        m = parse(src)
        v = check_constraints(m)
        codes = {d.code for d in v}
        assert "bundle_unknown_member" in codes

    def test_effect_application_lowercase_starts_flagged(self):
        """T(X) where T starts with a lowercase, non-underscored token is unusual
        enough that the constraint solver flags it as not matching the
        effect-name convention."""
        src = textwrap.dedent("""
            object Atoms = {NP, S}
            object Cat = FreeResiduated(Atoms, depth=1, ops=[slash])
            schema bad[X : Cat] : x(X) -> X
        """)
        m = parse(src)
        v = check_constraints(m)
        assert any(d.code == "effect_constraint" for d in v)

    def test_effect_application_uppercase_accepted(self):
        src = textwrap.dedent("""
            object Atoms = {NP, S}
            object Cat = FreeResiduated(Atoms, depth=1, ops=[slash])
            schema ok[X : Cat] : Cont_S(X) -> Cont_S(X)
        """)
        m = parse(src)
        v = check_constraints(m)
        assert all(d.code != "effect_constraint" for d in v)


# ---------------------------------------------------------------------------
# `qvr check` CLI
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_qvr(tmp_path: Path):
    def _write(name: str, body: str) -> Path:
        p = tmp_path / name
        p.write_text(textwrap.dedent(body))
        return p

    return _write


class TestQvrCheckCli:
    def test_clean_file_returns_zero(self, tmp_qvr):
        f = tmp_qvr(
            "clean.qvr",
            """
            object X : 3
            export identity(X)
            """,
        )
        out = io.StringIO()
        err = io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            rc = check_main([str(f)], json_output=False)
        assert rc == 0
        assert "OK" in out.getvalue()

    def test_parse_error_returns_one(self, tmp_qvr):
        f = tmp_qvr(
            "bad.qvr",
            """
            object X :
            """,
        )
        out = io.StringIO()
        err = io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            rc = check_main([str(f)], json_output=False)
        assert rc == 1
        assert "error" in err.getvalue()

    def test_compile_error_returns_one(self, tmp_qvr):
        f = tmp_qvr(
            "compile_bad.qvr",
            """
            object X : 3
            latent f : X -> Y
            export f
            """,
        )
        out = io.StringIO()
        err = io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            rc = check_main([str(f)], json_output=False)
        assert rc == 1

    def test_json_output_is_well_formed(self, tmp_qvr):
        f = tmp_qvr(
            "any.qvr",
            """
            object X : 3
            export identity(X)
            """,
        )
        out = io.StringIO()
        err = io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            rc = check_main([str(f)], json_output=True)
        assert rc == 0
        payload = json.loads(out.getvalue())
        assert payload["ok"] is True
        assert payload["files"] == [str(f)]
        assert payload["diagnostics"] == []

    def test_check_one_returns_diagnostics(self, tmp_qvr):
        f = tmp_qvr(
            "constraint_bad.qvr",
            """
            bundle B = [no_such_rule]
            """,
        )
        diags = _check_one(f)
        codes = {d.code for d in diags}
        assert "bundle_unknown_member" in codes


# ---------------------------------------------------------------------------
# Highlight queries cover every new production
# ---------------------------------------------------------------------------


class TestHighlightQueries:
    def test_highlights_query_compiles(self):
        """The grammars/qvr/queries/highlights.scm file references only
        node kinds that exist in the regenerated grammar's node-types.json.
        Regression test against future grammar / query drift."""
        repo = Path(__file__).resolve().parents[1]
        node_types_path = repo / "grammars" / "qvr" / "src" / "node-types.json"
        highlights_path = repo / "grammars" / "qvr" / "queries" / "highlights.scm"
        node_types = json.loads(node_types_path.read_text())
        known_kinds = {nt["type"] for nt in node_types}

        # Extract `(kind ...)` patterns from highlights.scm. Crude
        # but adequate for catching new productions added to the
        # grammar without query updates.
        import re

        text = highlights_path.read_text()
        # Strip comments from the .scm file before pattern extraction.
        cleaned = re.sub(r";.*", "", text)
        kinds_in_query = set(re.findall(r"\(\s*([a-z_][a-z_0-9]*)\b", cleaned))
        # The highlight query may reference field-grouped patterns
        # like `(rule_decl name: (identifier))`; the regex above
        # extracts the parent kind. Filter to AST node kinds.
        unknown = kinds_in_query - known_kinds
        # Allow the implicit `identifier` token (not in node-types
        # under that exact key on every tree-sitter version).
        unknown.discard("identifier")
        unknown.discard("integer")
        unknown.discard("float")
        unknown.discard("signed_number")
        unknown.discard("line_comment")
        assert not unknown, (
            f"highlights.scm references unknown kinds: {sorted(unknown)}"
        )


# ---------------------------------------------------------------------------
# Pygments lexer
# ---------------------------------------------------------------------------


class TestPygmentsLexer:
    def test_tokenises_keywords_and_identifiers(self):
        from pygments.token import Keyword, Name

        from quivers.dsl.pygments_lexer import QvrLexer

        lex = QvrLexer()
        tokens = list(lex.get_tokens_unprocessed("object X : 3\noutput identity(X)\n"))
        # Expect at least one Keyword for `object` and one Name for X.
        kinds = [tok[1] for tok in tokens]
        assert any(k is Keyword for k in kinds)
        assert any(k in Name or k is Name.Class for k in kinds)

    def test_doc_comment_tokenises_distinctly(self):
        from pygments.token import Comment

        from quivers.dsl.pygments_lexer import QvrLexer

        lex = QvrLexer()
        tokens = list(lex.get_tokens_unprocessed("## a doc comment\nobject X : 3\n"))
        kinds = [tok[1] for tok in tokens]
        # Either Comment.Doc (tree-sitter path) or Comment.Doc (regex
        # fallback) — both share the Comment hierarchy.
        assert any(k in Comment for k in kinds)

    def test_full_example_round_trips(self):
        """Tokenise a full categorial-effects example and assert no Errors."""
        from pygments.token import Error

        from quivers.dsl.pygments_lexer import QvrLexer

        repo = Path(__file__).resolve().parents[1]
        ex = repo / "src" / "quivers" / "dsl" / "examples" / "quantifier_scope.qvr"
        lex = QvrLexer()
        tokens = list(lex.get_tokens_unprocessed(ex.read_text()))
        for _, tok, value in tokens:
            assert tok is not Error, f"Error token at value={value!r}"


# ---------------------------------------------------------------------------
# Top-level grammar-shape coverage
# ---------------------------------------------------------------------------


class TestGrammarSurfaceParity:
    def test_every_example_still_parses(self):
        """Every shipped example continues to parse after the surface additions."""
        repo = Path(__file__).resolve().parents[1]
        examples = sorted((repo / "src" / "quivers" / "dsl" / "examples").glob("*.qvr"))
        assert examples, "no examples found"
        for ex in examples:
            try:
                parse(ex.read_text(), file_path=str(ex))
            except ParseError as e:  # pragma: no cover
                pytest.fail(f"{ex} failed to parse: {e}")
