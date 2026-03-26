"""Tests for the kleisli DSL (lexer, parser, compiler, loader)."""

import tempfile
from pathlib import Path

import torch
import pytest

from quivers.dsl import loads, load, parse, LexError, ParseError, CompileError
from quivers.dsl.lexer import Lexer
from quivers.dsl.tokens import TokenType
from quivers.dsl.parser import Parser
from quivers.dsl.compiler import Compiler
from quivers.dsl.ast_nodes import (
    Module,
    QuantaleDecl,
    CategoryDecl,
    RuleDecl,
    CatPatternName,
    CatPatternSlash,
    CatPatternProduct,
    ObjectDecl,
    MorphismDecl,
    SpaceDecl,
    ContinuousMorphismDecl,
    StochasticMorphismDecl,
    DiscretizeDecl,
    EmbedDecl,
    DrawStep,
    LetStep,
    ProgramDecl,
    LetDecl,
    OutputDecl,
    TypeName,
    TypeProduct,
    TypeCoproduct,
    SpaceConstructor,
    SpaceProduct,
    ExprIdent,
    ExprIdentity,
    ExprCompose,
    ExprTensorProduct,
    ExprMarginalize,
    ExprFan,
    ExprRepeat,
    ExprStack,
    ExprScan,
    ExprParser,
)
from quivers.continuous.programs import MonadicProgram
from quivers.continuous.inline import (
    FixedDistribution,
    DirectBernoulli,
    DirectTruncatedNormal,
)
from quivers.core.objects import FinSet, ProductSet, CoproductSet
from quivers.core.quantales import PRODUCT_FUZZY, BOOLEAN
from quivers.program import Program


# ===== lexer tests ==========================================================


class TestLexer:
    def test_empty_source(self):
        """Empty source produces only EOF."""
        tokens = Lexer("").tokenize()
        assert len(tokens) == 1
        assert tokens[0].type == TokenType.EOF

    def test_comment_only(self):
        """Comment-only source produces only EOF (plus maybe newlines)."""
        tokens = Lexer("# this is a comment\n").tokenize()
        types = [t.type for t in tokens if t.type != TokenType.NEWLINE]
        assert types == [TokenType.EOF]

    def test_keywords(self):
        """All keywords are recognized."""
        source = "quantale object latent observed let output identity marginalize"
        tokens = Lexer(source).tokenize()
        expected = [
            TokenType.QUANTALE, TokenType.OBJECT, TokenType.LATENT,
            TokenType.OBSERVED, TokenType.LET, TokenType.OUTPUT,
            TokenType.IDENTITY, TokenType.MARGINALIZE, TokenType.EOF,
        ]
        actual = [t.type for t in tokens]
        assert actual == expected

    def test_operators(self):
        """All operators are correctly tokenized."""
        source = ": -> >> * + @ = . ( ) [ ] ,"
        tokens = Lexer(source).tokenize()
        expected = [
            TokenType.COLON, TokenType.ARROW, TokenType.COMPOSE,
            TokenType.PRODUCT, TokenType.COPRODUCT, TokenType.TENSOR,
            TokenType.EQUALS, TokenType.DOT, TokenType.LPAREN,
            TokenType.RPAREN, TokenType.LBRACKET, TokenType.RBRACKET,
            TokenType.COMMA, TokenType.EOF,
        ]
        actual = [t.type for t in tokens]
        assert actual == expected

    def test_identifiers_and_integers(self):
        """Identifiers and integer literals are recognized."""
        source = "foo bar_baz 42 100"
        tokens = Lexer(source).tokenize()
        expected = [
            (TokenType.IDENT, "foo"),
            (TokenType.IDENT, "bar_baz"),
            (TokenType.INT, "42"),
            (TokenType.INT, "100"),
            (TokenType.EOF, ""),
        ]
        actual = [(t.type, t.value) for t in tokens]
        assert actual == expected

    def test_line_tracking(self):
        """Line numbers are tracked correctly."""
        source = "object X : 3\nobject Y : 4"
        tokens = Lexer(source).tokenize()
        # first 'object' on line 1, second 'object' on line 2
        object_tokens = [t for t in tokens if t.type == TokenType.OBJECT]
        assert object_tokens[0].line == 1
        assert object_tokens[1].line == 2

    def test_unexpected_character(self):
        """Unexpected characters raise LexError."""
        with pytest.raises(LexError, match="unexpected character"):
            Lexer("object X : 3 $").tokenize()

    def test_inline_comments(self):
        """Comments after code are handled."""
        source = "object X : 3 # this is X"
        tokens = Lexer(source).tokenize()
        types = [t.type for t in tokens if t.type not in (TokenType.NEWLINE, TokenType.EOF)]
        assert types == [TokenType.OBJECT, TokenType.IDENT, TokenType.COLON, TokenType.INT]

    def test_full_program(self):
        """A complete program tokenizes without errors."""
        source = """
        # model definition
        quantale product_fuzzy
        object X : 3
        object Y : 4
        latent f : X -> Y
        let g = f >> identity(X)
        output f
        """
        tokens = Lexer(source).tokenize()
        assert tokens[-1].type == TokenType.EOF


# ===== parser tests =========================================================


class TestParser:
    def _parse(self, source: str) -> Module:
        tokens = Lexer(source).tokenize()
        return Parser(tokens).parse()

    def test_quantale_decl(self):
        """Parse a quantale declaration."""
        mod = self._parse("quantale boolean")
        assert len(mod.statements) == 1
        stmt = mod.statements[0]
        assert isinstance(stmt, QuantaleDecl)
        assert stmt.name == "boolean"

    def test_object_decl_int(self):
        """Parse an object declaration with integer cardinality."""
        mod = self._parse("object X : 3")
        assert len(mod.statements) == 1
        stmt = mod.statements[0]
        assert isinstance(stmt, ObjectDecl)
        assert stmt.name == "X"
        assert isinstance(stmt.type_expr, TypeName)
        assert stmt.type_expr.name == "3"

    def test_object_decl_product(self):
        """Parse an object declaration with product type."""
        mod = self._parse("object X : 3\nobject Y : 4\nobject XY : X * Y")
        stmts = mod.statements
        assert len(stmts) == 3
        xy_stmt = stmts[2]
        assert isinstance(xy_stmt, ObjectDecl)
        assert isinstance(xy_stmt.type_expr, TypeProduct)
        assert len(xy_stmt.type_expr.components) == 2

    def test_object_decl_coproduct(self):
        """Parse an object declaration with coproduct type."""
        mod = self._parse("object X : 3\nobject Y : 4\nobject XY : X + Y")
        xy_stmt = mod.statements[2]
        assert isinstance(xy_stmt.type_expr, TypeCoproduct)

    def test_latent_morphism(self):
        """Parse a latent morphism declaration."""
        mod = self._parse("object X : 3\nlatent f : X -> X")
        stmt = mod.statements[1]
        assert isinstance(stmt, MorphismDecl)
        assert stmt.kind == "latent"
        assert stmt.name == "f"

    def test_observed_morphism_with_identity(self):
        """Parse an observed morphism with identity init."""
        mod = self._parse("object X : 3\nobserved h : X -> X = identity(X)")
        stmt = mod.statements[1]
        assert isinstance(stmt, MorphismDecl)
        assert stmt.kind == "observed"
        assert isinstance(stmt.init_expr, ExprIdentity)
        assert stmt.init_expr.object_name == "X"

    def test_morphism_with_options(self):
        """Parse a morphism with bracket options."""
        mod = self._parse("object X : 3\nlatent f : X -> X [scale=0.3]")
        stmt = mod.statements[1]
        assert isinstance(stmt, MorphismDecl)
        assert stmt.options == {"scale": "0.3"}

    def test_let_compose(self):
        """Parse a let binding with composition."""
        source = "object X : 3\nlatent f : X -> X\nlatent g : X -> X\nlet h = f >> g"
        mod = self._parse(source)
        let_stmt = mod.statements[3]
        assert isinstance(let_stmt, LetDecl)
        assert isinstance(let_stmt.expr, ExprCompose)

    def test_let_tensor_product(self):
        """Parse a let binding with tensor product."""
        source = "object X : 3\nlatent f : X -> X\nlatent g : X -> X\nlet h = f @ g"
        mod = self._parse(source)
        let_stmt = mod.statements[3]
        assert isinstance(let_stmt.expr, ExprTensorProduct)

    def test_let_marginalize(self):
        """Parse a let binding with marginalization."""
        source = (
            "object X : 3\nobject Y : 4\n"
            "latent f : X -> X\n"
            "let m = f.marginalize(X)"
        )
        mod = self._parse(source)
        let_stmt = mod.statements[3]
        assert isinstance(let_stmt.expr, ExprMarginalize)
        assert let_stmt.expr.names == ("X",)

    def test_output_decl(self):
        """Parse an output declaration."""
        source = "object X : 3\nlatent f : X -> X\noutput f"
        mod = self._parse(source)
        out = mod.statements[2]
        assert isinstance(out, OutputDecl)
        assert isinstance(out.expr, ExprIdent)

    def test_parenthesized_expr(self):
        """Parse parenthesized expressions."""
        source = (
            "object X : 3\nlatent f : X -> X\n"
            "latent g : X -> X\nlatent h : X -> X\n"
            "let r = (f >> g) @ h"
        )
        mod = self._parse(source)
        let_stmt = mod.statements[4]
        assert isinstance(let_stmt.expr, ExprTensorProduct)
        assert isinstance(let_stmt.expr.left, ExprCompose)

    def test_parse_error_expected_type(self):
        """ParseError on malformed type expression."""
        with pytest.raises(ParseError, match="expected type expression"):
            self._parse("object X : >>")

    def test_parse_error_expected_statement(self):
        """ParseError on unexpected token at statement level."""
        with pytest.raises(ParseError, match="expected statement"):
            self._parse("42")

    def test_chained_compose(self):
        """Parse chained composition: f >> g >> h."""
        source = (
            "object X : 3\n"
            "latent f : X -> X\nlatent g : X -> X\nlatent h : X -> X\n"
            "let r = f >> g >> h"
        )
        mod = self._parse(source)
        let_stmt = mod.statements[4]
        # left-associative: (f >> g) >> h
        assert isinstance(let_stmt.expr, ExprCompose)
        assert isinstance(let_stmt.expr.left, ExprCompose)

    def test_product_type_three_components(self):
        """Parse a 3-component product type."""
        source = "object X : 2\nobject Y : 3\nobject Z : 4\nobject XYZ : X * Y * Z"
        mod = self._parse(source)
        xyz_stmt = mod.statements[3]
        assert isinstance(xyz_stmt.type_expr, TypeProduct)
        assert len(xyz_stmt.type_expr.components) == 3


# ===== compiler tests =======================================================


class TestCompiler:
    def test_simple_latent(self):
        """Compile a single latent morphism."""
        prog = loads("""
            object X : 3
            object Y : 4
            latent f : X -> Y
            output f
        """)
        assert isinstance(prog, Program)
        assert prog().shape == torch.Size([3, 4])

    def test_composition(self):
        """Compile sequential composition."""
        prog = loads("""
            object X : 3
            object Y : 4
            object Z : 2
            latent f : X -> Y
            latent g : Y -> Z
            output f >> g
        """)
        assert prog().shape == torch.Size([3, 2])

    def test_tensor_product(self):
        """Compile tensor product."""
        prog = loads("""
            object X : 2
            object Y : 3
            latent f : X -> X
            latent g : Y -> Y
            output f @ g
        """)
        # (X * Y) -> (X * Y) = (2, 3, 2, 3)
        assert prog().shape == torch.Size([2, 3, 2, 3])

    def test_identity_morphism(self):
        """Compile identity morphism."""
        prog = loads("""
            object X : 3
            observed h : X -> X = identity(X)
            output h
        """)
        out = prog()
        expected = torch.eye(3)
        torch.testing.assert_close(out, expected)

    def test_marginalization(self):
        """Compile morphism with marginalization."""
        prog = loads("""
            object X : 2
            object Y : 3
            latent f : X -> X
            latent g : Y -> Y
            let par = f @ g
            let m = par.marginalize(Y)
            output m
        """)
        # (X * Y) -> X, marginalized over Y
        assert prog().shape == torch.Size([2, 3, 2])

    def test_let_binding(self):
        """Let bindings can be referenced later."""
        prog = loads("""
            object X : 3
            latent f : X -> X
            let g = f >> f
            output g
        """)
        assert prog().shape == torch.Size([3, 3])

    def test_quantale_boolean(self):
        """Compile with boolean quantale."""
        prog = loads("""
            quantale boolean
            object X : 2
            observed h : X -> X = identity(X)
            output h
        """)
        out = prog()
        expected = torch.eye(2)
        torch.testing.assert_close(out, expected)

    def test_quantale_godel(self):
        """Compile with Godel quantale."""
        prog = loads("""
            quantale godel
            object X : 2
            observed h : X -> X = identity(X)
            output h
        """)
        out = prog()
        expected = torch.eye(2)
        torch.testing.assert_close(out, expected)

    def test_product_object(self):
        """Compile product object type."""
        prog = loads("""
            object X : 2
            object Y : 3
            object XY : X * Y
            latent f : XY -> X
            output f
        """)
        assert prog().shape == torch.Size([2, 3, 2])

    def test_coproduct_object(self):
        """Compile coproduct object type."""
        prog = loads("""
            object X : 2
            object Y : 3
            object XY : X + Y
            latent f : XY -> X
            output f
        """)
        # coproduct has shape (5,) → (2,)
        assert prog().shape == torch.Size([5, 2])

    def test_trainable(self):
        """Compiled program has trainable parameters."""
        prog = loads("""
            object X : 3
            object Y : 4
            latent f : X -> Y
            output f
        """)
        params = list(prog.parameters())
        assert len(params) > 0
        assert all(p.requires_grad for p in params)

    def test_gradient_flow(self):
        """Gradients flow through composed morphisms."""
        prog = loads("""
            object X : 2
            object Y : 3
            object Z : 2
            latent f : X -> Y
            latent g : Y -> Z
            output f >> g
        """)
        out = prog()
        loss = out.sum()
        loss.backward()

        for p in prog.parameters():
            assert p.grad is not None

    def test_init_scale(self):
        """Morphism options (scale) are respected."""
        # just check it doesn't error
        prog = loads("""
            object X : 3
            latent f : X -> X [scale=0.1]
            output f
        """)
        assert prog().shape == torch.Size([3, 3])

    def test_undefined_object_error(self):
        """CompileError for undefined object reference."""
        with pytest.raises(CompileError, match="undefined object"):
            loads("""
                latent f : X -> Y
                output f
            """)

    def test_undefined_morphism_error(self):
        """CompileError for undefined morphism reference."""
        with pytest.raises(CompileError, match="undefined morphism"):
            loads("""
                object X : 3
                output f
            """)

    def test_no_output_error(self):
        """CompileError when no output declaration."""
        with pytest.raises(CompileError, match="no output"):
            loads("""
                object X : 3
                latent f : X -> X
            """)

    def test_duplicate_object_error(self):
        """CompileError on duplicate object name."""
        with pytest.raises(CompileError, match="already declared"):
            loads("""
                object X : 3
                object X : 4
                latent f : X -> X
                output f
            """)

    def test_unknown_quantale_error(self):
        """CompileError for unknown quantale name."""
        with pytest.raises(CompileError, match="unknown quantale"):
            loads("""
                quantale nonexistent
                object X : 3
                latent f : X -> X
                output f
            """)

    def test_observed_without_init_error(self):
        """CompileError for observed morphism without initializer."""
        with pytest.raises(CompileError, match="requires"):
            loads("""
                object X : 3
                observed f : X -> X
                output f
            """)

    def test_multiple_output_error(self):
        """CompileError on multiple output declarations."""
        with pytest.raises(CompileError, match="multiple output"):
            loads("""
                object X : 3
                latent f : X -> X
                output f
                output f
            """)


# ===== loader tests =========================================================


class TestLoader:
    def test_load_file(self, tmp_path):
        """Load a .kl file from disk."""
        kl_file = tmp_path / "test_model.kl"
        kl_file.write_text(
            "object X : 3\n"
            "object Y : 4\n"
            "latent f : X -> Y\n"
            "output f\n"
        )
        prog = load(kl_file)
        assert isinstance(prog, Program)
        assert prog().shape == torch.Size([3, 4])

    def test_load_string_path(self, tmp_path):
        """Load accepts string paths."""
        kl_file = tmp_path / "model.kl"
        kl_file.write_text(
            "object X : 2\nlatent f : X -> X\noutput f\n"
        )
        prog = load(str(kl_file))
        assert isinstance(prog, Program)

    def test_file_not_found(self):
        """FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            load("/nonexistent/path/model.kl")


# ===== parse function tests =================================================


class TestParse:
    def test_parse_returns_module(self):
        """The parse function returns a Module AST."""
        mod = parse("object X : 3\nlatent f : X -> X\noutput f")
        assert isinstance(mod, Module)
        assert len(mod.statements) == 3

    def test_parse_preserves_line_info(self):
        """AST nodes carry source location info."""
        mod = parse("object X : 3\nlatent f : X -> X")
        assert mod.statements[0].line == 1
        assert mod.statements[1].line == 2


# ===== compile_env tests ====================================================


class TestCompileEnv:
    def test_compile_env_objects(self):
        """compile_env returns objects in the environment."""
        ast = parse("""
            object X : 3
            object Y : 4
            latent f : X -> Y
        """)
        compiler = Compiler(ast)
        env = compiler.compile_env()

        assert "X" in env
        assert isinstance(env["X"], FinSet)
        assert env["X"].cardinality == 3

    def test_compile_env_morphisms(self):
        """compile_env returns morphisms in the environment."""
        ast = parse("""
            object X : 3
            object Y : 4
            latent f : X -> Y
        """)
        compiler = Compiler(ast)
        env = compiler.compile_env()

        assert "f" in env

    def test_compile_env_quantale(self):
        """compile_env includes the active quantale."""
        ast = parse("quantale boolean\nobject X : 2")
        compiler = Compiler(ast)
        env = compiler.compile_env()
        assert env["__quantale__"] is BOOLEAN


# ===== integration tests ====================================================


class TestIntegration:
    def test_full_pipeline(self):
        """End-to-end: parse, compile, forward, backward."""
        source = """
            # a simple category
            quantale product_fuzzy

            object Phoneme : 40
            object Feature : 12
            object Word : 100

            # learnable morphisms
            latent encode : Phoneme -> Feature
            latent decode : Feature -> Word

            # composition
            let model = encode >> decode

            output model
        """
        prog = loads(source)

        # forward
        out = prog()
        assert out.shape == torch.Size([40, 100])
        assert (out >= 0).all() and (out <= 1).all()

        # backward
        loss = prog.bce_loss(torch.rand(40, 100))
        loss.backward()

        for p in prog.parameters():
            assert p.grad is not None

    def test_complex_pipeline(self):
        """Complex model with products, composition, and tensor product."""
        source = """
            object X : 3
            object Y : 4
            object Z : 2

            latent f : X -> Y
            latent g : X -> Z

            # parallel composition
            let par = f @ g

            output par
        """
        prog = loads(source)
        out = prog()
        # (X, X) -> (Y, Z) = (3, 3, 4, 2)
        assert out.shape == torch.Size([3, 3, 4, 2])

    def test_nll_loss(self):
        """NLL loss works through compiled program."""
        prog = loads("""
            object X : 5
            object Y : 3
            latent f : X -> Y
            output f
        """)

        domain_idx = torch.tensor([0, 1, 2])
        codomain_idx = torch.tensor([0, 1, 2])
        loss = prog.nll_loss(domain_idx, codomain_idx)
        assert loss.ndim == 0  # scalar
        loss.backward()

    def test_comments_and_whitespace(self):
        """Comments and extra whitespace are handled gracefully."""
        source = """
            # this is a model

            object X : 3  # input space
            object Y : 4  # output space

            # learnable morphism
            latent f : X -> Y

            # the output
            output f
        """
        prog = loads(source)
        assert prog().shape == torch.Size([3, 4])

    def test_composition_chain(self):
        """Long composition chains compile correctly."""
        source = """
            object A : 2
            object B : 3
            object C : 4
            object D : 5

            latent ab : A -> B
            latent bc : B -> C
            latent cd : C -> D

            let chain = ab >> bc >> cd

            output chain
        """
        prog = loads(source)
        assert prog().shape == torch.Size([2, 5])

    def test_product_type_morphism(self):
        """Morphisms with product-typed domains work."""
        source = """
            object X : 2
            object Y : 3
            object XY : X * Y

            latent f : XY -> X

            output f
        """
        prog = loads(source)
        assert prog().shape == torch.Size([2, 3, 2])

    def test_marginalize_multi(self):
        """Marginalize over multiple dimensions."""
        source = """
            object A : 2
            object B : 3
            object C : 4
            object ABC : A * B * C

            latent f : A -> ABC

            let m = f.marginalize(B, C)
            output m
        """
        prog = loads(source)
        # A -> (A * B * C) marginalize B, C → A -> A
        assert prog().shape == torch.Size([2, 2])


# ===== new lexer tests for continuous extensions ============================


class TestLexerContinuous:
    def test_new_keywords(self):
        """New keywords: space, continuous, stochastic, discretize, embed."""
        source = "space continuous stochastic discretize embed"
        tokens = Lexer(source).tokenize()
        expected = [
            TokenType.SPACE, TokenType.CONTINUOUS, TokenType.STOCHASTIC,
            TokenType.DISCRETIZE, TokenType.EMBED, TokenType.EOF,
        ]
        actual = [t.type for t in tokens]
        assert actual == expected

    def test_tilde_operator(self):
        """The ~ (tilde) operator is lexed."""
        tokens = Lexer("~").tokenize()
        assert tokens[0].type == TokenType.TILDE
        assert tokens[0].value == "~"


# ===== new parser tests for continuous extensions ===========================


class TestParserContinuous:
    def _parse(self, source: str) -> Module:
        tokens = Lexer(source).tokenize()
        return Parser(tokens).parse()

    def test_space_decl_euclidean(self):
        """Parse a Euclidean space declaration."""
        mod = self._parse("space R3 : Euclidean(3)")
        stmt = mod.statements[0]
        assert isinstance(stmt, SpaceDecl)
        assert stmt.name == "R3"
        assert isinstance(stmt.space_expr, SpaceConstructor)
        assert stmt.space_expr.constructor == "Euclidean"
        assert stmt.space_expr.args == ("3",)

    def test_space_decl_with_kwargs(self):
        """Parse a space declaration with keyword arguments."""
        mod = self._parse("space B : Euclidean(2, low=0.0, high=1.0)")
        stmt = mod.statements[0]
        assert isinstance(stmt.space_expr, SpaceConstructor)
        assert stmt.space_expr.args == ("2",)
        assert stmt.space_expr.kwargs == {"low": "0.0", "high": "1.0"}

    def test_space_decl_unit_interval(self):
        """Parse UnitInterval (no args)."""
        mod = self._parse("space U : UnitInterval()")
        stmt = mod.statements[0]
        assert isinstance(stmt.space_expr, SpaceConstructor)
        assert stmt.space_expr.constructor == "UnitInterval"
        assert stmt.space_expr.args == ()

    def test_space_decl_simplex(self):
        """Parse Simplex space."""
        mod = self._parse("space S4 : Simplex(4)")
        stmt = mod.statements[0]
        assert stmt.space_expr.constructor == "Simplex"
        assert stmt.space_expr.args == ("4",)

    def test_space_product(self):
        """Parse product space: A * B."""
        mod = self._parse(
            "space R3 : Euclidean(3)\n"
            "space S4 : Simplex(4)\n"
            "space PS : R3 * S4"
        )
        stmt = mod.statements[2]
        assert isinstance(stmt.space_expr, SpaceProduct)
        assert len(stmt.space_expr.components) == 2

    def test_space_reference(self):
        """Parse space reference (bare identifier)."""
        mod = self._parse(
            "space R3 : Euclidean(3)\n"
            "space alias : R3"
        )
        stmt = mod.statements[1]
        assert isinstance(stmt.space_expr, TypeName)
        assert stmt.space_expr.name == "R3"

    def test_continuous_decl(self):
        """Parse continuous morphism declaration."""
        mod = self._parse(
            "object X : 3\n"
            "space R3 : Euclidean(3)\n"
            "continuous f : X -> R3 ~ Normal"
        )
        stmt = mod.statements[2]
        assert isinstance(stmt, ContinuousMorphismDecl)
        assert stmt.name == "f"
        assert stmt.domain.name == "X"
        assert stmt.codomain.name == "R3"
        assert stmt.family == "Normal"
        assert stmt.options == {}

    def test_continuous_decl_with_options(self):
        """Parse continuous morphism with bracket options."""
        mod = self._parse(
            "object X : 3\n"
            "space R3 : Euclidean(3)\n"
            "continuous fl : X -> R3 ~ Flow [n_layers=6, hidden_dim=32]"
        )
        stmt = mod.statements[2]
        assert isinstance(stmt, ContinuousMorphismDecl)
        assert stmt.family == "Flow"
        assert stmt.options == {"n_layers": "6", "hidden_dim": "32"}

    def test_stochastic_decl(self):
        """Parse stochastic morphism declaration."""
        mod = self._parse(
            "object X : 3\n"
            "object Y : 4\n"
            "stochastic s : X -> Y"
        )
        stmt = mod.statements[2]
        assert isinstance(stmt, StochasticMorphismDecl)
        assert stmt.name == "s"

    def test_discretize_decl(self):
        """Parse discretize declaration."""
        mod = self._parse(
            "space B : Euclidean(1, low=0.0, high=1.0)\n"
            "discretize d : B -> 10"
        )
        stmt = mod.statements[1]
        assert isinstance(stmt, DiscretizeDecl)
        assert stmt.space_name == "B"
        assert stmt.n_bins == 10

    def test_embed_decl(self):
        """Parse embed declaration."""
        mod = self._parse(
            "object X : 5\n"
            "space R3 : Euclidean(3)\n"
            "embed e : X -> R3"
        )
        stmt = mod.statements[2]
        assert isinstance(stmt, EmbedDecl)
        assert stmt.domain_name == "X"
        assert stmt.codomain_name == "R3"


# ===== new compiler tests for continuous extensions =========================


class TestCompilerContinuous:
    def test_space_euclidean(self):
        """Compile a Euclidean space."""
        from quivers.continuous.spaces import Euclidean

        ast = parse("space R3 : Euclidean(3)")
        compiler = Compiler(ast)
        env = compiler.compile_env()
        assert "R3" in env
        assert isinstance(env["R3"], Euclidean)
        assert env["R3"].dim == 3

    def test_space_simplex(self):
        """Compile a Simplex space."""
        from quivers.continuous.spaces import Simplex

        ast = parse("space S : Simplex(4)")
        compiler = Compiler(ast)
        env = compiler.compile_env()
        assert isinstance(env["S"], Simplex)
        assert env["S"].dim == 4

    def test_space_positive_reals(self):
        """Compile a PositiveReals space."""
        from quivers.continuous.spaces import PositiveReals

        ast = parse("space P : PositiveReals(2)")
        compiler = Compiler(ast)
        env = compiler.compile_env()
        assert isinstance(env["P"], PositiveReals)
        assert env["P"].dim == 2

    def test_space_unit_interval(self):
        """Compile a UnitInterval space."""
        from quivers.continuous.spaces import Euclidean

        ast = parse("space U : UnitInterval()")
        compiler = Compiler(ast)
        env = compiler.compile_env()
        assert isinstance(env["U"], Euclidean)
        assert env["U"].dim == 1

    def test_space_bounded_euclidean(self):
        """Compile a bounded Euclidean space."""
        from quivers.continuous.spaces import Euclidean

        ast = parse("space B : Euclidean(2, low=0.0, high=1.0)")
        compiler = Compiler(ast)
        env = compiler.compile_env()
        assert isinstance(env["B"], Euclidean)
        assert env["B"].dim == 2
        assert env["B"].low == 0.0
        assert env["B"].high == 1.0

    def test_space_product(self):
        """Compile a product space."""
        from quivers.continuous.spaces import ProductSpace

        ast = parse(
            "space R3 : Euclidean(3)\n"
            "space S4 : Simplex(4)\n"
            "space PS : R3 * S4"
        )
        compiler = Compiler(ast)
        env = compiler.compile_env()
        assert isinstance(env["PS"], ProductSpace)
        assert env["PS"].dim == 7

    def test_continuous_normal(self):
        """Compile a continuous Normal morphism."""
        from quivers.continuous.families import ConditionalNormal

        ast = parse(
            "object X : 5\n"
            "space R3 : Euclidean(3)\n"
            "continuous f : X -> R3 ~ Normal"
        )
        compiler = Compiler(ast)
        env = compiler.compile_env()
        assert "f" in env
        assert isinstance(env["f"], ConditionalNormal)

    def test_continuous_dirichlet(self):
        """Compile a continuous Dirichlet morphism."""
        from quivers.continuous.families import ConditionalDirichlet

        ast = parse(
            "object X : 3\n"
            "space S : Simplex(4)\n"
            "continuous g : X -> S ~ Dirichlet"
        )
        compiler = Compiler(ast)
        env = compiler.compile_env()
        assert isinstance(env["g"], ConditionalDirichlet)

    def test_continuous_flow(self):
        """Compile a ConditionalFlow."""
        from quivers.continuous.flows import ConditionalFlow

        ast = parse(
            "object X : 5\n"
            "space R4 : Euclidean(4)\n"
            "continuous fl : X -> R4 ~ Flow [n_layers=6, hidden_dim=32]"
        )
        compiler = Compiler(ast)
        env = compiler.compile_env()
        assert isinstance(env["fl"], ConditionalFlow)

    def test_continuous_beta(self):
        """Compile a continuous Beta morphism."""
        from quivers.continuous.families import ConditionalBeta

        ast = parse(
            "object X : 3\n"
            "space R2 : Euclidean(2)\n"
            "continuous b : X -> R2 ~ Beta"
        )
        compiler = Compiler(ast)
        env = compiler.compile_env()
        assert isinstance(env["b"], ConditionalBeta)

    def test_continuous_laplace(self):
        """Compile a Laplace morphism (factory-generated)."""
        from quivers.continuous.families import ConditionalLaplace

        ast = parse(
            "object X : 3\n"
            "space R2 : Euclidean(2)\n"
            "continuous l : X -> R2 ~ Laplace"
        )
        compiler = Compiler(ast)
        env = compiler.compile_env()
        assert isinstance(env["l"], ConditionalLaplace)

    def test_stochastic_morphism(self):
        """Compile a stochastic morphism."""
        from quivers.stochastic import StochasticMorphism

        ast = parse(
            "object X : 3\n"
            "object Y : 4\n"
            "stochastic s : X -> Y"
        )
        compiler = Compiler(ast)
        env = compiler.compile_env()
        assert isinstance(env["s"], StochasticMorphism)

    def test_discretize(self):
        """Compile a Discretize boundary morphism."""
        from quivers.continuous.boundaries import Discretize

        ast = parse(
            "space B : Euclidean(1, low=0.0, high=1.0)\n"
            "discretize d : B -> 10"
        )
        compiler = Compiler(ast)
        env = compiler.compile_env()
        assert isinstance(env["d"], Discretize)

    def test_embed(self):
        """Compile an Embed boundary morphism."""
        from quivers.continuous.boundaries import Embed

        ast = parse(
            "object X : 5\n"
            "space R3 : Euclidean(3)\n"
            "embed e : X -> R3"
        )
        compiler = Compiler(ast)
        env = compiler.compile_env()
        assert isinstance(env["e"], Embed)

    def test_unknown_family_error(self):
        """CompileError for unknown distribution family."""
        with pytest.raises(CompileError, match="unknown distribution family"):
            ast = parse(
                "object X : 3\n"
                "space R : Euclidean(2)\n"
                "continuous f : X -> R ~ Nonexistent\n"
                "output f"
            )
            Compiler(ast).compile()

    def test_unknown_space_constructor_error(self):
        """CompileError for unknown space constructor."""
        with pytest.raises(CompileError, match="unknown space constructor"):
            ast = parse("space R : FakeSpace(3)")
            Compiler(ast).compile_env()

    def test_undefined_space_error(self):
        """CompileError for undefined space in discretize."""
        with pytest.raises(CompileError, match="undefined space"):
            ast = parse("discretize d : missing -> 10")
            Compiler(ast).compile_env()

    def test_undefined_object_in_embed_error(self):
        """CompileError for undefined object in embed."""
        with pytest.raises(CompileError, match="undefined object"):
            ast = parse(
                "space R : Euclidean(2)\n"
                "embed e : missing -> R"
            )
            Compiler(ast).compile_env()


# ===== continuous DSL integration tests =====================================


class TestContinuousDSLIntegration:
    def test_discrete_to_continuous_pipeline(self):
        """Full pipeline: discrete -> continuous via DSL."""
        ast = parse("""
            object X : 5
            space R3 : Euclidean(3)

            continuous f : X -> R3 ~ Normal

            output f
        """)
        compiler = Compiler(ast)
        prog = compiler.compile()

        # the output is a ContinuousMorphism, so Program wraps it
        assert isinstance(prog, Program)

        # continuous programs support rsample/log_prob
        x = torch.arange(5)
        y = prog.rsample(x)
        assert y.shape == (5, 3)

    def test_stochastic_then_continuous(self):
        """Stochastic >> Continuous composition via DSL."""
        ast = parse("""
            object A : 5
            object B : 3
            space R2 : Euclidean(2)

            stochastic s : A -> B
            continuous g : B -> R2 ~ Normal

            let pipeline = s >> g
            output pipeline
        """)
        compiler = Compiler(ast)
        env = compiler.compile_env()
        assert "pipeline" in env

    def test_embed_then_continuous(self):
        """Embed >> Continuous composition via DSL."""
        ast = parse("""
            object A : 4
            space R2 : Euclidean(2)
            space R1 : Euclidean(1)

            embed e : A -> R2
            continuous g : R2 -> R1 ~ Normal

            let pipeline = e >> g
            output pipeline
        """)
        compiler = Compiler(ast)
        env = compiler.compile_env()
        assert "pipeline" in env

    def test_continuous_rsample(self):
        """Compiled continuous morphism can rsample."""
        ast = parse("""
            object X : 5
            space R3 : Euclidean(3)

            continuous f : X -> R3 ~ Normal
        """)
        compiler = Compiler(ast)
        env = compiler.compile_env()
        f = env["f"]

        x = torch.tensor([0, 1, 2])
        samples = f.rsample(x)
        assert samples.shape == (3, 3)

    def test_continuous_log_prob(self):
        """Compiled continuous morphism can compute log_prob."""
        ast = parse("""
            object X : 3
            space R2 : Euclidean(2)

            continuous f : X -> R2 ~ Normal
        """)
        compiler = Compiler(ast)
        env = compiler.compile_env()
        f = env["f"]

        x = torch.tensor([0, 1, 2])
        y = torch.randn(3, 2)
        lp = f.log_prob(x, y)
        assert lp.shape == (3,)
        assert torch.isfinite(lp).all()

    def test_flow_via_dsl(self):
        """ConditionalFlow via DSL."""
        ast = parse("""
            object X : 3
            space R4 : Euclidean(4)

            continuous fl : X -> R4 ~ Flow [n_layers=4, hidden_dim=16]
        """)
        compiler = Compiler(ast)
        env = compiler.compile_env()
        fl = env["fl"]

        x = torch.tensor([0, 1, 2])
        samples = fl.rsample(x)
        assert samples.shape == (3, 4)

    def test_all_families_via_dsl(self):
        """Verify every distribution family is accessible via DSL."""
        # families that work with unbounded Euclidean
        unbounded_families = [
            "Normal", "LogitNormal", "Beta",
            "Cauchy", "Laplace", "Gumbel", "LogNormal", "StudentT",
            "Exponential", "Gamma", "Chi2", "HalfCauchy", "HalfNormal",
            "InverseGamma", "Weibull", "Pareto",
            "Kumaraswamy", "ContinuousBernoulli",
            "FisherSnedecor",
        ]

        for family in unbounded_families:
            source = (
                f"object X : 3\n"
                f"space R : Euclidean(2)\n"
                f"continuous f : X -> R ~ {family}\n"
            )
            ast = parse(source)
            env = Compiler(ast).compile_env()
            assert "f" in env, f"family {family} not compiled"

        # families that need bounded codomain
        for family in ["TruncatedNormal", "Uniform"]:
            source = (
                f"object X : 3\n"
                f"space R : Euclidean(2, low=0.0, high=1.0)\n"
                f"continuous f : X -> R ~ {family}\n"
            )
            ast = parse(source)
            env = Compiler(ast).compile_env()
            assert "f" in env, f"bounded family {family} not compiled"

    def test_multivariate_families_via_dsl(self):
        """MultivariateNormal and LowRankMVN via DSL."""
        for family in ["MultivariateNormal", "LowRankMVN"]:
            source = (
                f"object X : 3\n"
                f"space R : Euclidean(4)\n"
                f"continuous f : X -> R ~ {family}\n"
            )
            ast = parse(source)
            env = Compiler(ast).compile_env()
            assert "f" in env

    def test_dirichlet_via_dsl(self):
        """Dirichlet distribution via DSL."""
        ast = parse("""
            object X : 3
            space S : Simplex(4)
            continuous f : X -> S ~ Dirichlet
        """)
        compiler = Compiler(ast)
        env = compiler.compile_env()

        x = torch.tensor([0, 1, 2])
        samples = env["f"].rsample(x)
        assert samples.shape == (3, 4)
        assert torch.allclose(samples.sum(dim=-1), torch.ones(3), atol=1e-5)

    def test_full_mixed_pipeline(self):
        """Full mixed pipeline: discrete -> stochastic -> continuous."""
        ast = parse("""
            object A : 5
            object B : 3
            space R2 : Euclidean(2)

            stochastic s : A -> B
            continuous g : B -> R2 ~ Normal

            let pipeline = s >> g
        """)
        compiler = Compiler(ast)
        env = compiler.compile_env()

        pipeline = env["pipeline"]
        x = torch.tensor([0, 1, 2, 3, 4])
        samples = pipeline.rsample(x)
        assert samples.shape == (5, 2)

    def test_comments_in_continuous_program(self):
        """Comments and whitespace work in continuous programs."""
        source = """
            # continuous model
            object X : 3  # input

            space R3 : Euclidean(3)  # output space

            # learnable conditional distribution
            continuous f : X -> R3 ~ Normal

            output f
        """
        ast = parse(source)
        compiler = Compiler(ast)
        prog = compiler.compile()
        assert isinstance(prog, Program)


# ===== monadic program tests ================================================


class TestLexerProgram:
    def test_program_keyword(self):
        """Lexer recognizes 'program' keyword."""
        tokens = Lexer("program").tokenize()
        assert tokens[0].type == TokenType.PROGRAM

    def test_draw_keyword(self):
        """Lexer recognizes 'draw' keyword."""
        tokens = Lexer("draw").tokenize()
        assert tokens[0].type == TokenType.DRAW

    def test_return_keyword(self):
        """Lexer recognizes 'return' keyword."""
        tokens = Lexer("return").tokenize()
        assert tokens[0].type == TokenType.RETURN


class TestParserProgram:
    def test_simple_program(self):
        """Parse a minimal program block."""
        ast = parse("""
            object X : 3
            space R : Euclidean(2)
            continuous f : X -> R ~ Normal

            program p : X -> R
                draw y ~ f
                return y
        """)
        prog_stmt = ast.statements[3]
        assert isinstance(prog_stmt, ProgramDecl)
        assert prog_stmt.name == "p"
        assert prog_stmt.domain.name == "X"
        assert prog_stmt.codomain.name == "R"
        assert len(prog_stmt.draws) == 1
        assert prog_stmt.draws[0].vars == ("y",)
        assert prog_stmt.draws[0].morphism == "f"
        assert prog_stmt.draws[0].args is None
        assert prog_stmt.return_vars == ("y",)

    def test_program_with_arg(self):
        """Parse draw step with explicit argument."""
        ast = parse("""
            object X : 3
            space R : Euclidean(2)
            space S : Euclidean(4)
            continuous f : X -> R ~ Normal
            continuous g : R -> S ~ Normal

            program p : X -> S
                draw y ~ f
                draw z ~ g(y)
                return z
        """)
        prog_stmt = ast.statements[5]
        assert isinstance(prog_stmt, ProgramDecl)
        assert len(prog_stmt.draws) == 2
        assert prog_stmt.draws[1].args == ("y",)
        assert prog_stmt.return_vars == ("z",)

    def test_program_fan_out(self):
        """Parse program with fan-out (multiple draws from input)."""
        ast = parse("""
            object X : 2
            space B : UnitInterval(1)
            continuous p1 : X -> B ~ LogitNormal
            continuous p2 : X -> B ~ LogitNormal

            program fan_prog : X -> B
                draw a ~ p1
                draw b ~ p2
                return a
        """)
        prog_stmt = ast.statements[4]
        assert isinstance(prog_stmt, ProgramDecl)
        assert len(prog_stmt.draws) == 2
        assert prog_stmt.draws[0].args is None
        assert prog_stmt.draws[1].args is None

    def test_program_missing_draw_error(self):
        """Program with no draw steps is a parse error."""
        with pytest.raises(ParseError):
            parse("""
                object X : 3
                space R : Euclidean(2)
                continuous f : X -> R ~ Normal

                program p : X -> R
                    return y
            """)


class TestCompilerProgram:
    def test_simple_program_compiles(self):
        """Simple program compiles to MonadicProgram."""
        ast = parse("""
            object X : 3
            space R : Euclidean(2)
            continuous f : X -> R ~ Normal

            program p : X -> R
                draw y ~ f
                return y
        """)
        env = Compiler(ast).compile_env()
        assert "p" in env
        assert isinstance(env["p"], MonadicProgram)

    def test_chained_draws(self):
        """Chained draws compile and produce correct output shape."""
        ast = parse("""
            object X : 3
            space R : Euclidean(2)
            space S : Euclidean(4)
            continuous f : X -> R ~ Normal
            continuous g : R -> S ~ Normal

            program chain : X -> S
                draw y ~ f
                draw z ~ g(y)
                return z
        """)
        env = Compiler(ast).compile_env()
        prog = env["chain"]

        x = torch.tensor([0, 1, 2])
        out = prog.rsample(x)
        assert out.shape == (3, 4)

    def test_fan_out_independent_draws(self):
        """Fan-out: multiple draws from same input produce independent values."""
        ast = parse("""
            object X : 2
            space B : UnitInterval(1)
            continuous p1 : X -> B ~ LogitNormal
            continuous p2 : X -> B ~ LogitNormal
            continuous p3 : X -> B ~ LogitNormal

            program prior : X -> B
                draw x ~ p1
                draw y ~ p2
                draw z ~ p3
                return x
        """)
        env = Compiler(ast).compile_env()
        prog = env["prior"]

        inp = torch.tensor([0, 1])
        out = prog.rsample(inp)
        assert out.shape == (2, 1)

    def test_parameters_visible(self):
        """All step morphism parameters are accessible from the program."""
        ast = parse("""
            object X : 2
            space R : Euclidean(2)
            space S : Euclidean(3)
            continuous f : X -> R ~ Normal
            continuous g : R -> S ~ Normal

            program p : X -> S
                draw y ~ f
                draw z ~ g(y)
                return z
        """)
        env = Compiler(ast).compile_env()
        prog = env["p"]

        n_params = sum(p.numel() for p in prog.parameters())
        assert n_params > 0

    def test_gradient_flow(self):
        """Gradients flow through the monadic program."""
        ast = parse("""
            object X : 3
            space R : Euclidean(2)
            continuous f : X -> R ~ Normal

            program p : X -> R
                draw y ~ f
                return y
        """)
        env = Compiler(ast).compile_env()
        prog = env["p"]

        x = torch.tensor([0, 1, 2])
        out = prog.rsample(x)
        loss = out.sum()
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in prog.parameters()
        )
        assert has_grad

    def test_log_joint(self):
        """log_joint computes joint density given all intermediates."""
        ast = parse("""
            object X : 3
            space R : Euclidean(2)
            space S : Euclidean(4)
            continuous f : X -> R ~ Normal
            continuous g : R -> S ~ Normal

            program p : X -> S
                draw y ~ f
                draw z ~ g(y)
                return z
        """)
        env = Compiler(ast).compile_env()
        prog = env["p"]

        x = torch.tensor([0, 1, 2])
        y = env["f"].rsample(x)
        z = env["g"].rsample(y)

        lj = prog.log_joint(x, {"y": y, "z": z})
        assert lj.shape == (3,)
        assert torch.isfinite(lj).all()

    def test_log_prob_raises(self):
        """log_prob raises NotImplementedError for monadic programs."""
        ast = parse("""
            object X : 3
            space R : Euclidean(2)
            continuous f : X -> R ~ Normal

            program p : X -> R
                draw y ~ f
                return y
        """)
        env = Compiler(ast).compile_env()
        prog = env["p"]

        x = torch.tensor([0, 1, 2])
        y = torch.randn(3, 2)

        with pytest.raises(NotImplementedError):
            prog.log_prob(x, y)

    def test_undefined_morphism_error(self):
        """Draw from undefined morphism raises CompileError."""
        with pytest.raises(CompileError):
            Compiler(parse("""
                object X : 3
                space R : Euclidean(2)

                program p : X -> R
                    draw y ~ nonexistent
                    return y
            """)).compile_env()

    def test_undefined_variable_error(self):
        """Draw from undefined variable raises CompileError."""
        with pytest.raises(CompileError):
            Compiler(parse("""
                object X : 3
                space R : Euclidean(2)
                space S : Euclidean(4)
                continuous f : X -> R ~ Normal
                continuous g : R -> S ~ Normal

                program p : X -> S
                    draw y ~ f
                    draw z ~ g(w)
                    return z
            """)).compile_env()

    def test_undefined_return_var_error(self):
        """Return of unbound variable raises CompileError."""
        with pytest.raises(CompileError):
            Compiler(parse("""
                object X : 3
                space R : Euclidean(2)
                continuous f : X -> R ~ Normal

                program p : X -> R
                    draw y ~ f
                    return w
            """)).compile_env()

    def test_duplicate_variable_error(self):
        """Duplicate variable name in draw raises CompileError."""
        with pytest.raises(CompileError):
            Compiler(parse("""
                object X : 3
                space R : Euclidean(2)
                continuous f : X -> R ~ Normal

                program p : X -> R
                    draw y ~ f
                    draw y ~ f
                    return y
            """)).compile_env()

    def test_program_as_output(self):
        """Program can be used as the output expression."""
        ast = parse("""
            object X : 3
            space R : Euclidean(2)
            continuous f : X -> R ~ Normal

            program model : X -> R
                draw y ~ f
                return y

            output model
        """)
        prog = Compiler(ast).compile()
        assert isinstance(prog, Program)

        x = torch.tensor([0, 1, 2])
        out = prog.rsample(x)
        assert out.shape == (3, 2)

    def test_program_in_let_composition(self):
        """Program can be composed via let with >>."""
        ast = parse("""
            object X : 5
            object Y : 3
            space R : Euclidean(2)

            stochastic s : X -> Y
            continuous f : Y -> R ~ Normal

            program gen : Y -> R
                draw z ~ f
                return z

            let pipeline = s >> gen
        """)
        env = Compiler(ast).compile_env()
        pipeline = env["pipeline"]

        x = torch.tensor([0, 1, 2, 3, 4])
        out = pipeline.rsample(x)
        assert out.shape == (5, 2)


class TestPDSFactivityPattern:
    """Tests modeled on the PDS factivity architecture."""

    def test_factivity_prior_fan_out(self):
        """Three independent LogitNormal priors from same entity input.

        This mirrors the PDS pattern::

            let' x (LogitNormal 0 1)
            (let' y (LogitNormal 0 1)
            (let' z (LogitNormal 0 1) ...))
        """
        ast = parse("""
            object Entity : 2
            space Belief : UnitInterval(1)

            continuous prior_x : Entity -> Belief ~ LogitNormal
            continuous prior_y : Entity -> Belief ~ LogitNormal
            continuous prior_z : Entity -> Belief ~ LogitNormal

            program factivity_prior : Entity -> Belief
                draw x ~ prior_x
                draw y ~ prior_y
                draw z ~ prior_z
                return x
        """)
        env = Compiler(ast).compile_env()
        prog = env["factivity_prior"]

        entity = torch.tensor([0, 1])
        out = prog.rsample(entity)
        assert out.shape == (2, 1)

        # all three priors have independent parameters
        n_params = sum(p.numel() for p in prog.parameters())
        assert n_params > 0

    def test_factivity_response(self):
        """Response function: belief -> bounded response via TruncatedNormal.

        Mirrors PDS::

            respond(lam x (Let sigma (Truncate Uniform 0 1)
                          (Truncate (Normal x sigma) 0 1)))
        """
        ast = parse("""
            object Entity : 2
            space Belief : UnitInterval(1)
            space Response : Euclidean(1, low=0.0, high=1.0)

            continuous prior : Entity -> Belief ~ LogitNormal
            continuous respond : Belief -> Response ~ TruncatedNormal

            program model : Entity -> Response
                draw x ~ prior
                draw r ~ respond(x)
                return r
        """)
        env = Compiler(ast).compile_env()
        prog = env["model"]

        entity = torch.tensor([0, 1])
        response = prog.rsample(entity)
        assert response.shape == (2, 1)

    def test_factivity_chained_dependency(self):
        """Chained dependency: x conditions y conditions z.

        Mirrors PDS::

            let' x (LogitNormal 0 1)
            (let' b (Bern x) ...)

        where a continuous draw conditions a later draw.
        """
        ast = parse("""
            object Entity : 2
            space Belief : UnitInterval(1)
            space Response : Euclidean(1, low=0.0, high=1.0)

            continuous prior : Entity -> Belief ~ LogitNormal
            continuous transform : Belief -> Belief ~ Beta
            continuous respond : Belief -> Response ~ TruncatedNormal

            program model : Entity -> Response
                draw x ~ prior
                draw y ~ transform(x)
                draw r ~ respond(y)
                return r
        """)
        env = Compiler(ast).compile_env()
        prog = env["model"]

        entity = torch.tensor([0, 1])
        response = prog.rsample(entity)
        assert response.shape == (2, 1)

    def test_factivity_log_joint_with_all_intermediates(self):
        """Joint density computation with all intermediates observed.

        This supports HMC/NUTS inference over latent variables,
        mirroring how PDS compiles to Stan.
        """
        ast = parse("""
            object Entity : 2
            space Belief : UnitInterval(1)
            space Response : Euclidean(1, low=0.0, high=1.0)

            continuous prior : Entity -> Belief ~ LogitNormal
            continuous respond : Belief -> Response ~ TruncatedNormal

            program model : Entity -> Response
                draw x ~ prior
                draw r ~ respond(x)
                return r
        """)
        env = Compiler(ast).compile_env()
        prog = env["model"]

        entity = torch.tensor([0, 1])
        belief = env["prior"].rsample(entity)
        response = env["respond"].rsample(belief)

        lj = prog.log_joint(entity, {"x": belief, "r": response})
        assert lj.shape == (2,)
        assert torch.isfinite(lj).all()

    def test_factivity_full_pipeline_with_comments(self):
        """Full pipeline with comments, matching PDS structure."""
        source = """
            # === PDS factivity model ===
            # discrete: entities, continuous: beliefs + responses

            object Entity : 2           # j, b

            space Belief : UnitInterval(1)  # belief strength
            space Response : Euclidean(1, low=0.0, high=1.0)

            # three LogitNormal priors (cf. PDS factivityPrior)
            continuous prior_x : Entity -> Belief ~ LogitNormal
            continuous prior_y : Entity -> Belief ~ LogitNormal
            continuous prior_z : Entity -> Belief ~ LogitNormal

            # response function (cf. PDS factivityRespond)
            continuous respond : Belief -> Response ~ TruncatedNormal

            # monadic program: sample priors, generate response
            program factivity : Entity -> Response
                draw x ~ prior_x
                draw y ~ prior_y
                draw z ~ prior_z
                draw r ~ respond(x)
                return r

            output factivity
        """
        prog = Compiler(parse(source)).compile()
        assert isinstance(prog, Program)

        entity = torch.tensor([0, 1])
        response = prog.rsample(entity)
        assert response.shape == (2, 1)


# ===== conditional bernoulli / categorical tests ==============================


class TestConditionalBernoulli:
    """Tests for ConditionalBernoulli (continuous -> discrete bridge)."""

    def test_bernoulli_basic_sample(self):
        """Bernoulli produces {0, 1} samples."""
        from quivers.continuous.families import ConditionalBernoulli
        from quivers.continuous.spaces import UnitInterval
        from quivers.core.objects import FinSet

        dom = UnitInterval(1)
        cod = FinSet("Truth", 2)
        bern = ConditionalBernoulli(dom, cod)

        x = torch.rand(8, 1)
        y = bern.rsample(x)
        assert y.shape == (8,)
        assert y.dtype == torch.long
        assert set(y.tolist()).issubset({0, 1})

    def test_bernoulli_log_prob(self):
        """Log-prob is well-defined for {0, 1} targets."""
        from quivers.continuous.families import ConditionalBernoulli
        from quivers.continuous.spaces import UnitInterval
        from quivers.core.objects import FinSet

        dom = UnitInterval(1)
        cod = FinSet("Truth", 2)
        bern = ConditionalBernoulli(dom, cod)

        x = torch.rand(8, 1)
        y = torch.randint(0, 2, (8,))
        lp = bern.log_prob(x, y)
        assert lp.shape == (8,)
        assert torch.isfinite(lp).all()
        assert (lp <= 0).all()

    def test_bernoulli_requires_finset2(self):
        """Codomain must be FinSet(2)."""
        from quivers.continuous.families import ConditionalBernoulli
        from quivers.continuous.spaces import UnitInterval
        from quivers.core.objects import FinSet

        dom = UnitInterval(1)
        with pytest.raises(ValueError, match="FinSet.*2"):
            ConditionalBernoulli(dom, FinSet("Bad", 3))

    def test_bernoulli_discrete_domain(self):
        """Bernoulli works with discrete (FinSet) domain too."""
        from quivers.continuous.families import ConditionalBernoulli
        from quivers.core.objects import FinSet

        dom = FinSet("Entity", 4)
        cod = FinSet("Truth", 2)
        bern = ConditionalBernoulli(dom, cod)

        x = torch.tensor([0, 1, 2, 3])
        y = bern.rsample(x)
        assert y.shape == (4,)
        assert y.dtype == torch.long

    def test_bernoulli_sample_shape(self):
        """sample_shape adds leading dimensions."""
        from quivers.continuous.families import ConditionalBernoulli
        from quivers.continuous.spaces import UnitInterval
        from quivers.core.objects import FinSet

        dom = UnitInterval(1)
        cod = FinSet("Truth", 2)
        bern = ConditionalBernoulli(dom, cod)

        x = torch.rand(4, 1)
        y = bern.rsample(x, sample_shape=torch.Size([10]))
        assert y.shape == (10, 4)

    def test_bernoulli_in_dsl(self):
        """Bernoulli is accessible from DSL syntax."""
        source = """
            object Entity : 3
            object Truth : 2
            space Belief : UnitInterval(1)

            continuous prior : Entity -> Belief ~ LogitNormal
            continuous bern : Belief -> Truth ~ Bernoulli

            program model : Entity -> Truth
                draw x ~ prior
                draw b ~ bern(x)
                return b
        """
        env = Compiler(parse(source)).compile_env()
        prog = env["model"]
        assert isinstance(prog, MonadicProgram)

        entity = torch.tensor([0, 1, 2])
        result = prog.rsample(entity)
        assert result.shape == (3,)
        assert result.dtype == torch.long
        assert set(result.tolist()).issubset({0, 1})

    def test_bernoulli_log_joint_in_program(self):
        """log_joint works with Bernoulli steps."""
        source = """
            object Entity : 2
            object Truth : 2
            space Belief : UnitInterval(1)

            continuous prior : Entity -> Belief ~ LogitNormal
            continuous bern : Belief -> Truth ~ Bernoulli

            program model : Entity -> Truth
                draw x ~ prior
                draw b ~ bern(x)
                return b
        """
        env = Compiler(parse(source)).compile_env()
        prog = env["model"]

        entity = torch.tensor([0, 1])
        belief = env["prior"].rsample(entity)
        truth = env["bern"].rsample(belief)

        lj = prog.log_joint(entity, {"x": belief, "b": truth})
        assert lj.shape == (2,)
        assert torch.isfinite(lj).all()


class TestConditionalCategorical:
    """Tests for ConditionalCategorical (continuous -> discrete, k >= 2)."""

    def test_categorical_basic_sample(self):
        """Categorical produces {0, ..., k-1} samples."""
        from quivers.continuous.families import ConditionalCategorical
        from quivers.continuous.spaces import Euclidean
        from quivers.core.objects import FinSet

        dom = Euclidean("X", 3)
        cod = FinSet("Color", 5)
        cat = ConditionalCategorical(dom, cod)

        x = torch.randn(8, 3)
        y = cat.rsample(x)
        assert y.shape == (8,)
        assert y.dtype == torch.long
        assert all(0 <= v <= 4 for v in y.tolist())

    def test_categorical_log_prob(self):
        """Log-prob is well-defined for valid categories."""
        from quivers.continuous.families import ConditionalCategorical
        from quivers.continuous.spaces import Euclidean
        from quivers.core.objects import FinSet

        dom = Euclidean("X", 2)
        cod = FinSet("Cat", 4)
        cat = ConditionalCategorical(dom, cod)

        x = torch.randn(8, 2)
        y = torch.randint(0, 4, (8,))
        lp = cat.log_prob(x, y)
        assert lp.shape == (8,)
        assert torch.isfinite(lp).all()
        assert (lp <= 0).all()

    def test_categorical_requires_finset(self):
        """Codomain must be a FinSet."""
        from quivers.continuous.families import ConditionalCategorical
        from quivers.continuous.spaces import Euclidean

        dom = Euclidean("X", 2)
        with pytest.raises(ValueError, match="FinSet"):
            ConditionalCategorical(dom, Euclidean("Y", 3))

    def test_categorical_in_dsl(self):
        """Categorical is accessible from DSL syntax."""
        source = """
            space Input : Euclidean(3)
            object Label : 5

            continuous classify : Input -> Label ~ Categorical
        """
        env = Compiler(parse(source)).compile_env()
        morph = env["classify"]

        x = torch.randn(4, 3)
        y = morph.rsample(x)
        assert y.shape == (4,)
        assert y.dtype == torch.long
        assert all(0 <= v <= 4 for v in y.tolist())

    def test_categorical_sample_shape(self):
        """sample_shape adds leading dimensions."""
        from quivers.continuous.families import ConditionalCategorical
        from quivers.continuous.spaces import Euclidean
        from quivers.core.objects import FinSet

        dom = Euclidean("X", 2)
        cod = FinSet("Cat", 4)
        cat = ConditionalCategorical(dom, cod)

        x = torch.randn(6, 2)
        y = cat.rsample(x, sample_shape=torch.Size([5]))
        assert y.shape == (5, 6)


class TestPDSFaithfulFactivity:
    """Faithful reproduction of the PDS factivity model structure.

    The PDS model (Grove & White 2025) has this structure:

        factivityPrior =
          let' x (LogitNormal 0 1)        -- content belief strength
          (let' y (LogitNormal 0 1)        -- ling projection strength
          (let' z (LogitNormal 0 1)        -- epi projection strength
          (let' b (Bern x)                 -- whether speaker knows content
          (Return (UpdCG ..., UpdTauKnow b)))))

    The inner UpdCG contains a nested monadic program:
          let' c (Bern y)                  -- ling projection draw
          let' d (Bern z)                  -- epi projection draw
          Return (UpdLing c, UpdEpi d)

    factivityRespond = respond (\\x -> Truncate (Normal x sigma) 0 1)
        where sigma ~ Truncate Uniform 0 1

    We model this as a flat program with three LogitNormal draws and
    three Bernoulli draws, since quivers does not yet support nested
    programs or state update constructors.
    """

    def test_pds_factivity_prior_structure(self):
        """The factivity prior has 6 draws: x, y, z (continuous) then b, c, d (discrete)."""
        source = """
            # PDS factivity prior (Grove & White 2025)
            # three LogitNormal draws for belief strengths
            # three Bernoulli draws for truth-value projections

            object Entity : 2
            object Truth : 2

            space Belief : UnitInterval(1)

            # continuous priors (content, ling, epi belief strengths)
            continuous prior_x : Entity -> Belief ~ LogitNormal
            continuous prior_y : Entity -> Belief ~ LogitNormal
            continuous prior_z : Entity -> Belief ~ LogitNormal

            # bernoulli draws (know, ling projection, epi projection)
            continuous bern_b : Belief -> Truth ~ Bernoulli
            continuous bern_c : Belief -> Truth ~ Bernoulli
            continuous bern_d : Belief -> Truth ~ Bernoulli

            # the full prior program
            program factivityPrior : Entity -> Truth
                draw x ~ prior_x
                draw y ~ prior_y
                draw z ~ prior_z
                draw b ~ bern_b(x)
                draw c ~ bern_c(y)
                draw d ~ bern_d(z)
                return b
        """
        env = Compiler(parse(source)).compile_env()
        prog = env["factivityPrior"]
        assert isinstance(prog, MonadicProgram)

        # 6 steps: x, y, z, b, c, d
        assert len(prog._step_specs) == 6

        entity = torch.tensor([0, 1])
        result = prog.rsample(entity)
        assert result.shape == (2,)
        assert result.dtype == torch.long
        assert set(result.tolist()).issubset({0, 1})

    def test_pds_factivity_respond(self):
        """The response function: Truncate(Normal(x, sigma), 0, 1)."""
        source = """
            space Belief : UnitInterval(1)
            space Response : Euclidean(1, low=0.0, high=1.0)

            # PDS: respond (\\x -> Truncate (Normal x sigma) 0 1)
            continuous respond : Belief -> Response ~ TruncatedNormal
        """
        env = Compiler(parse(source)).compile_env()
        respond = env["respond"]

        x = torch.rand(8, 1)
        y = respond.rsample(x)
        assert y.shape == (8, 1)
        # truncated to [0, 1]
        assert (y >= 0.0).all()
        assert (y <= 1.0).all()

    def test_pds_full_factivity_model(self):
        """Full PDS factivity: prior + response in one program."""
        source = """
            # PDS factivity model (Grove & White 2025)
            object Entity : 2
            object Truth : 2
            space Belief : UnitInterval(1)
            space Response : Euclidean(1, low=0.0, high=1.0)

            # prior components
            continuous prior_x : Entity -> Belief ~ LogitNormal
            continuous prior_y : Entity -> Belief ~ LogitNormal
            continuous prior_z : Entity -> Belief ~ LogitNormal

            # bernoulli draws
            continuous bern_b : Belief -> Truth ~ Bernoulli
            continuous bern_c : Belief -> Truth ~ Bernoulli
            continuous bern_d : Belief -> Truth ~ Bernoulli

            # response function
            continuous respond : Belief -> Response ~ TruncatedNormal

            # full model: sample priors, draw Bernoullis, produce response
            program factivity : Entity -> Response
                draw x ~ prior_x
                draw y ~ prior_y
                draw z ~ prior_z
                draw b ~ bern_b(x)
                draw c ~ bern_c(y)
                draw d ~ bern_d(z)
                draw r ~ respond(x)
                return r
        """
        env = Compiler(parse(source)).compile_env()
        prog = env["factivity"]

        entity = torch.tensor([0, 1])
        response = prog.rsample(entity)
        assert response.shape == (2, 1)
        assert (response >= 0.0).all()
        assert (response <= 1.0).all()

    def test_pds_factivity_log_joint(self):
        """log_joint is computable when all intermediates are given."""
        source = """
            object Entity : 2
            object Truth : 2
            space Belief : UnitInterval(1)
            space Response : Euclidean(1, low=0.0, high=1.0)

            continuous prior_x : Entity -> Belief ~ LogitNormal
            continuous prior_y : Entity -> Belief ~ LogitNormal
            continuous prior_z : Entity -> Belief ~ LogitNormal
            continuous bern_b : Belief -> Truth ~ Bernoulli
            continuous bern_c : Belief -> Truth ~ Bernoulli
            continuous bern_d : Belief -> Truth ~ Bernoulli
            continuous respond : Belief -> Response ~ TruncatedNormal

            program factivity : Entity -> Response
                draw x ~ prior_x
                draw y ~ prior_y
                draw z ~ prior_z
                draw b ~ bern_b(x)
                draw c ~ bern_c(y)
                draw d ~ bern_d(z)
                draw r ~ respond(x)
                return r
        """
        env = Compiler(parse(source)).compile_env()
        prog = env["factivity"]

        entity = torch.tensor([0, 1])
        x = env["prior_x"].rsample(entity)
        y = env["prior_y"].rsample(entity)
        z = env["prior_z"].rsample(entity)
        b = env["bern_b"].rsample(x)
        c = env["bern_c"].rsample(y)
        d = env["bern_d"].rsample(z)
        r = env["respond"].rsample(x)

        intermediates = {
            "x": x, "y": y, "z": z,
            "b": b, "c": c, "d": d,
            "r": r,
        }
        lj = prog.log_joint(entity, intermediates)
        assert lj.shape == (2,)
        assert torch.isfinite(lj).all()

    def test_pds_factivity_parameters_learnable(self):
        """All morphism parameters are visible to optimizer."""
        source = """
            object Entity : 2
            object Truth : 2
            space Belief : UnitInterval(1)

            continuous prior_x : Entity -> Belief ~ LogitNormal
            continuous bern_b : Belief -> Truth ~ Bernoulli

            program model : Entity -> Truth
                draw x ~ prior_x
                draw b ~ bern_b(x)
                return b
        """
        env = Compiler(parse(source)).compile_env()
        prog = env["model"]

        params = list(prog.parameters())
        assert len(params) > 0

        # prior_x has embedding params, bern_b has neural net params
        param_names = [n for n, _ in prog.named_parameters()]
        assert any("prior_x" in n or "_step_x" in n for n in param_names)
        assert any("bern_b" in n or "_step_b" in n for n in param_names)


# ===== tuple returns, product types, named params, destructuring =============


class TestParserTupleFeatures:
    """Parser tests for tuple returns, product headers, named params."""

    def test_tuple_return(self):
        """Parse return (x, y, z)."""
        ast = parse("""
            object X : 2
            object T : 2
            space B : UnitInterval(1)
            continuous f : X -> B ~ LogitNormal
            continuous g : B -> T ~ Bernoulli

            program p : X -> T
                draw x ~ f
                draw b ~ g(x)
                return (x, b)
        """)
        prog_stmt = ast.statements[5]
        assert isinstance(prog_stmt, ProgramDecl)
        assert prog_stmt.return_vars == ("x", "b")

    def test_product_codomain(self):
        """Parse program with product codomain: A -> B * C."""
        ast = parse("""
            object X : 2
            object T : 2
            space B : UnitInterval(1)
            continuous f : X -> B ~ LogitNormal
            continuous g : B -> T ~ Bernoulli

            program p : X -> B * T
                draw x ~ f
                draw b ~ g(x)
                return (x, b)
        """)
        prog_stmt = ast.statements[5]
        assert isinstance(prog_stmt, ProgramDecl)
        assert isinstance(prog_stmt.codomain, TypeProduct)
        assert len(prog_stmt.codomain.components) == 2

    def test_product_domain(self):
        """Parse program with product domain: A * B -> C."""
        ast = parse("""
            object T : 2
            space B : UnitInterval(1)
            continuous bern : B -> T ~ Bernoulli

            program p : B * B -> T
                draw c ~ bern
                return c
        """)
        prog_stmt = ast.statements[3]
        assert isinstance(prog_stmt, ProgramDecl)
        assert isinstance(prog_stmt.domain, TypeProduct)

    def test_named_params(self):
        """Parse program with named parameters."""
        ast = parse("""
            object T : 2
            space B : UnitInterval(1)
            continuous bern : B -> T ~ Bernoulli

            program p(y, z) : B * B -> T * T
                draw c ~ bern(y)
                draw d ~ bern(z)
                return (c, d)
        """)
        prog_stmt = ast.statements[3]
        assert isinstance(prog_stmt, ProgramDecl)
        assert prog_stmt.params == ("y", "z")

    def test_destructuring_draw(self):
        """Parse draw (a, b) ~ morphism."""
        ast = parse("""
            object T : 2
            space B : UnitInterval(1)
            continuous f : T -> B ~ LogitNormal
            continuous g : B -> T ~ Bernoulli

            program sub : T -> B
                draw x ~ f
                return x

            program p : T -> B
                draw (a,) ~ sub
                return a
        """)
        # statements: object, space, continuous f, continuous g, program sub, program p
        prog_stmt = ast.statements[5]
        # the draw (a,) is a single-element tuple for testing syntax
        assert prog_stmt.draws[0].vars == ("a",)

    def test_multi_arg_draw(self):
        """Parse draw z ~ f(x, y)."""
        ast = parse("""
            object T : 2
            space B : UnitInterval(1)
            continuous f : T -> B ~ LogitNormal
            continuous g : B -> T ~ Bernoulli

            program sub(a, b) : B * B -> T
                draw c ~ g(a)
                return c

            program p : T -> T
                draw x ~ f
                draw y ~ f
                draw z ~ sub(x, y)
                return z
        """)
        # statements: object, space, continuous f, continuous g, program sub, program p
        prog_stmt = ast.statements[5]
        assert prog_stmt.draws[2].args == ("x", "y")


class TestCompilerTupleFeatures:
    """Compiler tests for tuple returns, product types, named params."""

    def test_tuple_return_compiles(self):
        """Tuple-returning program compiles."""
        source = """
            object Entity : 2
            space Belief : UnitInterval(1)
            continuous f : Entity -> Belief ~ LogitNormal

            program p : Entity -> Belief * Belief
                draw x ~ f
                draw y ~ f
                return (x, y)
        """
        env = Compiler(parse(source)).compile_env()
        prog = env["p"]
        assert isinstance(prog, MonadicProgram)
        assert not prog._return_is_single
        assert prog._return_vars == ("x", "y")

    def test_named_params_compiles(self):
        """Named-param sub-program compiles and runs."""
        source = """
            object Truth : 2
            space Belief : UnitInterval(1)
            continuous bern : Belief -> Truth ~ Bernoulli

            program sub(y, z) : Belief * Belief -> Truth * Truth
                draw c ~ bern(y)
                draw d ~ bern(z)
                return (c, d)
        """
        env = Compiler(parse(source)).compile_env()
        prog = env["sub"]
        assert isinstance(prog, MonadicProgram)
        assert prog._params == ("y", "z")

    def test_named_params_count_mismatch(self):
        """Error when param count doesn't match domain components."""
        source = """
            object Truth : 2
            space Belief : UnitInterval(1)
            continuous bern : Belief -> Truth ~ Bernoulli

            program sub(y, z, w) : Belief * Belief -> Truth
                draw c ~ bern(y)
                return c
        """
        with pytest.raises(CompileError, match="3 params"):
            Compiler(parse(source)).compile_env()

    def test_unbound_return_var_error(self):
        """Error when return variable is not bound."""
        source = """
            object Entity : 2
            space Belief : UnitInterval(1)
            continuous f : Entity -> Belief ~ LogitNormal

            program p : Entity -> Belief * Belief
                draw x ~ f
                return (x, y)
        """
        with pytest.raises(CompileError, match="not bound"):
            Compiler(parse(source)).compile_env()

    def test_destructuring_draw_compiles(self):
        """Destructuring draw from sub-program compiles."""
        source = """
            object Entity : 2
            object Truth : 2
            space Belief : UnitInterval(1)

            continuous prior : Entity -> Belief ~ LogitNormal
            continuous bern : Belief -> Truth ~ Bernoulli

            program sub(y, z) : Belief * Belief -> Truth * Truth
                draw c ~ bern(y)
                draw d ~ bern(z)
                return (c, d)

            program outer : Entity -> Truth * Truth
                draw y ~ prior
                draw z ~ prior
                draw (c, d) ~ sub(y, z)
                return (c, d)
        """
        env = Compiler(parse(source)).compile_env()
        outer = env["outer"]
        assert isinstance(outer, MonadicProgram)


class TestExecutionTupleFeatures:
    """Execution tests for tuple returns, named params, destructuring."""

    def test_tuple_return_rsample(self):
        """Tuple-returning program rsample returns dict."""
        source = """
            object Entity : 2
            space Belief : UnitInterval(1)
            continuous f : Entity -> Belief ~ LogitNormal
            continuous g : Entity -> Belief ~ LogitNormal

            program p : Entity -> Belief * Belief
                draw x ~ f
                draw y ~ g
                return (x, y)
        """
        env = Compiler(parse(source)).compile_env()
        prog = env["p"]

        entity = torch.tensor([0, 1])
        result = prog.rsample(entity)

        assert isinstance(result, dict)
        assert set(result.keys()) == {"x", "y"}
        assert result["x"].shape == (2, 1)
        assert result["y"].shape == (2, 1)

    def test_single_return_still_tensor(self):
        """Single-return program still returns a tensor (backward compat)."""
        source = """
            object Entity : 2
            space Belief : UnitInterval(1)
            continuous f : Entity -> Belief ~ LogitNormal

            program p : Entity -> Belief
                draw x ~ f
                return x
        """
        env = Compiler(parse(source)).compile_env()
        prog = env["p"]

        entity = torch.tensor([0, 1])
        result = prog.rsample(entity)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 1)

    def test_named_params_rsample(self):
        """Named-param sub-program splits input correctly."""
        source = """
            object Truth : 2
            space Belief : UnitInterval(1)
            continuous bern_c : Belief -> Truth ~ Bernoulli
            continuous bern_d : Belief -> Truth ~ Bernoulli

            program sub(y, z) : Belief * Belief -> Truth * Truth
                draw c ~ bern_c(y)
                draw d ~ bern_d(z)
                return (c, d)
        """
        env = Compiler(parse(source)).compile_env()
        prog = env["sub"]

        # input is (batch, 2) — two beliefs stacked
        x = torch.rand(4, 2)
        result = prog.rsample(x)

        assert isinstance(result, dict)
        assert set(result.keys()) == {"c", "d"}
        assert result["c"].shape == (4,)
        assert result["d"].shape == (4,)
        assert result["c"].dtype == torch.long
        assert result["d"].dtype == torch.long

    def test_multi_arg_draw(self):
        """Multi-arg draw stacks inputs for sub-program."""
        source = """
            object Entity : 2
            object Truth : 2
            space Belief : UnitInterval(1)

            continuous prior : Entity -> Belief ~ LogitNormal
            continuous bern_c : Belief -> Truth ~ Bernoulli
            continuous bern_d : Belief -> Truth ~ Bernoulli

            program sub(y, z) : Belief * Belief -> Truth * Truth
                draw c ~ bern_c(y)
                draw d ~ bern_d(z)
                return (c, d)

            program outer : Entity -> Truth * Truth
                draw y ~ prior
                draw z ~ prior
                draw (c, d) ~ sub(y, z)
                return (c, d)
        """
        env = Compiler(parse(source)).compile_env()
        prog = env["outer"]

        entity = torch.tensor([0, 1])
        result = prog.rsample(entity)

        assert isinstance(result, dict)
        assert set(result.keys()) == {"c", "d"}
        assert result["c"].shape == (2,)
        assert result["d"].shape == (2,)

    def test_log_joint_tuple_return(self):
        """log_joint works with tuple-returning programs."""
        source = """
            object Entity : 2
            space Belief : UnitInterval(1)
            continuous f : Entity -> Belief ~ LogitNormal
            continuous g : Entity -> Belief ~ LogitNormal

            program p : Entity -> Belief * Belief
                draw x ~ f
                draw y ~ g
                return (x, y)
        """
        env = Compiler(parse(source)).compile_env()
        prog = env["p"]

        entity = torch.tensor([0, 1])
        x = env["f"].rsample(entity)
        y = env["g"].rsample(entity)

        lj = prog.log_joint(entity, {"x": x, "y": y})
        assert lj.shape == (2,)
        assert torch.isfinite(lj).all()

    def test_log_joint_nested_programs(self):
        """log_joint works with nested sub-programs."""
        source = """
            object Entity : 2
            object Truth : 2
            space Belief : UnitInterval(1)

            continuous prior : Entity -> Belief ~ LogitNormal
            continuous bern_c : Belief -> Truth ~ Bernoulli
            continuous bern_d : Belief -> Truth ~ Bernoulli

            program sub(y, z) : Belief * Belief -> Truth * Truth
                draw c ~ bern_c(y)
                draw d ~ bern_d(z)
                return (c, d)

            program outer : Entity -> Truth * Truth
                draw y ~ prior
                draw z ~ prior
                draw (c, d) ~ sub(y, z)
                return (c, d)
        """
        env = Compiler(parse(source)).compile_env()
        prog = env["outer"]

        entity = torch.tensor([0, 1])
        y_val = env["prior"].rsample(entity)
        z_val = env["prior"].rsample(entity)
        c_val = env["bern_c"].rsample(y_val)
        d_val = env["bern_d"].rsample(z_val)

        lj = prog.log_joint(entity, {
            "y": y_val, "z": z_val,
            "c": c_val, "d": d_val,
        })
        assert lj.shape == (2,)
        assert torch.isfinite(lj).all()

    def test_pds_factivity_with_nesting(self):
        """Full PDS factivity with nested sub-programs and tuple returns."""
        source = """
            # PDS factivity model (Grove & White 2025)
            # with nested sub-programs for CG and TauKnow updates

            object Entity : 2
            object Truth : 2
            space Belief : UnitInterval(1)
            space Response : Euclidean(1, low=0.0, high=1.0)

            # prior morphisms
            continuous prior_x : Entity -> Belief ~ LogitNormal
            continuous prior_y : Entity -> Belief ~ LogitNormal
            continuous prior_z : Entity -> Belief ~ LogitNormal

            # bernoulli bridges (continuous -> discrete)
            continuous bern_b : Belief -> Truth ~ Bernoulli
            continuous bern_c : Belief -> Truth ~ Bernoulli
            continuous bern_d : Belief -> Truth ~ Bernoulli

            # response function
            continuous respond : Belief -> Response ~ TruncatedNormal

            # inner CG update sub-program
            # corresponds to PDS: let' c (Bern y) (let' d (Bern z) ...)
            program cg_update(y, z) : Belief * Belief -> Truth * Truth
                draw c ~ bern_c(y)
                draw d ~ bern_d(z)
                return (c, d)

            # outer factivity prior
            # corresponds to PDS factivityPrior
            program factivityPrior : Entity -> Truth * Truth * Truth * Response
                draw x ~ prior_x
                draw y ~ prior_y
                draw z ~ prior_z
                draw b ~ bern_b(x)
                draw (c, d) ~ cg_update(y, z)
                draw r ~ respond(x)
                return (b, c, d, r)

            output factivityPrior
        """
        prog = Compiler(parse(source)).compile()
        assert isinstance(prog, Program)

        entity = torch.tensor([0, 1])
        result = prog.rsample(entity)

        # result is a dict with b, c, d (discrete), r (continuous)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"b", "c", "d", "r"}

        # b, c, d are discrete truth values
        for key in ("b", "c", "d"):
            assert result[key].dtype == torch.long
            assert result[key].shape == (2,)
            assert set(result[key].tolist()).issubset({0, 1})

        # r is continuous response in [0, 1]
        assert result["r"].shape == (2, 1)
        assert (result["r"] >= 0.0).all()
        assert (result["r"] <= 1.0).all()

    def test_pds_factivity_log_joint_nested(self):
        """log_joint with the full nested PDS factivity model."""
        source = """
            object Entity : 2
            object Truth : 2
            space Belief : UnitInterval(1)
            space Response : Euclidean(1, low=0.0, high=1.0)

            continuous prior_x : Entity -> Belief ~ LogitNormal
            continuous prior_y : Entity -> Belief ~ LogitNormal
            continuous prior_z : Entity -> Belief ~ LogitNormal
            continuous bern_b : Belief -> Truth ~ Bernoulli
            continuous bern_c : Belief -> Truth ~ Bernoulli
            continuous bern_d : Belief -> Truth ~ Bernoulli
            continuous respond : Belief -> Response ~ TruncatedNormal

            program cg_update(y, z) : Belief * Belief -> Truth * Truth
                draw c ~ bern_c(y)
                draw d ~ bern_d(z)
                return (c, d)

            program factivityPrior : Entity -> Truth * Truth * Truth * Response
                draw x ~ prior_x
                draw y ~ prior_y
                draw z ~ prior_z
                draw b ~ bern_b(x)
                draw (c, d) ~ cg_update(y, z)
                draw r ~ respond(x)
                return (b, c, d, r)
        """
        env = Compiler(parse(source)).compile_env()
        prog = env["factivityPrior"]

        entity = torch.tensor([0, 1])
        x = env["prior_x"].rsample(entity)
        y = env["prior_y"].rsample(entity)
        z = env["prior_z"].rsample(entity)
        b = env["bern_b"].rsample(x)
        c = env["bern_c"].rsample(y)
        d = env["bern_d"].rsample(z)
        r = env["respond"].rsample(x)

        intermediates = {
            "x": x, "y": y, "z": z,
            "b": b, "c": c, "d": d, "r": r,
        }
        lj = prog.log_joint(entity, intermediates)
        assert lj.shape == (2,)
        assert torch.isfinite(lj).all()


# ==========================================================================
# inline distribution tests
# ==========================================================================


class TestParserInlineDistributions:
    """Test parsing of inline distribution syntax."""

    def test_draw_with_float_args(self):
        """Inline draw with all-float args."""
        source = """
            object Entity : 2
            space Belief : UnitInterval()
            program p : Entity -> Belief
                draw x ~ LogitNormal(0.0, 1.0)
                return x
        """
        mod = parse(source)
        prog = [s for s in mod.statements if isinstance(s, ProgramDecl)][0]
        draw = prog.draws[0]
        assert draw.morphism == "LogitNormal"
        assert draw.args == (0.0, 1.0)
        assert draw.vars == ("x",)

    def test_draw_with_variable_args(self):
        """Inline draw with variable reference."""
        source = """
            object Entity : 2
            object Truth : 2
            space Belief : UnitInterval()
            program p : Entity -> Truth
                draw x ~ LogitNormal(0.0, 1.0)
                draw b ~ Bernoulli(x)
                return b
        """
        mod = parse(source)
        prog = [s for s in mod.statements if isinstance(s, ProgramDecl)][0]
        draw_b = prog.draws[1]
        assert draw_b.morphism == "Bernoulli"
        assert draw_b.args == ("x",)

    def test_draw_with_mixed_args(self):
        """Inline draw with mixed variable and float args."""
        source = """
            object Entity : 2
            space Resp : Euclidean(1, low=0.0, high=1.0)
            space Belief : UnitInterval()
            program p : Entity -> Resp
                draw x ~ LogitNormal(0.0, 1.0)
                draw sigma ~ Uniform(0.0, 1.0)
                draw r ~ TruncatedNormal(x, sigma, 0.0, 1.0)
                return r
        """
        mod = parse(source)
        prog = [s for s in mod.statements if isinstance(s, ProgramDecl)][0]
        draw_r = prog.draws[2]
        assert draw_r.morphism == "TruncatedNormal"
        assert draw_r.args == ("x", "sigma", 0.0, 1.0)

    def test_draw_with_int_args(self):
        """INT tokens in args are parsed as floats."""
        source = """
            object Entity : 2
            space Belief : UnitInterval()
            program p : Entity -> Belief
                draw x ~ LogitNormal(0, 1)
                return x
        """
        mod = parse(source)
        prog = [s for s in mod.statements if isinstance(s, ProgramDecl)][0]
        draw = prog.draws[0]
        assert draw.args == (0.0, 1.0)


class TestParserLabeledReturns:
    """Test parsing of labeled return syntax."""

    def test_labeled_return(self):
        """Return with labels: return (a: x, b: y)."""
        source = """
            object Entity : 2
            object Truth : 2
            space Belief : UnitInterval()
            continuous f : Entity -> Belief ~ LogitNormal
            continuous g : Entity -> Belief ~ LogitNormal
            program p : Entity -> Truth * Truth
                draw x ~ f
                draw y ~ g
                return (state: x, prob: y)
        """
        mod = parse(source)
        prog = [s for s in mod.statements if isinstance(s, ProgramDecl)][0]
        assert prog.return_vars == ("x", "y")
        assert prog.return_labels == ("state", "prob")

    def test_unlabeled_return_still_works(self):
        """Regular unlabeled return (backward compat)."""
        source = """
            object Entity : 2
            space Belief : UnitInterval()
            continuous f : Entity -> Belief ~ LogitNormal
            continuous g : Entity -> Belief ~ LogitNormal
            program p : Entity -> Belief * Belief
                draw x ~ f
                draw y ~ g
                return (x, y)
        """
        mod = parse(source)
        prog = [s for s in mod.statements if isinstance(s, ProgramDecl)][0]
        assert prog.return_vars == ("x", "y")
        assert prog.return_labels is None

    def test_single_return_no_labels(self):
        """Single return can't have labels."""
        source = """
            object Entity : 2
            space Belief : UnitInterval()
            continuous f : Entity -> Belief ~ LogitNormal
            program p : Entity -> Belief
                draw x ~ f
                return x
        """
        mod = parse(source)
        prog = [s for s in mod.statements if isinstance(s, ProgramDecl)][0]
        assert prog.return_vars == ("x",)
        assert prog.return_labels is None


class TestCompilerInlineDistributions:
    """Test compilation of inline distribution draw steps."""

    def test_fixed_logitnormal(self):
        """All-float args compile to FixedDistribution."""
        source = """
            object Entity : 2
            space Belief : UnitInterval()
            program p : Entity -> Belief
                draw x ~ LogitNormal(0.0, 1.0)
                return x
        """
        env = Compiler(parse(source)).compile_env()
        prog = env["p"]
        assert isinstance(prog, MonadicProgram)

        # test sampling
        entity = torch.tensor([0, 1, 2])
        samples = prog.rsample(entity)
        assert samples.shape == (3, 1)
        # logitnormal values should be in (0, 1)
        assert (samples > 0).all() and (samples < 1).all()

    def test_fixed_uniform(self):
        """Fixed Uniform(0.0, 1.0)."""
        source = """
            object Entity : 2
            space Belief : UnitInterval()
            program p : Entity -> Belief
                draw sigma ~ Uniform(0.0, 1.0)
                return sigma
        """
        env = Compiler(parse(source)).compile_env()
        prog = env["p"]
        entity = torch.tensor([0, 1, 2])
        samples = prog.rsample(entity)
        assert samples.shape == (3, 1)
        assert (samples >= 0).all() and (samples <= 1).all()

    def test_direct_bernoulli(self):
        """Variable arg compiles to DirectBernoulli."""
        source = """
            object Entity : 2
            object Truth : 2
            space Belief : UnitInterval()
            program p : Entity -> Truth
                draw x ~ LogitNormal(0.0, 1.0)
                draw b ~ Bernoulli(x)
                return b
        """
        env = Compiler(parse(source)).compile_env()
        prog = env["p"]
        entity = torch.tensor([0, 1, 2])
        samples = prog.rsample(entity)
        # bernoulli output should be in {0, 1}
        assert samples.shape == (3,)
        assert set(samples.tolist()).issubset({0, 1})

    def test_direct_truncated_normal(self):
        """Mixed args: TruncatedNormal(var, var, float, float)."""
        source = """
            object Entity : 2
            space Resp : Euclidean(1, low=0.0, high=1.0)
            space Belief : UnitInterval()
            program p : Entity -> Resp
                draw mu ~ LogitNormal(0.0, 1.0)
                draw sigma ~ Uniform(0.0, 1.0)
                draw r ~ TruncatedNormal(mu, sigma, 0.0, 1.0)
                return r
        """
        env = Compiler(parse(source)).compile_env()
        prog = env["p"]
        entity = torch.tensor([0, 1, 2])
        samples = prog.rsample(entity)
        assert samples.shape == (3, 1)
        # truncated normal should be in [0, 1]
        assert (samples >= 0).all() and (samples <= 1).all()

    def test_inline_with_named_morphism_precedence(self):
        """Named morphism takes precedence over inline family."""
        source = """
            object Entity : 2
            space Belief : UnitInterval()
            continuous Uniform : Entity -> Belief ~ Uniform
            program p : Entity -> Belief
                draw x ~ Uniform
                return x
        """
        env = Compiler(parse(source)).compile_env()
        prog = env["p"]
        # should use the named morphism "Uniform", not inline
        entity = torch.tensor([0, 1])
        samples = prog.rsample(entity)
        assert samples.shape == (2, 1)

    def test_float_args_not_allowed_for_named_morphism(self):
        """Float literals in args for named morphisms should error."""
        source = """
            object Entity : 2
            space Belief : UnitInterval()
            continuous f : Entity -> Belief ~ Normal
            program p : Entity -> Belief
                draw x ~ f(0.0, 1.0)
                return x
        """
        with pytest.raises(CompileError, match="literal argument"):
            Compiler(parse(source)).compile_env()


class TestExecutionInlineDistributions:
    """Test end-to-end execution of inline distributions."""

    def test_two_stage_response(self):
        """sigma ~ Uniform then response ~ TruncatedNormal (PDS pattern)."""
        source = """
            object Entity : 2
            space Belief : UnitInterval()
            space Resp : Euclidean(1, low=0.0, high=1.0)
            program response_kernel : Entity -> Resp
                draw mu ~ LogitNormal(0.0, 1.0)
                draw sigma ~ Uniform(0.0, 1.0)
                draw r ~ TruncatedNormal(mu, sigma, 0.0, 1.0)
                return r
        """
        env = Compiler(parse(source)).compile_env()
        prog = env["response_kernel"]

        entity = torch.tensor([0, 1, 2, 3])
        samples = prog.rsample(entity)
        assert samples.shape == (4, 1)
        assert (samples >= 0).all() and (samples <= 1).all()

    def test_labeled_return_dict_keys(self):
        """Labeled returns produce dict with label keys."""
        source = """
            object Entity : 2
            object Truth : 2
            space Belief : UnitInterval()
            program p : Entity -> Truth * Belief
                draw x ~ LogitNormal(0.0, 1.0)
                draw b ~ Bernoulli(x)
                return (state: b, prob: x)
        """
        env = Compiler(parse(source)).compile_env()
        prog = env["p"]
        entity = torch.tensor([0, 1])
        result = prog.rsample(entity)
        assert isinstance(result, dict)
        assert "state" in result
        assert "prob" in result
        assert set(result["state"].tolist()).issubset({0, 1})
        assert (result["prob"] > 0).all() and (result["prob"] < 1).all()

    def test_full_pds_factivity_inline(self):
        """Full PDS factivity model with inline distributions.

        This is the corrected version using:
        - unconditional LogitNormal priors (fixed mu=0, sigma=1)
        - direct Bernoulli (x used as probability)
        - two-stage response: sigma ~ Uniform, r ~ TruncatedNormal
        - labeled returns for semantic structure
        - nested sub-programs
        """
        source = """
            object Entity : 2
            object Truth : 2

            space Belief : UnitInterval()
            space Resp : Euclidean(1, low=0.0, high=1.0)

            # inner CG update sub-program
            program cg_update(y, z) : Belief * Belief -> Truth * Truth
                draw c ~ Bernoulli(y)
                draw d ~ Bernoulli(z)
                return (c, d)

            # response kernel with two-stage randomness
            program response_kernel : Entity -> Resp
                draw mu ~ LogitNormal(0.0, 1.0)
                draw sigma ~ Uniform(0.0, 1.0)
                draw r ~ TruncatedNormal(mu, sigma, 0.0, 1.0)
                return r

            # full factivity prior
            program factivityPrior : Entity -> Truth * Truth * Truth * Resp
                draw x ~ LogitNormal(0.0, 1.0)
                draw y ~ LogitNormal(0.0, 1.0)
                draw z ~ LogitNormal(0.0, 1.0)
                draw b ~ Bernoulli(x)
                draw (c, d) ~ cg_update(y, z)
                draw r ~ response_kernel
                return (tau_know: b, cg_c: c, cg_d: d, response: r)
        """
        env = Compiler(parse(source)).compile_env()
        prog = env["factivityPrior"]

        # test forward sampling
        entity = torch.tensor([0, 1])
        result = prog.rsample(entity)
        assert isinstance(result, dict)

        # check labeled keys
        assert "tau_know" in result
        assert "cg_c" in result
        assert "cg_d" in result
        assert "response" in result

        # check value ranges
        assert set(result["tau_know"].tolist()).issubset({0, 1})
        assert set(result["cg_c"].tolist()).issubset({0, 1})
        assert set(result["cg_d"].tolist()).issubset({0, 1})
        assert (result["response"] >= 0).all()
        assert (result["response"] <= 1).all()

    def test_pds_factivity_log_joint(self):
        """log_joint works with inline distributions and labels."""
        source = """
            object Entity : 2
            object Truth : 2
            space Belief : UnitInterval()

            program p : Entity -> Truth * Belief
                draw x ~ LogitNormal(0.0, 1.0)
                draw b ~ Bernoulli(x)
                return (state: b, prob: x)
        """
        env = Compiler(parse(source)).compile_env()
        prog = env["p"]

        entity = torch.tensor([0, 1])

        # sample forward to get intermediates
        result = prog.rsample(entity)
        x_val = result["prob"]
        b_val = result["state"]

        # log_joint accepts variable names
        intermediates = {"x": x_val, "b": b_val}
        lj = prog.log_joint(entity, intermediates)
        assert lj.shape == (2,)
        assert torch.isfinite(lj).all()

    def test_pds_factivity_log_joint_with_labels(self):
        """log_joint also accepts label keys."""
        source = """
            object Entity : 2
            object Truth : 2
            space Belief : UnitInterval()

            program p : Entity -> Truth * Belief
                draw x ~ LogitNormal(0.0, 1.0)
                draw b ~ Bernoulli(x)
                return (state: b, prob: x)
        """
        env = Compiler(parse(source)).compile_env()
        prog = env["p"]

        entity = torch.tensor([0, 1])
        result = prog.rsample(entity)

        # pass intermediates with label keys
        intermediates = {"prob": result["prob"], "state": result["state"]}
        lj = prog.log_joint(entity, intermediates)
        assert lj.shape == (2,)
        assert torch.isfinite(lj).all()

    def test_inline_fixed_bernoulli(self):
        """Fixed Bernoulli(0.5) with literal probability."""
        source = """
            object Entity : 2
            object Truth : 2
            program p : Entity -> Truth
                draw b ~ Bernoulli(0.5)
                return b
        """
        env = Compiler(parse(source)).compile_env()
        prog = env["p"]
        entity = torch.tensor([0, 1, 2])
        samples = prog.rsample(entity)
        assert samples.shape == (3,)
        assert set(samples.tolist()).issubset({0, 1})

    def test_inline_fixed_normal(self):
        """Fixed Normal(0.0, 1.0)."""
        source = """
            object Entity : 2
            space R : Euclidean(1)
            program p : Entity -> R
                draw x ~ Normal(0.0, 1.0)
                return x
        """
        env = Compiler(parse(source)).compile_env()
        prog = env["p"]
        entity = torch.tensor([0, 1, 2])
        samples = prog.rsample(entity)
        assert samples.shape == (3, 1)

    def test_multiple_unconditional_draws(self):
        """Multiple independent unconditional draws."""
        source = """
            object Entity : 2
            space Belief : UnitInterval()
            program p : Entity -> Belief * Belief * Belief
                draw x ~ LogitNormal(0.0, 1.0)
                draw y ~ LogitNormal(0.0, 1.0)
                draw z ~ LogitNormal(0.0, 1.0)
                return (x, y, z)
        """
        env = Compiler(parse(source)).compile_env()
        prog = env["p"]
        entity = torch.tensor([0, 1])
        result = prog.rsample(entity)
        assert isinstance(result, dict)
        assert len(result) == 3
        for v in result.values():
            assert v.shape == (2, 1)
            assert (v > 0).all() and (v < 1).all()


# ── let bindings inside programs ─────────────────────────────────────


class TestParserLetSteps:
    """Parser tests for let bindings inside program blocks."""

    def test_let_float_literal(self):
        """let x = 0.5 parses to LetStep with float value."""
        source = """
            object A : 2
            space B : UnitInterval()
            program p : A -> B
                let x = 0.5
                draw y ~ LogitNormal(0.0, 1.0)
                return y
        """
        mod = parse(source)
        prog = [s for s in mod.statements if isinstance(s, ProgramDecl)][0]
        assert len(prog.draws) == 2
        step = prog.draws[0]
        assert isinstance(step, LetStep)
        assert step.name == "x"
        assert step.value == 0.5

    def test_let_int_literal(self):
        """let x = 1 parses to LetStep with float(1) value."""
        source = """
            object A : 2
            object B : 2
            program p : A -> B
                let x = 1
                draw y ~ LogitNormal(0.0, 1.0)
                return y
        """
        mod = parse(source)
        prog = [s for s in mod.statements if isinstance(s, ProgramDecl)][0]
        step = prog.draws[0]
        assert isinstance(step, LetStep)
        assert step.value == 1.0

    def test_let_variable_reference(self):
        """let y = x parses to LetStep with str value."""
        source = """
            object A : 2
            space B : UnitInterval()
            program p : A -> B * B
                draw x ~ LogitNormal(0.0, 1.0)
                let y = x
                return (x, y)
        """
        mod = parse(source)
        prog = [s for s in mod.statements if isinstance(s, ProgramDecl)][0]
        step = prog.draws[1]
        assert isinstance(step, LetStep)
        assert step.name == "y"
        assert step.value == "x"

    def test_let_interleaved_with_draws(self):
        """let steps can appear between draw steps."""
        source = """
            object A : 2
            space B : UnitInterval()
            object C : 2
            program p : A -> C * B
                draw x ~ LogitNormal(0.0, 1.0)
                let c = 1
                draw b ~ Bernoulli(x)
                return (b, x)
        """
        mod = parse(source)
        prog = [s for s in mod.statements if isinstance(s, ProgramDecl)][0]
        assert len(prog.draws) == 3
        assert isinstance(prog.draws[0], DrawStep)
        assert isinstance(prog.draws[1], LetStep)
        assert isinstance(prog.draws[2], DrawStep)


class TestCompilerLetSteps:
    """Compiler tests for let bindings inside program blocks."""

    def test_let_constant_compiles(self):
        """Program with a let constant compiles without error."""
        source = """
            object A : 2
            space B : UnitInterval()
            program p : A -> B
                let c = 0.5
                draw x ~ LogitNormal(0.0, 1.0)
                return x
        """
        env = Compiler(parse(source)).compile_env()
        assert "p" in env

    def test_let_alias_compiles(self):
        """Program with a let alias compiles without error."""
        source = """
            object A : 2
            space B : UnitInterval()
            program p : A -> B * B
                draw x ~ LogitNormal(0.0, 1.0)
                let y = x
                return (x, y)
        """
        env = Compiler(parse(source)).compile_env()
        assert "p" in env

    def test_let_duplicate_name_error(self):
        """let binding to an already-bound name is an error."""
        source = """
            object A : 2
            space B : UnitInterval()
            program p : A -> B
                draw x ~ LogitNormal(0.0, 1.0)
                let x = 1
                return x
        """
        with pytest.raises(CompileError, match="already bound"):
            Compiler(parse(source)).compile_env()

    def test_let_undefined_reference_error(self):
        """let alias to an undefined variable is an error."""
        source = """
            object A : 2
            space B : UnitInterval()
            program p : A -> B
                let y = z
                draw x ~ LogitNormal(0.0, 1.0)
                return x
        """
        with pytest.raises(CompileError, match="undefined variable"):
            Compiler(parse(source)).compile_env()


class TestExecutionLetSteps:
    """Execution tests for let bindings inside program blocks."""

    def test_let_constant_in_rsample(self):
        """let constant produces correct value in rsample output."""
        source = """
            object A : 2
            object B : 2
            space C : UnitInterval()
            program p : A -> B * C
                let c = 1
                draw x ~ LogitNormal(0.0, 1.0)
                return (c, x)
        """
        env = Compiler(parse(source)).compile_env()
        prog = env["p"]
        result = prog.rsample(torch.tensor([0, 1]))
        assert isinstance(result, dict)
        assert (result["c"] == 1.0).all()
        assert result["c"].shape == (2,)

    def test_let_alias_in_rsample(self):
        """let alias produces same value as the referenced variable."""
        source = """
            object A : 2
            space B : UnitInterval()
            program p : A -> B * B
                draw x ~ LogitNormal(0.0, 1.0)
                let y = x
                return (x, y)
        """
        env = Compiler(parse(source)).compile_env()
        prog = env["p"]
        result = prog.rsample(torch.tensor([0, 1]))
        assert torch.equal(result["x"], result["y"])

    def test_let_constant_zero_log_joint(self):
        """let bindings contribute zero to log_joint."""
        source = """
            object A : 2
            space B : UnitInterval()
            program p : A -> B
                let c = 1
                draw x ~ LogitNormal(0.0, 1.0)
                return x
        """
        env = Compiler(parse(source)).compile_env()
        prog = env["p"]
        inp = torch.tensor([0, 1])
        result = prog.rsample(inp)

        # log_joint with just the draw variable
        x_val = result if isinstance(result, torch.Tensor) else result["x"]
        lj = prog.log_joint(inp, {"x": x_val, "c": torch.ones(2)})
        assert lj.shape == (2,)
        assert torch.isfinite(lj).all()

    def test_let_in_labeled_return(self):
        """let-bound variables work with labeled returns."""
        source = """
            object A : 2
            object B : 2
            space C : UnitInterval()
            program p : A -> B * C
                draw x ~ LogitNormal(0.0, 1.0)
                let cg = 1
                return (cg_status: cg, belief: x)
        """
        env = Compiler(parse(source)).compile_env()
        prog = env["p"]
        result = prog.rsample(torch.tensor([0, 1]))
        assert "cg_status" in result
        assert "belief" in result
        assert (result["cg_status"] == 1.0).all()

    def test_let_used_as_draw_arg(self):
        """let-bound variable can be used as argument to a draw."""
        source = """
            object A : 2
            object B : 2
            space C : UnitInterval()
            program p : A -> B
                let prob = 0.5
                draw b ~ Bernoulli(prob)
                return b
        """
        env = Compiler(parse(source)).compile_env()
        prog = env["p"]
        result = prog.rsample(torch.tensor([0, 1]))
        assert set(result.tolist()).issubset({0, 1})

    def test_faithful_pds_factivity(self):
        """Faithful PDS factivity model with deterministic CG presupposition.

        This is the corrected model where:
        - theta priors are unconditional LogitNormal(0, 1)
        - cg_complement is deterministically 1 (factive presupposition)
        - tau_know and cg_matrix are Bernoulli draws
        - response has two-stage noise
        """
        source = """
            object Entity : 2
            object Truth : 2
            space Belief : UnitInterval()
            space Resp : Euclidean(1, low=0.0, high=1.0)

            program factivity : Entity -> Truth * Truth * Truth * Resp
                draw theta_know ~ LogitNormal(0.0, 1.0)
                draw theta_cg ~ LogitNormal(0.0, 1.0)
                let cg_complement = 1
                draw tau_know ~ Bernoulli(theta_know)
                draw cg_matrix ~ Bernoulli(theta_cg)
                draw sigma ~ Uniform(0.0, 1.0)
                draw response ~ TruncatedNormal(theta_know, sigma, 0.0, 1.0)
                return (tau_know: tau_know, cg_complement: cg_complement, cg_matrix: cg_matrix, response: response)
        """
        env = Compiler(parse(source)).compile_env()
        prog = env["factivity"]

        entity = torch.tensor([0, 1])
        result = prog.rsample(entity)

        assert isinstance(result, dict)
        assert set(result.keys()) == {
            "tau_know", "cg_complement", "cg_matrix", "response",
        }

        # cg_complement is deterministically 1 (factive presupposition)
        assert (result["cg_complement"] == 1.0).all()

        # truth values are {0, 1}
        assert set(result["tau_know"].tolist()).issubset({0, 1})
        assert set(result["cg_matrix"].tolist()).issubset({0, 1})

        # response in [0, 1]
        assert (result["response"] >= 0).all()
        assert (result["response"] <= 1).all()

    def test_faithful_pds_log_joint(self):
        """log_joint for PDS factivity: let contributes 0, draws contribute density."""
        source = """
            object Entity : 2
            object Truth : 2
            space Belief : UnitInterval()

            program p : Entity -> Truth * Belief
                draw theta ~ LogitNormal(0.0, 1.0)
                let cg = 1
                draw b ~ Bernoulli(theta)
                return (cg_status: cg, truth: b, belief: theta)
        """
        env = Compiler(parse(source)).compile_env()
        prog = env["p"]

        entity = torch.tensor([0, 1])
        result = prog.rsample(entity)

        # log_joint should work with labeled keys
        lj = prog.log_joint(entity, {
            "belief": result["belief"],
            "truth": result["truth"],
            "cg_status": result["cg_status"],
        })
        assert lj.shape == (2,)
        assert torch.isfinite(lj).all()

        # log_joint should also work with variable name keys
        lj2 = prog.log_joint(entity, {
            "theta": result["belief"],
            "b": result["truth"],
            "cg": result["cg_status"],
        })
        assert torch.allclose(lj, lj2)

    def test_repr_with_let(self):
        """__repr__ shows let bindings."""
        source = """
            object A : 2
            space B : UnitInterval()
            program p : A -> B
                let c = 1
                draw x ~ LogitNormal(0.0, 1.0)
                return x
        """
        env = Compiler(parse(source)).compile_env()
        prog = env["p"]
        r = repr(prog)
        assert "let c = 1" in r


# ===== replicate / fan / repeat tests =========================================


class TestLexerCombinators:
    """Lexer tests for fan and repeat keywords."""

    def test_fan_token(self):
        """fan is tokenized as FAN keyword."""
        tokens = Lexer("fan").tokenize()
        assert tokens[0].type == TokenType.FAN

    def test_repeat_token(self):
        """repeat is tokenized as REPEAT keyword."""
        tokens = Lexer("repeat").tokenize()
        assert tokens[0].type == TokenType.REPEAT


class TestParserCombinators:
    """Parser tests for replicate, fan, and repeat."""

    def test_parse_replicated_continuous(self):
        """Parse continuous with replication count."""
        ast = parse("""
            space A : Euclidean(4)
            space B : Euclidean(2)
            continuous head[4] : A -> B ~ Normal
            output head_0
        """)
        decl = ast.statements[2]
        assert isinstance(decl, ContinuousMorphismDecl)
        assert decl.name == "head"
        assert decl.replicate == 4

    def test_parse_replicated_stochastic(self):
        """Parse stochastic with replication count."""
        ast = parse("""
            object S : 8
            stochastic trans[3] : S -> S
            output trans_0
        """)
        decl = ast.statements[1]
        assert isinstance(decl, StochasticMorphismDecl)
        assert decl.name == "trans"
        assert decl.replicate == 3

    def test_parse_replicated_embed(self):
        """Parse embed with replication count."""
        ast = parse("""
            object T : 32
            space H : Euclidean(16)
            embed e[2] : T -> H
            output e_0
        """)
        decl = ast.statements[2]
        assert isinstance(decl, EmbedDecl)
        assert decl.name == "e"
        assert decl.replicate == 2

    def test_parse_fan_expression(self):
        """Parse fan(f, g, h) expression."""
        ast = parse("""
            space A : Euclidean(4)
            space B : Euclidean(2)
            continuous f : A -> B ~ Normal
            continuous g : A -> B ~ Normal
            let fanned = fan(f, g)
            output fanned
        """)
        let_decl = ast.statements[4]
        assert isinstance(let_decl, LetDecl)
        assert isinstance(let_decl.expr, ExprFan)
        assert len(let_decl.expr.exprs) == 2

    def test_parse_repeat_expression(self):
        """Parse repeat(f, 3) expression."""
        ast = parse("""
            object S : 4
            stochastic t : S -> S
            let chain = repeat(t, 5)
            output chain
        """)
        let_decl = ast.statements[2]
        assert isinstance(let_decl, LetDecl)
        assert isinstance(let_decl.expr, ExprRepeat)
        assert let_decl.expr.count == 5

    def test_parse_fan_composed(self):
        """Parse fan(...) >> combine composition."""
        ast = parse("""
            space A : Euclidean(4)
            space B : Euclidean(2)
            space C : Euclidean(4)
            continuous f : A -> B ~ Normal
            continuous g : A -> B ~ Normal
            continuous h : C -> A ~ Normal
            let pipeline = fan(f, g) >> h
            output pipeline
        """)
        let_decl = ast.statements[6]
        assert isinstance(let_decl, LetDecl)
        assert isinstance(let_decl.expr, ExprCompose)
        assert isinstance(let_decl.expr.left, ExprFan)


class TestCompilerCombinators:
    """Compiler tests for replicate, fan, and repeat."""

    def test_replicate_continuous_creates_n_morphisms(self):
        """Replicated continuous creates N independent morphisms."""
        source = """
            space A : Euclidean(4)
            space B : Euclidean(2)
            continuous head[3] : A -> B ~ Normal
            output head_0
        """
        compiler = Compiler(parse(source))
        compiler.compile()
        morphisms = compiler.morphisms
        assert "head_0" in morphisms
        assert "head_1" in morphisms
        assert "head_2" in morphisms
        assert "head" not in morphisms

    def test_replicate_independent_parameters(self):
        """Each replicated morphism has independent parameters."""
        source = """
            space A : Euclidean(4)
            space B : Euclidean(2)
            continuous head[2] : A -> B ~ Normal
            output head_0
        """
        compiler = Compiler(parse(source))
        compiler.compile()
        m = compiler.morphisms
        # different objects (independent parameter sets)
        assert m["head_0"] is not m["head_1"]

    def test_fan_explicit_morphisms(self):
        """fan(f, g) copies input and concatenates outputs."""
        source = """
            space A : Euclidean(4)
            space B : Euclidean(2)
            continuous f : A -> B ~ Normal
            continuous g : A -> B ~ Normal
            let fanned = fan(f, g)
            output fanned
        """
        prog = Compiler(parse(source)).compile()
        morph = prog.morphism
        x = torch.randn(8, 4)
        y = morph.rsample(x)
        # output should be B * B = 2 + 2 = 4
        assert y.shape == (8, 4)

    def test_fan_group_expansion(self):
        """fan(group_name) expands to all group members."""
        source = """
            space A : Euclidean(4)
            space B : Euclidean(2)
            continuous head[3] : A -> B ~ Normal
            let fanned = fan(head)
            output fanned
        """
        prog = Compiler(parse(source)).compile()
        morph = prog.morphism
        x = torch.randn(8, 4)
        y = morph.rsample(x)
        # output should be B * B * B = 2 + 2 + 2 = 6
        assert y.shape == (8, 6)

    def test_fan_compose(self):
        """fan(heads) >> combine works end-to-end."""
        source = """
            object Token : 32
            space Latent : Euclidean(16)
            space Value : Euclidean(4)

            embed tok_embed : Token -> Latent
            continuous head[4] : Latent -> Value ~ Normal
            continuous combine : Latent -> Latent ~ Normal [scale=0.1]

            let multi_head = fan(head) >> combine
            let model = tok_embed >> multi_head

            output model
        """
        prog = Compiler(parse(source)).compile()
        morph = prog.morphism
        x = torch.randn(4, 32)
        y = morph.rsample(x)
        assert y.shape[-1] == 16  # latent dim

    def test_repeat_stochastic(self):
        """repeat(transition, N) composes N times."""
        source = """
            object S : 4
            stochastic t : S -> S
            let chain = repeat(t, 3)
            output chain
        """
        prog = Compiler(parse(source)).compile()
        out = prog()
        assert out.shape == (4, 4)

    def test_repeat_continuous(self):
        """repeat works with continuous morphisms."""
        source = """
            space H : Euclidean(8)
            continuous layer : H -> H ~ Normal [scale=0.1]
            let deep = repeat(layer, 4)
            output deep
        """
        prog = Compiler(parse(source)).compile()
        morph = prog.morphism
        x = torch.randn(4, 8)
        y = morph.rsample(x)
        assert y.shape == (4, 8)

    def test_repeat_one(self):
        """repeat(f, 1) is the same as f."""
        source = """
            object S : 4
            object O : 8
            stochastic t : S -> O
            let one = repeat(t, 1)
            output one
        """
        prog = Compiler(parse(source)).compile()
        out = prog()
        assert out.shape == (4, 8)

    def test_replicate_stochastic(self):
        """Replicated stochastic creates N independent morphisms."""
        source = """
            object S : 4
            stochastic layer[3] : S -> S
            let chain = layer_0 >> layer_1 >> layer_2
            output chain
        """
        prog = Compiler(parse(source)).compile()
        out = prog()
        assert out.shape == (4, 4)

    def test_fan_log_prob(self):
        """FanOutMorphism log_prob sums component log-probs."""
        source = """
            space A : Euclidean(4)
            space B : Euclidean(2)
            continuous f : A -> B ~ Normal
            continuous g : A -> B ~ Normal
            let fanned = fan(f, g)
            output fanned
        """
        prog = Compiler(parse(source)).compile()
        morph = prog.morphism
        x = torch.randn(8, 4)
        y = morph.rsample(x)
        lp = morph.log_prob(x, y)
        assert lp.shape == (8,)
        assert torch.isfinite(lp).all()


class TestMixedInlineDistributions:
    """Tests for MixedInlineDistribution — general literal/variable mixing."""

    def test_normal_var_lit(self):
        """Normal(variable, literal) — loc is variable, scale is fixed."""
        source = """
            object Unit : 1
            space H : Euclidean(1)

            program test : Unit -> H
                draw mu ~ Normal(0.0, 1.0)
                draw x ~ Normal(mu, 0.5)
                return x
            output test
        """
        prog = Compiler(parse(source)).compile()
        y = prog.rsample(torch.randn(8, 1))
        assert y.shape == (8, 1)

    def test_normal_lit_var(self):
        """Normal(literal, variable) — loc is fixed, scale is variable."""
        source = """
            object Unit : 1
            space H : Euclidean(1)

            program test : Unit -> H
                draw sigma ~ HalfNormal(0.5)
                draw x ~ Normal(0.0, sigma)
                return x
            output test
        """
        prog = Compiler(parse(source)).compile()
        y = prog.rsample(torch.randn(8, 1))
        assert y.shape == (8, 1)

    def test_normal_var_var(self):
        """Normal(variable, variable) — both variable (existing case)."""
        source = """
            object Unit : 1
            space H : Euclidean(1)

            program test : Unit -> H
                draw mu ~ Normal(0.0, 1.0)
                draw sigma ~ HalfNormal(0.5)
                draw x ~ Normal(mu, sigma)
                return x
            output test
        """
        prog = Compiler(parse(source)).compile()
        y = prog.rsample(torch.randn(8, 1))
        assert y.shape == (8, 1)

    def test_normal_lit_lit(self):
        """Normal(literal, literal) — all fixed (existing case)."""
        source = """
            object Unit : 1
            space H : Euclidean(1)

            program test : Unit -> H
                draw x ~ Normal(0.0, 1.0)
                return x
            output test
        """
        prog = Compiler(parse(source)).compile()
        y = prog.rsample(torch.randn(8, 1))
        assert y.shape == (8, 1)

    def test_truncated_normal_var_lit_lit_lit(self):
        """TruncatedNormal(var, lit, lit, lit) — only mu is variable."""
        source = """
            object Unit : 1
            object Resp : 1

            program test : Unit -> Resp
                draw mu ~ Normal(0.0, 1.0)
                observe y ~ TruncatedNormal(mu, 0.5, 0.0, 1.0)
                return y
            output test
        """
        prog = Compiler(parse(source)).compile()
        y = prog.rsample(torch.randn(8, 1))
        assert y.shape == (8, 1)

    def test_mixed_normal_log_joint(self):
        """Mixed-arg Normal works with log_joint (inference path)."""
        source = """
            object Unit : 1
            space H : Euclidean(1)

            program test : Unit -> H
                draw mu ~ Normal(0.0, 1.0)
                draw x ~ Normal(mu, 0.5)
                return x
            output test
        """
        prog = Compiler(parse(source)).compile()
        morph = prog.morphism
        inp = torch.randn(8, 1)
        # sample to get intermediates
        y = morph.rsample(inp)
        # log_joint needs intermediate values
        mu_sample = torch.randn(8, 1)
        x_sample = torch.randn(8, 1)
        lj = morph.log_joint(inp, {"mu": mu_sample, "x": x_sample})
        assert lj.shape == (8,)
        assert torch.isfinite(lj).all()

    def test_mixed_normal_chained(self):
        """Chain of mixed-arg draws: mu -> x -> y."""
        source = """
            object Unit : 1
            space H : Euclidean(1)

            program test : Unit -> H
                draw mu ~ Normal(0.0, 2.0)
                draw x ~ Normal(mu, 1.0)
                draw y ~ Normal(x, 0.1)
                return y
            output test
        """
        prog = Compiler(parse(source)).compile()
        y = prog.rsample(torch.randn(8, 1))
        assert y.shape == (8, 1)

    def test_mixed_normal_scale_positive(self):
        """Normal(var, lit) with small scale produces tight samples."""
        source = """
            object Unit : 1
            space H : Euclidean(1)

            program test : Unit -> H
                draw mu ~ Normal(5.0, 0.001)
                draw x ~ Normal(mu, 0.001)
                return x
            output test
        """
        prog = Compiler(parse(source)).compile()
        y = prog.rsample(torch.randn(64, 1))
        # samples should cluster near 5.0
        assert (y - 5.0).abs().mean() < 1.0

    def test_gru_pattern(self):
        """The GRU pattern: draw gates, compute reset_hidden, draw candidate."""
        source = """
            object Unit : 1
            space Hidden : Euclidean(48)

            program gru : Unit -> Hidden
                draw h_prev ~ Normal(0.0, 1.0)
                draw z ~ LogitNormal(0.0, 1.0)
                draw r ~ LogitNormal(0.0, 1.0)
                let reset_hidden = r * h_prev
                draw h_cand ~ Normal(reset_hidden, 0.5)
                let z_complement = 1.0 - z
                let h_new = z_complement * h_prev + z * h_cand
                return h_new
            output gru
        """
        prog = Compiler(parse(source)).compile()
        y = prog.rsample(torch.randn(8, 1))
        # inline draws produce scalar outputs; arithmetic
        # broadcasts element-wise. verify finite output.
        assert y.dim() == 2
        assert y.shape[0] == 8
        assert torch.isfinite(y).all()

    def test_lstm_pattern(self):
        """The LSTM pattern: gates + candidate dependent on h_prev."""
        source = """
            object Unit : 1
            space Hidden : Euclidean(64)

            program lstm : Unit -> Hidden
                draw c_prev ~ Normal(0.0, 1.0)
                draw h_prev ~ Normal(0.0, 1.0)
                draw i_gate ~ LogitNormal(0.0, 1.0)
                draw f_gate ~ LogitNormal(0.0, 0.5)
                draw o_gate ~ LogitNormal(0.0, 1.0)
                draw g_cand ~ Normal(h_prev, 0.5)
                let c_new = f_gate * c_prev + i_gate * g_cand
                let two_c = 2.0 * c_new
                let sig_2c = sigmoid(two_c)
                let tanh_c = 2.0 * sig_2c - 1.0
                let h_new = o_gate * tanh_c
                return h_new
            output lstm
        """
        prog = Compiler(parse(source)).compile()
        y = prog.rsample(torch.randn(8, 1))
        assert y.dim() == 2
        assert y.shape[0] == 8
        assert torch.isfinite(y).all()


# ===== ergonomic syntax extensions ==========================================


class TestStackCombinator:
    """Tests for stack(f, N) — independent multi-layer composition."""

    def test_stack_produces_expr_stack_ast(self):
        """stack(f, 3) parses to ExprStack node."""
        ast = parse("""
            object X : 4
            space H : Euclidean(8)
            continuous f : H -> H ~ Normal
            let model = stack(f, 3)
            output model
        """)
        let_decl = ast.statements[3]
        assert isinstance(let_decl.expr, ExprStack)
        assert let_decl.expr.count == 3

    def test_stack_independent_params(self):
        """stack creates independent parameters, unlike repeat."""
        repeat_prog = Compiler(parse("""
            object X : 4
            space H : Euclidean(8)
            continuous f : H -> H ~ Normal [scale=0.1]
            let model = repeat(f, 3)
            output model
        """)).compile()

        stack_prog = Compiler(parse("""
            object X : 4
            space H : Euclidean(8)
            continuous f : H -> H ~ Normal [scale=0.1]
            let model = stack(f, 3)
            output model
        """)).compile()

        repeat_params = sum(p.numel() for p in repeat_prog.parameters())
        stack_params = sum(p.numel() for p in stack_prog.parameters())

        # stack(f, 3) should have 3x the morphism params vs repeat(f, 3)
        assert stack_params > repeat_params

    def test_stack_rsample_shape(self):
        """stack(f, 3) produces correct output shape."""
        prog = Compiler(parse("""
            object X : 4
            space H : Euclidean(8)
            embed e : X -> H
            continuous f : H -> H ~ Normal [scale=0.1]
            let model = e >> stack(f, 3)
            output model
        """)).compile()
        y = prog.rsample(torch.zeros(4, 4))
        assert y.shape[0] == 4
        assert torch.isfinite(y).all()

    def test_stack_count_one(self):
        """stack(f, 1) is equivalent to a single fresh copy."""
        prog = Compiler(parse("""
            object X : 4
            space H : Euclidean(8)
            continuous f : H -> H ~ Normal
            let model = stack(f, 1)
            output model
        """)).compile()
        y = prog.rsample(torch.zeros(2, 8))
        assert torch.isfinite(y).all()

    def test_stack_in_composition(self):
        """stack can be composed with >> and other combinators."""
        prog = Compiler(parse("""
            object X : 4
            space H : Euclidean(8)
            space Out : Euclidean(2)
            embed e : X -> H
            continuous f : H -> H ~ Normal [scale=0.1]
            continuous g : H -> Out ~ Normal [scale=0.1]
            let model = e >> stack(f, 4) >> g
            output model
        """)).compile()
        y = prog.rsample(torch.zeros(3, 4))
        assert y.shape == torch.Size([3, 4, 2])
        assert torch.isfinite(y).all()


class TestArrowSyntax:
    """Tests for <- do-notation style bind in programs."""

    def test_arrow_basic(self):
        """x <- Normal(0.0, 1.0) works as draw replacement."""
        prog = Compiler(parse("""
            object Unit : 1
            space H : Euclidean(4)
            program test : Unit -> H
                x <- Normal(0.0, 1.0)
                return x
            output test
        """)).compile()
        y = prog.rsample(torch.zeros(8, 1))
        assert y.shape[0] == 8
        assert torch.isfinite(y).all()

    def test_arrow_mixed_with_draw(self):
        """<- and draw can coexist in same program."""
        prog = Compiler(parse("""
            object Unit : 1
            space H : Euclidean(4)
            program test : Unit -> H
                draw x ~ Normal(0.0, 1.0)
                y <- Normal(x, 0.5)
                return y
            output test
        """)).compile()
        y = prog.rsample(torch.zeros(8, 1))
        assert torch.isfinite(y).all()

    def test_arrow_with_let(self):
        """<- works alongside let bindings."""
        prog = Compiler(parse("""
            object Unit : 1
            space H : Euclidean(4)
            program test : Unit -> H
                x <- Normal(0.0, 1.0)
                let doubled = 2.0 * x
                y <- Normal(doubled, 0.5)
                return y
            output test
        """)).compile()
        y = prog.rsample(torch.zeros(8, 1))
        assert torch.isfinite(y).all()


class TestBackwardComposition:
    """Tests for << backward composition operator."""

    def test_backward_compose_basic(self):
        """f << g produces g >> f."""
        prog = Compiler(parse("""
            object A : 3
            object B : 4
            object C : 5
            stochastic f : A -> B
            stochastic g : B -> C
            let h = g << f
            output h
        """)).compile()
        # g << f means f >> g: A -> C
        assert prog.morphism.domain.size == 3
        assert prog.morphism.codomain.size == 5

    def test_backward_compose_chain(self):
        """f << g << h produces h >> g >> f."""
        prog = Compiler(parse("""
            object A : 3
            object B : 4
            object C : 5
            object D : 6
            stochastic f : A -> B
            stochastic g : B -> C
            stochastic h : C -> D
            let chain = h << g << f
            output chain
        """)).compile()
        # h << g << f means f >> g >> h: A -> D
        assert prog.morphism.domain.size == 3
        assert prog.morphism.codomain.size == 6


class TestKleisliComposition:
    """Tests for >=> as synonym for >>."""

    def test_kleisli_basic(self):
        """f >=> g is equivalent to f >> g."""
        prog = Compiler(parse("""
            object A : 3
            object B : 4
            object C : 5
            stochastic f : A -> B
            stochastic g : B -> C
            let h = f >=> g
            output h
        """)).compile()
        assert prog.morphism.domain.size == 3
        assert prog.morphism.codomain.size == 5

    def test_kleisli_mixed_with_compose(self):
        """>=> and >> can be mixed in same expression."""
        prog = Compiler(parse("""
            object A : 3
            object B : 4
            object C : 5
            object D : 6
            stochastic f : A -> B
            stochastic g : B -> C
            stochastic h : C -> D
            let chain = f >=> g >> h
            output chain
        """)).compile()
        assert prog.morphism.domain.size == 3
        assert prog.morphism.codomain.size == 6


class TestTypeAlias:
    """Tests for type as alias for space."""

    def test_type_basic(self):
        """type H = Euclidean(8) works like space H : Euclidean(8)."""
        prog = Compiler(parse("""
            object X : 4
            type H = Euclidean(8)
            embed e : X -> H
            continuous f : H -> H ~ Normal
            let model = e >> f
            output model
        """)).compile()
        y = prog.rsample(torch.zeros(2, 4))
        assert y.shape[-1] == 8

    def test_type_parens_free(self):
        """type H = Euclidean 8 (parens-optional constructor)."""
        prog = Compiler(parse("""
            object X : 4
            type H = Euclidean 8
            embed e : X -> H
            output e
        """)).compile()
        y = prog.rsample(torch.zeros(2, 4))
        assert y.shape[-1] == 8


class TestParensFreeConstructor:
    """Tests for parens-optional space constructors."""

    def test_space_parens_free(self):
        """space H : Euclidean 8 works without parentheses."""
        prog = Compiler(parse("""
            object X : 4
            space H : Euclidean 8
            embed e : X -> H
            output e
        """)).compile()
        y = prog.rsample(torch.zeros(2, 4))
        assert y.shape[-1] == 8

    def test_space_parens_still_work(self):
        """Euclidean(8) still works with parentheses."""
        prog = Compiler(parse("""
            object X : 4
            space H : Euclidean(8)
            embed e : X -> H
            output e
        """)).compile()
        y = prog.rsample(torch.zeros(2, 4))
        assert y.shape[-1] == 8


class TestWhereClause:
    """Tests for where clauses on let bindings."""

    def test_where_basic(self):
        """let x = expr where let y = expr."""
        prog = Compiler(parse("""
            object A : 3
            object B : 4
            object C : 5
            stochastic f : A -> B
            stochastic g : B -> C
            let model = f >> chain
            where
                let chain = g
            output model
        """)).compile()
        assert prog.morphism.domain.size == 3
        assert prog.morphism.codomain.size == 5

    def test_where_multiple_bindings(self):
        """where can have multiple let bindings."""
        prog = Compiler(parse("""
            object A : 3
            object B : 4
            object C : 5
            object D : 6
            stochastic f : A -> B
            stochastic g : B -> C
            stochastic h : C -> D
            let model = first >> second
            where
                let first = f >> g
                let second = h
            output model
        """)).compile()
        assert prog.morphism.domain.size == 3
        assert prog.morphism.codomain.size == 6

    def test_where_with_stack(self):
        """where clause with stack combinator."""
        prog = Compiler(parse("""
            object X : 4
            space H : Euclidean(8)
            space Out : Euclidean(2)
            embed e : X -> H
            continuous f : H -> H ~ Normal [scale=0.1]
            continuous g : H -> Out ~ Normal [scale=0.1]
            let model = e >> layers >> g
            where
                let layers = stack(f, 3)
            output model
        """)).compile()
        y = prog.rsample(torch.zeros(2, 4))
        assert y.shape[-1] == 2
        assert torch.isfinite(y).all()


# ===== scan combinator tests ================================================


class TestScanCombinator:
    """Tests for the scan combinator (temporal recurrence)."""

    def test_scan_ast_node(self):
        """scan(expr) produces an ExprScan AST node."""
        ast = parse("""
            type A = Euclidean 4
            type H = Euclidean 8
            continuous cell : A * H -> H ~ Normal
            let rnn = scan(cell)
            output rnn
        """)
        let_decl = ast.statements[3]
        assert isinstance(let_decl.expr, ExprScan)
        assert let_decl.expr.init == "zeros"

    def test_scan_ast_init_learned(self):
        """scan(expr, init=learned) sets init strategy."""
        ast = parse("""
            type A = Euclidean 4
            type H = Euclidean 8
            continuous cell : A * H -> H ~ Normal
            let rnn = scan(cell, init=learned)
            output rnn
        """)
        let_decl = ast.statements[3]
        assert isinstance(let_decl.expr, ExprScan)
        assert let_decl.expr.init == "learned"

    def test_scan_basic_shape(self):
        """scan(cell) threads hidden state and returns correct shape."""
        prog = Compiler(parse("""
            type Input = Euclidean 4
            type Hidden = Euclidean 8
            continuous cell : Input * Hidden -> Hidden ~ Normal [scale=0.1]
            let rnn = scan(cell)
            output rnn
        """)).compile()

        # (batch=3, seq_len=5, input_dim=4)
        x = torch.randn(3, 5, 4)
        out = prog.rsample(x)
        assert out.shape == (3, 8)
        assert torch.isfinite(out).all()

    def test_scan_with_embed(self):
        """embed >> scan(cell) processes tokenized sequences."""
        prog = Compiler(parse("""
            object Token : 16
            type Hidden = Euclidean 8
            type Embedded = Euclidean 4
            embed tok_embed : Token -> Embedded
            continuous cell : Embedded * Hidden -> Hidden ~ Normal [scale=0.1]
            let rnn = tok_embed >> scan(cell)
            output rnn
        """)).compile()

        # (batch=2, seq_len=4) integer tokens
        tokens = torch.tensor([[0, 5, 3, 1], [2, 7, 15, 0]])
        out = prog.rsample(tokens)
        assert out.shape == (2, 8)
        assert torch.isfinite(out).all()

    def test_scan_full_pipeline(self):
        """embed >> scan(cell) >> output_proj is a full RNN pipeline."""
        prog = Compiler(parse("""
            object Token : 32
            type Embedded = Euclidean 16
            type Hidden = Euclidean 32
            type Output = Euclidean 8
            embed tok_embed : Token -> Embedded
            continuous cell : Embedded * Hidden -> Hidden ~ Normal [scale=0.1]
            continuous output_proj : Hidden -> Output ~ Normal [scale=0.1]
            let rnn = tok_embed >> scan(cell) >> output_proj
            output rnn
        """)).compile()

        tokens = torch.tensor([[5, 12, 3, 27, 0]])
        out = prog.rsample(tokens)
        assert out.shape == (1, 8)
        assert torch.isfinite(out).all()

    def test_scan_learned_init(self):
        """scan(cell, init=learned) has a learnable initial state."""
        prog = Compiler(parse("""
            type Input = Euclidean 4
            type Hidden = Euclidean 8
            continuous cell : Input * Hidden -> Hidden ~ Normal [scale=0.1]
            let rnn = scan(cell, init=learned)
            output rnn
        """)).compile()

        # check that h0 is a parameter
        from quivers.continuous.scan import ScanMorphism
        scan_morph = prog.morphism
        assert isinstance(scan_morph, ScanMorphism)
        assert hasattr(scan_morph, '_h0')

        x = torch.randn(2, 5, 4)
        out = prog.rsample(x)
        assert out.shape == (2, 8)

    def test_scan_different_seq_lengths(self):
        """scan handles different sequence lengths producing same output shape."""
        prog = Compiler(parse("""
            type Input = Euclidean 4
            type Hidden = Euclidean 8
            continuous cell : Input * Hidden -> Hidden ~ Normal [scale=0.1]
            let rnn = scan(cell)
            output rnn
        """)).compile()

        # varying sequence lengths
        x3 = torch.randn(1, 3, 4)
        x7 = torch.randn(1, 7, 4)
        x1 = torch.randn(1, 1, 4)

        assert prog.rsample(x3).shape == (1, 8)
        assert prog.rsample(x7).shape == (1, 8)
        assert prog.rsample(x1).shape == (1, 8)

    def test_scan_with_monadic_cell(self):
        """scan works with a monadic program cell (e.g. GRU)."""
        prog = Compiler(parse("""
            object Token : 16
            type Embedded = Euclidean 8
            type Hidden = Euclidean 16

            embed tok_embed : Token -> Embedded

            continuous gate_z : Embedded * Hidden -> Hidden ~ LogitNormal
            continuous gate_r : Embedded * Hidden -> Hidden ~ LogitNormal
            continuous cand : Embedded * Hidden -> Hidden ~ Normal [scale=0.1]

            program gru_cell(x_t, h_prev) : Embedded * Hidden -> Hidden
                draw z ~ gate_z(x_t, h_prev)
                draw r ~ gate_r(x_t, h_prev)
                let reset_h = r * h_prev
                draw h_cand ~ cand(x_t, reset_h)
                let z_c = 1.0 - z
                let h_new = z_c * h_prev + z * h_cand
                return h_new

            let gru = tok_embed >> scan(gru_cell)
            output gru
        """)).compile()

        tokens = torch.tensor([[0, 5, 3, 1], [2, 7, 15, 0]])
        out = prog.rsample(tokens)
        assert out.shape == (2, 16)
        assert torch.isfinite(out).all()

    def test_scan_independent_params(self):
        """scan cell parameters are trainable."""
        prog = Compiler(parse("""
            type Input = Euclidean 4
            type Hidden = Euclidean 8
            continuous cell : Input * Hidden -> Hidden ~ Normal [scale=0.1]
            let rnn = scan(cell)
            output rnn
        """)).compile()

        # verify parameters exist and are trainable
        params = list(prog.parameters())
        assert len(params) > 0
        assert all(p.requires_grad for p in params)

    def test_scan_in_composition_with_stack(self):
        """scan(cell) >> stack(layer, N) composes correctly."""
        prog = Compiler(parse("""
            object Token : 16
            type Embedded = Euclidean 8
            type Hidden = Euclidean 16
            type Output = Euclidean 4

            embed tok_embed : Token -> Embedded
            continuous cell : Embedded * Hidden -> Hidden ~ Normal [scale=0.1]
            continuous layer : Hidden -> Hidden ~ Normal [scale=0.1]
            continuous proj : Hidden -> Output ~ Normal [scale=0.1]

            let model = tok_embed >> scan(cell) >> stack(layer, 2) >> proj
            output model
        """)).compile()

        tokens = torch.tensor([[0, 5, 3, 1]])
        out = prog.rsample(tokens)
        assert out.shape == (1, 4)
        assert torch.isfinite(out).all()

    def test_scan_product_domain_error(self):
        """scan requires cell with product domain."""
        with pytest.raises(CompileError):
            Compiler(parse("""
                type H = Euclidean 8
                continuous cell : H -> H ~ Normal [scale=0.1]
                let rnn = scan(cell)
                output rnn
            """)).compile()

    def test_scan_log_joint(self):
        """scan.log_joint scores all intermediate hidden states."""
        from quivers.continuous.scan import ScanMorphism
        from quivers.continuous.spaces import Euclidean, ProductSpace
        from quivers.continuous.families import ConditionalNormal

        A = Euclidean("input", 4)
        H = Euclidean("hidden", 8)
        cell = ConditionalNormal(ProductSpace(A, H), H)
        scan_morph = ScanMorphism(cell, init="zeros")

        x = torch.randn(3, 5, 4)        # (batch=3, seq_len=5, input=4)
        hs = torch.randn(3, 5, 8)       # (batch=3, seq_len=5, hidden=8)
        lj = scan_morph.log_joint(x, hs)
        assert lj.shape == (3,)
        assert torch.isfinite(lj).all()


# ===== multi-line expression tests =========================================

class TestLexerMultiLine:
    """Lexer suppresses NEWLINE tokens inside balanced () and []."""

    def test_no_newlines_inside_parens(self):
        """Newlines inside () are suppressed."""
        source = "foo(\n  bar,\n  baz\n)"
        tokens = Lexer(source).tokenize()
        types = [t.type for t in tokens]
        assert TokenType.NEWLINE not in types

    def test_no_newlines_inside_brackets(self):
        """Newlines inside [] are suppressed."""
        source = "[\n  a,\n  b\n]"
        tokens = Lexer(source).tokenize()
        types = [t.type for t in tokens]
        assert TokenType.NEWLINE not in types

    def test_no_newlines_nested_brackets(self):
        """Newlines inside nested () and [] are suppressed."""
        source = "foo(\n  x=[\n    a,\n    b\n  ]\n)"
        tokens = Lexer(source).tokenize()
        types = [t.type for t in tokens]
        assert TokenType.NEWLINE not in types

    def test_newlines_outside_brackets_preserved(self):
        """Newlines outside () and [] are still emitted."""
        source = "foo\nbar"
        tokens = Lexer(source).tokenize()
        types = [t.type for t in tokens]
        assert TokenType.NEWLINE in types

    def test_newlines_after_close_paren_preserved(self):
        """Newlines after closing ) are emitted normally."""
        source = "foo(x)\nbar"
        tokens = Lexer(source).tokenize()
        types = [t.type for t in tokens]
        # should have exactly one newline (between the two statements)
        assert types.count(TokenType.NEWLINE) == 1


class TestParserMultiLine:
    """Parser handles multi-line function call expressions."""

    def test_multiline_parser(self):
        """parser() with arguments on separate lines parses correctly."""
        ast = parse("""
            object Token : 256
            let p = parser(
                categories=[S, NP, N, VP, PP],
                rules=[evaluation, harmonic_composition],
                start=S
            )
            output p
        """)
        stmts = ast.statements
        let_stmt = stmts[1]
        assert isinstance(let_stmt, LetDecl)
        parser_expr = let_stmt.expr
        assert isinstance(parser_expr, ExprParser)
        assert parser_expr.categories == ("S", "NP", "N", "VP", "PP")
        assert parser_expr.rules == ("evaluation", "harmonic_composition")
        assert parser_expr.start == "S"

    def test_multiline_parser_with_depth_constructors(self):
        """parser() with depth and constructors on separate lines."""
        ast = parse("""
            object Token : 256
            let p = parser(
                categories=[S, NP, N],
                rules=[evaluation, adjunction_units],
                constructors=[slash, diamond],
                depth=2,
                start=S
            )
            output p
        """)
        let_stmt = ast.statements[1]
        parser_expr = let_stmt.expr
        assert isinstance(parser_expr, ExprParser)
        assert parser_expr.depth == 2
        assert parser_expr.constructors == ("slash", "diamond")

    def test_multiline_fan(self):
        """fan() with arguments on separate lines."""
        ast = parse("""
            object X : 3
            object Y : 4
            latent f : X -> Y
            latent g : X -> Y
            latent h : X -> Y
            let par = fan(
                f,
                g,
                h
            )
            output par
        """)
        let_stmt = ast.statements[5]
        assert isinstance(let_stmt, LetDecl)
        assert isinstance(let_stmt.expr, ExprFan)

    def test_multiline_parser_morphism_rules(self):
        """parser() with morphism rules on separate lines."""
        ast = parse("""
            object N : 10
            object T : 64
            stochastic binary : N -> N * N
            stochastic lexical : N -> T
            let pcfg = parser(
                rules=[binary, lexical],
                start=0
            )
            output pcfg
        """)
        let_stmt = ast.statements[4]
        assert isinstance(let_stmt, LetDecl)

    def test_multiline_categories_list(self):
        """Category list split across lines."""
        ast = parse("""
            object Token : 256
            let p = parser(
                categories=[
                    S,
                    NP,
                    N,
                    VP,
                    PP
                ],
                rules=[evaluation],
                start=S
            )
            output p
        """)
        parser_expr = ast.statements[1].expr
        assert parser_expr.categories == ("S", "NP", "N", "VP", "PP")

    def test_multiline_compile_roundtrip(self):
        """Multi-line parser() compiles to a working ChartParser."""
        prog = loads("""
            object Token : 256
            let p = parser(
                categories=[S, NP, N, VP, PP],
                rules=[evaluation, harmonic_composition, crossed_composition],
                terminal=Token,
                start=S
            )
            output p
        """)
        tokens = torch.randint(0, 256, (2, 4))
        result = prog.morphism(tokens)
        assert result.shape == (2,)

    def test_multiline_program_body_unaffected(self):
        """Newlines in program bodies (outside brackets) still work."""
        prog = loads("""
            object Unit : 1
            object Obs : 1
            program model : Unit -> Obs
                draw x ~ Normal(0.0, 1.0)
                draw y ~ Normal(
                    x,
                    0.5
                )
                return y
            output model
        """)


class TestCategoryDecl:
    """Tests for the `category` keyword."""

    def test_parse_category_decl(self):
        """category <name> parses to CategoryDecl."""
        ast = parse("""
            category S
            category NP
        """)
        stmts = ast.statements
        assert len(stmts) == 2
        assert isinstance(stmts[0], CategoryDecl)
        assert stmts[0].name == "S"
        assert isinstance(stmts[1], CategoryDecl)
        assert stmts[1].name == "NP"

    def test_parse_category_decl_comma_separated(self):
        """category S, NP, N parses to multiple CategoryDecl nodes."""
        ast = parse("""
            category S, NP, N, VP, PP
        """)
        stmts = ast.statements
        assert len(stmts) == 5
        assert all(isinstance(s, CategoryDecl) for s in stmts)
        assert [s.name for s in stmts] == ["S", "NP", "N", "VP", "PP"]

    def test_category_comma_compile_roundtrip(self):
        """Comma-separated category declarations compile correctly."""
        prog = loads("""
            category S, NP, N, VP, PP
            object Token : 256
            let g = parser(
                rules=[evaluation, harmonic_composition],
                terminal=Token,
                start=S
            )
            output g
        """)
        tokens = torch.randint(0, 256, (2, 4))
        result = prog.morphism(tokens)
        assert result.shape == (2,)

    def test_category_decl_with_parser(self):
        """category declarations provide atoms for parser()."""
        ast = parse("""
            category S
            category NP
            category N
            object Token : 256
            let g = parser(
                rules=[evaluation],
                start=S
            )
            output g
        """)
        stmts = ast.statements
        assert isinstance(stmts[0], CategoryDecl)
        assert isinstance(stmts[4], LetDecl)
        parser_expr = stmts[4].expr
        assert isinstance(parser_expr, ExprParser)
        # categories not set inline — will come from category decls
        assert parser_expr.categories == ()

    def test_category_compile_roundtrip(self):
        """category declarations compile to a working ChartParser."""
        prog = loads("""
            category S, NP, N, VP, PP
            object Token : 256
            let g = parser(
                rules=[evaluation, harmonic_composition, crossed_composition],
                terminal=Token,
                start=S
            )
            output g
        """)
        tokens = torch.randint(0, 256, (2, 4))
        result = prog.morphism(tokens)
        assert result.shape == (2,)

    def test_category_duplicate_error(self):
        """Duplicate category declarations raise CompileError."""
        with pytest.raises(CompileError):
            loads("""
                category S
                category S
                object Token : 256
                let g = parser(rules=[evaluation], terminal=Token, start=S)
                output g
            """)

    def test_no_categories_error(self):
        """Schema rules without categories raise CompileError."""
        with pytest.raises(CompileError):
            loads("""
                object Token : 256
                let g = parser(
                    rules=[evaluation],
                    terminal=Token,
                    start=S
                )
                output g
            """)

    def test_no_terminal_error(self):
        """Schema rules without terminal= raise CompileError."""
        with pytest.raises(CompileError):
            loads("""
                category S
                object Token : 256
                let g = parser(
                    rules=[evaluation],
                    start=S
                )
                output g
            """)

    def test_inline_categories_still_work(self):
        """Inline categories=[...] still works."""
        prog = loads("""
            object Token : 256
            let g = parser(
                categories=[S, NP, N],
                rules=[evaluation],
                terminal=Token,
                start=S
            )
            output g
        """)
        tokens = torch.randint(0, 256, (2, 4))
        result = prog.morphism(tokens)
        assert result.shape == (2,)


class TestRuleDecl:
    """Tests for the `rule` keyword (rules of inference)."""

    def test_parse_binary_rule(self):
        """rule with two premises parses to RuleDecl."""
        ast = parse("""
            rule forward_app(X, Y) : X/Y, Y => X
        """)
        stmts = ast.statements
        assert len(stmts) == 1
        decl = stmts[0]
        assert isinstance(decl, RuleDecl)
        assert decl.name == "forward_app"
        assert decl.variables == ("X", "Y")
        assert len(decl.premises) == 2
        assert isinstance(decl.conclusion, CatPatternName)
        assert decl.conclusion.name == "X"

    def test_parse_unary_rule(self):
        """rule with one premise parses to RuleDecl."""
        ast = parse("""
            rule left_proj(A, B) : A * B => A
        """)
        stmts = ast.statements
        assert len(stmts) == 1
        decl = stmts[0]
        assert isinstance(decl, RuleDecl)
        assert decl.name == "left_proj"
        assert decl.variables == ("A", "B")
        assert len(decl.premises) == 1
        # premise is a product pattern
        assert isinstance(decl.premises[0], CatPatternProduct)
        assert isinstance(decl.conclusion, CatPatternName)

    def test_parse_slash_patterns(self):
        """Forward and backward slash patterns parse correctly."""
        ast = parse(r"""
            rule fwd(X, Y) : X/Y, Y => X
            rule bwd(X, Y) : Y, X\Y => X
        """)
        stmts = ast.statements
        assert len(stmts) == 2

        # forward: X/Y in first premise
        fwd = stmts[0]
        assert isinstance(fwd.premises[0], CatPatternSlash)
        assert fwd.premises[0].direction == "/"

        # backward: X\Y in second premise
        bwd = stmts[1]
        assert isinstance(bwd.premises[1], CatPatternSlash)
        assert bwd.premises[1].direction == "\\"

    def test_parse_composition_rule(self):
        """Composition rule with three variables parses correctly."""
        ast = parse("""
            rule fwd_comp(X, Y, Z) : X/Y, Y/Z => X/Z
        """)
        decl = ast.statements[0]
        assert isinstance(decl, RuleDecl)
        assert decl.variables == ("X", "Y", "Z")
        # conclusion is a slash pattern
        assert isinstance(decl.conclusion, CatPatternSlash)
        assert decl.conclusion.direction == "/"

    def test_compile_binary_rule(self):
        """Binary rule compiles to PatternBinarySchema."""
        from quivers.stochastic.schema import PatternBinarySchema

        src = """
            rule fwd(X, Y) : X/Y, Y => X
            category S, NP
            object Token : 10
            let g = parser(rules=[fwd], terminal=Token, start=S)
            output g
        """
        compiler = Compiler(parse(src))
        compiler.compile()
        rules = compiler.rules
        assert "fwd" in rules
        assert isinstance(rules["fwd"], PatternBinarySchema)

    def test_compile_unary_rule(self):
        """Unary rule compiles to PatternUnarySchema."""
        from quivers.stochastic.schema import PatternUnarySchema

        src = """
            rule proj(A, B) : A * B => A
            category S, NP
            object Token : 10
            let g = parser(
                rules=[evaluation, proj],
                terminal=Token,
                start=S,
                constructors=[slash, product]
            )
            output g
        """
        compiler = Compiler(parse(src))
        compiler.compile()
        rules = compiler.rules
        assert "proj" in rules
        assert isinstance(rules["proj"], PatternUnarySchema)

    def test_compile_duplicate_rule_error(self):
        """Duplicate rule names raise CompileError."""
        with pytest.raises(CompileError):
            loads("""
                rule fwd(X, Y) : X/Y, Y => X
                rule fwd(X, Y) : X/Y, Y => X
                category S
                object Token : 10
                let g = parser(rules=[fwd], terminal=Token, start=S)
                output g
            """)

    def test_rule_roundtrip(self):
        """DSL-declared forward/backward application produces a working parser."""
        prog = loads(r"""
            category S, NP, N

            rule fwd(X, Y) : X/Y, Y => X
            rule bwd(X, Y) : Y, X\Y => X

            object Token : 64

            let g = parser(
                rules=[fwd, bwd],
                terminal=Token,
                start=S
            )
            output g
        """)
        tokens = torch.randint(0, 64, (2, 4))
        result = prog.morphism(tokens)
        assert result.shape == (2,)

    def test_rule_matches_builtin_evaluation(self):
        """DSL-declared fwd+bwd produces the same rule count as built-in evaluation."""
        # built-in evaluation schema
        builtin = loads("""
            category S, NP, N
            object Token : 32
            let g = parser(
                rules=[evaluation],
                terminal=Token,
                start=S
            )
            output g
        """)

        # equivalent DSL-declared rules
        custom = loads(r"""
            category S, NP, N

            rule fwd(X, Y) : X/Y, Y => X
            rule bwd(X, Y) : Y, X\Y => X

            object Token : 32
            let g = parser(
                rules=[fwd, bwd],
                terminal=Token,
                start=S
            )
            output g
        """)

        # both should produce parsers with the same number of binary rules
        assert builtin.morphism.n_rules == custom.morphism.n_rules

    def test_mix_declared_and_builtin_rules(self):
        """DSL-declared rules and built-in schemas can be mixed."""
        prog = loads(r"""
            category S, NP, N

            rule fwd(X, Y) : X/Y, Y => X
            rule bwd(X, Y) : Y, X\Y => X

            object Token : 32

            let g = parser(
                rules=[fwd, bwd, harmonic_composition],
                terminal=Token,
                start=S
            )
            output g
        """)
        tokens = torch.randint(0, 32, (2, 4))
        result = prog.morphism(tokens)
        assert result.shape == (2,)

    def test_three_premise_rule_error(self):
        """Rules with 3 or more premises raise CompileError."""
        with pytest.raises(CompileError, match="3 premises"):
            loads("""
                rule bad(X, Y, Z) : X, Y, Z => X
                category S
                object Token : 10
                let g = parser(rules=[bad], terminal=Token, start=S)
                output g
            """)
