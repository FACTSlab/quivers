"""Compiler: transform a quivers DSL AST into a trainable Program."""

from __future__ import annotations
from quivers.core.quantales import PRODUCT_FUZZY, Quantale
from quivers.core.objects import SetObject
from quivers.program import Program
from quivers.dsl.ast_nodes import (
    AliasDecl,
    BundleDecl,
    CategoryDecl,
    ContractionDecl,
    DecoderDecl,
    DeductionDecl,
    DiscretizeDecl,
    EmbedDecl,
    EncoderDecl,
    ExportDecl,
    Expr,
    KernelDecl,
    LetDecl,
    LossDecl,
    Module,
    MorphismDecl,
    ObjectDecl,
    ProgramDecl,
    QuantaleDecl,
    RuleDecl,
    SchemaDecl,
    SignatureDecl,
    SpaceDecl,
    Statement,
    TypeExpr,
)
from quivers.dsl.compiler._prelude import (
    CompileError,
    _CompiledContraction,
    _build_default_trans_constructors,
    _build_default_trans_singletons,
    _register_extra_quantales,
)
from quivers.dsl.compiler.declarations import _DeclarationsMixin
from quivers.dsl.compiler.programs import _ProgramsMixin
from quivers.dsl.compiler.structural import _StructuralMixin
from quivers.dsl.compiler.deductions import _DeductionsMixin
from quivers.dsl.compiler.resolution import _ResolutionMixin
from quivers.dsl.compiler.expressions import _ExpressionsMixin


class Compiler(
    _DeclarationsMixin,
    _ProgramsMixin,
    _StructuralMixin,
    _DeductionsMixin,
    _ResolutionMixin,
    _ExpressionsMixin,
):
    """Compile a quivers DSL AST into a Program.

    The compiler maintains three environments:

    - objects: name -> SetObject (discrete finite sets)
    - spaces: name -> ContinuousSpace
    - morphisms: name -> Morphism or ContinuousMorphism (any morphism-like)

    It processes statements in order and compiles the output
    expression into a Program wrapping the morphism DAG.

    Parameters
    ----------
    module : Module
        The parsed AST.
    """

    def __init__(self, module: Module) -> None:
        self._module = module
        self._quantale: Quantale = PRODUCT_FUZZY
        self._categories: list[str] = []
        self._rules: dict = {}
        self._bundles: dict[str, tuple[str, ...]] = {}
        self._aliases: dict[str, TypeExpr] = {}
        self._alias_names: set[str] = set()
        # Built-in transformation catalog: singletons looked up by
        # bare name (``expectation``, ``log_prob``, …) and
        # constructors invoked with arguments (``softmax(B)``,
        # ``bayes_invert(prior)``).  Disjoint from
        # :attr:`_transformations`, which holds user let-bound
        # transformations defined inside the module.
        self._trans_singletons: dict = _build_default_trans_singletons()
        self._trans_constructors: dict = _build_default_trans_constructors()
        # User-defined transformations bound via ``let t = …``.
        # Disjoint from :attr:`_morphisms`: a ``let`` whose RHS
        # resolves to a transformation lands here; a ``let`` whose
        # RHS resolves to a morphism lands in ``_morphisms``.
        self._transformations: dict = {}
        self._objects: dict[str, SetObject] = {}
        self._spaces: dict = {}
        self._morphisms: dict = {}
        self._groups: dict[str, list[str]] = {}
        self._output_expr: Expr | None = None
        # Parametric-program templates: dependent kernels Π(p:P).Kern(dom(p),cod(p))
        # stored as their unsubstituted AST decl. Instantiated at each call
        # site by parameter substitution + α-renaming of internal latents.
        self._program_templates: dict[str, ProgramDecl] = {}
        # Operadic contraction declarations. Each entry is callable
        # from the DSL at let-binding sites; the value records the
        # compiled :class:`EinsumWiring` plus the declared domain /
        # codomain typing for shape-checking at invocation time.
        self._contractions: dict[str, "_CompiledContraction"] = {}

    @property
    def categories(self) -> list[str]:
        """The declared category atoms."""
        return list(self._categories)

    @property
    def rules(self) -> dict:
        """The compiled rule schemas from ``rule`` declarations."""
        return dict(self._rules)

    @property
    def objects(self) -> dict[str, SetObject]:
        """The compiled object environment."""
        return dict(self._objects)

    @property
    def spaces(self) -> dict:
        """The compiled space environment."""
        return dict(self._spaces)

    @property
    def morphisms(self) -> dict:
        """The compiled morphism environment."""
        return dict(self._morphisms)

    @property
    def quantale(self) -> Quantale:
        """The active quantale."""
        return self._quantale

    def compile(self) -> Program:
        """Compile the module into a trainable Program.

        Returns
        -------
        Program
            The compiled nn.Module wrapping the morphism DAG.

        Raises
        ------
        CompileError
            On semantic errors (undefined names, type mismatches, etc.).
        """
        _register_extra_quantales()
        for stmt in self._module.statements:
            self._compile_statement(stmt)
        if self._output_expr is None:
            # A module may declare only structural artifacts
            # (signatures, encoders, decoders, losses) with no
            # exported morphism; the returned Program is a container
            # carrying those artifacts.
            program = Program(None)
        else:
            root_morphism = self._compile_expr(self._output_expr)
            program = Program(root_morphism)
        # Attach the compiler's deduction and posterior registries to
        # the Program so downstream callers can reach them after
        # `quivers.dsl.load(...)`.
        program.deductions = getattr(self, "_deductions", {})
        program.posteriors = getattr(self, "_posteriors", {})
        program.signatures = getattr(self, "_signatures", {})
        program.encoders = getattr(self, "_encoders", {})
        program.decoders = getattr(self, "_decoders", {})
        program.losses = getattr(self, "_loss_registry", None)
        # Wire the loss registry into every compiled deduction so the
        # agenda's rule-firing and chart-completion paths can fire
        # rule-attached and chart-attached losses automatically.
        if program.losses is not None:
            for name, system in program.deductions.items():
                system._loss_registry = program.losses  # type: ignore[attr-defined]
                system._deduction_name = name  # type: ignore[attr-defined]
        return program

    def compile_env(self) -> dict:
        """Compile all statements and return the full environment.

        Useful for inspection without requiring an output declaration.

        Returns
        -------
        dict
            Combined environment of objects, spaces, morphisms, and the quantale.
        """
        _register_extra_quantales()
        for stmt in self._module.statements:
            self._compile_statement(stmt)
        env: dict = {}
        env["__quantale__"] = self._quantale
        for name, obj in self._objects.items():
            env[name] = obj
        for name, space in self._spaces.items():
            env[name] = space
        for name, morph in self._morphisms.items():
            env[name] = morph
        for name, rule in self._rules.items():
            env[name] = rule
        return env

    def _compile_statement(self, stmt: Statement) -> None:
        """Dispatch to the appropriate statement compiler."""
        if isinstance(stmt, QuantaleDecl):
            self._compile_quantale(stmt)
        elif isinstance(stmt, CategoryDecl):
            self._compile_category(stmt)
        elif isinstance(stmt, RuleDecl):
            self._compile_rule(stmt)
        elif isinstance(stmt, SchemaDecl):
            self._compile_schema(stmt)
        elif isinstance(stmt, AliasDecl):
            self._compile_alias(stmt)
        elif isinstance(stmt, BundleDecl):
            self._compile_bundle(stmt)
        elif isinstance(stmt, ObjectDecl):
            self._compile_object(stmt)
        elif isinstance(stmt, MorphismDecl):
            self._compile_morphism(stmt)
        elif isinstance(stmt, SpaceDecl):
            self._compile_space(stmt)
        elif isinstance(stmt, KernelDecl):
            self._compile_kernel(stmt)
        elif isinstance(stmt, DiscretizeDecl):
            self._compile_discretize(stmt)
        elif isinstance(stmt, EmbedDecl):
            self._compile_embed(stmt)
        elif isinstance(stmt, ProgramDecl):
            self._compile_program(stmt)
        elif isinstance(stmt, ContractionDecl):
            self._compile_contraction(stmt)
        elif isinstance(stmt, LetDecl):
            self._compile_let(stmt)
        elif isinstance(stmt, ExportDecl):
            self._compile_export(stmt)
        elif isinstance(stmt, DeductionDecl):
            self._compile_deduction(stmt)
        elif isinstance(stmt, SignatureDecl):
            self._compile_signature(stmt)
        elif isinstance(stmt, EncoderDecl):
            self._compile_encoder(stmt)
        elif isinstance(stmt, DecoderDecl):
            self._compile_decoder(stmt)
        elif isinstance(stmt, LossDecl):
            self._compile_loss(stmt)
        else:
            raise CompileError(f"unknown statement type: {type(stmt).__name__}")
