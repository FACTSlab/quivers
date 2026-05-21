"""Compiler: transform a quivers DSL AST into a trainable Program."""

from __future__ import annotations
from quivers.core.algebras import PRODUCT_FUZZY, Algebra
from quivers.core.objects import SetObject
from quivers.program import Program
from quivers.dsl.ast_nodes import (
    BundleDecl,
    CategoryDecl,
    CompositionDecl,
    ContractionDecl,
    DecoderDecl,
    DeductionDecl,
    EncoderDecl,
    ExportDecl,
    Expr,
    ExprIdent,
    LetDecl,
    LossDecl,
    Module,
    MorphismDecl,
    ProgramDecl,
    RuleDecl,
    SchemaDecl,
    SignatureDecl,
    Statement,
    ObjectDecl,
    ObjectExpr,
)
from quivers.dsl.compiler._prelude import (
    CompileError,
    _CompiledContraction,
    _build_default_trans_constructors,
    _build_default_trans_singletons,
    _register_extra_algebras,
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
        self._algebra: Algebra = PRODUCT_FUZZY
        self._categories: list[str] = []
        self._rules: dict = {}
        self._bundles: dict[str, tuple[str, ...]] = {}
        self._aliases: dict[str, ObjectExpr] = {}
        self._alias_names: set[str] = set()
        # Built-in transformation catalog: singletons looked up by
        # bare name (``expectation``, ``log_prob``, …) and
        # constructors invoked with arguments (``softmax(B)``,
        # ``bayes_invert(prior)``).  Disjoint from
        # `_transformations`, which holds user let-bound
        # transformations defined inside the module.
        self._trans_singletons: dict = _build_default_trans_singletons()
        self._trans_constructors: dict = _build_default_trans_constructors()
        # User-defined transformations bound via ``let t = …``.
        # Disjoint from `_morphisms`: a ``let`` whose RHS
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
        # compiled `EinsumWiring` plus the declared domain /
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
    def algebra(self) -> Algebra:
        """The active algebra."""
        return self._algebra

    @property
    def programs(self) -> dict:
        """Declared parametric program templates (``program NAME(...)``)."""
        return dict(getattr(self, "_program_templates", {}))

    @property
    def deductions(self) -> dict:
        """Declared deduction systems (``deduction NAME : ...``)."""
        return dict(getattr(self, "_deductions", {}))

    @property
    def signatures(self) -> dict:
        """Declared signatures (``signature NAME``)."""
        return dict(getattr(self, "_signatures", {}))

    @property
    def encoders(self) -> dict:
        """Declared encoders (``encoder NAME : Sig``)."""
        return dict(getattr(self, "_encoders", {}))

    @property
    def decoders(self) -> dict:
        """Declared decoders (``decoder NAME : Sig``)."""
        return dict(getattr(self, "_decoders", {}))

    @property
    def losses(self) -> dict:
        """Declared loss heads (``loss NAME : ... [on=...]``).

        Keyed by entry name; values are the registered :class:`LossEntry`
        records. Empty when no ``loss`` decl appears in the module.
        """
        reg = getattr(self, "_loss_registry", None)
        if reg is None:
            return {}
        return {entry.name: entry for entry in reg.entries()}

    @property
    def bundles(self) -> dict[str, tuple[str, ...]]:
        """Declared bundle aliases (``bundle NAME = R1 | R2 | ...``)."""
        return dict(self._bundles)

    @property
    def contractions(self) -> dict:
        """Declared tensor-network contractions
        (``contraction NAME : ... [wiring=...]``)."""
        return dict(getattr(self, "_contractions", {}))

    @property
    def transformations(self) -> dict:
        """User-declared transformation constructors / singletons."""
        return dict(self._transformations)

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
        _register_extra_algebras()
        for stmt in self._module.statements:
            self._compile_statement(stmt)
        if self._output_expr is None:
            # A module may declare only structural artifacts
            # (signatures, encoders, decoders, losses) with no
            # exported morphism; the returned Program is a container
            # carrying those artifacts.
            program = Program(None)
        elif isinstance(
            self._output_expr, ExprIdent
        ) and self._output_expr.name in getattr(self, "_program_templates", {}):
            # The export names a parametric program template. A
            # template has no root morphism on its own (a function
            # ``Pi (p : P). Kern(dom(p), cod(p))`` rather than a
            # single Kleisli arrow); the returned Program is a
            # container holding the template, which the caller
            # instantiates by attribute access.
            tmpl_name = self._output_expr.name
            program = Program(None)
            invoker = self._make_template_invoker(tmpl_name)
            program.templates = {tmpl_name: invoker}
            # Convenience: ``program.<name>(...)`` works directly.
            object.__setattr__(program, tmpl_name, invoker)
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
            Combined environment of every declared atom: the active
            algebra, objects, spaces, morphisms, rules, programs
            (parametric templates), deductions, signatures, encoders,
            decoders, losses, bundles, contractions, transformations.
        """
        _register_extra_algebras()
        for stmt in self._module.statements:
            self._compile_statement(stmt)
        env: dict = {}
        env["__algebra__"] = self._algebra
        for name, obj in self._objects.items():
            env[name] = obj
        for name, space in self._spaces.items():
            env[name] = space
        for name, morph in self._morphisms.items():
            env[name] = morph
        for name, rule in self._rules.items():
            env[name] = rule
        for name, tmpl in getattr(self, "_program_templates", {}).items():
            env[name] = tmpl
        for name, system in getattr(self, "_deductions", {}).items():
            env[name] = system
        for name, sig in getattr(self, "_signatures", {}).items():
            env[name] = sig
        for name, enc in getattr(self, "_encoders", {}).items():
            env[name] = enc
        for name, dec in getattr(self, "_decoders", {}).items():
            env[name] = dec
        reg = getattr(self, "_loss_registry", None)
        if reg is not None:
            for entry in reg.entries():
                env[entry.name] = entry
        for name, bundle in self._bundles.items():
            env[name] = bundle
        for name, contr in getattr(self, "_contractions", {}).items():
            env[name] = contr
        return env

    def _compile_statement(self, stmt: Statement) -> None:
        """Dispatch to the appropriate statement compiler."""
        if isinstance(stmt, CompositionDecl):
            self._compile_composition(stmt)
        elif isinstance(stmt, CategoryDecl):
            self._compile_category(stmt)
        elif isinstance(stmt, RuleDecl):
            self._compile_rule(stmt)
        elif isinstance(stmt, SchemaDecl):
            self._compile_schema(stmt)
        elif isinstance(stmt, ObjectDecl):
            self._compile_type(stmt)
        elif isinstance(stmt, BundleDecl):
            self._compile_bundle(stmt)
        elif isinstance(stmt, MorphismDecl):
            self._compile_morphism(stmt)
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
