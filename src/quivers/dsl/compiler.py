"""Compiler: transform a quivers DSL AST into a trainable Program.

The compiler walks the AST in declaration order, building up an
environment of objects, spaces, and morphisms, then compiles the
output expression into a quivers.Program (nn.Module).

Supports both discrete (FinSet-based) and continuous (ContinuousSpace-
based) morphisms, including stochastic (Markov kernels), boundary
(Discretize/Embed), and parameterized family distributions.
"""

from __future__ import annotations
from collections.abc import Callable
import torch
from quivers.continuous.spaces import ContinuousSpace
from quivers.continuous.morphisms import AnySpace
from quivers.core.objects import SetObject, FinSet, ProductSet
from quivers.core.quantales import Quantale, PRODUCT_FUZZY, BOOLEAN
from quivers.core.morphisms import morphism as make_latent, identity as make_identity
from quivers.program import Program
from quivers.dsl.ast_nodes import (
    Module,
    Statement,
    QuantaleDecl,
    CategoryDecl,
    RuleDecl,
    ObjectDecl,
    MorphismDecl,
    SpaceDecl,
    ContinuousMorphismDecl,
    StochasticMorphismDecl,
    AliasDecl,
    BundleDecl,
    DiscretizeDecl,
    EmbedDecl,
    EnumSetLiteral,
    FreeMonoidExpr,
    FreeResiduatedExpr,
    SchemaDecl,
    LetStep,
    LetExprBinOp,
    LetExprUnaryOp,
    LetExprCall,
    LetExprLiteral,
    LetExprVar,
    LetExprNode,
    ProgramDecl,
    LetDecl,
    OutputDecl,
    TypeExpr,
    TypeName,
    TypeProduct,
    SpaceExpr,
    SpaceConstructor,
    Expr,
    ExprIdent,
    ExprIdentity,
    ExprCompose,
    ExprTensorProduct,
    ExprChartFold,
    ExprCurry,
    ExprMarginalize,
    ExprFan,
    ExprRepeat,
    ExprStack,
    ExprScan,
    ExprParser,
)

_QUANTALE_REGISTRY: dict[str, Quantale] = {
    "product_fuzzy": PRODUCT_FUZZY,
    "boolean": BOOLEAN,
}


def _register_extra_quantales() -> None:
    """Lazily register extra quantales if the module is available."""
    if "lukasiewicz" not in _QUANTALE_REGISTRY:
        try:
            from quivers.core.extra_quantales import LUKASIEWICZ, GODEL, TROPICAL

            _QUANTALE_REGISTRY["lukasiewicz"] = LUKASIEWICZ
            _QUANTALE_REGISTRY["godel"] = GODEL
            _QUANTALE_REGISTRY["tropical"] = TROPICAL
        except ImportError:
            pass
    if "markov" not in _QUANTALE_REGISTRY:
        try:
            from quivers.stochastic import MARKOV

            _QUANTALE_REGISTRY["markov"] = MARKOV
        except ImportError:
            pass


_FAMILY_REGISTRY: dict[str, type] | None = None


def _get_family_registry() -> dict[str, type]:
    """Lazily build the distribution family registry."""
    global _FAMILY_REGISTRY
    if _FAMILY_REGISTRY is not None:
        return _FAMILY_REGISTRY
    from quivers.continuous.families import (
        ConditionalNormal,
        ConditionalLogitNormal,
        ConditionalBeta,
        ConditionalTruncatedNormal,
        ConditionalDirichlet,
        ConditionalCauchy,
        ConditionalLaplace,
        ConditionalGumbel,
        ConditionalLogNormal,
        ConditionalStudentT,
        ConditionalExponential,
        ConditionalGamma,
        ConditionalChi2,
        ConditionalHalfCauchy,
        ConditionalHalfNormal,
        ConditionalInverseGamma,
        ConditionalWeibull,
        ConditionalPareto,
        ConditionalKumaraswamy,
        ConditionalContinuousBernoulli,
        ConditionalFisherSnedecor,
        ConditionalUniform,
        ConditionalMultivariateNormal,
        ConditionalLowRankMVN,
        ConditionalRelaxedBernoulli,
        ConditionalRelaxedOneHotCategorical,
        ConditionalWishart,
        ConditionalBernoulli,
        ConditionalCategorical,
    )

    _FAMILY_REGISTRY = {
        "Normal": ConditionalNormal,
        "LogitNormal": ConditionalLogitNormal,
        "Beta": ConditionalBeta,
        "TruncatedNormal": ConditionalTruncatedNormal,
        "Dirichlet": ConditionalDirichlet,
        "Cauchy": ConditionalCauchy,
        "Laplace": ConditionalLaplace,
        "Gumbel": ConditionalGumbel,
        "LogNormal": ConditionalLogNormal,
        "StudentT": ConditionalStudentT,
        "Exponential": ConditionalExponential,
        "Gamma": ConditionalGamma,
        "Chi2": ConditionalChi2,
        "HalfCauchy": ConditionalHalfCauchy,
        "HalfNormal": ConditionalHalfNormal,
        "InverseGamma": ConditionalInverseGamma,
        "Weibull": ConditionalWeibull,
        "Pareto": ConditionalPareto,
        "Kumaraswamy": ConditionalKumaraswamy,
        "ContinuousBernoulli": ConditionalContinuousBernoulli,
        "FisherSnedecor": ConditionalFisherSnedecor,
        "Uniform": ConditionalUniform,
        "MultivariateNormal": ConditionalMultivariateNormal,
        "LowRankMVN": ConditionalLowRankMVN,
        "RelaxedBernoulli": ConditionalRelaxedBernoulli,
        "RelaxedOneHotCategorical": ConditionalRelaxedOneHotCategorical,
        "Wishart": ConditionalWishart,
        "Bernoulli": ConditionalBernoulli,
        "Categorical": ConditionalCategorical,
    }
    try:
        from quivers.continuous.families import ConditionalGeneralizedPareto

        _FAMILY_REGISTRY["GeneralizedPareto"] = ConditionalGeneralizedPareto
    except ImportError, AttributeError:
        pass
    return _FAMILY_REGISTRY


_SPACE_CONSTRUCTORS: (
    dict[str, type[ContinuousSpace] | Callable[..., ContinuousSpace]] | None
) = None


def _get_space_constructors() -> dict[
    str, type[ContinuousSpace] | Callable[..., ContinuousSpace]
]:
    """Lazily build the space constructor registry."""
    global _SPACE_CONSTRUCTORS
    if _SPACE_CONSTRUCTORS is not None:
        return _SPACE_CONSTRUCTORS
    from quivers.continuous.spaces import (
        Euclidean,
        Simplex,
        PositiveReals,
        UnitInterval,
        ProductSpace,
    )

    _SPACE_CONSTRUCTORS = {
        "Euclidean": Euclidean,
        "Simplex": Simplex,
        "PositiveReals": PositiveReals,
        "UnitInterval": UnitInterval,
        "ProductSpace": ProductSpace,
    }
    return _SPACE_CONSTRUCTORS


class _ChartHandlerComposite(torch.nn.Module):
    """Post-handler composition over a chart parser's output.

    Wraps a base ``InsideAlgorithm`` (or any callable returning a
    ``(batch, N)``-shaped tensor of log-probabilities over the start
    symbol's enriched category cell) and composes one or more
    handler morphisms on the output. Each handler's tensor is taken
    as a ``N × N'`` log-probability transition that reduces the
    effect stack on the output cell.
    """

    def __init__(self, base, handler) -> None:
        super().__init__()
        self._base = base
        self._handler = handler
        # Register the handler's module so parameters and buffers are
        # tracked through training.
        if hasattr(handler, "module"):
            self._handler_mod = handler.module()
        else:
            self._handler_mod = handler

    def forward(self, tokens):
        base_out = self._base(tokens)
        # base_out shape: (batch,) for the start-symbol log-prob, or
        # (batch, N) for the cell distribution. Handlers reduce the
        # cell distribution along the N axis via log-space matrix
        # multiplication; if base_out is scalar, the handler is a
        # no-op identity on the start-symbol axis.
        if base_out.dim() == 1:
            return base_out
        log_handler = torch.log(self._handler.tensor.clamp(min=1e-30))
        # log[batch, B] = logsumexp_A(base_out[batch, A] + log_handler[A, B])
        return torch.logsumexp(
            base_out.unsqueeze(2) + log_handler.unsqueeze(0), dim=1
        )

    def __repr__(self) -> str:
        return f"ChartHandlerComposite({self._base!r} ; {self._handler!r})"


class CompileError(Exception):
    """Raised when the compiler encounters a semantic error.

    Parameters
    ----------
    message : str
        Error description.
    line : int
        Source line number (0 if unknown).
    col : int
        Source column number (0 if unknown).
    """

    def __init__(self, message: str, line: int = 0, col: int = 0) -> None:
        self.line = line
        self.col = col
        loc = f"line {line}, col {col}: " if line else ""
        super().__init__(f"{loc}{message}")


class Compiler:
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
        self._objects: dict[str, SetObject] = {}
        self._spaces: dict = {}
        self._morphisms: dict = {}
        self._groups: dict[str, list[str]] = {}
        self._output_expr: Expr | None = None

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
            raise CompileError("no output declaration found")
        root_morphism = self._compile_expr(self._output_expr)
        return Program(root_morphism)

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
        elif isinstance(stmt, ContinuousMorphismDecl):
            self._compile_continuous_morphism(stmt)
        elif isinstance(stmt, StochasticMorphismDecl):
            self._compile_stochastic_morphism(stmt)
        elif isinstance(stmt, DiscretizeDecl):
            self._compile_discretize(stmt)
        elif isinstance(stmt, EmbedDecl):
            self._compile_embed(stmt)
        elif isinstance(stmt, ProgramDecl):
            self._compile_program(stmt)
        elif isinstance(stmt, LetDecl):
            self._compile_let(stmt)
        elif isinstance(stmt, OutputDecl):
            self._compile_output(stmt)
        else:
            raise CompileError(f"unknown statement type: {type(stmt).__name__}")

    def _compile_quantale(self, decl: QuantaleDecl) -> None:
        """Set the active quantale."""
        name = decl.name.lower()
        if name not in _QUANTALE_REGISTRY:
            raise CompileError(
                f"unknown quantale {decl.name!r}; available: {', '.join(sorted(_QUANTALE_REGISTRY))}",
                decl.line,
                decl.col,
            )
        self._quantale = _QUANTALE_REGISTRY[name]

    def _compile_category(self, decl: CategoryDecl) -> None:
        """Register a category atom declaration.

        Category atoms are generators for a free categorical structure,
        distinct from finite set objects.  They are used by the parser
        compiler to build a ``CategorySystem``.
        """
        if decl.name in self._categories:
            raise CompileError(
                f"category {decl.name!r} already declared", decl.line, decl.col
            )
        self._categories.append(decl.name)

    def _compile_rule(self, decl: RuleDecl) -> None:
        """Compile a rule-of-inference declaration into a RuleSchema.

        Creates a ``PatternBinarySchema`` (2 premises) or
        ``PatternUnarySchema`` (1 premise) and registers it by name
        so it can be resolved in ``parser(rules=[...])``.
        """
        from quivers.stochastic.schema import (
            PatternBinarySchema,
            PatternUnarySchema,
            SCHEMA_REGISTRY,
        )

        if decl.name in self._rules:
            raise CompileError(
                f"rule {decl.name!r} already declared", decl.line, decl.col
            )
        if decl.name in SCHEMA_REGISTRY:
            raise CompileError(
                f"rule {decl.name!r} shadows a built-in schema; choose a different name",
                decl.line,
                decl.col,
            )
        variables = frozenset(decl.variables)
        n_premises = len(decl.premises)
        if n_premises == 2:
            schema = PatternBinarySchema(
                left_pattern=decl.premises[0],
                right_pattern=decl.premises[1],
                conclusion_pattern=decl.conclusion,
                variables=variables,
                name=decl.name,
            )
        elif n_premises == 1:
            schema = PatternUnarySchema(
                premise_pattern=decl.premises[0],
                conclusion_pattern=decl.conclusion,
                variables=variables,
                name=decl.name,
            )
        else:
            raise CompileError(
                f"rule {decl.name!r} has {n_premises} premises; only unary (1) and binary (2) rules are supported",
                decl.line,
                decl.col,
            )
        self._rules[decl.name] = schema

    def _compile_schema(self, decl: SchemaDecl) -> None:
        """Compile a pattern-polymorphic schema declaration.

        Creates a ``PatternBinarySchema`` when the declared domain is a
        :class:`TypeProduct` with two components, otherwise a
        ``PatternUnarySchema``. Pattern variables are the union of the
        ``names`` lists across all :class:`SchemaParameter` entries; the
        parameter type-expression is consulted only for well-formedness
        (it must reference a residuated universe in scope; the
        type-checker does not yet enforce this — the chart-parser
        catches mismatches at firing time).
        """
        from quivers.stochastic.schema import (
            PatternBinarySchema,
            PatternUnarySchema,
            SCHEMA_REGISTRY,
        )

        if decl.name in self._rules:
            raise CompileError(
                f"schema {decl.name!r} already declared",
                decl.line,
                decl.col,
            )
        if decl.name in SCHEMA_REGISTRY:
            raise CompileError(
                f"schema {decl.name!r} shadows a built-in schema; choose a different name",
                decl.line,
                decl.col,
            )

        variables: frozenset[str] = frozenset(
            n for group in decl.parameter_names for n in group
        )

        # Decide arity from the domain shape:
        #  - top-level TypeProduct with exactly 2 components → binary
        #  - any other shape (TypeName, TypeSlash, TypeEffectApply,
        #    or a non-binary TypeProduct) → unary
        if isinstance(decl.domain, TypeProduct) and len(decl.domain.components) == 2:
            left, right = decl.domain.components
            schema = PatternBinarySchema(
                left_pattern=left,
                right_pattern=right,
                conclusion_pattern=decl.codomain,
                variables=variables,
                name=decl.name,
            )
        else:
            schema = PatternUnarySchema(
                premise_pattern=decl.domain,
                conclusion_pattern=decl.codomain,
                variables=variables,
                name=decl.name,
            )

        self._rules[decl.name] = schema

    def _compile_alias(self, decl) -> None:
        """Compile an ``alias Foo = ...`` type-level alias.

        Two cases:

        - The right-hand side resolves cleanly as a :class:`SetObject`
          (TypeName / TypeProduct / TypeCoproduct over named objects).
          The alias binds to that SetObject in :attr:`self._objects`,
          so ``Foo`` is usable wherever an ordinary object reference
          is — `latent f : Foo -> Bar`, `parser(rules=..., terminal=Foo)`
          etc.
        - The right-hand side is a residuated pattern (TypeSlash /
          TypeEffectApply) or otherwise fails SetObject resolution.
          The alias is recorded in :attr:`self._aliases` for textual
          substitution at use site (inside schema patterns).
        """
        if decl.name in self._alias_names:
            raise CompileError(
                f"alias {decl.name!r} already declared",
                decl.line,
                decl.col,
            )
        if decl.name in self._objects:
            raise CompileError(
                f"alias {decl.name!r} shadows an existing object",
                decl.line,
                decl.col,
            )
        self._alias_names.add(decl.name)
        try:
            resolved = self._resolve_type(decl.type_expr, decl.name)
        except TypeError, KeyError:
            # Residuated / effect-typed RHS: record as a syntactic
            # alias for substitution at schema-pattern use site.
            self._aliases[decl.name] = decl.type_expr
            return
        self._objects[decl.name] = resolved

    def _compile_bundle(self, decl) -> None:
        """Compile a ``bundle CCG = [r1, r2, ...]`` rule bundle.

        Each entry must resolve at compile time as either a previously-
        declared rule / schema or as a built-in entry of
        :data:`SCHEMA_REGISTRY`. The bundle is recorded under its name
        in ``self._bundles`` so ``parser(rules=CCG)`` and
        ``chart_fold(binary=CCG)`` can splice its members.
        """
        from quivers.stochastic.schema import SCHEMA_REGISTRY

        if decl.name in self._bundles:
            raise CompileError(
                f"bundle {decl.name!r} already declared",
                decl.line,
                decl.col,
            )
        if decl.name in self._rules or decl.name in SCHEMA_REGISTRY:
            raise CompileError(
                f"bundle {decl.name!r} shadows a rule / built-in schema",
                decl.line,
                decl.col,
            )
        # Member references are resolved lazily at use-site (in the
        # parser-rules expander) so that bundles can forward-reference
        # other bundles. Cycles surface as ``cycle through ...`` errors
        # at expansion time.
        self._bundles[decl.name] = tuple(decl.rules)

    def _compile_object(self, decl: ObjectDecl) -> None:
        """Compile an object declaration into the environment.

        Three surface forms are recognised:

        - ``object X : <type_expr>`` — resolves via the
          :class:`TypeExprToSetObject` lens.
        - ``object Atoms = {NP, S, VP}`` — constructs an
          :class:`EnumSet`.
        - ``object Cat = FreeResiduated(Atoms, depth=, ops=[...])`` —
          constructs a :class:`FreeResiduated` over a previously-declared
          :class:`EnumSet`.
        """
        from quivers.core.objects import EnumSet, FreeResiduated

        if decl.name in self._objects:
            raise CompileError(
                f"object {decl.name!r} already declared", decl.line, decl.col
            )

        if decl.type_expr is not None:
            obj = self._resolve_type(decl.type_expr, decl.name)
            self._objects[decl.name] = obj
            return

        if decl.init is None:
            raise CompileError(
                f"object {decl.name!r} has no type or initializer",
                decl.line,
                decl.col,
            )

        if isinstance(decl.init, EnumSetLiteral):
            self._objects[decl.name] = EnumSet(
                name=decl.name, elements=decl.init.elements
            )
            return

        if isinstance(decl.init, FreeMonoidExpr):
            from quivers.core.objects import FinSet, FreeMonoid

            gen = self._objects.get(decl.init.generators)
            if not isinstance(gen, FinSet):
                raise CompileError(
                    f"FreeMonoid generators {decl.init.generators!r} must "
                    f"reference a previously-declared FinSet (got "
                    f"{type(gen).__name__ if gen else 'undefined'})",
                    decl.line,
                    decl.col,
                )
            self._objects[decl.name] = FreeMonoid(
                generators=gen, max_length=decl.init.max_length
            )
            return

        if isinstance(decl.init, FreeResiduatedExpr):
            gen = self._objects.get(decl.init.generators)
            if not isinstance(gen, EnumSet):
                raise CompileError(
                    f"FreeResiduated generators {decl.init.generators!r} must "
                    f"reference a previously-declared EnumSet (got "
                    f"{type(gen).__name__ if gen else 'undefined'})",
                    decl.line,
                    decl.col,
                )
            self._objects[decl.name] = FreeResiduated(
                generators=gen,
                depth=decl.init.depth,
                ops=decl.init.ops,
            )
            return

        raise CompileError(
            f"unrecognised object initializer for {decl.name!r}",
            decl.line,
            decl.col,
        )

    def _compile_morphism(self, decl: MorphismDecl) -> None:
        """Compile a morphism declaration into the environment."""
        if decl.name in self._morphisms:
            raise CompileError(
                f"morphism {decl.name!r} already declared", decl.line, decl.col
            )
        domain = self._resolve_type(decl.domain)
        codomain = self._resolve_type(decl.codomain)
        if decl.morphism_kind == "latent":
            scale = float(decl.options.get("scale", "0.5"))
            morph = make_latent(
                domain, codomain, init_scale=scale, quantale=self._quantale
            )
        elif decl.morphism_kind == "observed":
            if decl.init_expr is not None:
                morph = self._compile_expr(decl.init_expr)
                if morph.domain != domain or morph.codomain != codomain:
                    raise CompileError(
                        f"morphism {decl.name!r} init expression has type {morph.domain!r} -> {morph.codomain!r}, expected {domain!r} -> {codomain!r}",
                        decl.line,
                        decl.col,
                    )
            else:
                raise CompileError(
                    f"observed morphism {decl.name!r} requires an initializer (e.g. = identity({decl.domain}))",
                    decl.line,
                    decl.col,
                )
        else:
            raise CompileError(
                f"unknown morphism kind {decl.morphism_kind!r}", decl.line, decl.col
            )
        self._morphisms[decl.name] = morph

    def _compile_space(self, decl: SpaceDecl) -> None:
        """Compile a space declaration into the space environment."""
        if decl.name in self._spaces:
            raise CompileError(
                f"space {decl.name!r} already declared", decl.line, decl.col
            )
        space = self._resolve_space(decl.space_expr, decl.name)
        self._spaces[decl.name] = space

    def _compile_continuous_morphism(self, decl: ContinuousMorphismDecl) -> None:
        """Compile a continuous morphism declaration.

        If ``decl.replicate`` is set, creates N independent copies
        named ``name_0`` through ``name_{N-1}`` and registers the
        base name as a group.
        """
        if decl.name in self._morphisms:
            raise CompileError(
                f"morphism {decl.name!r} already declared", decl.line, decl.col
            )
        domain = self._resolve_any_space(decl.domain)
        codomain = self._resolve_any_space(decl.codomain)
        count = decl.replicate if decl.replicate is not None else 1
        names = (
            [f"{decl.name}_{i}" for i in range(count)]
            if decl.replicate is not None
            else [decl.name]
        )
        for name in names:
            morph = self._make_continuous_morphism(
                domain, codomain, decl.family, decl.options, decl
            )
            self._morphisms[name] = morph
        if decl.replicate is not None:
            self._groups[decl.name] = names

    def _make_continuous_morphism(
        self, domain, codomain, family_name: str, options: dict[str, str], decl
    ):
        """Create a single continuous morphism from a family name."""
        if family_name == "Flow":
            from quivers.continuous.flows import ConditionalFlow

            n_layers = int(options.get("n_layers", "4"))
            hidden_dim = int(options.get("hidden_dim", "64"))
            return ConditionalFlow(
                domain, codomain, n_layers=n_layers, hidden_dim=hidden_dim
            )
        registry = _get_family_registry()
        if family_name not in registry:
            raise CompileError(
                f"unknown distribution family {family_name!r}; available: {', '.join(sorted(registry))}",
                decl.line,
                decl.col,
            )
        cls = registry[family_name]
        hidden_dim = int(options.get("hidden_dim", "64"))
        kwargs: dict = {"hidden_dim": hidden_dim}
        if "rank" in options:
            kwargs["rank"] = int(options["rank"])
        if "temperature" in options:
            kwargs["temperature"] = float(options["temperature"])
        return cls(domain, codomain, **kwargs)

    def _compile_stochastic_morphism(self, decl: StochasticMorphismDecl) -> None:
        """Compile a stochastic morphism declaration."""
        if decl.name in self._morphisms:
            raise CompileError(
                f"morphism {decl.name!r} already declared", decl.line, decl.col
            )
        from quivers.stochastic import StochasticMorphism

        domain = self._resolve_type(decl.domain)
        codomain = self._resolve_type(decl.codomain)
        count = decl.replicate if decl.replicate is not None else 1
        names = (
            [f"{decl.name}_{i}" for i in range(count)]
            if decl.replicate is not None
            else [decl.name]
        )
        for name in names:
            morph = StochasticMorphism(domain, codomain)
            self._morphisms[name] = morph
        if decl.replicate is not None:
            self._groups[decl.name] = names

    def _compile_discretize(self, decl: DiscretizeDecl) -> None:
        """Compile a discretize boundary morphism."""
        if decl.name in self._morphisms:
            raise CompileError(
                f"morphism {decl.name!r} already declared", decl.line, decl.col
            )
        if decl.space_name not in self._spaces:
            raise CompileError(
                f"undefined space {decl.space_name!r}", decl.line, decl.col
            )
        from quivers.continuous.boundaries import Discretize

        space = self._spaces[decl.space_name]
        morph = Discretize(space, n_bins=decl.n_bins)
        self._morphisms[decl.name] = morph

    def _compile_embed(self, decl: EmbedDecl) -> None:
        """Compile an embed boundary morphism."""
        if decl.name in self._morphisms:
            raise CompileError(
                f"morphism {decl.name!r} already declared", decl.line, decl.col
            )
        if decl.domain_name not in self._objects:
            raise CompileError(
                f"undefined object {decl.domain_name!r}", decl.line, decl.col
            )
        if decl.codomain_name not in self._spaces:
            raise CompileError(
                f"undefined space {decl.codomain_name!r}", decl.line, decl.col
            )
        from quivers.continuous.boundaries import Embed

        domain = self._objects[decl.domain_name]
        codomain = self._spaces[decl.codomain_name]
        count = decl.replicate if decl.replicate is not None else 1
        names = (
            [f"{decl.name}_{i}" for i in range(count)]
            if decl.replicate is not None
            else [decl.name]
        )
        for name in names:
            assert isinstance(domain, FinSet)
            morph = Embed(domain, codomain)
            self._morphisms[name] = morph
        if decl.replicate is not None:
            self._groups[decl.name] = names

    def _compile_program(self, decl: ProgramDecl) -> None:
        """Compile a monadic program block into a MonadicProgram."""
        if decl.name in self._morphisms:
            raise CompileError(
                f"morphism {decl.name!r} already declared", decl.line, decl.col
            )
        from quivers.continuous.programs import MonadicProgram

        domain = self._resolve_any_space(decl.domain)
        codomain = self._resolve_any_space(decl.codomain)
        from quivers.continuous.spaces import ProductSpace as _PS

        if decl.params is not None:
            if isinstance(domain, (ProductSet, _PS)):
                if len(decl.params) != len(domain.components):
                    raise CompileError(
                        f"program has {len(decl.params)} params but domain has {len(domain.components)} components",
                        decl.line,
                        decl.col,
                    )
            elif len(decl.params) != 1:
                raise CompileError(
                    f"program has {len(decl.params)} params but domain is not a product type",
                    decl.line,
                    decl.col,
                )
        bound_vars: dict[str, AnySpace | None] = {}
        if decl.params is not None:
            if isinstance(domain, (ProductSet, _PS)):
                for pname, factor in zip(decl.params, domain.components):
                    bound_vars[pname] = factor
            else:
                bound_vars[decl.params[0]] = domain
        steps: list[tuple] = []
        for step in decl.draws:
            if isinstance(step, LetStep):
                if step.name in bound_vars:
                    raise CompileError(
                        f"variable {step.name!r} already bound in program",
                        step.line,
                        step.col,
                    )
                if isinstance(step.value, LetExprVar):
                    if step.value.name not in bound_vars:
                        raise CompileError(
                            f"undefined variable {step.value.name!r} in let binding",
                            step.line,
                            step.col,
                        )
                    bound_vars[step.name] = bound_vars[step.value.name]
                    steps.append(((step.name,), None, step.value.name))
                elif isinstance(step.value, LetExprLiteral):
                    bound_vars[step.name] = None
                    steps.append(((step.name,), None, step.value.value))
                else:
                    self._validate_let_expr_vars(step.value, bound_vars, step)
                    compiled_fn = self._compile_let_expr(step.value)
                    bound_vars[step.name] = None
                    steps.append(((step.name,), None, compiled_fn))
                continue
            draw = step
            for v in draw.vars:
                if v in bound_vars:
                    raise CompileError(
                        f"variable {v!r} already bound in program", draw.line, draw.col
                    )
            morph, step_args = self._resolve_draw_morphism(draw, bound_vars, codomain)
            if step_args is not None:
                for arg_name in step_args:
                    if arg_name not in bound_vars:
                        raise CompileError(
                            f"undefined variable {arg_name!r} in draw step",
                            draw.line,
                            draw.col,
                        )
            if len(draw.vars) == 1:
                bound_vars[draw.vars[0]] = morph.codomain
            elif isinstance(morph, MonadicProgram) and (not morph._return_is_single):
                if len(draw.vars) != len(morph._return_vars):
                    raise CompileError(
                        f"destructuring {len(draw.vars)} vars but sub-program returns {len(morph._return_vars)}",
                        draw.line,
                        draw.col,
                    )
                for v in draw.vars:
                    bound_vars[v] = None
            elif isinstance(morph.codomain, ProductSet):
                if len(draw.vars) != len(morph.codomain.components):
                    raise CompileError(
                        f"destructuring {len(draw.vars)} vars but codomain has {len(morph.codomain.components)} components",
                        draw.line,
                        draw.col,
                    )
                for v, factor in zip(draw.vars, morph.codomain.components):
                    bound_vars[v] = factor
            else:
                raise CompileError(
                    f"cannot destructure non-product codomain {morph.codomain!r}",
                    draw.line,
                    draw.col,
                )
            steps.append((draw.vars, morph, step_args, draw.is_observed))
        for rv in decl.return_vars:
            if rv not in bound_vars:
                raise CompileError(
                    f"return variable {rv!r} not bound in program", decl.line, decl.col
                )
        prog = MonadicProgram(
            domain,
            codomain,
            steps,
            decl.return_vars,
            params=decl.params,
            return_labels=decl.return_labels,
        )
        self._morphisms[decl.name] = prog

    def _resolve_draw_morphism(
        self,
        draw,
        bound_vars: dict[str, AnySpace | None],
        program_codomain: SetObject | ContinuousSpace | None,
    ) -> tuple:
        """Resolve a draw step's morphism, handling both named morphisms
        and inline distribution families.

        Parameters
        ----------
        draw : DrawStep
            The draw step to resolve.
        bound_vars : dict
            Currently bound variable types.
        program_codomain : object
            The program's declared codomain (for type inference).

        Returns
        -------
        tuple of (morphism, step_args)
            The compiled morphism and the variable-only args for
            the step spec (None = use program input).
        """
        if draw.morphism in self._morphisms:
            morph = self._morphisms[draw.morphism]
            if draw.args is not None:
                for a in draw.args:
                    if isinstance(a, (int, float)):
                        raise CompileError(
                            f"literal argument {a} not allowed for named morphism {draw.morphism!r}",
                            draw.line,
                            draw.col,
                        )
            step_args = (
                tuple((str(a) for a in draw.args)) if draw.args is not None else None
            )
            return (morph, step_args)
        from quivers.continuous.inline import (
            get_inline_param_names,
            make_inline_distribution,
        )

        param_names = get_inline_param_names(draw.morphism)
        if param_names is not None:
            if draw.args is None:
                raise CompileError(
                    f"inline distribution {draw.morphism!r} requires arguments (e.g. {draw.morphism}(...))",
                    draw.line,
                    draw.col,
                )
            inline_codomain = self._infer_inline_codomain(
                draw.morphism, draw.args, draw.vars, program_codomain
            )
            morph, var_args = make_inline_distribution(
                draw.morphism,
                draw.args,
                inline_codomain,
                variable_types={k: v for k, v in bound_vars.items() if v is not None},
            )
            return (morph, var_args)
        registry = _get_family_registry()
        if draw.morphism in registry:
            raise CompileError(
                f"distribution family {draw.morphism!r} is not supported as an inline distribution; declare it as a continuous morphism instead",
                draw.line,
                draw.col,
            )
        raise CompileError(
            f"undefined morphism or distribution family {draw.morphism!r}",
            draw.line,
            draw.col,
        )

    def _infer_inline_codomain(
        self,
        family: str,
        args: tuple,
        var_names: tuple[str, ...],
        program_codomain: object,
    ):
        """Infer the codomain for an inline distribution.

        Parameters
        ----------
        family : str
            Distribution family name.
        args : tuple
            Arguments from the draw step.
        var_names : tuple[str, ...]
            Bound variable name(s).
        program_codomain : object
            The program's declared codomain.

        Returns
        -------
        AnySpace
            The inferred codomain.
        """
        from quivers.continuous.spaces import UnitInterval, Euclidean

        if family == "LogitNormal":
            return UnitInterval(f"_{var_names[0]}")
        elif family == "Bernoulli":
            return FinSet(name=f"_{var_names[0]}", cardinality=2)
        elif family == "Uniform":
            float_args = [a for a in args if isinstance(a, (int, float))]
            if len(float_args) >= 2:
                low, high = (float(float_args[0]), float(float_args[1]))
                if low == 0.0 and high == 1.0:
                    return UnitInterval(f"_{var_names[0]}")
                return Euclidean(name=f"_{var_names[0]}", dim=1, low=low, high=high)
            return UnitInterval(f"_{var_names[0]}")
        elif family == "TruncatedNormal":
            float_args = {
                i: a for i, a in enumerate(args) if isinstance(a, (int, float))
            }
            if 2 in float_args and 3 in float_args:
                low, high = (float(float_args[2]), float(float_args[3]))
                return Euclidean(name=f"_{var_names[0]}", dim=1, low=low, high=high)
            return UnitInterval(f"_{var_names[0]}")
        elif family == "Normal":
            return Euclidean(name=f"_{var_names[0]}", dim=1)
        elif family == "Beta":
            return UnitInterval(f"_{var_names[0]}")
        elif family == "Exponential":
            from quivers.continuous.spaces import PositiveReals

            return PositiveReals(name=f"_{var_names[0]}", dim=1)
        elif family in ("HalfCauchy", "HalfNormal", "LogNormal", "Gamma"):
            from quivers.continuous.spaces import PositiveReals

            return PositiveReals(name=f"_{var_names[0]}", dim=1)
        else:
            return Euclidean(name=f"_{var_names[0]}", dim=1)

    def _validate_let_expr_vars(
        self, node: LetExprNode, bound_vars: dict[str, AnySpace | None], step: LetStep
    ) -> None:
        """Validate that all variables in a let expression are bound.

        Parameters
        ----------
        node : LetExprNode
            Expression tree to validate.
        bound_vars : dict
            Currently bound variables.
        step : LetStep
            The let step (for error reporting).
        """
        if isinstance(node, LetExprVar):
            if node.name not in bound_vars:
                raise CompileError(
                    f"undefined variable {node.name!r} in let expression",
                    step.line,
                    step.col,
                )
        elif isinstance(node, LetExprBinOp):
            self._validate_let_expr_vars(node.left, bound_vars, step)
            self._validate_let_expr_vars(node.right, bound_vars, step)
        elif isinstance(node, LetExprUnaryOp):
            self._validate_let_expr_vars(node.operand, bound_vars, step)
        elif isinstance(node, LetExprCall):
            for arg in node.args:
                self._validate_let_expr_vars(arg, bound_vars, step)

    @staticmethod
    def _compile_let_expr(
        node: LetExprNode,
    ) -> Callable[[dict[str, torch.Tensor]], torch.Tensor]:
        """Compile a let expression tree into a callable.

        The returned callable takes a dict[str, torch.Tensor] (the
        variable environment) and returns a torch.Tensor.

        Parameters
        ----------
        node : LetExprNode
            Expression tree to compile.

        Returns
        -------
        callable
            A function env -> torch.Tensor.
        """
        import torch

        if isinstance(node, LetExprLiteral):
            val = node.value

            def _literal(env: dict) -> torch.Tensor:
                for v in env.values():
                    if isinstance(v, torch.Tensor):
                        return torch.tensor(val, device=v.device)
                return torch.tensor(val)

            return _literal
        if isinstance(node, LetExprVar):
            name = node.name

            def _var(env: dict) -> torch.Tensor:
                return env[name]

            return _var
        if isinstance(node, LetExprBinOp):
            left_fn = Compiler._compile_let_expr(node.left)
            right_fn = Compiler._compile_let_expr(node.right)
            op = node.op

            def _binop(env: dict) -> torch.Tensor:
                l = left_fn(env)
                r = right_fn(env)
                l, r = torch.broadcast_tensors(l, r)
                if op == "+":
                    return l + r
                elif op == "-":
                    return l - r
                elif op == "*":
                    return l * r
                elif op == "/":
                    return l / r
                raise ValueError(f"unknown operator: {op}")

            return _binop
        if isinstance(node, LetExprUnaryOp):
            inner_fn = Compiler._compile_let_expr(node.operand)

            def _neg(env: dict) -> torch.Tensor:
                return -inner_fn(env)

            return _neg
        if isinstance(node, LetExprCall):
            func_name = node.func
            arg_fns = [Compiler._compile_let_expr(a) for a in node.args]

            def _call(env: dict) -> torch.Tensor:
                args = [fn(env) for fn in arg_fns]
                if func_name == "sigmoid":
                    return torch.sigmoid(args[0])
                elif func_name == "exp":
                    return torch.exp(args[0])
                elif func_name == "log":
                    return torch.log(args[0])
                elif func_name == "abs":
                    return torch.abs(args[0])
                elif func_name == "softplus":
                    return torch.nn.functional.softplus(args[0])
                raise ValueError(f"unknown function: {func_name}")

            return _call
        raise CompileError(f"unknown let expression node: {type(node).__name__}")

    def _compile_let(self, decl: LetDecl) -> None:
        """Compile a let-binding with optional where clause."""
        if hasattr(decl, "where") and decl.where:
            for where_decl in decl.where:
                self._compile_let(where_decl)
        if decl.name in self._morphisms:
            raise CompileError(f"name {decl.name!r} already bound", decl.line, decl.col)
        morph = self._compile_expr(decl.expr)
        self._morphisms[decl.name] = morph

    def _compile_output(self, decl: OutputDecl) -> None:
        """Record the output expression."""
        if self._output_expr is not None:
            raise CompileError("multiple output declarations", decl.line, decl.col)
        self._output_expr = decl.expr

    def _resolve_type(self, texpr: TypeExpr, bind_name: str | None = None) -> SetObject:
        """Resolve a type expression into a SetObject.

        Delegates to :class:`~quivers.dsl.resolution.TypeExprToSetObject`,
        a :class:`didactic.api.Lens` parameterised by the current object
        environment. Integer-literal :class:`TypeName` nodes that aren't
        in the environment use ``bind_name`` (falling back to
        ``"_<value>"``) as the synthesised :class:`FinSet` name; this
        thin wrapper is kept so the literal-naming policy stays in
        compiler control.
        """
        from quivers.dsl.resolution import TypeExprToSetObject

        if (
            isinstance(texpr, TypeName)
            and texpr.name.isdigit()
            and texpr.name not in self._objects
            and bind_name is not None
        ):
            return FinSet(name=bind_name, cardinality=int(texpr.name))

        try:
            resolved, _ = TypeExprToSetObject(self._objects).forward(texpr)
        except KeyError as e:
            line = getattr(texpr, "line", 0)
            col = getattr(texpr, "col", 0)
            raise CompileError(str(e).strip("'\""), line, col) from e
        return resolved

    def _resolve_any_space(self, texpr: TypeExpr):
        """Resolve a type expression to either a SetObject or ContinuousSpace.

        Continuous morphism domains/codomains can be either discrete
        objects, continuous spaces, or product types.

        Parameters
        ----------
        texpr : TypeExpr
            The type expression to resolve (TypeName, TypeProduct, etc.).

        Returns
        -------
        SetObject or ContinuousSpace
            The resolved domain/codomain.
        """
        if isinstance(texpr, TypeProduct):
            from quivers.core.objects import ProductSet
            from quivers.continuous.spaces import ContinuousSpace, ProductSpace

            components = [self._resolve_any_space(c) for c in texpr.components]
            if any((isinstance(c, ContinuousSpace) for c in components)):
                return ProductSpace(components=tuple(components))
            return ProductSet(components=tuple(components))
        if not isinstance(texpr, TypeName):
            raise CompileError(
                f"unsupported type expression in domain/codomain: {type(texpr).__name__}",
                getattr(texpr, "line", 0),
                getattr(texpr, "col", 0),
            )
        name = texpr.name
        if name in self._objects:
            return self._objects[name]
        if name in self._spaces:
            return self._spaces[name]
        raise CompileError(f"undefined object or space {name!r}", texpr.line, texpr.col)

    def _resolve_space(self, sexpr: SpaceExpr, bind_name: str | None = None):
        """Resolve a space expression into a ContinuousSpace.

        Delegates to :class:`~quivers.dsl.resolution.SpaceExprToContinuousSpace`,
        a :class:`didactic.api.Lens` parameterised by both the space and
        object environments (a bare identifier may resolve to either).
        """
        from quivers.dsl.resolution import SpaceExprToContinuousSpace

        if isinstance(sexpr, SpaceConstructor):
            constructors = _get_space_constructors()
            cname = sexpr.constructor
            if cname not in constructors:
                raise CompileError(
                    f"unknown space constructor {cname!r}; available: "
                    f"{', '.join(sorted(constructors))}",
                    sexpr.line,
                    sexpr.col,
                )

        try:
            resolved, _ = SpaceExprToContinuousSpace(
                env_spaces=self._spaces,
                env_objects=self._objects,
                name=bind_name or "_anon",
            ).forward(sexpr)
        except (KeyError, ValueError) as e:
            line = getattr(sexpr, "line", 0)
            col = getattr(sexpr, "col", 0)
            raise CompileError(str(e).strip("'\""), line, col) from e
        return resolved

    def _compile_expr(self, expr: Expr):
        """Compile a value expression into a morphism.

        Parameters
        ----------
        expr : Expr
            The expression to compile.

        Returns
        -------
        Morphism or ContinuousMorphism
            The compiled morphism (possibly a DAG of compositions).
        """
        if isinstance(expr, ExprIdent):
            if expr.name not in self._morphisms:
                raise CompileError(
                    f"undefined morphism {expr.name!r}", expr.line, expr.col
                )
            return self._morphisms[expr.name]
        elif isinstance(expr, ExprIdentity):
            if expr.object_name not in self._objects:
                raise CompileError(
                    f"undefined object {expr.object_name!r}", expr.line, expr.col
                )
            obj = self._objects[expr.object_name]
            return make_identity(obj, quantale=self._quantale)
        elif isinstance(expr, ExprCompose):
            left = self._compile_expr(expr.left)
            right = self._compile_expr(expr.right)
            try:
                return left >> right
            except TypeError as e:
                raise CompileError(str(e), expr.line, expr.col) from e
        elif isinstance(expr, ExprTensorProduct):
            left = self._compile_expr(expr.left)
            right = self._compile_expr(expr.right)
            return left @ right
        elif isinstance(expr, ExprMarginalize):
            inner = self._compile_expr(expr.inner)
            sets = []
            for name in expr.names:
                if name not in self._objects:
                    raise CompileError(
                        f"undefined object {name!r} in marginalize", expr.line, expr.col
                    )
                sets.append(self._objects[name])
            try:
                return inner.marginalize(*sets)
            except (TypeError, ValueError) as e:
                raise CompileError(str(e), expr.line, expr.col) from e
        elif isinstance(expr, ExprCurry):
            from quivers.core.morphisms import CurriedMorphism

            inner = self._compile_expr(expr.inner)
            try:
                return CurriedMorphism(inner, direction=expr.direction)
            except (TypeError, ValueError) as e:
                raise CompileError(str(e), expr.line, expr.col) from e
        elif isinstance(expr, ExprChartFold):
            return self._compile_chart_fold(expr)
        elif isinstance(expr, ExprFan):
            from quivers.continuous.morphisms import FanOutMorphism

            components = []
            for sub_expr in expr.exprs:
                if isinstance(sub_expr, ExprIdent) and sub_expr.name in self._groups:
                    for member_name in self._groups[sub_expr.name]:
                        components.append(self._morphisms[member_name])
                else:
                    morph = self._compile_expr(sub_expr)
                    components.append(morph)
            try:
                return FanOutMorphism(components)
            except (TypeError, ValueError) as e:
                raise CompileError(str(e), expr.line, expr.col) from e
        elif isinstance(expr, ExprRepeat):
            morph = self._compile_expr(expr.expr)
            if expr.count is None:
                from quivers.core.morphisms import RepeatMorphism

                try:
                    return RepeatMorphism(morph, n=1)
                except (TypeError, ValueError) as e:
                    raise CompileError(str(e), expr.line, expr.col) from e
            result = morph
            for _ in range(expr.count - 1):
                try:
                    result = result >> morph
                except TypeError as e:
                    raise CompileError(str(e), expr.line, expr.col) from e
            return result
        elif isinstance(expr, ExprStack):
            import copy

            morph = self._compile_expr(expr.expr)
            result = copy.deepcopy(morph)
            for _ in range(expr.count - 1):
                clone = copy.deepcopy(morph)
                try:
                    result = result >> clone
                except TypeError as e:
                    raise CompileError(str(e), expr.line, expr.col) from e
            return result
        elif isinstance(expr, ExprScan):
            from quivers.continuous.scan import ScanMorphism

            cell = self._compile_expr(expr.expr)
            try:
                return ScanMorphism(cell, init=expr.init)
            except TypeError as e:
                raise CompileError(str(e), expr.line, expr.col) from e
        elif isinstance(expr, ExprParser):
            from quivers.stochastic.schema import SCHEMA_REGISTRY

            schemas: list = []
            morphisms: list = []

            def _expand(name: str, seen: frozenset[str]) -> list[str]:
                """Recursively expand a bundle reference into rule names.

                Cycle detection: if ``name`` already appears in ``seen``,
                raises CompileError.
                """
                if name not in self._bundles:
                    return [name]
                if name in seen:
                    raise CompileError(
                        f"bundle cycle through {name!r}",
                        expr.line,
                        expr.col,
                    )
                expanded: list[str] = []
                for member in self._bundles[name]:
                    expanded.extend(_expand(member, seen | {name}))
                return expanded

            resolved_rules: list[str] = []
            for rule_name in expr.rules:
                resolved_rules.extend(_expand(rule_name, frozenset()))

            for rule_name in resolved_rules:
                if rule_name in self._rules:
                    schemas.append(self._rules[rule_name])
                elif (schema_obj := SCHEMA_REGISTRY.get(rule_name)) is not None:
                    schemas.append(schema_obj)
                elif rule_name in self._morphisms:
                    morphisms.append(self._morphisms[rule_name])
                else:
                    raise CompileError(
                        f"unknown rule {rule_name!r}; not a declared rule, schema primitive ({', '.join(sorted(SCHEMA_REGISTRY))}), or a declared morphism",
                        expr.line,
                        expr.col,
                    )
            if schemas and morphisms:
                raise CompileError(
                    "parser() rules must be all schema primitives or all morphism references, not a mix",
                    expr.line,
                    expr.col,
                )
            if morphisms:
                return self._compile_parser_morphisms(morphisms, expr)
            if not schemas:
                raise CompileError(
                    "parser() requires at least one rule", expr.line, expr.col
                )
            return self._compile_parser_schemas(schemas, expr)
        else:
            raise CompileError(f"unknown expression type: {type(expr).__name__}")

    def _compile_parser_morphisms(self, morphisms: list, expr: ExprParser):
        """Compile parser from user-declared morphisms via type inspection.

        Classifies each morphism by its type signature:

        - ``N → N ⊗ N`` (codomain is a product of the domain with
          itself) contributes binary deductions.
        - ``N → T`` (codomain differs from domain) contributes
          lexical axioms.

        Parameters
        ----------
        morphisms : list
            Compiled morphism objects.
        expr : ExprParser
            The AST node (for error reporting).
        """
        from quivers.core.objects import ProductSet
        from quivers.stochastic.inside import InsideAlgorithm

        binary = None
        lexical = None
        for morph in morphisms:
            cod = morph.codomain
            if (
                isinstance(cod, ProductSet)
                and len(cod.components) == 2
                and all((c == morph.domain for c in cod.components))
            ):
                if binary is not None:
                    raise CompileError(
                        "parser() received multiple binary morphisms (codomain = domain ⊗ domain); expected one",
                        expr.line,
                        expr.col,
                    )
                binary = morph
            else:
                if lexical is not None:
                    raise CompileError(
                        "parser() received multiple lexical morphisms; expected one",
                        expr.line,
                        expr.col,
                    )
                lexical = morph
        if binary is None:
            raise CompileError(
                "parser() requires a binary morphism (type N → N ⊗ N) among its rules",
                expr.line,
                expr.col,
            )
        if lexical is None:
            raise CompileError(
                "parser() requires a lexical morphism (type N → T) among its rules",
                expr.line,
                expr.col,
            )
        try:
            start = expr.start if isinstance(expr.start, int) else 0
            return InsideAlgorithm(binary, lexical, start=start)
        except TypeError as e:
            raise CompileError(str(e), expr.line, expr.col) from e

    def _compile_chart_fold(self, expr):
        """Compile a chart_fold(...) primitive expression.

        chart_fold is the explicit form of which the legacy
        parser(rules=...) is sugar. Given a lexical morphism
        ``lex : Token -> Cat`` plus a binary morphism (and optional
        unary morphism) on Cat, it constructs an InsideAlgorithm-based
        chart parser. The user-visible structure of the parser is
        therefore expressible from primitives — no opaque parser()
        call required.

        Effect-typed chart cells (``effect_depth`` > 0) extend the
        category universe to ``Cat × EffectStack_{≤d}`` via the
        class-driven lifting machinery in
        :mod:`quivers.stochastic.effect_lifts`; the caller is expected
        to have constructed ``binary`` (and any ``unary``) over this
        enlarged universe, typically via
        :func:`quivers.stochastic.effect_lifts.lift_rule_set` over the
        declared :class:`EffectDecl` instances in scope. The
        ``effect_depth`` integer flows through to the parser as the
        depth bound used for any depth-truncating reductions over
        intermediate cells.

        Handler firings (``handlers=`` argument) are applied as a
        post-composition step on the parser's denotation: the final
        chart cell is routed through each handler's :meth:`run`
        morphism in declared order, reducing the effect stack as the
        handlers compose.
        """
        from quivers.stochastic.inside import InsideAlgorithm

        lex = self._compile_expr(expr.lex)
        if expr.binary is None:
            raise CompileError(
                "chart_fold(...) requires a binary= argument (a morphism "
                "Cat * Cat -> Cat representing the union of binary rule "
                "schemas)",
                expr.line,
                expr.col,
            )
        binary = self._compile_expr(expr.binary)

        unary = self._compile_expr(expr.unary) if expr.unary is not None else None

        handlers_morphisms: list = []
        for h_expr in getattr(expr, "handlers", ()) or ():
            handlers_morphisms.append(self._compile_expr(h_expr))

        try:
            start = expr.start if isinstance(expr.start, int) else 0
            parser = InsideAlgorithm(
                binary, lex, start=start, unary=unary
            )
        except (TypeError, ValueError) as e:
            raise CompileError(str(e), expr.line, expr.col) from e

        # Compose handlers as post-applications on the parser's output.
        # Each handler is a morphism Cat → Cat (or a more refined effect
        # reduction); composition is right-to-left in declaration order.
        result = parser
        for handler in handlers_morphisms:
            result = _ChartHandlerComposite(result, handler)
        return result

    def _compile_parser_schemas(self, schemas: list, expr: ExprParser):
        """Compile parser from schema functors over a category system.

        Parameters
        ----------
        schemas : list
            Schema objects from ``SCHEMA_REGISTRY``.
        expr : ExprParser
            The AST node.
        """
        from quivers.stochastic.categories import CategorySystem
        from quivers.stochastic.parsers import ChartParser

        from quivers.core.objects import FreeResiduated

        if expr.categories:
            categories = list(expr.categories)
        elif self._categories:
            categories = list(self._categories)
        else:
            # Look for a FreeResiduated object in scope and use its
            # generators' atom names. If exactly one residuated universe
            # is declared, this avoids the user having to spell out
            # `categories=[NP, S, VP, ...]` redundantly.
            residuated = [
                obj for obj in self._objects.values() if isinstance(obj, FreeResiduated)
            ]
            if len(residuated) == 1:
                categories = list(residuated[0].generators.elements)
            elif len(residuated) > 1:
                raise CompileError(
                    "parser() with schema rules: multiple FreeResiduated "
                    "objects in scope; pass categories=[...] explicitly to "
                    "select the atom set",
                    expr.line,
                    expr.col,
                )
            else:
                raise CompileError(
                    "parser() with schema rules requires category atoms — declare them via `object Atoms = {NP, S, VP, ...}` plus `object Cat = FreeResiduated(Atoms, ...)`, or pass categories=[NP, S, VP, ...] inline",
                    expr.line,
                    expr.col,
                )
        if expr.constructors is not None:
            cs = CategorySystem.from_generators(
                atoms=categories,
                constructors=list(expr.constructors),
                max_depth=expr.depth,
            )
        else:
            cs = CategorySystem.from_atoms_and_slash_depth(
                categories, max_depth=expr.depth
            )
        schema = schemas[0]
        for piece in schemas[1:]:
            schema = schema | piece
        if expr.terminal is None:
            raise CompileError(
                "parser() with schema rules requires terminal=<object> — the declared object serving as the terminal vocabulary",
                expr.line,
                expr.col,
            )
        if expr.terminal not in self._objects:
            raise CompileError(
                f"terminal={expr.terminal!r} does not refer to a declared object",
                expr.line,
                expr.col,
            )
        n_term = self._objects[expr.terminal].size
        try:
            start = expr.start if isinstance(expr.start, str) else "S"
            return ChartParser.from_schema(schema, cs, n_terminals=n_term, start=start)
        except (TypeError, ValueError) as e:
            raise CompileError(str(e), expr.line, expr.col) from e
