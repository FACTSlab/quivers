"""Compiler: transform a quivers DSL AST into a trainable Program.

The compiler walks the AST in declaration order, building up an
environment of objects, spaces, and morphisms, then compiles the
output expression into a quivers.Program (nn.Module).

Supports both discrete (FinSet-based) and continuous (ContinuousSpace-
based) morphisms, including stochastic (Markov kernels), boundary
(Discretize/Embed), and parameterized family distributions.
"""

from __future__ import annotations

from quivers.core.objects import SetObject, FinSet, ProductSet, CoproductSet
from quivers.core.quantales import (
    Quantale,
    PRODUCT_FUZZY,
    BOOLEAN,
)
from quivers.core.morphisms import (
    morphism as make_latent,
    identity as make_identity,
)
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
    DiscretizeDecl,
    EmbedDecl,
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
    TypeCoproduct,
    SpaceExpr,
    SpaceConstructor,
    SpaceProduct,
    Expr,
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


# quantale name -> singleton lookup
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


# distribution family name -> class lookup (lazily populated)
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
        # canonical names (PascalCase, matching class suffix)
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
        # discrete-valued
        "Bernoulli": ConditionalBernoulli,
        "Categorical": ConditionalCategorical,
    }

    # try adding GeneralizedPareto (optional)
    try:
        from quivers.continuous.families import ConditionalGeneralizedPareto

        _FAMILY_REGISTRY["GeneralizedPareto"] = ConditionalGeneralizedPareto

    except (ImportError, AttributeError):
        pass

    return _FAMILY_REGISTRY


# space constructor name -> factory (lazily populated)
_SPACE_CONSTRUCTORS: dict[str, type] | None = None


def _get_space_constructors() -> dict[str, type]:
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
        self._categories: list[str] = []  # category atom names
        self._rules: dict = {}  # name -> RuleSchema (from rule declarations)
        self._objects: dict[str, SetObject] = {}
        self._spaces: dict = {}  # name -> ContinuousSpace
        self._morphisms: dict = {}  # name -> Morphism or ContinuousMorphism
        self._groups: dict[str, list[str]] = {}  # base_name -> [name_0, name_1, ...]
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

    # -- statement compilation -----------------------------------------------

    def _compile_statement(self, stmt: Statement) -> None:
        """Dispatch to the appropriate statement compiler."""
        if isinstance(stmt, QuantaleDecl):
            self._compile_quantale(stmt)

        elif isinstance(stmt, CategoryDecl):
            self._compile_category(stmt)

        elif isinstance(stmt, RuleDecl):
            self._compile_rule(stmt)

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
                f"unknown quantale {decl.name!r}; "
                f"available: {', '.join(sorted(_QUANTALE_REGISTRY))}",
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
                f"category {decl.name!r} already declared",
                decl.line,
                decl.col,
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
                f"rule {decl.name!r} already declared",
                decl.line,
                decl.col,
            )

        if decl.name in SCHEMA_REGISTRY:
            raise CompileError(
                f"rule {decl.name!r} shadows a built-in schema; "
                f"choose a different name",
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
                f"rule {decl.name!r} has {n_premises} premises; "
                f"only unary (1) and binary (2) rules are supported",
                decl.line,
                decl.col,
            )

        self._rules[decl.name] = schema

    def _compile_object(self, decl: ObjectDecl) -> None:
        """Compile an object declaration into the environment."""
        if decl.name in self._objects:
            raise CompileError(
                f"object {decl.name!r} already declared",
                decl.line,
                decl.col,
            )

        obj = self._resolve_type(decl.type_expr, decl.name)
        self._objects[decl.name] = obj

    def _compile_morphism(self, decl: MorphismDecl) -> None:
        """Compile a morphism declaration into the environment."""
        if decl.name in self._morphisms:
            raise CompileError(
                f"morphism {decl.name!r} already declared",
                decl.line,
                decl.col,
            )

        domain = self._resolve_type(decl.domain)
        codomain = self._resolve_type(decl.codomain)

        if decl.kind == "latent":
            scale = float(decl.options.get("scale", "0.5"))
            morph = make_latent(
                domain, codomain, init_scale=scale, quantale=self._quantale
            )

        elif decl.kind == "observed":
            if decl.init_expr is not None:
                morph = self._compile_expr(decl.init_expr)

                # verify domain/codomain match
                if morph.domain != domain or morph.codomain != codomain:
                    raise CompileError(
                        f"morphism {decl.name!r} init expression has "
                        f"type {morph.domain!r} -> {morph.codomain!r}, "
                        f"expected {domain!r} -> {codomain!r}",
                        decl.line,
                        decl.col,
                    )

            else:
                raise CompileError(
                    f"observed morphism {decl.name!r} requires "
                    f"an initializer (e.g. = identity({decl.domain}))",
                    decl.line,
                    decl.col,
                )

        else:
            raise CompileError(
                f"unknown morphism kind {decl.kind!r}",
                decl.line,
                decl.col,
            )

        self._morphisms[decl.name] = morph

    def _compile_space(self, decl: SpaceDecl) -> None:
        """Compile a space declaration into the space environment."""
        if decl.name in self._spaces:
            raise CompileError(
                f"space {decl.name!r} already declared",
                decl.line,
                decl.col,
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
                f"morphism {decl.name!r} already declared",
                decl.line,
                decl.col,
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
                domain,
                codomain,
                decl.family,
                decl.options,
                decl,
            )
            self._morphisms[name] = morph

        if decl.replicate is not None:
            self._groups[decl.name] = names

    def _make_continuous_morphism(
        self,
        domain,
        codomain,
        family_name: str,
        options: dict[str, str],
        decl,
    ):
        """Create a single continuous morphism from a family name."""
        # handle Flow specially
        if family_name == "Flow":
            from quivers.continuous.flows import ConditionalFlow

            n_layers = int(options.get("n_layers", "4"))
            hidden_dim = int(options.get("hidden_dim", "64"))
            return ConditionalFlow(
                domain,
                codomain,
                n_layers=n_layers,
                hidden_dim=hidden_dim,
            )

        # look up family
        registry = _get_family_registry()

        if family_name not in registry:
            raise CompileError(
                f"unknown distribution family {family_name!r}; "
                f"available: {', '.join(sorted(registry))}",
                decl.line,
                decl.col,
            )

        cls = registry[family_name]
        hidden_dim = int(options.get("hidden_dim", "64"))

        # some families have extra constructor args
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
                f"morphism {decl.name!r} already declared",
                decl.line,
                decl.col,
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
                f"morphism {decl.name!r} already declared",
                decl.line,
                decl.col,
            )

        if decl.space_name not in self._spaces:
            raise CompileError(
                f"undefined space {decl.space_name!r}",
                decl.line,
                decl.col,
            )

        from quivers.continuous.boundaries import Discretize

        space = self._spaces[decl.space_name]
        morph = Discretize(space, n_bins=decl.n_bins)
        self._morphisms[decl.name] = morph

    def _compile_embed(self, decl: EmbedDecl) -> None:
        """Compile an embed boundary morphism."""
        if decl.name in self._morphisms:
            raise CompileError(
                f"morphism {decl.name!r} already declared",
                decl.line,
                decl.col,
            )

        if decl.domain_name not in self._objects:
            raise CompileError(
                f"undefined object {decl.domain_name!r}",
                decl.line,
                decl.col,
            )

        if decl.codomain_name not in self._spaces:
            raise CompileError(
                f"undefined space {decl.codomain_name!r}",
                decl.line,
                decl.col,
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
            morph = Embed(domain, codomain)
            self._morphisms[name] = morph

        if decl.replicate is not None:
            self._groups[decl.name] = names

    def _compile_program(self, decl: ProgramDecl) -> None:
        """Compile a monadic program block into a MonadicProgram."""
        if decl.name in self._morphisms:
            raise CompileError(
                f"morphism {decl.name!r} already declared",
                decl.line,
                decl.col,
            )

        from quivers.continuous.programs import MonadicProgram

        domain = self._resolve_any_space(decl.domain)
        codomain = self._resolve_any_space(decl.codomain)

        # if named params are given, validate count matches product domain
        from quivers.continuous.spaces import ProductSpace as _PS

        if decl.params is not None:
            if isinstance(domain, (ProductSet, _PS)):
                if len(decl.params) != len(domain.components):
                    raise CompileError(
                        f"program has {len(decl.params)} params but domain "
                        f"has {len(domain.components)} components",
                        decl.line,
                        decl.col,
                    )

            elif len(decl.params) != 1:
                raise CompileError(
                    f"program has {len(decl.params)} params but domain "
                    f"is not a product type",
                    decl.line,
                    decl.col,
                )

        # resolve each draw step
        # track the codomain of each bound variable for type checking
        # named params are also bound variables
        bound_vars: dict[str, object] = {}

        if decl.params is not None:
            if isinstance(domain, (ProductSet, _PS)):
                for pname, factor in zip(decl.params, domain.components):
                    bound_vars[pname] = factor

            else:
                bound_vars[decl.params[0]] = domain

        steps: list[tuple] = []

        for step in decl.draws:
            if isinstance(step, LetStep):
                # deterministic let binding
                if step.name in bound_vars:
                    raise CompileError(
                        f"variable {step.name!r} already bound in program",
                        step.line,
                        step.col,
                    )

                if isinstance(step.value, str):
                    # simple variable alias
                    if step.value not in bound_vars:
                        raise CompileError(
                            f"undefined variable {step.value!r} in let binding",
                            step.line,
                            step.col,
                        )

                    bound_vars[step.name] = bound_vars[step.value]
                    steps.append(((step.name,), None, step.value))

                elif isinstance(step.value, (int, float)):
                    # numeric literal
                    bound_vars[step.name] = None
                    steps.append(((step.name,), None, step.value))

                else:
                    # expression tree: compile to callable
                    self._validate_let_expr_vars(step.value, bound_vars, step)
                    compiled_fn = self._compile_let_expr(step.value)
                    bound_vars[step.name] = None
                    steps.append(((step.name,), None, compiled_fn))

                continue

            # draw step
            draw = step

            # check for duplicate variable names
            for v in draw.vars:
                if v in bound_vars:
                    raise CompileError(
                        f"variable {v!r} already bound in program",
                        draw.line,
                        draw.col,
                    )

            morph, step_args = self._resolve_draw_morphism(
                draw,
                bound_vars,
                codomain,
            )

            # validate string arguments (variable references)
            if step_args is not None:
                for arg_name in step_args:
                    if arg_name not in bound_vars:
                        raise CompileError(
                            f"undefined variable {arg_name!r} in draw step",
                            draw.line,
                            draw.col,
                        )

            # record the variable codomain(s)
            if len(draw.vars) == 1:
                bound_vars[draw.vars[0]] = morph.codomain

            else:
                # destructuring: morphism must return a product or
                # be a tuple-returning MonadicProgram
                if isinstance(morph, MonadicProgram) and not morph._return_is_single:
                    # the sub-program returns a dict; bind each var
                    if len(draw.vars) != len(morph._return_vars):
                        raise CompileError(
                            f"destructuring {len(draw.vars)} vars but "
                            f"sub-program returns {len(morph._return_vars)}",
                            draw.line,
                            draw.col,
                        )

                    for v in draw.vars:
                        bound_vars[v] = None  # type info not tracked for dict returns

                elif isinstance(morph.codomain, ProductSet):
                    if len(draw.vars) != len(morph.codomain.components):
                        raise CompileError(
                            f"destructuring {len(draw.vars)} vars but "
                            f"codomain has {len(morph.codomain.components)} components",
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

        # validate return variables
        for rv in decl.return_vars:
            if rv not in bound_vars:
                raise CompileError(
                    f"return variable {rv!r} not bound in program",
                    decl.line,
                    decl.col,
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
        bound_vars: dict[str, object],
        program_codomain: object,
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
        # first check if it's an existing named morphism
        if draw.morphism in self._morphisms:
            morph = self._morphisms[draw.morphism]

            # validate args are all variable names (no float literals
            # for named morphisms)
            if draw.args is not None:
                for a in draw.args:
                    if isinstance(a, (int, float)):
                        raise CompileError(
                            f"literal argument {a} not allowed for "
                            f"named morphism {draw.morphism!r}",
                            draw.line,
                            draw.col,
                        )

            # args are all strings (or None)
            step_args = (
                tuple(str(a) for a in draw.args) if draw.args is not None else None
            )
            return morph, step_args

        # check if it's an inline distribution family
        from quivers.continuous.inline import (
            get_inline_param_names,
            make_inline_distribution,
        )

        param_names = get_inline_param_names(draw.morphism)

        if param_names is not None:
            # it's an inline family
            if draw.args is None:
                raise CompileError(
                    f"inline distribution {draw.morphism!r} requires "
                    f"arguments (e.g. {draw.morphism}(...))",
                    draw.line,
                    draw.col,
                )

            # determine codomain for the inline distribution
            inline_codomain = self._infer_inline_codomain(
                draw.morphism,
                draw.args,
                draw.vars,
                program_codomain,
            )

            # build the morphism
            morph, var_args = make_inline_distribution(
                draw.morphism,
                draw.args,
                inline_codomain,
                variable_types={k: v for k, v in bound_vars.items() if v is not None},
            )
            return morph, var_args

        # also check the family registry (for families not in
        # inline specs but that could still be used inline)
        registry = _get_family_registry()

        if draw.morphism in registry:
            raise CompileError(
                f"distribution family {draw.morphism!r} is not supported "
                f"as an inline distribution; declare it as a continuous "
                f"morphism instead",
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
            return FinSet(f"_{var_names[0]}", 2)

        elif family == "Uniform":
            # extract bounds from float args
            float_args = [a for a in args if isinstance(a, (int, float))]

            if len(float_args) >= 2:
                low, high = float(float_args[0]), float(float_args[1])

                if low == 0.0 and high == 1.0:
                    return UnitInterval(f"_{var_names[0]}")

                return Euclidean(
                    f"_{var_names[0]}",
                    1,
                    low=low,
                    high=high,
                )

            return UnitInterval(f"_{var_names[0]}")

        elif family == "TruncatedNormal":
            # extract bounds from last two float args
            float_args = {
                i: a for i, a in enumerate(args) if isinstance(a, (int, float))
            }

            if 2 in float_args and 3 in float_args:
                low, high = float(float_args[2]), float(float_args[3])
                return Euclidean(
                    f"_{var_names[0]}",
                    1,
                    low=low,
                    high=high,
                )

            return UnitInterval(f"_{var_names[0]}")

        elif family == "Normal":
            return Euclidean(f"_{var_names[0]}", 1)

        elif family == "Beta":
            return UnitInterval(f"_{var_names[0]}")

        elif family == "Exponential":
            from quivers.continuous.spaces import PositiveReals

            return PositiveReals(f"_{var_names[0]}", 1)

        elif family in ("HalfCauchy", "HalfNormal", "LogNormal", "Gamma"):
            from quivers.continuous.spaces import PositiveReals

            return PositiveReals(f"_{var_names[0]}", 1)

        else:
            return Euclidean(f"_{var_names[0]}", 1)

    def _validate_let_expr_vars(
        self,
        node: LetExprNode,
        bound_vars: dict[str, object],
        step: LetStep,
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
    def _compile_let_expr(node: LetExprNode) -> callable:
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
                # return a 0-dim scalar tensor that broadcasts with any shape
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

                # broadcast shapes so (batch,) op (batch, dim) works
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
        # handle where clause bindings first (they define names used in main expr)
        if hasattr(decl, "where") and decl.where:
            for where_decl in decl.where:
                self._compile_let(where_decl)

        if decl.name in self._morphisms:
            raise CompileError(
                f"name {decl.name!r} already bound",
                decl.line,
                decl.col,
            )

        morph = self._compile_expr(decl.expr)
        self._morphisms[decl.name] = morph

    def _compile_output(self, decl: OutputDecl) -> None:
        """Record the output expression."""
        if self._output_expr is not None:
            raise CompileError(
                "multiple output declarations",
                decl.line,
                decl.col,
            )

        self._output_expr = decl.expr

    # -- type resolution -----------------------------------------------------

    def _resolve_type(self, texpr: TypeExpr, bind_name: str | None = None) -> SetObject:
        """Resolve a type expression into a SetObject.

        Parameters
        ----------
        texpr : TypeExpr
            The type expression to resolve.
        bind_name : str or None
            If provided, used as the FinSet name for integer literals.

        Returns
        -------
        SetObject
            The resolved object.
        """
        if isinstance(texpr, TypeName):
            # integer literal -> FinSet
            if texpr.name.isdigit():
                name = bind_name if bind_name else f"_{texpr.name}"
                return FinSet(name, int(texpr.name))

            # lookup in object environment
            if texpr.name in self._objects:
                return self._objects[texpr.name]

            raise CompileError(
                f"undefined object {texpr.name!r}",
                texpr.line,
                texpr.col,
            )

        elif isinstance(texpr, TypeProduct):
            components = [self._resolve_type(c) for c in texpr.components]
            return ProductSet(*components)

        elif isinstance(texpr, TypeCoproduct):
            components = [self._resolve_type(c) for c in texpr.components]
            return CoproductSet(*components)

        else:
            raise CompileError(f"unknown type expression: {type(texpr).__name__}")

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

            # if any component is a continuous space, build ProductSpace
            if any(isinstance(c, ContinuousSpace) for c in components):
                return ProductSpace(*components)

            return ProductSet(*components)

        if not isinstance(texpr, TypeName):
            raise CompileError(
                f"unsupported type expression in domain/codomain: "
                f"{type(texpr).__name__}",
                getattr(texpr, "line", 0),
                getattr(texpr, "col", 0),
            )

        name = texpr.name

        # check objects first
        if name in self._objects:
            return self._objects[name]

        # check spaces
        if name in self._spaces:
            return self._spaces[name]

        raise CompileError(
            f"undefined object or space {name!r}",
            texpr.line,
            texpr.col,
        )

    def _resolve_space(self, sexpr: SpaceExpr, bind_name: str | None = None):
        """Resolve a space expression into a ContinuousSpace.

        Parameters
        ----------
        sexpr : SpaceExpr
            The space expression.
        bind_name : str or None
            Used as the space name for constructor calls.

        Returns
        -------
        ContinuousSpace
            The resolved space.
        """
        constructors = _get_space_constructors()

        if isinstance(sexpr, SpaceConstructor):
            cname = sexpr.constructor

            if cname not in constructors:
                raise CompileError(
                    f"unknown space constructor {cname!r}; "
                    f"available: {', '.join(sorted(constructors))}",
                    sexpr.line,
                    sexpr.col,
                )

            cls = constructors[cname]
            name = bind_name or cname

            # parse constructor arguments
            if cname == "UnitInterval":
                # UnitInterval(name?) -- no dim needed
                return cls(name)

            elif cname in ("Euclidean", "Simplex", "PositiveReals"):
                # first positional arg is dim
                if not sexpr.args:
                    raise CompileError(
                        f"{cname} requires a dimension argument",
                        sexpr.line,
                        sexpr.col,
                    )

                dim = int(sexpr.args[0])
                kwargs = {}

                for k, v in sexpr.kwargs.items():
                    try:
                        kwargs[k] = float(v)

                    except ValueError:
                        kwargs[k] = v

                return cls(name, dim, **kwargs)

            else:
                raise CompileError(
                    f"unsupported space constructor {cname!r}",
                    sexpr.line,
                    sexpr.col,
                )

        elif isinstance(sexpr, TypeName):
            # reference to previously declared space
            if sexpr.name in self._spaces:
                return self._spaces[sexpr.name]

            raise CompileError(
                f"undefined space {sexpr.name!r}",
                sexpr.line,
                sexpr.col,
            )

        elif isinstance(sexpr, SpaceProduct):
            from quivers.continuous.spaces import ProductSpace

            components = [self._resolve_space(c) for c in sexpr.components]
            return ProductSpace(*components)

        else:
            raise CompileError(f"unknown space expression: {type(sexpr).__name__}")

    # -- expression compilation ----------------------------------------------

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
                    f"undefined morphism {expr.name!r}",
                    expr.line,
                    expr.col,
                )

            return self._morphisms[expr.name]

        elif isinstance(expr, ExprIdentity):
            if expr.object_name not in self._objects:
                raise CompileError(
                    f"undefined object {expr.object_name!r}",
                    expr.line,
                    expr.col,
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
                        f"undefined object {name!r} in marginalize",
                        expr.line,
                        expr.col,
                    )

                sets.append(self._objects[name])

            try:
                return inner.marginalize(*sets)

            except (TypeError, ValueError) as e:
                raise CompileError(str(e), expr.line, expr.col) from e

        elif isinstance(expr, ExprFan):
            from quivers.continuous.morphisms import FanOutMorphism

            components = []

            for sub_expr in expr.exprs:
                # if a bare identifier refers to a group, expand it
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
                # runtime-variable repeat: create RepeatMorphism
                from quivers.core.morphisms import RepeatMorphism

                try:
                    return RepeatMorphism(morph, n=1)

                except (TypeError, ValueError) as e:
                    raise CompileError(str(e), expr.line, expr.col) from e

            # static unrolling: compile-time composition chain
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

            # partition rules into schemas and morphism references
            schemas: list = []
            morphisms: list = []

            for rule_name in expr.rules:
                # check DSL-declared rules first, then built-in schemas
                if rule_name in self._rules:
                    schemas.append(self._rules[rule_name])

                elif (schema_obj := SCHEMA_REGISTRY.get(rule_name)) is not None:
                    schemas.append(schema_obj)

                elif rule_name in self._morphisms:
                    morphisms.append(self._morphisms[rule_name])

                else:
                    raise CompileError(
                        f"unknown rule {rule_name!r}; not a declared rule, "
                        f"schema primitive ({', '.join(sorted(SCHEMA_REGISTRY))}), "
                        f"or a declared morphism",
                        expr.line,
                        expr.col,
                    )

            if schemas and morphisms:
                raise CompileError(
                    "parser() rules must be all schema primitives "
                    "or all morphism references, not a mix",
                    expr.line,
                    expr.col,
                )

            # morphism mode: type-driven dispatch
            if morphisms:
                return self._compile_parser_morphisms(
                    morphisms,
                    expr,
                )

            # schema mode: categories + rule schemas → ChartParser
            if not schemas:
                raise CompileError(
                    "parser() requires at least one rule",
                    expr.line,
                    expr.col,
                )

            return self._compile_parser_schemas(
                schemas,
                expr,
            )

        else:
            raise CompileError(f"unknown expression type: {type(expr).__name__}")

    def _compile_parser_morphisms(
        self,
        morphisms: list,
        expr: ExprParser,
    ):
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

            # N → N ⊗ N: codomain is a product whose components
            # are all equal to the domain
            if (
                isinstance(cod, ProductSet)
                and len(cod.components) == 2
                and all(c == morph.domain for c in cod.components)
            ):
                if binary is not None:
                    raise CompileError(
                        "parser() received multiple binary morphisms "
                        "(codomain = domain ⊗ domain); expected one",
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
            return InsideAlgorithm(
                binary,
                lexical,
                start=expr.start,
            )

        except TypeError as e:
            raise CompileError(
                str(e),
                expr.line,
                expr.col,
            ) from e

    def _compile_parser_schemas(
        self,
        schemas: list,
        expr: ExprParser,
    ):
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

        # resolve categories: explicit parameter or declared via `category`
        if expr.categories:
            categories = list(expr.categories)

        elif self._categories:
            categories = list(self._categories)

        else:
            raise CompileError(
                "parser() with schema rules requires category atoms — "
                "either declare them with `category S`, `category NP`, ... "
                "or pass categories=[S, NP, ...] inline",
                expr.line,
                expr.col,
            )

        # build category system from constructors + depth
        if expr.constructors is not None:
            cs = CategorySystem.from_generators(
                atoms=categories,
                constructors=list(expr.constructors),
                max_depth=expr.depth,
            )

        else:
            cs = CategorySystem.from_atoms_and_slash_depth(
                categories,
                max_depth=expr.depth,
            )

        # compose schemas via union
        schema = schemas[0]

        for piece in schemas[1:]:
            schema = schema | piece

        # resolve terminal vocabulary from explicit parameter
        if expr.terminal is None:
            raise CompileError(
                "parser() with schema rules requires terminal=<object> — "
                "the declared object serving as the terminal vocabulary",
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
            return ChartParser.from_schema(
                schema,
                cs,
                n_terminals=n_term,
                start=expr.start,
            )

        except (TypeError, ValueError) as e:
            raise CompileError(str(e), expr.line, expr.col) from e
