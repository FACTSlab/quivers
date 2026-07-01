"""Compiler mixin: declaration-level statements.

Handles the consolidated Statement family: composition, category,
rule, schema, bundle, type, morphism. The mixin dispatches on the
`TypeInitializer` variant inside a `ObjectDecl` and on
the ``role`` option inside a `MorphismDecl` to pick the
runtime construction.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

from quivers.analysis.init_spec import _algebra_init_spec
from quivers.core.algebras import (
    Algebra,
    BilinearForm,
    BooleanAlgebra,
    CompositionRule,
    CustomAlgebra,
    CustomBilinearForm,
    CustomSemigroupoid,
    GodelAlgebra,
    LukasiewiczAlgebra,
    ProbabilityAlgebra,
    ProductFuzzyAlgebra,
    Semigroupoid,
)
from quivers.continuous.boundaries import Discretize, Embed
from quivers.continuous.flows import ConditionalFlow
from quivers.continuous.spaces import ContinuousSpace
from quivers.core.morphisms import (
    ObservedMorphism,
    morphism as make_latent,
)
from quivers.core.objects import (
    EnumSet,
    FinSet,
    FreeMonoid,
    FreeResiduated,
    SetObject,
)
from quivers.dsl.ast_nodes import (
    AxisSpec,
    BundleDecl,
    CategoryDecl,
    CompositionDecl,
    CompositionRuleEntry,
    ExprIdent,
    MorphismDecl,
    RuleDecl,
    SchemaDecl,
    ObjectDecl,
    TypeEnumSet,
    ObjectExpr,
    TypeFreeMonoid,
    TypeFreeResiduated,
    TypeFromExpr,
    ObjectProduct,
)
from quivers.dsl.compiler._options import (
    get_option_float,
    get_option_int,
    get_option_name,
    get_option_name_list,
    get_option_string,
)
from quivers.dsl.compiler._prelude import (
    _ALGEBRA_REGISTRY,
    CompileError,
    _available_axes_for,
    _get_family_registry,
    _shape_size,
    _validate_axis_spec,
    _wrap_join_dim,
)
from quivers.dsl.compiler.programs import _ProgramsMixin
from quivers.stochastic import StochasticMorphism
from quivers.stochastic.schema import (
    SCHEMA_REGISTRY,
    PatternBinarySchema,
    PatternUnarySchema,
)


_VALID_ROLES: frozenset[str] = frozenset(
    {"latent", "observed", "kernel", "embed", "discretize", "let"}
)


def _apply_auto_init(morph, domain, codomain, algebra) -> None:
    """Apply the algebra's saturation-free init recipe to a freshly
    constructed `LatentMorphism`.

    The recipe is computed at depth 1 (a top-level latent declaration
    is, in isolation, a one-step morphism; downstream composition is
    out of scope for the static recipe) with the larger of the
    morphism's resolved domain / codomain numel as the intermediate
    axis size. The recipe is in value space; the raw parameter feeds
    through `LatentMorphism`'s sigmoid bijector, so for the
    sigmoid case we invert via ``logit`` before sampling; for
    algebras whose latent representation does not pass through a
    bijector (Markov, log-prob, real, max-plus, tropical) the recipe
    is applied to the raw parameter directly.
    """

    def _numel(obj) -> int:
        shape = getattr(obj, "shape", ())
        n = 1
        for s in shape:
            n *= int(s)
        return max(1, n)

    intermediate_size = max(_numel(domain), _numel(codomain))
    spec = _algebra_init_spec(
        algebra,
        depth=1,
        intermediate_size=intermediate_size,
    )
    raw = morph.raw
    is_sigmoid_bijected = isinstance(
        algebra,
        (
            ProductFuzzyAlgebra,
            BooleanAlgebra,
            GodelAlgebra,
            LukasiewiczAlgebra,
            ProbabilityAlgebra,
        ),
    )

    def _logit(p: float) -> float:
        p = max(min(p, 1.0 - 1e-6), 1e-6)
        return math.log(p / (1.0 - p))

    with torch.no_grad():
        if spec.distribution == "constant":
            target = _logit(spec.mean) if is_sigmoid_bijected else spec.mean
            raw.data.fill_(target)
        elif spec.distribution == "uniform":
            if is_sigmoid_bijected:
                lo = _logit(spec.lower)
                hi = _logit(spec.upper)
            else:
                lo = spec.lower
                hi = spec.upper
            raw.data.uniform_(lo, hi)
        else:
            mean = _logit(spec.mean) if is_sigmoid_bijected else spec.mean
            raw.data.normal_(mean=mean, std=max(spec.std, 1e-6))


class _DeclarationsMixin:
    """Mixin: declaration-level compilation methods.

    The compiler base supplies every environment slot below; the
    annotations let the type checker verify each access from a
    mixin method.
    """

    _algebra: CompositionRule
    _categories: list[str]
    _rules: dict
    _bundles: dict[str, tuple[str, ...]]
    _aliases: dict[str, ObjectExpr]
    _alias_names: set[str]
    _objects: dict[str, SetObject]
    _spaces: dict[str, ContinuousSpace]
    _morphisms: dict
    _groups: dict[str, list[str]]

    # ``_resolve_type``, ``_resolve_any_space``, ``_compile_expr``
    # come from `_ResolutionMixin` and
    # `_ExpressionsMixin` via the ``Compiler`` MRO.

    # ------------------------------------------------------------------
    # composition
    # ------------------------------------------------------------------

    def _compile_composition(self, decl: CompositionDecl) -> None:
        """Install ``decl`` as the active composition rule for this module.

        ``decl.level`` advertises the algebraic level at which the rule
        is offered (``algebra`` / ``semigroupoid`` / ``bilinear_form``
        / ``rule``). Two body forms:

        * empty body: ``decl.name`` references a rule in the registry;
          the compiler confirms its concrete type matches the
          declared level.
        * non-empty body: ``decl.body`` defines the rule's operations
          inline; the compiler synthesises a ``CustomAlgebra`` /
          ``CustomSemigroupoid`` / ``CustomBilinearForm`` of the
          declared level.
        """
        level = decl.level or "rule"
        if decl.body:
            rule = self._build_custom_composition_rule(decl, level)
        else:
            key = decl.name.lower()
            if key not in _ALGEBRA_REGISTRY:
                raise CompileError(
                    f"unknown {level} {decl.name!r}; available: "
                    f"{', '.join(sorted(_ALGEBRA_REGISTRY))}",
                    decl.line,
                    decl.col,
                )
            rule = _ALGEBRA_REGISTRY[key]
        self._verify_composition_rule_level(rule, decl, level)
        self._algebra = rule  # type: ignore[assignment]

    def _verify_composition_rule_level(
        self,
        rule: "CompositionRule",
        decl: CompositionDecl,
        level: str,
    ) -> None:
        required = {
            "algebra": Algebra,
            "semigroupoid": Semigroupoid,
            "bilinear_form": BilinearForm,
            "rule": CompositionRule,
        }
        required_class = required.get(level, CompositionRule)
        if not isinstance(rule, required_class):
            actual = (
                "Algebra"
                if isinstance(rule, Algebra)
                else "Semigroupoid"
                if isinstance(rule, Semigroupoid)
                else "BilinearForm"
                if isinstance(rule, BilinearForm)
                else "CompositionRule"
            )
            raise CompileError(
                f"composition {decl.name!r}: registered rule is a "
                f"{actual}, which is not at level {level!r}. Declare "
                f"with the matching `at <level>` clause or register a "
                f"rule at the right level.",
                decl.line,
                decl.col,
            )

    def _build_custom_composition_rule(
        self,
        decl: CompositionDecl,
        level: str,
    ) -> "CompositionRule":
        entries: dict[str, CompositionRuleEntry] = {}
        for entry in decl.body:
            if entry.key in entries:
                raise CompileError(
                    f"composition {decl.name!r}: duplicate entry {entry.key!r}",
                    entry.line,
                    entry.col,
                )
            entries[entry.key] = entry
        required_keys = {
            "algebra": {"tensor_op", "join", "unit", "zero"},
            "semigroupoid": {"tensor_op", "join"},
            "bilinear_form": {"tensor_op", "join"},
            "rule": {"tensor_op", "join"},
        }[level]
        missing = required_keys - set(entries)
        if missing:
            raise CompileError(
                f"composition {decl.name!r}: missing required entries "
                f"{sorted(missing)}",
                decl.line,
                decl.col,
            )
        compiled: dict[str, "Callable[..., torch.Tensor] | float"] = {}
        for key, entry in entries.items():
            compiled[key] = self._compile_composition_rule_entry(entry, decl)
        tensor_op = compiled["tensor_op"]
        join = compiled["join"]
        if not callable(tensor_op):
            raise CompileError(
                f"composition {decl.name!r}: ``tensor_op`` must be a "
                f"function entry like ``tensor_op(a, b) = …``",
                decl.line,
                decl.col,
            )
        if not callable(join):
            raise CompileError(
                f"composition {decl.name!r}: ``join`` must be a "
                f"function entry like ``join(t) = …``",
                decl.line,
                decl.col,
            )
        join_wrapped = _wrap_join_dim(join)
        if level == "algebra":
            unit = compiled["unit"]
            zero = compiled["zero"]
            if callable(unit) or callable(zero):
                raise CompileError(
                    f"composition {decl.name!r}: ``unit`` and ``zero`` "
                    f"must be value entries (no parens)",
                    decl.line,
                    decl.col,
                )
            negate_fn = compiled.get("negation")
            if negate_fn is not None and not callable(negate_fn):
                raise CompileError(
                    f"composition {decl.name!r}: ``negation`` must be "
                    f"a function entry like ``negation(a) = …``",
                    decl.line,
                    decl.col,
                )
            return CustomAlgebra(
                name=decl.name,
                tensor_op=tensor_op,
                join=join_wrapped,
                unit=float(unit),
                zero=float(zero),
                negate=negate_fn,
                verify=False,
            )
        if level == "semigroupoid":
            return CustomSemigroupoid(
                name=decl.name,
                tensor_op=tensor_op,
                join=join_wrapped,
                verify_associative=False,
            )
        return CustomBilinearForm(
            name=decl.name,
            tensor_op=tensor_op,
            join=join_wrapped,
        )

    def _compile_composition_rule_entry(
        self,
        entry: CompositionRuleEntry,
        decl: CompositionDecl,
    ) -> "Callable[..., torch.Tensor] | float":
        body_fn = _ProgramsMixin._compile_let_expr(entry.body)
        if not entry.params:
            try:
                return body_fn({})
            except Exception as exc:
                raise CompileError(
                    f"composition {decl.name!r}: value entry "
                    f"{entry.key!r} could not be evaluated: {exc}",
                    entry.line,
                    entry.col,
                ) from exc
        param_names = entry.params

        def _callable(*args):
            if len(args) != len(param_names):
                raise TypeError(
                    f"{entry.key}(): expected {len(param_names)} "
                    f"arguments, got {len(args)}"
                )
            env = dict(zip(param_names, args))
            return body_fn(env)

        return _callable

    # ------------------------------------------------------------------
    # category
    # ------------------------------------------------------------------

    def _compile_category(self, decl: CategoryDecl) -> None:
        """Register one or more category-atom generators.

        ``decl.names`` is a tuple to accommodate the comma-separated
        surface form ``category A, B, C``; each name is appended to
        the running category list.
        """
        for name in decl.names:
            if name in self._categories:
                raise CompileError(
                    f"category {name!r} already declared",
                    decl.line,
                    decl.col,
                )
            self._categories.append(name)

    # ------------------------------------------------------------------
    # rule / schema
    # ------------------------------------------------------------------

    def _compile_rule(self, decl: RuleDecl) -> None:
        """Compile an inference rule into a pattern schema."""
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
                f"rule {decl.name!r} has {n_premises} premises; only "
                f"unary (1) and binary (2) rules are supported",
                decl.line,
                decl.col,
            )
        self._rules[decl.name] = schema

    def _compile_schema(self, decl: SchemaDecl) -> None:
        """Compile a pattern-polymorphic schema declaration."""
        if decl.name in self._rules:
            raise CompileError(
                f"schema {decl.name!r} already declared",
                decl.line,
                decl.col,
            )
        if decl.name in SCHEMA_REGISTRY:
            raise CompileError(
                f"schema {decl.name!r} shadows a built-in schema; "
                f"choose a different name",
                decl.line,
                decl.col,
            )
        variables: frozenset[str] = frozenset(
            n for group in decl.parameters for n in group.names
        )
        if isinstance(decl.domain, ObjectProduct) and len(decl.domain.components) == 2:
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

    # ------------------------------------------------------------------
    # bundle
    # ------------------------------------------------------------------

    def _compile_bundle(self, decl: BundleDecl) -> None:
        """Register a rule bundle for later expansion at use site."""
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
        self._bundles[decl.name] = tuple(decl.rules)

    # ------------------------------------------------------------------
    # type
    # ------------------------------------------------------------------

    def _compile_type(self, decl: ObjectDecl) -> None:
        """Compile a ``type NAME : VALUE`` declaration.

        The init's tagged-union variant picks the construction:

        * `TypeEnumSet` -> `EnumSet`
        * `TypeFreeResiduated` -> `FreeResiduated`
        * `TypeFreeMonoid` -> `FreeMonoid`
        * `TypeFromExpr` -> the inner type expression is
          resolved via the unified resolver; the resulting object is
          either a `SetObject` (discrete) or a
          `ContinuousSpace`, bound under ``decl.name`` in the
          appropriate environment.
        """
        if decl.name in self._objects or decl.name in self._spaces:
            raise CompileError(
                f"type {decl.name!r} already declared",
                decl.line,
                decl.col,
            )
        init = decl.init
        if isinstance(init, TypeEnumSet):
            self._objects[decl.name] = EnumSet(
                name=decl.name,
                elements=init.elements,
            )
            return
        if isinstance(init, TypeFreeMonoid):
            gen = self._objects.get(init.generators)
            if not isinstance(gen, FinSet):
                raise CompileError(
                    f"FreeMonoid generators {init.generators!r} must "
                    f"reference a previously-declared FinSet (got "
                    f"{type(gen).__name__ if gen else 'undefined'})",
                    decl.line,
                    decl.col,
                )
            self._objects[decl.name] = FreeMonoid(
                generators=gen,
                max_length=init.max_length,
            )
            return
        if isinstance(init, TypeFreeResiduated):
            gen = self._objects.get(init.generators)
            if not isinstance(gen, EnumSet):
                raise CompileError(
                    f"FreeResiduated generators {init.generators!r} "
                    f"must reference a previously-declared EnumSet "
                    f"(got "
                    f"{type(gen).__name__ if gen else 'undefined'})",
                    decl.line,
                    decl.col,
                )
            self._objects[decl.name] = FreeResiduated(
                generators=gen,
                depth=init.depth,
                ops=init.ops,
            )
            return
        if isinstance(init, TypeFromExpr):
            expr = init.expr
            try:
                resolved = self._resolve_any_space(expr)
            except CompileError:
                # Residuated patterns and effect-typed RHS do not
                # resolve to a concrete object/space; record the
                # alias for use-site substitution inside schema
                # patterns.
                if decl.name in self._alias_names:
                    raise CompileError(
                        f"alias {decl.name!r} already declared",
                        decl.line,
                        decl.col,
                    )
                self._alias_names.add(decl.name)
                self._aliases[decl.name] = expr
                return
            if isinstance(resolved, ContinuousSpace):
                self._spaces[decl.name] = resolved
            else:
                self._objects[decl.name] = resolved
            return
        raise CompileError(
            f"unrecognized type initializer for {decl.name!r}: {type(init).__name__}",
            decl.line,
            decl.col,
        )

    # ------------------------------------------------------------------
    # morphism
    # ------------------------------------------------------------------

    def _compile_morphism(self, decl: MorphismDecl) -> None:
        """Compile a ``morphism NAME : DOM -> COD [options] [~ init]``.

        Required option ``role`` picks the runtime construction:

        * ``role=latent``  : learnable algebraic morphism on finite
          sets, optionally re-initialised by the algebra's auto
          recipe when ``init=auto`` is present.
        * ``role=observed``: fixed structural morphism whose tensor
          comes from the ``~ <expr>`` initializer; the expression's
          domain/codomain may be anonymous and rebinds to the
          declared types when their numel matches.
        * ``role=kernel``  : Markov kernel. Without ``~ Family``, a
          lookup-table `StochasticMorphism` on finite sets;
          with ``~ Family(...)``, a parametric continuous kernel.
        * ``role=embed``   : `Embed` boundary, finite-set to
          continuous space.
        * ``role=discretize``: `Discretize` boundary,
          continuous space to finite set; ``[bins=N]`` is required.
        * ``role=let``     : deterministic morphism whose value is
          ``~ <expr>`` (composition pipeline, contraction call,
          transformation invocation, etc.).
        """
        if decl.name in self._morphisms:
            raise CompileError(
                f"morphism {decl.name!r} already declared",
                decl.line,
                decl.col,
            )
        role = get_option_name(
            decl.options,
            "role",
            line=decl.line,
            col=decl.col,
        )
        if role is None:
            raise CompileError(
                f"morphism {decl.name!r}: required option ``role`` is "
                f"missing; expected one of "
                f"{sorted(_VALID_ROLES)}",
                decl.line,
                decl.col,
            )
        if role not in _VALID_ROLES:
            raise CompileError(
                f"morphism {decl.name!r}: unknown role {role!r}; "
                f"expected one of {sorted(_VALID_ROLES)}",
                decl.line,
                decl.col,
            )
        replicate = get_option_int(
            decl.options,
            "replicate",
            line=decl.line,
            col=decl.col,
        )
        count = 1 if replicate is None else int(replicate)
        names = (
            [f"{decl.name}_{i}" for i in range(count)]
            if replicate is not None
            else [decl.name]
        )
        if role == "latent":
            self._compile_latent_role(decl, names)
        elif role == "observed":
            self._compile_observed_role(decl, names)
        elif role == "kernel":
            self._compile_kernel_role(decl, names)
        elif role == "embed":
            self._compile_embed_role(decl, names)
        elif role == "discretize":
            self._compile_discretize_role(decl, names)
        else:
            self._compile_let_role(decl, names)
        if replicate is not None:
            self._groups[decl.name] = names

    # role-specific lowerings ------------------------------------------

    def _compile_latent_role(
        self,
        decl: MorphismDecl,
        names: list[str],
    ) -> None:
        if decl.init_expr is not None:
            raise CompileError(
                f"latent morphism {decl.name!r}: ``~ <expression>`` "
                f"init is reserved for ``role=let`` and ``role="
                f"observed``; latent priors take a ``~ Family(...)`` "
                f"form instead",
                decl.line,
                decl.col,
            )
        if decl.init_family is not None:
            self._validate_family_axes(decl, decl.init_family.family)
        domain = self._resolve_type(decl.domain)
        codomain = self._resolve_type(decl.codomain)
        scale = get_option_float(
            decl.options,
            "scale",
            line=decl.line,
            col=decl.col,
            default=0.5,
        )
        init_mode = get_option_name(
            decl.options,
            "init",
            line=decl.line,
            col=decl.col,
        )
        for name in names:
            morph = make_latent(
                domain,
                codomain,
                init_scale=float(scale),
                algebra=self._algebra,
            )
            if init_mode == "auto":
                _apply_auto_init(morph, domain, codomain, self._algebra)
            self._morphisms[name] = morph

    def _compile_observed_role(
        self,
        decl: MorphismDecl,
        names: list[str],
    ) -> None:
        if decl.init_expr is None and decl.init_family is None:
            raise CompileError(
                f"observed morphism {decl.name!r} requires an "
                f"initializer (e.g. ``~ identity({decl.domain})``)",
                decl.line,
                decl.col,
            )
        if decl.init_family is not None:
            raise CompileError(
                f"observed morphism {decl.name!r}: ``~ Family(...)`` "
                f"is a stochastic-kernel prior; use ``role=kernel`` "
                f"or supply a deterministic ``~ <expression>`` "
                f"initializer instead",
                decl.line,
                decl.col,
            )
        domain = self._resolve_type(decl.domain)
        codomain = self._resolve_type(decl.codomain)
        for name in names:
            morph = self._compile_expr(decl.init_expr)
            morph = self._coerce_observed_shape(morph, domain, codomain, decl)
            self._morphisms[name] = morph

    def _coerce_observed_shape(
        self,
        morph,
        domain,
        codomain,
        decl: MorphismDecl,
    ):
        """Rebind ``morph``'s declared domain/codomain when its
        underlying tensor's numel matches the declared types.

        Categorically, ``B = B1 * ... * Bk`` and a flat ``B'`` whose
        cardinality is ``|B1| * ... * |Bk|`` are isomorphic objects;
        the storage is identical up to reshape. The compiler accepts
        an init expression whose anonymous shape matches that
        invariant and rebinds the morphism to the declared factored
        shape.
        """
        if morph.domain == domain and morph.codomain == codomain:
            return morph

        def _numel(shape) -> int:
            n = 1
            for s in shape:
                n *= int(s)
            return n

        init_d = _numel(morph.domain.shape)
        init_c = _numel(morph.codomain.shape)
        decl_d = _numel(domain.shape)
        decl_c = _numel(codomain.shape)
        if init_d == decl_d and init_c == decl_c:
            target_shape = tuple(domain.shape) + tuple(codomain.shape)
            reshaped = morph.tensor.reshape(target_shape)
            return ObservedMorphism(
                domain,
                codomain,
                reshaped,
                algebra=morph.algebra,
            )
        raise CompileError(
            f"morphism {decl.name!r} init expression has type "
            f"{morph.domain!r} -> {morph.codomain!r} (numel {init_d} "
            f"-> {init_c}), expected {domain!r} -> {codomain!r} "
            f"(numel {decl_d} -> {decl_c})",
            decl.line,
            decl.col,
        )

    def _compile_kernel_role(
        self,
        decl: MorphismDecl,
        names: list[str],
    ) -> None:
        family = decl.init_family.family if decl.init_family is not None else None
        # ``~ Normal`` (bare identifier, no parens) parses as
        # ``init_expr`` rather than ``init_family``. If the bare
        # initializer names a registered family, promote it.
        if (
            family is None
            and decl.init_expr is not None
            and isinstance(decl.init_expr, ExprIdent)
            and decl.init_expr.name in _get_family_registry()
        ):
            family = decl.init_expr.name
        if family is not None:
            self._validate_family_axes(decl, family)
        if family is None:
            domain = self._resolve_type(decl.domain)
            codomain = self._resolve_type(decl.codomain)
            for name in names:
                self._morphisms[name] = StochasticMorphism(domain, codomain)
            return
        domain = self._resolve_any_space(decl.domain)
        codomain = self._resolve_any_space(decl.codomain)
        for name in names:
            morph = self._make_continuous_morphism(
                domain,
                codomain,
                family,
                decl,
            )
            self._morphisms[name] = morph

    def _compile_embed_role(
        self,
        decl: MorphismDecl,
        names: list[str],
    ) -> None:
        domain = self._resolve_type(decl.domain)
        if not isinstance(domain, FinSet):
            raise CompileError(
                f"embed morphism {decl.name!r}: domain must be a "
                f"FinSet, got {type(domain).__name__}",
                decl.line,
                decl.col,
            )
        codomain = self._resolve_any_space(decl.codomain)
        if not isinstance(codomain, ContinuousSpace):
            raise CompileError(
                f"embed morphism {decl.name!r}: codomain must be a "
                f"ContinuousSpace, got {type(codomain).__name__}",
                decl.line,
                decl.col,
            )
        for name in names:
            self._morphisms[name] = Embed(domain, codomain)

    def _compile_discretize_role(
        self,
        decl: MorphismDecl,
        names: list[str],
    ) -> None:
        space = self._resolve_any_space(decl.domain)
        if not isinstance(space, ContinuousSpace):
            raise CompileError(
                f"discretize morphism {decl.name!r}: domain must be a "
                f"ContinuousSpace, got {type(space).__name__}",
                decl.line,
                decl.col,
            )
        bins = get_option_int(
            decl.options,
            "bins",
            line=decl.line,
            col=decl.col,
        )
        if bins is None:
            raise CompileError(
                f"discretize morphism {decl.name!r}: required option "
                f"``bins`` is missing",
                decl.line,
                decl.col,
            )
        for name in names:
            self._morphisms[name] = Discretize(space, n_bins=bins)

    def _compile_let_role(
        self,
        decl: MorphismDecl,
        names: list[str],
    ) -> None:
        if decl.init_expr is None:
            raise CompileError(
                f"let morphism {decl.name!r}: ``role=let`` requires an "
                f"``~ <expression>`` initializer",
                decl.line,
                decl.col,
            )
        morph = self._compile_expr(decl.init_expr)
        for name in names:
            self._morphisms[name] = morph

    # family-construction plumbing ------------------------------------

    def _validate_family_axes(
        self,
        decl: MorphismDecl,
        family: str,
    ) -> None:
        """Check the option block's ``over``/``iid`` axis lists against
        the family's declared event/batch decomposition."""
        over = get_option_name_list(
            decl.options,
            "over",
            line=decl.line,
            col=decl.col,
        )
        iid = get_option_name_list(
            decl.options,
            "iid",
            line=decl.line,
            col=decl.col,
        )
        if not over and not iid:
            return
        axes_spec = AxisSpec(
            over=over,
            iid_over=iid,
            line=decl.line,
            col=decl.col,
        )
        _validate_axis_spec(
            axes_spec,
            family,
            _available_axes_for(decl.domain, decl.codomain),
            decl.line,
            decl.col,
        )

    def _make_continuous_morphism(
        self,
        domain,
        codomain,
        family_name: str,
        decl: MorphismDecl,
    ):
        if family_name == "Flow":
            n_layers = get_option_int(
                decl.options,
                "n_layers",
                line=decl.line,
                col=decl.col,
                default=4,
            )
            hidden_dim = get_option_int(
                decl.options,
                "hidden_dim",
                line=decl.line,
                col=decl.col,
                default=64,
            )
            return ConditionalFlow(
                domain,
                codomain,
                n_layers=int(n_layers),
                hidden_dim=int(hidden_dim),
            )
        registry = _get_family_registry()
        if family_name not in registry:
            raise CompileError(
                f"unknown distribution family {family_name!r}; "
                f"available: {', '.join(sorted(registry))}",
                decl.line,
                decl.col,
            )
        cls = registry[family_name]
        hidden_dim = get_option_int(
            decl.options,
            "hidden_dim",
            line=decl.line,
            col=decl.col,
            default=64,
        )
        kwargs: dict = {"hidden_dim": int(hidden_dim)}
        # Optional `[param_source=<kind>[(...)]]` DSL surface for
        # picking the parameter-source architecture (linear, MLP,
        # attention, identity). The default MLP with `hidden_dim`
        # matches the pre-abstraction behaviour. The kwarg is
        # threaded through to the conditional family's `__init__`,
        # which uses `param_source_from_option` internally to build
        # the concrete `ParamSource` once `param_dim` is knowable.
        param_source_opt = get_option_name(
            decl.options,
            "param_source",
            line=decl.line,
            col=decl.col,
        )
        if param_source_opt is None:
            param_source_opt = get_option_string(
                decl.options,
                "param_source",
                line=decl.line,
                col=decl.col,
            )
        if param_source_opt is not None:
            kwargs["param_source_option"] = param_source_opt
        rank = get_option_int(
            decl.options,
            "rank",
            line=decl.line,
            col=decl.col,
        )
        if rank is not None:
            kwargs["rank"] = int(rank)
        temperature = get_option_float(
            decl.options,
            "temperature",
            line=decl.line,
            col=decl.col,
        )
        if temperature is not None:
            kwargs["temperature"] = float(temperature)
        over = get_option_name_list(
            decl.options,
            "over",
            line=decl.line,
            col=decl.col,
        )
        if family_name == "MatrixNormal" and over:
            if len(over) != 2:
                raise CompileError(
                    f"MatrixNormal requires ``over=[rows_axis, "
                    f"cols_axis]``; got over={list(over)!r}",
                    decl.line,
                    decl.col,
                )
            rows_axis, cols_axis = over
            kwargs["rows"] = self._axis_dim(decl, rows_axis)
            kwargs["cols"] = self._axis_dim(decl, cols_axis)
        return cls(domain, codomain, **kwargs)

    def _axis_dim(self, decl: MorphismDecl, axis_name: str) -> int:
        if axis_name == "dom":
            return _shape_size(self._resolve_any_space(decl.domain))
        if axis_name == "cod":
            return _shape_size(self._resolve_any_space(decl.codomain))
        if axis_name in self._objects:
            return int(self._objects[axis_name].cardinality)
        if axis_name in self._spaces:
            return _shape_size(self._spaces[axis_name])
        raise CompileError(
            f"axis {axis_name!r}: not a declared object/space and "
            f"not a ``dom``/``cod`` shortcut",
            decl.line,
            decl.col,
        )


__all__ = ["_DeclarationsMixin", "_apply_auto_init"]
