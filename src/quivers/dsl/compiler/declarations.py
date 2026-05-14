"""Compiler mixin: declaration-level statements.

Handles quantale, category, rule, schema, alias, bundle, object,
morphism, space, kernel, discretize, and embed declarations.
"""
from __future__ import annotations
from collections.abc import Callable
import torch
from quivers.core.objects import FinSet
from quivers.core.quantales import (
    BilinearForm,
    CompositionRule,
    CustomBilinearForm,
    CustomQuantale,
    CustomSemigroupoid,
    Quantale,
    Semigroupoid,
)
from quivers.core.morphisms import morphism as make_latent
from quivers.stochastic import StochasticMorphism
from quivers.dsl.ast_nodes import (
    CategoryDecl,
    CompositionRuleEntry,
    DiscretizeDecl,
    EmbedDecl,
    EnumSetLiteral,
    FreeMonoidExpr,
    FreeResiduatedExpr,
    KernelDecl,
    MorphismDecl,
    ObjectDecl,
    QuantaleDecl,
    RuleDecl,
    SchemaDecl,
    SpaceDecl,
    TypeProduct,
)
from quivers.dsl.compiler._prelude import (
    CompileError,
    _QUANTALE_REGISTRY,
    _available_axes_for,
    _get_family_registry,
    _shape_size,
    _validate_axis_spec,
    _wrap_join_dim,
)
from quivers.dsl.compiler.programs import _ProgramsMixin


class _DeclarationsMixin:
    """Mixin: declaration-level compilation methods."""

    def _compile_quantale(self, decl: QuantaleDecl) -> None:
        """Set the active composition rule for this module.

        Four surface forms (distinguished by ``decl.declared_level``):

        * ``quantale X`` — X must be a Quantale.
        * ``semigroupoid X`` — X must be a Semigroupoid.
        * ``bilinear_form X`` — X must be a BilinearForm.
        * ``composition_rule X`` — X must be any CompositionRule.

        Two body forms:

        * **No body** — ``X`` names a registered rule in the
          composition-rule registry; the compiler verifies it
          matches the declared level.
        * **With body** — ``decl.body`` is a list of entries
          defining the rule's operations from scratch. The
          compiler evaluates each entry's expression and builds
          a fresh ``CustomQuantale``, ``CustomSemigroupoid``, or
          ``CustomBilinearForm`` of the declared level, then
          registers it under the supplied name.
        """
        level = decl.declared_level
        if decl.body:
            rule = self._build_custom_composition_rule(decl)
        else:
            name = decl.name.lower()
            if name not in _QUANTALE_REGISTRY:
                raise CompileError(
                    f"unknown {level} {decl.name!r}; available: "
                    f"{', '.join(sorted(_QUANTALE_REGISTRY))}",
                    decl.line,
                    decl.col,
                )
            rule = _QUANTALE_REGISTRY[name]
        self._verify_composition_rule_level(rule, decl, level)
        self._quantale = rule  # type: ignore[assignment]

    def _verify_composition_rule_level(
        self,
        rule: "CompositionRule",
        decl: QuantaleDecl,
        level: str,
    ) -> None:
        """Confirm ``rule`` satisfies the algebraic level declared
        by the keyword."""
        required = {
            "quantale": Quantale,
            "semigroupoid": Semigroupoid,
            "bilinear_form": BilinearForm,
            "composition_rule": CompositionRule,
        }
        required_class = required.get(level, CompositionRule)
        if not isinstance(rule, required_class):
            actual = (
                "Quantale"
                if isinstance(rule, Quantale)
                else "Semigroupoid"
                if isinstance(rule, Semigroupoid)
                else "BilinearForm"
                if isinstance(rule, BilinearForm)
                else "CompositionRule"
            )
            raise CompileError(
                f"{level} {decl.name!r}: registered rule is a "
                f"{actual}, which is not at level {level!r}. "
                f"Either declare it with the matching keyword or "
                f"register a rule at the right level.",
                decl.line,
                decl.col,
            )

    def _build_custom_composition_rule(self, decl: QuantaleDecl) -> "CompositionRule":
        """Build a fresh CompositionRule instance from a
        ``quantale name { … }`` body, dispatching on the declared
        level."""
        entries: dict[str, "CompositionRuleEntry"] = {}
        for entry in decl.body:
            if entry.key in entries:
                raise CompileError(
                    f"{decl.declared_level} {decl.name!r}: duplicate "
                    f"entry {entry.key!r}",
                    entry.line,
                    entry.col,
                )
            entries[entry.key] = entry
        level = decl.declared_level
        required_keys = {
            "quantale": {"tensor_op", "join", "unit", "zero"},
            "semigroupoid": {"tensor_op", "join"},
            "bilinear_form": {"tensor_op", "join"},
            "composition_rule": {"tensor_op", "join"},
        }[level]
        missing = required_keys - set(entries)
        if missing:
            raise CompileError(
                f"{level} {decl.name!r}: missing required entries {sorted(missing)}",
                decl.line,
                decl.col,
            )
        # Compile each entry to a Python callable (function-valued
        # entry like ``tensor_op(a, b) = …``) or a numeric value
        # (literal-valued entry like ``unit = 1.0``). The union of
        # callable and float is the natural type here; we keep
        # them separate downstream so the level builders type-check
        # each slot precisely.
        compiled: dict[str, "Callable[..., torch.Tensor] | float"] = {}
        for key, entry in entries.items():
            compiled[key] = self._compile_composition_rule_entry(entry, decl)
        # Build the concrete rule at the declared level.
        tensor_op = compiled["tensor_op"]
        join = compiled["join"]
        if not callable(tensor_op):
            raise CompileError(
                f"{level} {decl.name!r}: ``tensor_op`` must be a "
                f"function entry like ``tensor_op(a, b) = …``",
                decl.line,
                decl.col,
            )
        if not callable(join):
            raise CompileError(
                f"{level} {decl.name!r}: ``join`` must be a "
                f"function entry like ``join(t) = …``",
                decl.line,
                decl.col,
            )
        join_wrapped = _wrap_join_dim(join)
        if level == "quantale":
            unit = compiled["unit"]
            zero = compiled["zero"]
            if callable(unit) or callable(zero):
                raise CompileError(
                    f"quantale {decl.name!r}: ``unit`` and ``zero`` "
                    f"must be value entries (no parens)",
                    decl.line,
                    decl.col,
                )
            negate_fn = compiled.get("negation")
            if negate_fn is not None and not callable(negate_fn):
                raise CompileError(
                    f"quantale {decl.name!r}: ``negation`` must be "
                    f"a function entry like ``negation(a) = …``",
                    decl.line,
                    decl.col,
                )
            return CustomQuantale(
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
        # ``bilinear_form`` and ``composition_rule`` both build a
        # CustomBilinearForm: ``bilinear_form`` is the weakest
        # named level (no associativity promise), and
        # ``composition_rule`` is the permissive surface that
        # admits any rule, for which BilinearForm is the right
        # default since it makes no claim beyond the
        # CompositionRule interface.
        return CustomBilinearForm(
            name=decl.name,
            tensor_op=tensor_op,
            join=join_wrapped,
        )

    def _compile_composition_rule_entry(
        self,
        entry: "CompositionRuleEntry",
        decl: QuantaleDecl,
    ) -> "Callable[..., torch.Tensor] | float":
        """Compile one ``key(params) = body`` or ``key = body``
        entry to a Python callable (when ``entry.params`` is
        non-empty) or to its evaluated numeric value (when
        ``entry.params`` is empty)."""
        body_fn = _ProgramsMixin._compile_let_expr(entry.body)
        if not entry.params:
            try:
                return body_fn({})
            except Exception as exc:
                raise CompileError(
                    f"{decl.declared_level} {decl.name!r}: value "
                    f"entry {entry.key!r} could not be evaluated: {exc}",
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

        Three surface forms are recognized:

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
            f"unrecognized object initializer for {decl.name!r}",
            decl.line,
            decl.col,
        )

    def _compile_morphism(self, decl: MorphismDecl) -> None:
        """Compile a morphism declaration into the environment."""
        if decl.name in self._morphisms:
            raise CompileError(
                f"morphism {decl.name!r} already declared", decl.line, decl.col
            )
        if decl.prior is not None:
            if decl.morphism_kind != "latent":
                raise CompileError(
                    f"morphism prior `~ Family(...)` is legal only on "
                    f"`latent` declarations; got {decl.morphism_kind!r}",
                    decl.line, decl.col,
                )
            if decl.prior.axes is not None:
                _validate_axis_spec(
                    decl.prior.axes,
                    decl.prior.family,
                    _available_axes_for(decl.domain, decl.codomain),
                    decl.line, decl.col,
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
                # The init expression's domain/codomain may be
                # anonymous (e.g. ``from_data(...)`` synthesizes
                # them from the tensor shape). Accept a shape
                # match and rebind to the user-declared types so
                # downstream code sees the correct named objects.
                #
                # Compatibility is checked at the storage level
                # rather than the type level: a flat init tensor
                # whose total numel matches the declared product
                # codomain's numel is accepted, then reshaped to
                # the declared factored shape. This is the
                # categorical view that ``B = B1 * B2 * ... * Bk``
                # and ``B'`` of cardinality ``|B1| * ... * |Bk|``
                # are isomorphic objects; the tensor storage is
                # the same up to reshape.
                if morph.domain != domain or morph.codomain != codomain:

                    def _numel(shape):
                        n = 1
                        for s in shape:
                            n *= int(s)
                        return n

                    init_d = _numel(morph.domain.shape)
                    init_c = _numel(morph.codomain.shape)
                    decl_d = _numel(domain.shape)
                    decl_c = _numel(codomain.shape)
                    if init_d == decl_d and init_c == decl_c:
                        from quivers.core.morphisms import (
                            ObservedMorphism as _Obs,
                        )

                        # Reshape the tensor to match the declared
                        # factored shape. ``Tensor.reshape`` is a
                        # no-op when the storage already matches.
                        target_shape = tuple(domain.shape) + tuple(codomain.shape)
                        reshaped = morph.tensor.reshape(target_shape)
                        morph = _Obs(
                            domain,
                            codomain,
                            reshaped,
                            quantale=morph.quantale,
                        )
                    else:
                        raise CompileError(
                            f"morphism {decl.name!r} init expression has "
                            f"type {morph.domain!r} -> {morph.codomain!r} "
                            f"(numel {init_d} -> {init_c}), expected "
                            f"{domain!r} -> {codomain!r} "
                            f"(numel {decl_d} -> {decl_c})",
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

    def _compile_kernel(self, decl: KernelDecl) -> None:
        """Compile a Markov-kernel declaration ``kernel f : A -> B [~ F ...]``.

        Without a ``~`` clause the declaration is a lookup-table
        kernel on finite sets, realised as a
        :class:`quivers.stochastic.StochasticMorphism`.  With a ``~``
        clause it is a parametric kernel whose family parameters are
        produced from the input by a parameter network at sample
        time; the ``decl.axes`` clause configures the family's
        event/batch decomposition over codomain factors.  Replicate
        counts produce N independent copies named ``name_0`` through
        ``name_{N-1}`` with the base name registered as a group.
        """
        if decl.name in self._morphisms:
            raise CompileError(
                f"morphism {decl.name!r} already declared", decl.line, decl.col
            )
        count = decl.replicate if decl.replicate is not None else 1
        names = (
            [f"{decl.name}_{i}" for i in range(count)]
            if decl.replicate is not None
            else [decl.name]
        )
        if decl.axes is not None:
            if decl.family is None:
                raise CompileError(
                    "axis-role clause requires a `~ Family` clause on the "
                    "kernel declaration; lookup-table kernels do not carry "
                    "a parametric family.",
                    decl.line, decl.col,
                )
            _validate_axis_spec(
                decl.axes,
                decl.family,
                _available_axes_for(decl.domain, decl.codomain),
                decl.line, decl.col,
            )
        if decl.family is None:
            domain = self._resolve_type(decl.domain)
            codomain = self._resolve_type(decl.codomain)
            for name in names:
                self._morphisms[name] = StochasticMorphism(domain, codomain)
        else:
            domain = self._resolve_any_space(decl.domain)
            codomain = self._resolve_any_space(decl.codomain)
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
        # For event_rank-2 matrix families (MatrixNormal) the
        # constructor needs explicit row/column dims, which come
        # from the axis-role clause's named factors.  When the
        # user wrote ``~ MatrixNormal over (X, Y)``, X resolves
        # to the rows axis and Y to the cols axis (positional
        # ordering corresponds positionally to the family's
        # declared event-axis ordering: rows first, cols second).
        axes = getattr(decl, "axes", None)
        if family_name == "MatrixNormal" and axes is not None:
            if len(axes.over) != 2:
                raise CompileError(
                    f"MatrixNormal requires `over (rows_axis, cols_axis)`; "
                    f"got over={axes.over!r}",
                    decl.line, decl.col,
                )
            rows_axis, cols_axis = axes.over
            kwargs["rows"] = self._axis_dim(decl, rows_axis)
            kwargs["cols"] = self._axis_dim(decl, cols_axis)
        return cls(domain, codomain, **kwargs)

    def _axis_dim(self, decl, axis_name: str) -> int:
        """Resolve an axis name to its dimension.

        ``axis_name`` is either a declared factor name of the
        morphism's dom or cod, or the reserved shortcut ``dom`` /
        ``cod``.  Returns the cardinality / dim of the resolved
        object or space.
        """
        # The dom/cod shortcuts resolve to the morphism's dom/cod
        # objects directly.
        if axis_name == "dom":
            obj = self._resolve_any_space(decl.domain)
            return _shape_size(obj)
        if axis_name == "cod":
            obj = self._resolve_any_space(decl.codomain)
            return _shape_size(obj)
        # Otherwise the axis is a named factor of dom or cod.  For
        # an unfactored side whose argument carries the same name
        # (``Euclidean(D)`` with object D in scope) the axis name
        # is the object's name; resolve it directly.
        if axis_name in self._objects:
            return int(self._objects[axis_name].cardinality)
        if axis_name in self._spaces:
            return _shape_size(self._spaces[axis_name])
        raise CompileError(
            f"axis-role clause: cannot resolve axis name {axis_name!r} to a "
            f"dimension; not a declared object/space, and not a `dom`/`cod` "
            f"shortcut", decl.line, decl.col,
        )

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
