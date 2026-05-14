"""Compiler mixin: expression compilation.

Handles expression compilation, transformation expressions,
composition, parsers, chart folds, exports, and data binding.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from quivers.core.morphisms import identity as make_identity
from quivers.core.morphisms import cap as _make_cap, cup as _make_cup
from quivers.core.quantales import (
    BilinearForm,
    Quantale,
    Semigroupoid,
)
from quivers.core.trans import TransSeq
from quivers.core.morphism_transformations import (
    bayes_invert as _bayes_invert,
    l1_normalize as _l1_normalize,
    l2_normalize as _l2_normalize,
    softmax as _softmax,
)
from quivers.core.quantale_morphisms import (
    COUNTING_FROM_REAL,
    COUNTING_TO_REAL,
    EXPECTATION,
    LOG_PROB as _LOG_PROB_HOM,
    MATERIAL_IMPLICATION,
    MAX_PLUS as _MAX_PLUS_HOM,
    PROBABILITY_CLAMP,
    PROBABILITY_TO_REAL,
    embedding as _make_embedding,
    threshold as _make_threshold,
)
from quivers.dsl.ast_nodes import (
    ExportDecl,
    Expr,
    ExprCap,
    ExprChangeBase,
    ExprChartFold,
    ExprCompose,
    ExprCup,
    ExprCurry,
    ExprDagger,
    ExprFan,
    ExprFreeze,
    ExprFromData,
    ExprIdent,
    ExprIdentity,
    ExprMarginalize,
    ExprMorphismCall,
    ExprParser,
    ExprRepeat,
    ExprScan,
    ExprStack,
    ExprTensorProduct,
    ExprTrace,
    ExprTransCompose,
)
from quivers.dsl.compiler._prelude import (
    CompileError,
    _ChartHandlerComposite,
    _QUANTALE_REGISTRY,
)


class _ExpressionsMixin:
    """Mixin: expression compilation methods."""

    def _compile_export(self, decl: ExportDecl) -> None:
        """Record an exported expression.

        Replaces v0.4's single-output model: a module may declare
        any number of ``export`` statements, each selecting a
        top-level binding for the module's public surface. The
        compiled output runner picks the first export; further
        exports become additional accessible morphisms on the
        compiled object.
        """
        if not hasattr(self, "_exports"):
            self._exports = []
        self._exports.append(decl.expr)
        # Maintain backwards compatibility with internal helpers
        # that consult `_output_expr`: the first export wins.
        if self._output_expr is None:
            self._output_expr = decl.expr

    def bind_data(self, data: dict) -> None:
        """Bind a runtime data dictionary for ``from_data("KEY")``
        initializers.

        Each key in ``data`` maps to a tensor (or tensor-like
        object) that supplies the morphism tensor for any
        ``from_data("KEY")`` expression that references it. The
        bindings are consulted at compile time; supply them BEFORE
        calling :meth:`compile`.
        """
        self._data_bindings = dict(data)

    def _require_quantale(self, op_name: str, line: int, col: int) -> None:
        """Raise ``CompileError`` if the module's composition rule
        isn't a :class:`Quantale`.

        Operations that need a unit, zero, dagger, or compact-closed
        structure (``identity``, ``cup``, ``cap``, ``.dagger``,
        ``.trace``) call this before constructing their result.
        The error names the operation and the active rule so the
        user can decide whether to switch the file's
        ``quantale`` / ``semigroupoid`` / ``bilinear_form``
        declaration or drop the operation.
        """
        rule = self._quantale
        if isinstance(rule, Quantale):
            return
        kind = (
            "semigroupoid"
            if isinstance(rule, Semigroupoid)
            else (
                "bilinear_form"
                if isinstance(rule, BilinearForm)
                else "composition_rule"
            )
        )
        raise CompileError(
            f"{op_name}: requires a Quantale composition rule (needs "
            f"identity / unit / zero); the module declares {kind} "
            f"{rule.name!r}, which does not have those. Switch the "
            f"file-level declaration to ``quantale X`` for a "
            f"Quantale X, or drop the {op_name} operation.",
            line,
            col,
        )

    def _compile_trans_expr(self, expr):
        """Compile an expression that evaluates to a transformation.

        Recognised expression shapes:

        * :class:`ExprIdent` — a bare name resolved against
          :attr:`_transformations` (user let-bindings) and then
          :attr:`_trans_singletons` (built-in singletons).
        * :class:`ExprMorphismCall` — a constructor call whose
          callee names an entry of :attr:`_trans_constructors`;
          each argument is resolved against the surrounding scope
          (objects, morphisms, quantales).
        * :class:`ExprTransCompose` — ``t1 >>> t2``; recursively
          compiles each side, flattens nested sequences, and
          type-checks each adjacent ``target == source`` boundary,
          returning a :class:`TransSeq` (or, for two single-step
          values, an unboxed TransSeq of length 2).

        Returns a value that the runtime understands: a
        :class:`QuantaleHomomorphism`, a
        :class:`MorphismTransformation`, or a :class:`TransSeq`.
        Raises :class:`CompileError` on any unresolved reference
        or type mismatch.
        """
        if isinstance(expr, ExprIdent):
            if expr.name in self._transformations:
                return self._transformations[expr.name]
            if expr.name in self._trans_singletons:
                return self._trans_singletons[expr.name]
            if expr.name in self._trans_constructors:
                raise CompileError(
                    f"change_base: {expr.name!r} is a transformation "
                    f"constructor and needs arguments; call it like "
                    f"`{expr.name}(axis_object)`",
                    expr.line,
                    expr.col,
                )
            raise CompileError(
                f"change_base: undefined transformation "
                f"{expr.name!r}; available singletons: "
                f"{sorted(self._trans_singletons)}; constructors: "
                f"{sorted(self._trans_constructors)}; let-bound: "
                f"{sorted(self._transformations)}",
                expr.line,
                expr.col,
            )
        if isinstance(expr, ExprMorphismCall):
            factory = self._trans_constructors.get(expr.callee)
            if factory is None:
                raise CompileError(
                    f"change_base: undefined transformation "
                    f"constructor {expr.callee!r}; available: "
                    f"{sorted(self._trans_constructors)}",
                    expr.line,
                    expr.col,
                )
            resolved: list = []
            for arg_name in expr.args:
                value = self._resolve_trans_constructor_argument(
                    arg_name,
                    expr.line,
                    expr.col,
                    constructor_name=expr.callee,
                )
                resolved.append(value)
            try:
                return factory(*resolved)
            except (TypeError, ValueError) as e:
                raise CompileError(
                    f"change_base: constructor {expr.callee!r} "
                    f"rejected arguments {expr.args!r}: {e}",
                    expr.line,
                    expr.col,
                ) from e
        if isinstance(expr, ExprTransCompose):
            left_val = self._compile_trans_expr(expr.left)
            right_val = self._compile_trans_expr(expr.right)
            left_steps = (
                left_val.steps if isinstance(left_val, TransSeq) else (left_val,)
            )
            right_steps = (
                right_val.steps if isinstance(right_val, TransSeq) else (right_val,)
            )
            steps = left_steps + right_steps
            # Type-check the seam between each adjacent step.
            for i in range(len(steps) - 1):
                tgt = steps[i].target
                src = steps[i + 1].source
                if type(tgt) is not type(src):
                    raise CompileError(
                        f">>>: target of step {i} ({tgt.name!r}) "
                        f"does not match source of step {i + 1} "
                        f"({src.name!r})",
                        expr.line,
                        expr.col,
                    )
            return TransSeq(steps)
        raise CompileError(
            f"change_base: expression of kind "
            f"{type(expr).__name__!r} does not denote a "
            f"transformation",
            expr.line,
            expr.col,
        )

    def _resolve_trans_constructor_argument(
        self,
        name: str,
        line: int,
        col: int,
        *,
        constructor_name: str,
    ):
        """Resolve a named constructor argument to its compiled
        value.  Tries the value sorts that show up in
        transformation constructors (objects, morphisms,
        continuous morphisms, quantales) and surfaces a clear
        error if the name doesn't resolve.
        """
        if name in self._objects:
            return self._objects[name]
        if name in self._morphisms:
            return self._morphisms[name]
        if (
            hasattr(self, "_continuous_morphisms")
            and name in self._continuous_morphisms
        ):
            return self._continuous_morphisms[name]
        if name in _QUANTALE_REGISTRY:
            return _QUANTALE_REGISTRY[name]
        raise CompileError(
            f"change_base: constructor {constructor_name!r} "
            f"argument {name!r} unresolved (must name an object, "
            f"morphism, or quantale)",
            line,
            col,
        )

    def _apply_trans(self, inner_morph, phi):
        """Apply a transformation value (a singleton, constructor
        result, or :class:`TransSeq`) to a morphism via the
        morphism's own ``change_base``.

        Sequences are unfolded into iterated change_base calls;
        each intermediate result feeds the next step.
        """
        if isinstance(phi, TransSeq):
            current = inner_morph
            for step in phi.steps:
                current = current.change_base(step)
            return current
        return inner_morph.change_base(phi)

    def _compose_with_op(self, left, right, op: str):
        """Dispatch a composition expression to the quantale
        implied by the surface operator.

        Each composition operator carries an enrichment quantale.
        ``>>``, ``<<`` (already swapped to forward), and ``>=>``
        all use the operands' shared quantale (the existing
        :meth:`Morphism.__rshift__` path, which raises
        ``incompatible quantales`` if they differ).

        The new operators (``*>``, ``~>``, ``||>``, ``?>``,
        ``&&>``, ``+>``) each fix the composition quantale at the
        operator and re-tag the operands accordingly. If the
        operands' declared quantales already match the operator's
        target, no base change is needed; otherwise the user must
        have applied an explicit ``.change_base(φ)`` upstream.
        """
        from quivers.core.quantales import (
            COUNTING,
            GODEL,
            LOG_PROB,
            LUKASIEWICZ,
            MAX_PLUS,
            PROBABILITY,
            REAL,
        )
        from quivers.core.morphisms import ComposedMorphism, Morphism
        from quivers.core.quantales import BOOLEAN
        from quivers.stochastic.quantale import MARKOV

        del COUNTING  # exposed via module-level `quantale counting` only
        op_to_quantale: dict[str, object] = {
            ">>": None,  # use operands' shared quantale
            ">=>": None,
            "*>": MARKOV,
            "~>": LOG_PROB,
            "||>": GODEL,
            "?>": MAX_PLUS,
            "&&>": BOOLEAN,
            "+>": LUKASIEWICZ,
            "$>": REAL,
            "%>": PROBABILITY,
        }
        if op not in op_to_quantale:
            raise CompileError(f"unknown composition operator {op!r}", 0, 0)
        target_quantale = op_to_quantale[op]
        if target_quantale is None:
            # ``>>`` and ``>=>``: fall through to the operands' own
            # composition machinery, which uses the shared quantale
            # (and errors on a mismatch as before).
            return left >> right
        # Validate both operands carry the operator's target
        # quantale. The operator does NOT auto-base-change; the
        # user must have applied ``.change_base(...)`` upstream to
        # bring both operands to the target quantale before
        # composing.
        if not isinstance(left, Morphism) or not isinstance(right, Morphism):
            raise TypeError(
                f"composition operator {op!r}: both operands must be "
                f"Morphism instances; got "
                f"{type(left).__name__} {op} {type(right).__name__}"
            )
        for label, m in (("left", left), ("right", right)):
            if type(m.quantale) is not type(target_quantale):
                raise TypeError(
                    f"composition operator {op!r}: {label} operand's "
                    f"quantale is {m.quantale.name!r}, but the "
                    f"operator dispatches to "
                    f"{target_quantale.name!r}; apply "  # type: ignore[union-attr]
                    f"`.change_base(...)` first to convert "
                    f"{label} into the operator's quantale"
                )
        if left.codomain != right.domain:
            raise TypeError(
                f"composition operator {op!r}: cannot compose "
                f"codomain {left.codomain!r} != domain "
                f"{right.domain!r}"
            )
        return ComposedMorphism(left, right)

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
        elif isinstance(expr, ExprMorphismCall):
            return self._compile_morphism_call(expr)
        elif isinstance(expr, ExprIdentity):
            if expr.object_name not in self._objects:
                raise CompileError(
                    f"undefined object {expr.object_name!r}", expr.line, expr.col
                )
            self._require_quantale("identity", expr.line, expr.col)
            obj = self._objects[expr.object_name]
            return make_identity(obj, quantale=self._quantale)
        elif isinstance(expr, ExprDagger):
            inner = self._compile_expr(expr.inner)
            self._require_quantale("dagger", expr.line, expr.col)
            return inner.dagger
        elif isinstance(expr, ExprTrace):
            inner = self._compile_expr(expr.inner)
            if expr.object_name not in self._objects:
                raise CompileError(
                    f"trace: undefined object {expr.object_name!r}",
                    expr.line,
                    expr.col,
                )
            self._require_quantale("trace", expr.line, expr.col)
            try:
                return inner.trace(self._objects[expr.object_name])
            except TypeError as e:
                raise CompileError(str(e), expr.line, expr.col) from e
        elif isinstance(expr, ExprChangeBase):
            inner = self._compile_expr(expr.inner)
            phi = self._compile_trans_expr(expr.phi)
            try:
                return self._apply_trans(inner, phi)
            except TypeError as e:
                raise CompileError(str(e), expr.line, expr.col) from e
        elif isinstance(expr, ExprCup):
            if expr.object_name not in self._objects:
                raise CompileError(
                    f"cup: undefined object {expr.object_name!r}",
                    expr.line,
                    expr.col,
                )
            self._require_quantale("cup", expr.line, expr.col)
            return _make_cup(self._objects[expr.object_name], quantale=self._quantale)
        elif isinstance(expr, ExprCap):
            if expr.object_name not in self._objects:
                raise CompileError(
                    f"cap: undefined object {expr.object_name!r}",
                    expr.line,
                    expr.col,
                )
            self._require_quantale("cap", expr.line, expr.col)
            return _make_cap(self._objects[expr.object_name], quantale=self._quantale)
        elif isinstance(expr, ExprFreeze):
            inner = self._compile_expr(expr.inner)
            # Materialise the inner morphism's tensor with detach()
            # and wrap as an ObservedMorphism. Gradient flow stops
            # at this boundary; the constituent morphisms' parameters
            # are not part of the result's parameter set.
            from quivers.core.morphisms import (
                Morphism as _CatMorph,
                ObservedMorphism as _Obs,
            )

            if not isinstance(inner, _CatMorph):
                raise CompileError(
                    f"freeze: inner expression compiled to "
                    f"{type(inner).__name__}; expected a Morphism",
                    expr.line,
                    expr.col,
                )
            frozen = _Obs(
                inner.domain,
                inner.codomain,
                inner.tensor.detach().clone(),
                quantale=inner.quantale,
            )
            return frozen
        elif isinstance(expr, ExprFromData):
            # ``from_data("KEY")`` cannot resolve the tensor at
            # compile time because the runtime data dict is only
            # available at fit time. We must therefore emit a
            # late-binding ObservedMorphism whose tensor is filled
            # in via a fit-time hook on the program's data
            # dictionary. For the standalone ``let``-expression
            # case this is materialised as a deferred binding the
            # compiler can resolve once the data dict is known.
            #
            # The data dictionary is stored on the Compiler under
            # ``self._data_bindings``; users supply it via
            # :meth:`Compiler.bind_data` BEFORE calling compile()
            # (or via the high-level ``loads()`` ``data=`` kwarg).
            data_dict = getattr(self, "_data_bindings", None)
            if data_dict is None or expr.key not in data_dict:
                available = sorted(data_dict.keys()) if data_dict else []
                raise CompileError(
                    f"from_data: unknown data key {expr.key!r}; "
                    f"available: {available}. Bind the data dict via "
                    f"``Compiler.bind_data(...)`` or the ``data=`` "
                    f"keyword on ``loads()`` before compiling.",
                    expr.line,
                    expr.col,
                )
            from quivers.core.morphisms import ObservedMorphism as _Obs

            tensor = data_dict[expr.key]
            if not isinstance(tensor, torch.Tensor):
                tensor = torch.as_tensor(tensor)
            # The user-supplied domain/codomain on the morphism
            # declaration is in the parent ``observed`` decl; here
            # we synthesize a one-shot ObservedMorphism whose
            # domain/codomain are inferred from the tensor's shape
            # split halfway. For the common ``observed f : A -> B =
            # from_data("KEY")`` pattern the parent decl supplies
            # the correct domain/codomain post-substitution; this
            # let-expression path is for ``let h = from_data(...)``
            # which the surrounding context constrains.
            from quivers.core.objects import FinSet as _FS

            if tensor.dim() < 1:
                raise CompileError(
                    f"from_data: tensor at key {expr.key!r} has "
                    f"rank 0; cannot infer a morphism",
                    expr.line,
                    expr.col,
                )
            # Synthetic anonymous domain/codomain for the
            # let-expression path. The morphism's downstream
            # usage typically discards this synthetic typing
            # because it is reassigned to a properly typed
            # ``observed`` slot, OR used as a tensor source.
            dom = _FS(name=f"_data_dom_{expr.key}", cardinality=tensor.shape[0])
            if tensor.dim() == 1:
                cod = _FS(name="_data_unit", cardinality=1)
                tensor = tensor.unsqueeze(-1)
            else:
                cod_size = int(torch.tensor(tensor.shape[1:]).prod().item())
                cod = _FS(name=f"_data_cod_{expr.key}", cardinality=cod_size)
                if tensor.dim() > 2:
                    tensor = tensor.reshape(tensor.shape[0], -1)
            return _Obs(dom, cod, tensor, quantale=self._quantale)
        elif isinstance(expr, ExprCompose):
            left = self._compile_expr(expr.left)
            right = self._compile_expr(expr.right)
            try:
                return self._compose_with_op(left, right, expr.op)
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
            parser = InsideAlgorithm(binary, lex, start=start, unary=unary)
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
