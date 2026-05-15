"""Compiler mixin: program / contraction / let compilation.

Handles bind-step expansion, effect verification, template inlining,
program bodies, contractions, and let-expression compilation.
"""

from __future__ import annotations
from collections.abc import Callable
from itertools import product as _cartesian_product
from typing import cast
import torch
from quivers.continuous.morphisms import AnySpace, ContinuousMorphism
from quivers.core.morphisms import Morphism

from quivers.continuous.plate import marginalize_grouped
from quivers.continuous.spaces import (
    ContinuousSpace,
    Euclidean,
    PositiveReals,
    Simplex,
    UnitInterval,
)
from quivers.core.objects import FinSet, ProductSet, SetObject
from quivers.core.wiring import EinsumWiring
from quivers.dsl.ast_nodes import (
    BindStep,
    ContractionDecl,
    DrawStep,
    ExprIdent,
    ExprMorphismCall,
    ExprTransCompose,
    GroupedBodyObserveStep,
    GroupedLatentInitStep,
    GroupedObserveEntry,
    LetDecl,
    LetExprBinOp,
    LetExprCall,
    LetExprIndex,
    LetExprLambda,
    LetExprFactor,
    LetExprList,
    LetFactorCase,
    LetExprLiteral,
    LetExprMethodCall,
    LetExprNode,
    LetExprString,
    LetExprUnaryOp,
    LetExprVar,
    LetStep,
    MarginalizeStep,
    MorphismParam,
    ObjectParam,
    PlateDrawStep,
    ProgramDecl,
    ProgramStep,
    ScalarParam,
    TypeExpr,
    TypeName,
    TypeProduct,
    VectorisedObserveStep,
)
from quivers.dsl.compiler._prelude import (
    CompileError,
    _CompiledContraction,
    _ALGEBRA_REGISTRY,
    _get_family_registry,
    _numel_shape,
)


# Value carried by a let-binding at compile time.  The let
# sublanguage is a small typed lambda calculus over heterogeneous
# values: numeric literals lift to torch tensors; identifier
# references resolve to whatever is bound (morphisms, programs,
# chart views, structured tuples); composite expressions return
# tensors of derived shape; lambdas wrap functions.  The union
# below enumerates the kinds the program-theory check admits at a
# let position.
type LetValue = (
    torch.Tensor
    | int
    | float
    | bool
    | str
    | tuple["LetValue", ...]
    | list["LetValue"]
    | Morphism
    | ContinuousMorphism
    | Callable[[dict[str, "LetValue"]], "LetValue"]
)


class _ProgramsMixin:
    """Mixin: program / contraction / let compilation methods."""

    def _expand_bind_steps(
        self, steps: tuple[ProgramStep, ...]
    ) -> tuple[ProgramStep, ...]:
        """Translate v0.5 :class:`BindStep` IR into the compiler's
        internal step-IR (:class:`DrawStep`, :class:`PlateDrawStep`,
        :class:`VectorisedObserveStep`, :class:`MarginalizeStep`).

        The expansion is purely a syntactic refinement: each
        BindStep dispatches on its ``mode`` and ``index`` fields
        to one of the four internal step shapes. Marginalize binds
        additionally inline a synthesized sample step for the
        coordinate, followed by the scope's recursively-expanded
        steps, followed by a :class:`MarginalizeStep` reduction.

        ``LetStep`` passes through unchanged. The expansion
        preserves the Kleisli-arrow denotation of the program body
        — it is a reorganisation of the surface IR, not a change
        of semantics.
        """
        out: list[ProgramStep] = []
        for step in steps:
            if isinstance(step, LetStep):
                out.append(step)
                continue
            if not isinstance(step, BindStep):
                # Pass-through for any internal-IR step that has
                # already been expanded (e.g., template-inlined
                # bodies that synthesized internal steps directly).
                out.append(step)
                continue
            if step.mode == "sample":
                if step.index is None:
                    out.append(
                        DrawStep(
                            vars=step.vars,
                            morphism=step.morphism,
                            args=step.args,
                            is_observed=False,
                            line=step.line,
                            col=step.col,
                        )
                    )
                else:
                    if len(step.vars) != 1:
                        raise CompileError(
                            "indexed sample bind must bind a single name",
                            step.line,
                            step.col,
                        )
                    # The per-row codomain for a plate-draw is taken
                    # from the family's natural codomain at compile
                    # time; the IR carries a `codomain` field that
                    # the compiler's PlateDrawStep handler resolves
                    # via the family's domain/codomain dimensions.
                    # For the v0.5 unified surface, the index annotation
                    # `: A` declares the index set; the per-row codomain
                    # is implicit (taken from the family). We supply a
                    # placeholder `TypeName("1")` which the family
                    # resolver interprets as "scalar per-row codomain"
                    # (Euclidean(1)); families that declare richer
                    # codomains override this.
                    out.append(
                        PlateDrawStep(
                            name=step.vars[0],
                            index=step.index,
                            codomain=TypeName(name="1", line=step.line, col=step.col),
                            morphism=step.morphism,
                            args=step.args,
                            line=step.line,
                            col=step.col,
                        )
                    )
            elif step.mode == "score":
                if step.index is None:
                    out.append(
                        DrawStep(
                            vars=step.vars,
                            morphism=step.morphism,
                            args=step.args,
                            is_observed=True,
                            line=step.line,
                            col=step.col,
                        )
                    )
                else:
                    if len(step.vars) != 1:
                        raise CompileError(
                            "indexed observe bind must bind a single name",
                            step.line,
                            step.col,
                        )
                    out.append(
                        VectorisedObserveStep(
                            index_var=step.vars[0],
                            index_set=step.index,
                            morphism=step.morphism,
                            args=step.args,
                            response_var=step.vars[0],
                            fibration_var=step.via,
                            fibration_axes=step.via_axes,
                            line=step.line,
                            col=step.col,
                        )
                    )
            elif step.mode == "marginal":
                if len(step.vars) != 1:
                    raise CompileError(
                        "marginalize bind must bind a single name",
                        step.line,
                        step.col,
                    )
                # Grouped form: `over G` (single plate) or
                # `over G * H` (product grouping plate). The
                # fibration data lives on each observe inside the
                # body via that observe's `via <idx>` (or
                # `via product(...)`) clause; the header carries
                # only the grouping plate.
                has_over = step.over is not None or step.over_objs is not None
                has_grouping = has_over
                # Normalize the grouping plate into tuple form so
                # the downstream code handles single + product
                # uniformly.
                over_names: tuple[str, ...] | None = None
                if step.over_objs is not None:
                    over_names = step.over_objs
                elif step.over is not None:
                    over_names = (step.over,)
                # Resolve the per-class size from the latent's index
                # annotation when present. The grouped form requires
                # an explicit `: K` annotation so the prior's class
                # axis can be sized at compile time.
                class_size = 0
                if step.index is not None:
                    if isinstance(step.index, TypeName):
                        nm = step.index.name
                        if nm.isdigit():
                            class_size = int(nm)
                        elif nm in self._objects:
                            class_size = int(self._objects[nm].size)
                if has_over and class_size == 0:
                    raise CompileError(
                        "grouped marginalize requires an explicit class-set "
                        "annotation (e.g. `marginalize c : K <- ...`) so the "
                        "class axis is sized at compile time",
                        step.line,
                        step.col,
                    )
                if over_names is not None:
                    for nm in over_names:
                        if nm not in self._objects:
                            raise CompileError(
                                f"grouped marginalize: `over` object "
                                f"{nm!r} is not a declared object",
                                step.line,
                                step.col,
                            )
                # Reduction defaults to logsumexp when unspecified.
                reduction = step.reduction or "logsumexp"
                if reduction not in ("logsumexp", "sum", "mean"):
                    raise CompileError(
                        f"grouped marginalize: unknown reduction "
                        f"{reduction!r}; expected one of logsumexp, "
                        f"sum, mean",
                        step.line,
                        step.col,
                    )
                # Extract the categorical prior's `probs` argument.
                probs_var: str | None = None
                if has_over:
                    if not step.args:
                        raise CompileError(
                            "grouped marginalize requires the latent's "
                            "categorical family to carry a probs argument "
                            "(e.g. `Categorical(probs)`)",
                            step.line,
                            step.col,
                        )
                    first = step.args[0]
                    if not isinstance(first, str):
                        raise CompileError(
                            "grouped marginalize: the categorical family's "
                            "first argument must be a named probs tensor "
                            f"(got literal {first!r})",
                            step.line,
                            step.col,
                        )
                    probs_var = first
                # Introduce the coordinate. For an *ungrouped*
                # marginalize block the latent is a real sample
                # from its categorical prior; for a *grouped*
                # block it is the implicit class-index vector
                # ``torch.arange(K)`` so any downstream let-step
                # expression that references the latent broadcasts
                # across the class axis. The terminal observe in
                # the body (rewritten below) overwrites this slot
                # with the per-(N, K) log-likelihood tensor the
                # marginalize step consumes.
                latent_name = step.vars[0]
                if has_grouping:
                    out.append(
                        GroupedLatentInitStep(
                            latent_name=latent_name,
                            class_size=class_size,
                            line=step.line,
                            col=step.col,
                        )
                    )
                elif step.index is None:
                    out.append(
                        DrawStep(
                            vars=step.vars,
                            morphism=step.morphism,
                            args=step.args,
                            is_observed=False,
                            line=step.line,
                            col=step.col,
                        )
                    )
                else:
                    out.append(
                        PlateDrawStep(
                            name=step.vars[0],
                            index=step.index,
                            codomain=TypeName(name="1", line=step.line, col=step.col),
                            morphism=step.morphism,
                            args=step.args,
                            line=step.line,
                            col=step.col,
                        )
                    )
                # Scope's steps. For a grouped block, rewrite every
                # VectorisedObserveStep in the expanded body as a
                # GroupedBodyObserveStep that writes its per-row
                # per-class log-likelihood to a dedicated env slot.
                # The surrounding MarginalizeStep's runtime callable
                # collects each slot, pairs it with that observe's
                # `via <idx>` fibration, and scatter-sums each
                # contribution into the shared `(|G|, K)`
                # accumulator before the reduction.
                scope_steps = step.scope if step.scope is not None else ()
                expanded_scope = list(self._expand_bind_steps(scope_steps))
                body_observes: list[GroupedObserveEntry] = []
                if has_grouping:
                    # Walk the body forward; rewrite each observe
                    # into a GroupedBodyObserveStep with a unique
                    # ll_slot and capture its per-observe fibration.
                    for j, scoped_step in enumerate(expanded_scope):
                        if isinstance(scoped_step, VectorisedObserveStep):
                            if (
                                scoped_step.fibration_var is None
                                and scoped_step.fibration_axes is None
                            ):
                                raise CompileError(
                                    "grouped marginalize: every observe "
                                    "inside the body must carry its own "
                                    "`via <idx>` clause "
                                    "(e.g. `observe r : N via idx <- ...`); "
                                    f"observe {scoped_step.response_var!r} "
                                    "has no `via`",
                                    scoped_step.line,
                                    scoped_step.col,
                                )
                            ll_slot = f"_grouped_ll_{latent_name}_{len(body_observes)}"
                            expanded_scope[j] = GroupedBodyObserveStep(
                                response_var=scoped_step.response_var,
                                morphism=scoped_step.morphism,
                                args=scoped_step.args,
                                index_set=scoped_step.index_set,
                                index_var=scoped_step.index_var,
                                latent_name=latent_name,
                                fibration_var=scoped_step.fibration_var,
                                fibration_axes=scoped_step.fibration_axes,
                                ll_slot=ll_slot,
                                line=scoped_step.line,
                                col=scoped_step.col,
                            )
                            body_observes.append(
                                GroupedObserveEntry(
                                    ll_slot=ll_slot,
                                    fibration_var=scoped_step.fibration_var,
                                    fibration_axes=scoped_step.fibration_axes,
                                )
                            )
                    if not body_observes:
                        # Nested case: a grouped marginalize block whose
                        # body's only contribution to the per-group
                        # accumulator is an inner grouped block.  The
                        # inner block's MarginalizeStep produces the
                        # (N_outer, K_outer) tensor the outer block
                        # consumes.  Re-point the inner's
                        # ``body_ll_var`` at the outer latent so the
                        # outer codegen finds the tensor at the
                        # expected slot, and record a single
                        # body_observes entry whose ll_slot is the
                        # outer latent's name (no per-observe
                        # fibration, since the inner block already
                        # performed its own scatter-add).
                        nested_marg_idx: int | None = None
                        for j in range(len(expanded_scope) - 1, -1, -1):
                            if isinstance(expanded_scope[j], MarginalizeStep):
                                nested_marg_idx = j
                                break
                        if nested_marg_idx is None:
                            raise CompileError(
                                "grouped marginalize: the body must "
                                "contain at least one `observe` step "
                                "(or an inner grouped marginalize) "
                                "whose per-row log-likelihood produces "
                                "the per-group accumulator's "
                                "contribution",
                                step.line,
                                step.col,
                            )
                        inner_marg = expanded_scope[nested_marg_idx]
                        assert isinstance(inner_marg, MarginalizeStep)
                        expanded_scope[nested_marg_idx] = MarginalizeStep(
                            var_name=inner_marg.var_name,
                            class_size=inner_marg.class_size,
                            probs_var=inner_marg.probs_var,
                            over_obj=inner_marg.over_obj,
                            over_objs=inner_marg.over_objs,
                            body_ll_var=latent_name,
                            body_observes=inner_marg.body_observes,
                            reduction=inner_marg.reduction,
                            line=inner_marg.line,
                            col=inner_marg.col,
                        )
                        body_observes.append(GroupedObserveEntry(ll_slot=latent_name))
                out.extend(expanded_scope)
                # Pushforward reduction. When grouped, the
                # MarginalizeStep carries the list of per-observe
                # (ll_slot, fibration) entries the runtime callable
                # consumes; the legacy single-fibration fields are
                # gone.
                single_over = (
                    over_names[0]
                    if over_names is not None and len(over_names) == 1
                    else None
                )
                product_overs = (
                    over_names
                    if over_names is not None and len(over_names) > 1
                    else None
                )
                out.append(
                    MarginalizeStep(
                        var_name=step.vars[0],
                        class_size=class_size,
                        probs_var=probs_var,
                        over_obj=single_over,
                        over_objs=product_overs,
                        body_ll_var=step.vars[0],
                        body_observes=(tuple(body_observes) if has_grouping else None),
                        reduction=step.reduction,
                        line=step.line,
                        col=step.col,
                    )
                )
            else:
                raise CompileError(
                    f"unknown bind mode {step.mode!r}",
                    step.line,
                    step.col,
                )
        return tuple(out)

    def _verify_effects(
        self, decl: ProgramDecl, steps: tuple[ProgramStep, ...]
    ) -> None:
        """Verify the program body's effect usage matches `! effects`.

        Each program step contributes to the program's *actual*
        effect set:

        * ``DrawStep`` / ``PlateDrawStep`` with ``is_observed=False``
          contribute ``Sample``.
        * ``DrawStep`` / ``VectorisedObserveStep`` with score-bind
          shape contribute ``Score``.
        * ``MarginalizeStep`` contributes ``Marginal``.
        * ``LetStep`` contributes nothing (purely deterministic).

        If the declaration includes ``! effects``, the actual set
        must be a subset of the declared set. A declared
        ``Pure`` rejects any of {Sample, Score, Marginal}.
        """
        if decl.effects is None:
            return  # unannotated → no verification
        declared = decl.effects
        actual: set[str] = set()
        for step in steps:
            if isinstance(step, (DrawStep, PlateDrawStep)):
                if isinstance(step, DrawStep) and step.is_observed:
                    actual.add("Score")
                else:
                    actual.add("Sample")
            elif isinstance(step, VectorisedObserveStep):
                actual.add("Score")
            elif isinstance(step, MarginalizeStep):
                actual.add("Marginal")
            # LetStep contributes nothing.
        if "Pure" in declared and actual:
            raise CompileError(
                f"program {decl.name!r} is declared `! Pure` but body "
                f"uses effects {sorted(actual)}",
                decl.line,
                decl.col,
            )
        unaccounted = actual - declared - {"Pure"}
        if unaccounted:
            raise CompileError(
                f"program {decl.name!r} body uses effects {sorted(unaccounted)} "
                f"not listed in `! {{{', '.join(sorted(declared))}}}`",
                decl.line,
                decl.col,
            )

    def _expand_template_calls(
        self, steps: tuple[ProgramStep, ...]
    ) -> tuple[ProgramStep, ...]:
        """Inline parametric-program-template call sites in a step list.

        A ``draw v ~ T(args)`` step whose morphism name ``T`` is a
        registered parametric program denotes the instantiation of
        the dependent kernel at the supplied arguments. The body of
        ``T`` is substituted (formal parameters → actual arguments)
        and α-renamed (every locally-bound name is prefixed by
        ``v$``, except the return-variable which is renamed to ``v``
        directly so the call's binding receives the template's
        return value). The renamed step list replaces the call site.

        Recursive template calls (a template body that itself calls
        another template) are handled by post-expansion: after a
        template is inlined its expanded steps are themselves
        recursively expanded, with cycle detection.
        """
        expanded: list[ProgramStep] = []
        for step in steps:
            if (
                isinstance(step, DrawStep)
                and not step.is_observed
                and step.morphism in self._program_templates
            ):
                tmpl = self._program_templates[step.morphism]
                if len(step.vars) != 1:
                    raise CompileError(
                        f"template call {step.morphism!r} may bind only one "
                        f"variable, got tuple {step.vars}",
                        step.line,
                        step.col,
                    )
                bind_name = step.vars[0]
                args = step.args or ()
                inst = self._instantiate_template(tmpl, bind_name, args, step)
                # Recursively expand any nested template calls in the
                # inlined body.
                expanded.extend(self._expand_template_calls(inst))
                continue
            if (
                isinstance(step, PlateDrawStep)
                and step.morphism in self._program_templates
            ):
                raise CompileError(
                    f"template {step.morphism!r} cannot be called from a "
                    f"plate-draw step; use a bare 'draw' inside the template "
                    f"body for the plate or wrap the call in a per-index "
                    f"helper",
                    step.line,
                    step.col,
                )
            expanded.append(step)
        return tuple(expanded)

    def _instantiate_template(
        self,
        tmpl: ProgramDecl,
        bind_name: str,
        args: tuple,
        call_site: ProgramStep,
    ) -> tuple[ProgramStep, ...]:
        """Realise one call site of a parametric program template.

        Categorical denotation: given the dependent kernel
        :math:`\\Pi (p_i : P_i).\\ \\mathbf{Kern}(\\mathrm{dom}(p),\\, \\mathrm{cod}(p))`
        carried by ``tmpl``, return the concrete Kern-morphism at
        ``args`` (a section of the family). The morphism is
        represented as the renamed step list whose internal latents
        contribute their own factors to the caller's joint kernel,
        with the return-variable renamed to ``bind_name`` so the
        call's binding receives the template's output value.

        Substitution + α-renaming together realise the categorical
        substitution lemma: substituting actuals for formals
        commutes with denotation up to renaming-equivalence.
        """
        type_params = tmpl.type_params or ()
        if len(args) != len(type_params):
            raise CompileError(
                f"template {tmpl.name!r} expects {len(type_params)} arguments, "
                f"got {len(args)}",
                call_site.line,
                call_site.col,
            )
        # Build the parameter-substitution environment.
        type_subst: dict[str, TypeExpr] = {}
        value_subst: dict[str, str | float] = {}
        for param, arg in zip(type_params, args):
            if isinstance(param, ObjectParam):
                if not isinstance(arg, str):
                    raise CompileError(
                        f"template {tmpl.name!r}: parameter {param.name!r} "
                        f"({param.universe}) requires a type-name argument, "
                        f"got {arg!r}",
                        call_site.line,
                        call_site.col,
                    )
                # Validate the named object/space matches the declared
                # universe.
                if param.universe == "FinSet" and arg not in self._objects:
                    raise CompileError(
                        f"template {tmpl.name!r}: parameter {param.name!r} : "
                        f"FinSet expects a finite-set object, but {arg!r} is "
                        f"not a declared object",
                        call_site.line,
                        call_site.col,
                    )
                if param.universe == "Space" and arg not in self._spaces:
                    raise CompileError(
                        f"template {tmpl.name!r}: parameter {param.name!r} : "
                        f"Space expects a continuous space, but {arg!r} is "
                        f"not a declared space",
                        call_site.line,
                        call_site.col,
                    )
                if (
                    param.universe == "Object"
                    and arg not in self._objects
                    and arg not in self._spaces
                ):
                    raise CompileError(
                        f"template {tmpl.name!r}: parameter {param.name!r} : "
                        f"Object expects a declared object or space, got {arg!r}",
                        call_site.line,
                        call_site.col,
                    )
                type_subst[param.name] = TypeName(
                    name=arg, line=call_site.line, col=call_site.col
                )
            elif isinstance(param, ScalarParam):
                if isinstance(arg, str):
                    # Scalar parameter passed as a name (e.g., a previously
                    # let-bound scalar in the caller). Pass through as a
                    # string reference; the caller's bound_vars will
                    # resolve it at draw-site time.
                    value_subst[param.name] = arg
                else:
                    value_subst[param.name] = float(arg)
            elif isinstance(param, MorphismParam):
                if not isinstance(arg, str):
                    raise CompileError(
                        f"template {tmpl.name!r}: parameter {param.name!r} : "
                        f"Mor[...] expects a morphism name, got {arg!r}",
                        call_site.line,
                        call_site.col,
                    )
                if arg not in self._morphisms and arg not in self._program_templates:
                    raise CompileError(
                        f"template {tmpl.name!r}: parameter {param.name!r}: "
                        f"morphism {arg!r} is not declared",
                        call_site.line,
                        call_site.col,
                    )
                value_subst[param.name] = arg
            else:
                raise CompileError(
                    f"template {tmpl.name!r}: unknown parameter kind for "
                    f"{getattr(param, 'name', '?')!r}",
                    call_site.line,
                    call_site.col,
                )
        # Collect all locally-bound names in the template body (latents
        # drawn, plate-draws, lets, observe loop-vars). These are
        # α-renamed to live in the caller's namespace.
        local_names = self._collect_template_local_names(tmpl)
        # The return variable (if a single identifier) receives the
        # call's binding name directly; other locals are namespaced.
        return_var = tmpl.return_vars[0] if len(tmpl.return_vars) == 1 else None
        rename: dict[str, str] = {}
        for nm in local_names:
            if nm == return_var:
                rename[nm] = bind_name
            else:
                rename[nm] = f"{bind_name}${nm}"
        # Expand the template body's BindStep IR into the
        # compiler's internal step shapes first (so the rename pass
        # operates on a uniform IR).
        expanded_body = self._expand_bind_steps(tmpl.draws)
        # Walk the expanded body, applying parameter substitution +
        # α-renaming step by step.
        return tuple(
            self._rename_step(step, type_subst, value_subst, rename)
            for step in expanded_body
        )

    def _collect_template_local_names(self, tmpl: ProgramDecl) -> set[str]:
        """All names bound inside the template body (latents + lets).

        Walks the *unexpanded* v0.5 BindStep / LetStep surface; the
        BindStep covers sample / score / marginal modes, contributing
        all bound names to the local-name set for α-renaming.
        """
        out: set[str] = set()

        def _walk(steps):
            for step in steps:
                if isinstance(step, BindStep):
                    out.update(step.vars)
                    if step.scope is not None:
                        _walk(step.scope)
                elif isinstance(step, LetStep):
                    out.add(step.name)
                # Internal IR (post-expand) — also covered, for the
                # case where _collect_local_names is invoked on
                # already-expanded steps.
                elif isinstance(step, DrawStep):
                    out.update(step.vars)
                elif isinstance(step, PlateDrawStep):
                    out.add(step.name)
                elif isinstance(step, VectorisedObserveStep):
                    out.add(step.index_var)
                    if step.response_var:
                        out.add(step.response_var)
                elif isinstance(step, MarginalizeStep):
                    out.add(step.var_name)
                    if step.body_ll_var is not None:
                        out.add(step.body_ll_var)
                elif isinstance(step, GroupedLatentInitStep):
                    out.add(step.latent_name)
                elif isinstance(step, GroupedBodyObserveStep):
                    out.add(step.index_var)
                    if step.response_var:
                        out.add(step.response_var)
                    if step.latent_name:
                        out.add(step.latent_name)

        _walk(tmpl.draws)
        return out

    def _rename_type(
        self, texpr: TypeExpr, type_subst: dict[str, TypeExpr]
    ) -> TypeExpr:
        """Substitute object parameters inside a type expression."""
        if isinstance(texpr, TypeName):
            if texpr.name in type_subst:
                return type_subst[texpr.name]
            return texpr
        if isinstance(texpr, TypeProduct):
            return TypeProduct(
                components=tuple(
                    self._rename_type(c, type_subst) for c in texpr.components
                ),
                line=texpr.line,
                col=texpr.col,
            )
        return texpr

    def _rename_args(
        self,
        args: tuple | None,
        value_subst: dict[str, str | float],
        rename: dict[str, str],
    ) -> tuple | None:
        """Apply parameter substitution and α-renaming inside a draw-arg list."""
        if args is None:
            return None
        out: list = []
        for a in args:
            if isinstance(a, str):
                if a in value_subst:
                    out.append(value_subst[a])
                elif a in rename:
                    out.append(rename[a])
                else:
                    out.append(a)
            else:
                out.append(a)
        return tuple(out)

    def _rename_step(
        self,
        step: ProgramStep,
        type_subst: dict[str, TypeExpr],
        value_subst: dict[str, str | float],
        rename: dict[str, str],
    ) -> ProgramStep:
        """Apply parameter substitution + α-renaming to a single step."""
        if isinstance(step, DrawStep):
            new_vars = tuple(rename.get(v, v) for v in step.vars)
            new_morph = value_subst.get(step.morphism, step.morphism)
            if not isinstance(new_morph, str):
                raise CompileError(
                    f"draw step morphism {step.morphism!r} substituted to a "
                    f"non-string value {new_morph!r}",
                    step.line,
                    step.col,
                )
            return DrawStep(
                vars=new_vars,
                morphism=new_morph,
                args=self._rename_args(step.args, value_subst, rename),
                is_observed=step.is_observed,
                line=step.line,
                col=step.col,
            )
        if isinstance(step, PlateDrawStep):
            new_morph = value_subst.get(step.morphism, step.morphism)
            if not isinstance(new_morph, str):
                raise CompileError(
                    f"plate-draw step morphism {step.morphism!r} substituted "
                    f"to a non-string value {new_morph!r}",
                    step.line,
                    step.col,
                )
            return PlateDrawStep(
                name=rename.get(step.name, step.name),
                index=self._rename_type(step.index, type_subst),
                codomain=self._rename_type(step.codomain, type_subst),
                morphism=new_morph,
                args=self._rename_args(step.args, value_subst, rename),
                line=step.line,
                col=step.col,
            )
        if isinstance(step, LetStep):
            return LetStep(
                name=rename.get(step.name, step.name),
                expr=self._rename_let_expr(step.expr, value_subst, rename),
                line=step.line,
                col=step.col,
            )
        if isinstance(step, VectorisedObserveStep):
            new_morph = value_subst.get(step.morphism, step.morphism)
            if not isinstance(new_morph, str):
                raise CompileError(
                    f"observe step morphism {step.morphism!r} substituted to "
                    f"a non-string value {new_morph!r}",
                    step.line,
                    step.col,
                )
            return VectorisedObserveStep(
                index_var=rename.get(step.index_var, step.index_var),
                index_set=self._rename_type(step.index_set, type_subst),
                morphism=new_morph,
                args=self._rename_args(step.args, value_subst, rename),
                response_var=rename.get(step.response_var, step.response_var),
                line=step.line,
                col=step.col,
            )
        if isinstance(step, MarginalizeStep):
            renamed_probs = (
                rename.get(step.probs_var, step.probs_var)
                if step.probs_var is not None
                else None
            )
            renamed_body_ll = (
                rename.get(step.body_ll_var, step.body_ll_var)
                if step.body_ll_var is not None
                else None
            )
            renamed_body_observes: tuple[GroupedObserveEntry, ...] | None = None
            if step.body_observes is not None:
                renamed_body_observes = tuple(
                    GroupedObserveEntry(
                        ll_slot=rename.get(entry.ll_slot, entry.ll_slot),
                        fibration_var=(
                            rename.get(entry.fibration_var, entry.fibration_var)
                            if entry.fibration_var is not None
                            else None
                        ),
                        fibration_axes=(
                            tuple(rename.get(v, v) for v in entry.fibration_axes)
                            if entry.fibration_axes is not None
                            else None
                        ),
                    )
                    for entry in step.body_observes
                )
            return MarginalizeStep(
                var_name=rename.get(step.var_name, step.var_name),
                class_size=step.class_size,
                probs_var=renamed_probs,
                over_obj=step.over_obj,
                over_objs=step.over_objs,
                body_ll_var=renamed_body_ll,
                body_observes=renamed_body_observes,
                reduction=step.reduction,
                line=step.line,
                col=step.col,
            )
        if isinstance(step, GroupedLatentInitStep):
            return GroupedLatentInitStep(
                latent_name=rename.get(step.latent_name, step.latent_name),
                class_size=step.class_size,
                line=step.line,
                col=step.col,
            )
        if isinstance(step, GroupedBodyObserveStep):
            return GroupedBodyObserveStep(
                response_var=rename.get(step.response_var, step.response_var),
                morphism=step.morphism,
                args=self._rename_args(step.args, value_subst, rename),
                index_set=self._rename_type(step.index_set, type_subst)
                if step.index_set is not None
                else None,
                index_var=rename.get(step.index_var, step.index_var),
                latent_name=rename.get(step.latent_name, step.latent_name),
                line=step.line,
                col=step.col,
            )
        raise CompileError(
            f"unsupported step kind in template body: {type(step).__name__}",
            getattr(step, "line", 0),
            getattr(step, "col", 0),
        )

    def _rename_let_expr(
        self,
        expr: LetExprNode,
        value_subst: dict[str, str | float],
        rename: dict[str, str],
    ) -> LetExprNode:
        """Apply parameter substitution + α-renaming inside a let RHS."""
        if isinstance(expr, LetExprVar):
            if expr.name in value_subst:
                val = value_subst[expr.name]
                if isinstance(val, str):
                    return LetExprVar(name=val, line=expr.line, col=expr.col)
                return LetExprLiteral(value=float(val), line=expr.line, col=expr.col)
            if expr.name in rename:
                return LetExprVar(name=rename[expr.name], line=expr.line, col=expr.col)
            return expr
        if isinstance(expr, LetExprLiteral):
            return expr
        if isinstance(expr, LetExprBinOp):
            return LetExprBinOp(
                op=expr.op,
                lhs=self._rename_let_expr(expr.lhs, value_subst, rename),
                rhs=self._rename_let_expr(expr.rhs, value_subst, rename),
                line=expr.line,
                col=expr.col,
            )
        if isinstance(expr, LetExprUnaryOp):
            return LetExprUnaryOp(
                op=expr.op,
                operand=self._rename_let_expr(expr.operand, value_subst, rename),
                line=expr.line,
                col=expr.col,
            )
        if isinstance(expr, LetExprCall):
            new_callee = value_subst.get(expr.callee, expr.callee)
            if not isinstance(new_callee, str):
                raise CompileError(
                    f"let-expression callee {expr.callee!r} substituted to "
                    f"non-string value {new_callee!r}",
                    expr.line,
                    expr.col,
                )
            return LetExprCall(
                callee=new_callee,
                args=tuple(
                    self._rename_let_expr(a, value_subst, rename) for a in expr.args
                ),
                line=expr.line,
                col=expr.col,
            )
        if isinstance(expr, LetExprIndex):
            new_arr = value_subst.get(expr.array, expr.array)
            if not isinstance(new_arr, str):
                raise CompileError(
                    f"let-expression array {expr.array!r} substituted to "
                    f"non-string value {new_arr!r}",
                    expr.line,
                    expr.col,
                )
            arr_name = (
                rename.get(new_arr, new_arr) if isinstance(new_arr, str) else new_arr
            )
            return LetExprIndex(
                array=arr_name,
                indices=tuple(
                    self._rename_let_expr(i, value_subst, rename) for i in expr.indices
                ),
                line=expr.line,
                col=expr.col,
            )
        if isinstance(expr, LetExprFactor):
            # Factor binders are local; their identifiers shouldn't
            # be alpha-renamed against the outer template's rename
            # dict.  Body and case values are recursively renamed.
            return LetExprFactor(
                binders=expr.binders,
                body=(
                    self._rename_let_expr(expr.body, value_subst, rename)
                    if expr.body is not None
                    else None
                ),
                cases=tuple(
                    LetFactorCase(
                        label=c.label,
                        value=self._rename_let_expr(c.value, value_subst, rename),
                        line=c.line,
                        col=c.col,
                    )
                    for c in expr.cases
                ),
            )
        return expr

    def _compile_morphism_call(self, expr: ExprMorphismCall):
        """Compile ``callee(arg1, arg2, …)`` — currently used to
        invoke a :class:`ContractionDecl` at a let-binding site.

        Resolves ``expr.callee`` against the registered
        contractions, validates that the argument count matches
        the contraction's expected arity, resolves each argument
        against the morphism scope, and runs the einsum-style
        contraction. Returns the result as an
        :class:`ObservedMorphism` with the contraction's declared
        domain and codomain.
        """
        from quivers.core.morphisms import ObservedMorphism

        contraction = self._contractions.get(expr.callee)
        if contraction is None:
            # Fall through to parametric-program template
            # invocation: ``let applied = p(f)`` is the existing
            # surface and should keep working when ``p`` is a
            # parametric program rather than a contraction.
            if expr.callee in self._program_templates:
                return self._compile_program_template_call(expr)
            raise CompileError(
                f"morphism-call: undefined {expr.callee!r}; "
                f"contractions: {sorted(self._contractions)}; "
                f"program templates: {sorted(self._program_templates)}",
                expr.line,
                expr.col,
            )
        if len(expr.args) != len(contraction.input_types):
            raise CompileError(
                f"contraction {expr.callee!r}: expected "
                f"{len(contraction.input_types)} arguments "
                f"({', '.join(n for n, _, _ in contraction.input_types)}), "
                f"got {len(expr.args)}",
                expr.line,
                expr.col,
            )
        resolved_morphs: list = []
        for arg_name, (param_name, exp_dom, exp_cod) in zip(
            expr.args, contraction.input_types
        ):
            if arg_name not in self._morphisms:
                raise CompileError(
                    f"contraction {expr.callee!r}: argument "
                    f"{arg_name!r} (for parameter {param_name!r}) "
                    f"is not a declared morphism",
                    expr.line,
                    expr.col,
                )
            morph = self._morphisms[arg_name]
            # Light shape check: matching numel for domain / codomain.
            if _numel_shape(morph.domain.shape) != _numel_shape(
                exp_dom.shape
            ) or _numel_shape(morph.codomain.shape) != _numel_shape(exp_cod.shape):
                raise CompileError(
                    f"contraction {expr.callee!r}: argument {arg_name!r} "
                    f"has shape {tuple(morph.domain.shape)} -> "
                    f"{tuple(morph.codomain.shape)} but parameter "
                    f"{param_name!r} declares {tuple(exp_dom.shape)} -> "
                    f"{tuple(exp_cod.shape)}",
                    expr.line,
                    expr.col,
                )
            resolved_morphs.append(morph)
        # Run the wiring.
        tensors = [m.tensor for m in resolved_morphs]
        result_tensor = contraction.wiring.apply(*tensors)
        return ObservedMorphism(
            contraction.domain,
            contraction.codomain,
            result_tensor,
            algebra=contraction.algebra,
        )

    def _compile_program_template_call(self, expr: ExprMorphismCall):
        """Instantiate a parametric program template at a let-binding
        site: ``let applied = p(f, …)`` substitutes the actual
        morphism / object / scalar arguments for the template's
        formal parameters, builds a synthetic non-parametric
        :class:`ProgramDecl`, and compiles it into a runtime
        :class:`MonadicProgram` morphism.

        Realises the dependent-kernel application
        :math:`\\Pi (p:P).\\ \\mathbf{Kern}(\\mathrm{dom}(p),\\, \\mathrm{cod}(p))`
        at the supplied section: the result is the concrete
        Kern-morphism the template denotes at those parameters,
        materialised as a standalone morphism rather than inlined
        into an enclosing program.
        """
        tmpl = self._program_templates[expr.callee]
        type_params = tmpl.type_params or ()
        args = expr.args
        if len(args) != len(type_params):
            raise CompileError(
                f"template {expr.callee!r} expects {len(type_params)} "
                f"arguments, got {len(args)}",
                expr.line,
                expr.col,
            )
        type_subst: dict[str, TypeExpr] = {}
        value_subst: dict[str, str | float] = {}
        for param, arg in zip(type_params, args):
            if isinstance(param, ObjectParam):
                if param.universe == "FinSet" and arg not in self._objects:
                    raise CompileError(
                        f"template {expr.callee!r}: parameter "
                        f"{param.name!r} : FinSet expects a finite-set "
                        f"object, but {arg!r} is not a declared object",
                        expr.line,
                        expr.col,
                    )
                if param.universe == "Space" and arg not in self._spaces:
                    raise CompileError(
                        f"template {expr.callee!r}: parameter "
                        f"{param.name!r} : Space expects a continuous "
                        f"space, but {arg!r} is not a declared space",
                        expr.line,
                        expr.col,
                    )
                if (
                    param.universe == "Object"
                    and arg not in self._objects
                    and arg not in self._spaces
                ):
                    raise CompileError(
                        f"template {expr.callee!r}: parameter "
                        f"{param.name!r} : Object expects a declared "
                        f"object or space, got {arg!r}",
                        expr.line,
                        expr.col,
                    )
                type_subst[param.name] = TypeName(
                    name=arg, line=expr.line, col=expr.col
                )
            elif isinstance(param, ScalarParam):
                value_subst[param.name] = arg
            elif isinstance(param, MorphismParam):
                if arg not in self._morphisms and arg not in self._program_templates:
                    raise CompileError(
                        f"template {expr.callee!r}: parameter "
                        f"{param.name!r}: morphism {arg!r} is not "
                        f"declared",
                        expr.line,
                        expr.col,
                    )
                value_subst[param.name] = arg
            else:
                raise CompileError(
                    f"template {expr.callee!r}: unknown parameter kind "
                    f"for {getattr(param, 'name', '?')!r}",
                    expr.line,
                    expr.col,
                )
        # Substitute parameters through the template body. We do not
        # α-rename locals here: the synthetic program owns its own
        # scope, so the template's local names (return-var, latents,
        # let-bindings) need no further refresh.
        expanded_body = self._expand_bind_steps(tmpl.draws)
        empty_rename: dict[str, str] = {}
        substituted_body = tuple(
            self._rename_step(step, type_subst, value_subst, empty_rename)
            for step in expanded_body
        )
        synth_domain = self._rename_type(tmpl.domain, type_subst)
        synth_codomain = self._rename_type(tmpl.codomain, type_subst)
        synth_name = f"__tmpl_call${expr.callee}${expr.line}${expr.col}"
        # Defensively dodge collisions if the same call site is
        # compiled twice for any reason.
        suffix = 0
        unique_name = synth_name
        while unique_name in self._morphisms:
            suffix += 1
            unique_name = f"{synth_name}#{suffix}"
        synth = ProgramDecl(
            name=unique_name,
            params=None,
            domain=synth_domain,
            codomain=synth_codomain,
            draws=substituted_body,
            return_vars=tmpl.return_vars,
            return_labels=tmpl.return_labels,
            effects=tmpl.effects,
            over_model=None,
            type_params=None,
            docs=(),
            line=expr.line,
            col=expr.col,
        )
        self._compile_program(synth)
        return self._morphisms.pop(unique_name)

    def _compile_contraction(self, decl: ContractionDecl) -> None:
        """Compile a ``contraction`` declaration into a callable
        registered under ``decl.name``.

        Categorically, a contraction is an n-ary operadic morphism:
        it takes ``len(decl.inputs)`` input morphisms, contracts
        them together under the named composition rule using the
        einsum-style wiring spec, and returns a fresh morphism
        with the declared ``domain -> codomain`` typing.

        The compiled callable accepts that many input morphisms (in
        the order declared in ``decl.inputs``) and returns an
        :class:`ObservedMorphism` whose tensor is the contraction
        result. The callable is registered in the morphism table
        so the user can invoke it like any other morphism:
        ``let out = op_apply(arg1, arg2, kernel)``.
        """
        if decl.name in self._morphisms or decl.name in self._program_templates:
            raise CompileError(
                f"contraction {decl.name!r} already declared as a morphism or program",
                decl.line,
                decl.col,
            )
        rule_name = decl.rule_name.lower()
        if rule_name not in _ALGEBRA_REGISTRY:
            raise CompileError(
                f"contraction {decl.name!r}: unknown rule "
                f"{decl.rule_name!r}; available: "
                f"{', '.join(sorted(_ALGEBRA_REGISTRY))}",
                decl.line,
                decl.col,
            )
        rule = _ALGEBRA_REGISTRY[rule_name]
        try:
            wiring = EinsumWiring(rule, decl.wiring_spec)
        except ValueError as exc:
            raise CompileError(
                f"contraction {decl.name!r}: invalid wiring "
                f"{decl.wiring_spec!r}: {exc}",
                decl.line,
                decl.col,
            ) from exc
        expected_arity = wiring.input_arity
        if expected_arity != len(decl.inputs):
            raise CompileError(
                f"contraction {decl.name!r}: wiring spec declares "
                f"{expected_arity} inputs but the parameter list "
                f"has {len(decl.inputs)}",
                decl.line,
                decl.col,
            )
        # Resolve domain / codomain object refs.
        domain_obj = self._resolve_type(decl.domain)
        codomain_obj = self._resolve_type(decl.codomain)
        # Resolve each input's domain / codomain for validation.
        input_types = [
            (
                inp.name,
                self._resolve_type(inp.input_domain),
                self._resolve_type(inp.input_codomain),
            )
            for inp in decl.inputs
        ]
        self._contractions[decl.name] = _CompiledContraction(
            name=decl.name,
            wiring=wiring,
            domain=domain_obj,
            codomain=codomain_obj,
            input_types=input_types,
            algebra=rule,
        )

    def _compile_program(self, decl: ProgramDecl) -> None:
        """Compile a monadic program block into a MonadicProgram.

        Parametric programs (those carrying ``type_params``) are not
        compiled into a runtime ``MonadicProgram`` directly. They
        denote a dependent kernel

        .. math::

            \\Pi (p_1 : P_1) \\ldots \\Pi (p_n : P_n).\\ \\mathbf{Kern}(\\mathrm{dom}(p),\\, \\mathrm{cod}(p))

        in the indexed family of Kleisli arrows over the parameter
        category, and are stored as templates. Each call site of a
        template (a ``draw v ~ template(args)`` step inside another
        program) is realised by substituting the actual arguments
        for the formal parameters and α-renaming all locally-bound
        latents under the call's binding name, then inlining the
        renamed body into the caller's step list. The freshness of
        latent names per call site is the syntactic shadow of the
        fact that distinct call sites contribute distinct factors
        to the parent's joint kernel.
        """
        if decl.type_params is not None:
            # Parametric program — store as a template; defer body
            # compilation until each call site instantiates it.
            if decl.name in self._morphisms or decl.name in self._program_templates:
                raise CompileError(
                    f"morphism {decl.name!r} already declared",
                    decl.line,
                    decl.col,
                )
            if decl.params is not None:
                raise CompileError(
                    f"parametric program {decl.name!r} cannot also take data parameters",
                    decl.line,
                    decl.col,
                )
            self._program_templates[decl.name] = decl
            return
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
        # First, expand the v0.5 unified surface (BindStep) into the
        # internal IR (DrawStep / PlateDrawStep / VectorisedObserveStep /
        # MarginalizeStep) that the rest of the compiler consumes.
        # The expansion translates each BindStep based on its mode +
        # index annotation, and inlines marginalize scopes.
        ir_draws = self._expand_bind_steps(decl.draws)
        # Effect-set verification: walk the expanded IR and check
        # that the declared `!` capability set is consistent with
        # the body's actual effect usage.
        self._verify_effects(decl, ir_draws)
        # Then expand parametric-template call sites by inlining the
        # substituted + α-renamed template body. This realises the
        # dependent-kernel application: each call site is a section
        # of the family Π(p:P).Kern(dom(p), cod(p)) at the supplied
        # arguments, contributing its own factors to the parent's
        # joint kernel.
        expanded_draws = self._expand_template_calls(ir_draws)
        steps: list[tuple] = []
        for step in expanded_draws:
            if isinstance(step, PlateDrawStep):
                # draw v : A -> B ~ Family(args).  By the natural iso
                # Kern(1, B^A) ≅ Kern(A, B), the plate variable IS a
                # Kern-morphism A → B; we realise it as a PlateDraw
                # whose codomain is the flat product space of
                # |A| copies of the per-row family's codomain.
                from quivers.continuous.plate import PlateDraw as _PlateDraw
                from quivers.continuous.spaces import Euclidean as _Euc

                idx_space = self._resolve_any_space(step.index)
                # The per-row codomain `B` is either a declared object /
                # space or an integer literal interpreted as
                # `Euclidean(N)` — the standard convention for
                # continuous per-row families.
                if (
                    isinstance(step.codomain, TypeName)
                    and step.codomain.name.isdigit()
                    and step.codomain.name not in self._objects
                ):
                    cod_space = _Euc(
                        name=f"_plate_codom_{step.name}",
                        dim=int(step.codomain.name),
                    )
                else:
                    cod_space = self._resolve_any_space(step.codomain)
                # Synthesize a DrawStep so we can reuse the inline /
                # family-registry resolution logic. The synthetic step
                # carries the plate's per-row codomain so the family
                # is built at the right dimensionality.
                _synth = DrawStep(
                    vars=(step.name,),
                    morphism=step.morphism,
                    args=step.args,
                    is_observed=False,
                    line=step.line,
                    col=step.col,
                )
                family, step_args = self._resolve_draw_morphism(
                    _synth, bound_vars, cod_space
                )
                plate = _PlateDraw(idx_space.size, family, domain=family.domain)
                if step.name in bound_vars:
                    raise CompileError(
                        f"variable {step.name!r} already bound in program",
                        step.line,
                        step.col,
                    )
                bound_vars[step.name] = plate.codomain
                steps.append(((step.name,), plate, step_args, False))
                continue
            if isinstance(step, GroupedLatentInitStep):
                # Initialise the latent's environment slot to
                # torch.arange(K) so any downstream let-step that
                # references the latent broadcasts across the class
                # axis.
                def _grouped_latent_init(
                    env: dict, _K: int = step.class_size
                ) -> torch.Tensor:
                    return torch.arange(_K, dtype=torch.long)

                if step.latent_name in bound_vars:
                    raise CompileError(
                        f"grouped marginalize: latent {step.latent_name!r} "
                        f"is already bound in scope",
                        step.line,
                        step.col,
                    )
                bound_vars[step.latent_name] = None
                steps.append(((step.latent_name,), None, _grouped_latent_init))
                continue
            if isinstance(step, GroupedBodyObserveStep):
                # The captured observe inside a grouped marginalize
                # block: compute the family's per-row log-likelihood
                # against the supplied response data, broadcasting
                # any (K,)-shaped parameters across the class axis,
                # and store the resulting (N, K) tensor at this
                # observe's dedicated env slot.  The surrounding
                # MarginalizeStep collects each slot, pairs it with
                # the observe's fibration, and scatter-sums each
                # contribution into the shared (|G|, K) accumulator.
                idx_space = self._resolve_any_space(step.index_set)
                _synth = DrawStep(
                    vars=(step.index_var,),
                    morphism=step.morphism,
                    args=step.args,
                    is_observed=True,
                    line=step.line,
                    col=step.col,
                )
                family, step_args = self._resolve_draw_morphism(
                    _synth, bound_vars, codomain
                )
                # Register the response variable so the runtime can
                # supply data via the observations dict.
                if step.response_var not in bound_vars:
                    bound_vars[step.response_var] = family.codomain
                resp_var = step.response_var
                ll_slot = step.ll_slot or step.latent_name
                num_rows = int(idx_space.size)

                def _captured_observe(
                    env: dict,
                    _family=family,
                    _args=step_args,
                    _resp=resp_var,
                    _slot=ll_slot,
                    _N=num_rows,
                ) -> torch.Tensor:
                    # Resolve theta from env using the same input-
                    # stacking rule as the standard runtime path.
                    if _args is None:
                        # No upstream variables to gather; the
                        # family takes the program input directly.
                        theta = env["_x_input"]
                    elif len(_args) == 1:
                        theta = env[_args[0]]
                    else:
                        parts = [env[a] for a in _args]
                        # Broadcast each part to a common shape so
                        # `stack` doesn't choke on mismatched event
                        # dimensions; this preserves the per-(N, K)
                        # broadcast pattern when one part is (N,)
                        # and another is (K,).
                        common_shape = torch.broadcast_shapes(*(p.shape for p in parts))
                        expanded = [p.expand(common_shape) for p in parts]
                        theta = torch.stack(expanded, dim=-1)
                    response = env[_resp]
                    ll = _family.log_prob(theta, response)
                    # Reshape to (N, K). When theta is shape (N,)
                    # (no K dependence) the family returns (N,) and
                    # we add a trailing axis so the latent slot is
                    # always 2-D. When theta is shape (N, K) the
                    # family returns (N, K) directly.
                    if ll.dim() == 1:
                        ll = ll.unsqueeze(-1)
                    elif ll.dim() == 0:
                        ll = ll.reshape(1, 1)
                    return ll

                bound_vars[ll_slot] = None
                steps.append(((ll_slot,), None, _captured_observe))
                continue
            if isinstance(step, VectorisedObserveStep):
                # observe r[n] ~ Family(args) for n in N — the batched-
                # likelihood kernel Φ → G_{≤1}(Φ) with score
                # ∏_n p_F(r_obs(n); θ(n, φ)). Realised as a
                # VectorisedObserve wrapping the per-row family;
                # threads through the existing _StepSpec(is_observed=True)
                # path. The response buffer is supplied at runtime
                # via the `observations` dict on the program.
                from quivers.continuous.plate import (
                    VectorisedObserve as _VectorisedObserve,
                )

                idx_space = self._resolve_any_space(step.index_set)
                _synth = DrawStep(
                    vars=(step.index_var,),
                    morphism=step.morphism,
                    args=step.args,
                    is_observed=True,
                    line=step.line,
                    col=step.col,
                )
                # Use the program's declared codomain as a fallback for
                # type inference on inline distributions; the family's
                # codomain is what actually matters.
                family, step_args = self._resolve_draw_morphism(
                    _synth, bound_vars, codomain
                )
                # Build a placeholder response of the right shape; the
                # actual values are supplied at fit time via the
                # `observations[response_var]` dict-entry. The buffer
                # only carries shape information here.
                resp_shape: tuple[int, ...]
                if hasattr(family.codomain, "dim"):
                    resp_shape = (idx_space.size, int(family.codomain.dim))
                else:
                    resp_shape = (idx_space.size,) + tuple(family.codomain.shape)
                import torch as _torch

                placeholder = _torch.zeros(*resp_shape)
                vec_obs = _VectorisedObserve(family, placeholder)
                # The step's response_var is the data column supplied
                # at fit time. We expose it as the bound name so the
                # runtime's observations[response_var] = data flow
                # automatically clamps the placeholder.
                if step.response_var not in bound_vars:
                    bound_vars[step.response_var] = family.codomain
                steps.append(((step.response_var,), vec_obs, step_args, True))
                continue
            if isinstance(step, MarginalizeStep):
                # marginalize v — pushforward G(π_{Φ\\C}). Realised as a
                # deterministic let-step that applies log-sum-exp across
                # the class axis of the named variable's per-class score
                # tensor. The runtime convention is that the body
                # populates `env[v]` with a per-class log-likelihood
                # tensor before this step executes.
                if step.var_name not in bound_vars:
                    raise CompileError(
                        f"marginalize: variable {step.var_name!r} not bound",
                        step.line,
                        step.col,
                    )
                target_var = step.var_name
                # Grouped form: `over G` (single plate) or
                # `over G * H` (product plate).  Per-observe
                # fibration data lives in ``body_observes``: each
                # entry is ``(ll_slot, fibration_var,
                # fibration_axes)``.  The codegen iterates the
                # entries, builds a parallel list of (ll, idx)
                # pairs from ``env``, and calls the multi-axis
                # form of ``marginalize_grouped``.
                has_grouping = step.over_obj is not None or step.over_objs is not None
                if has_grouping:
                    over_tuple: tuple[str, ...]
                    if step.over_objs is not None:
                        over_tuple = step.over_objs
                    else:
                        assert step.over_obj is not None
                        over_tuple = (step.over_obj,)
                    for over_name in over_tuple:
                        if over_name not in self._objects:
                            raise CompileError(
                                f"grouped marginalize: `over` object "
                                f"{over_name!r} is not a declared object",
                                step.line,
                                step.col,
                            )
                    if step.probs_var is None or step.probs_var not in bound_vars:
                        raise CompileError(
                            f"grouped marginalize: categorical prior "
                            f"{step.probs_var!r} is not bound in program scope",
                            step.line,
                            step.col,
                        )
                    if not step.body_observes:
                        raise CompileError(
                            "grouped marginalize: the body must contain "
                            "at least one observe step carrying its own "
                            "`via <idx>` clause",
                            step.line,
                            step.col,
                        )
                    is_product = len(over_tuple) > 1
                    # Verify each per-observe fibration's arity and
                    # bound-ness in scope.  Each entry is either
                    # single-fibration (``fibration_var`` set) or
                    # product-fibration (``fibration_axes`` set
                    # with length equal to ``len(over_tuple)``).
                    for entry in step.body_observes:
                        slot = entry.ll_slot
                        fib_var = entry.fibration_var
                        fib_axes = entry.fibration_axes
                        if fib_var is None and fib_axes is None:
                            # Nested-marginalize entry: the inner
                            # block has already performed its own
                            # scatter; the outer block consumes its
                            # already-(|G|, K)-shaped output
                            # directly with no further fibration.
                            continue
                        if is_product:
                            if fib_axes is None or len(fib_axes) != len(over_tuple):
                                raise CompileError(
                                    "grouped marginalize: observe writing "
                                    f"to slot {slot!r} must declare a "
                                    f"`via product(...)` clause of arity "
                                    f"{len(over_tuple)} matching the "
                                    "product grouping plate",
                                    step.line,
                                    step.col,
                                )
                            for axis_name in fib_axes:
                                if axis_name not in bound_vars:
                                    raise CompileError(
                                        f"grouped marginalize: per-observe "
                                        f"`via` axis {axis_name!r} is not "
                                        "bound in program scope",
                                        step.line,
                                        step.col,
                                    )
                        else:
                            if fib_axes is not None:
                                raise CompileError(
                                    "grouped marginalize: `via product(...)` "
                                    "on an observe requires the marginalize "
                                    "header to declare a product grouping "
                                    "plate `over G * H * ...`",
                                    step.line,
                                    step.col,
                                )
                            if fib_var not in bound_vars:
                                raise CompileError(
                                    f"grouped marginalize: per-observe "
                                    f"`via` variable {fib_var!r} is not "
                                    "bound in program scope",
                                    step.line,
                                    step.col,
                                )
                    group_sizes = tuple(
                        int(self._objects[name].size) for name in over_tuple
                    )
                    num_classes = step.class_size
                    probs_var = step.probs_var
                    reduction = step.reduction or "logsumexp"
                    observe_specs = tuple(
                        (entry.ll_slot, entry.fibration_var, entry.fibration_axes)
                        for entry in step.body_observes
                    )

                    def _marginalize_grouped_callable(
                        env: dict,
                        _specs: tuple[
                            tuple[str, str | None, tuple[str, ...] | None], ...
                        ] = observe_specs,
                        _probs: str = probs_var,
                        _sizes: tuple[int, ...] = group_sizes,
                        _k: int = num_classes,
                        _reduction: str = reduction,
                        _product: bool = is_product,
                    ) -> torch.Tensor:
                        # Collect per-observe (ll, idx) pairs from
                        # env, with the shape contract that each
                        # ll's trailing axis is the class axis.
                        ll_list: list[torch.Tensor] = []
                        idx_list: list[torch.Tensor | tuple[torch.Tensor, ...]] = []
                        for slot, fib_var, fib_axes in _specs:
                            ll = env[slot]
                            if ll.shape[-1] != _k:
                                raise ValueError(
                                    f"grouped marginalize: per-row "
                                    f"per-class log-likelihood at slot "
                                    f"{slot!r} must end with the class "
                                    f"axis of size {_k}; got "
                                    f"{tuple(ll.shape)}"
                                )
                            ll_list.append(ll)
                            if fib_axes is not None:
                                idx_tuple = []
                                for axis_name in fib_axes:
                                    idx = env[axis_name]
                                    if idx.dim() == 2 and idx.shape[-1] == 1:
                                        idx = idx.squeeze(-1)
                                    idx_tuple.append(idx.to(torch.long))
                                idx_list.append(tuple(idx_tuple))
                            elif fib_var is not None:
                                idx = env[fib_var]
                                if idx.dim() == 2 and idx.shape[-1] == 1:
                                    idx = idx.squeeze(-1)
                                idx_list.append(idx.to(torch.long))
                            else:
                                # Nested-marginalize entry: the
                                # inner block produced a
                                # (|G|, K)-shaped tensor; bypass
                                # scatter-add for this entry by
                                # using an identity fibration.
                                idx_list.append(
                                    torch.arange(
                                        int(ll.shape[0]),
                                        dtype=torch.long,
                                        device=ll.device,
                                    )
                                )
                        probs = env[_probs]
                        log_prior = torch.log(probs.clamp_min(1e-38))
                        result = marginalize_grouped(
                            ll_list,
                            idx_list,
                            log_prior,
                            _sizes if _product else _sizes[0],
                            reduction=_reduction,
                        )
                        # Outermost block returns a scalar; we wrap
                        # in length-1 so the surrounding joint
                        # accumulator can broadcast cleanly.
                        if result.dim() == 0:
                            return result.reshape(1)
                        return result

                    # In a nested-marginalize context, the outer
                    # block's _expand_bind_steps re-points
                    # ``body_ll_var`` at the *outer* latent name so
                    # the inner block's reduction output is bound
                    # there (where the outer block's runtime
                    # callable expects to find it). Otherwise we
                    # emit a fresh ``_marg_<var>`` slot.
                    marg_name = (
                        step.body_ll_var
                        if step.body_ll_var is not None
                        and step.body_ll_var != step.var_name
                        else f"_marg_{step.var_name}"
                    )
                    bound_vars[marg_name] = None
                    # 4-tuple with True flag = score step: the callable
                    # returns a log-density contribution that gets added
                    # to log_joint (not just bound to env).
                    steps.append(
                        ((marg_name,), None, _marginalize_grouped_callable, True)
                    )
                    continue

                def _marginalize_callable(
                    env: dict, _v: str = target_var
                ) -> torch.Tensor:
                    tensor = env[_v]
                    return torch.logsumexp(tensor, dim=-1)

                marg_name = (
                    step.body_ll_var
                    if step.body_ll_var is not None
                    and step.body_ll_var != step.var_name
                    else f"_marg_{step.var_name}"
                )
                bound_vars[marg_name] = None
                steps.append(((marg_name,), None, _marginalize_callable, True))
                continue
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
                    # Let-expressions inside a program body may
                    # reference compiled deductions by name (for
                    # `parse(D, ...)` calls), so we pass the
                    # compiler's deductions dict as the static
                    # `globals_` environment for variable
                    # resolution.
                    self._validate_let_expr_vars(step.value, bound_vars, step)
                    deductions_globals = dict(getattr(self, "_deductions", {}))
                    deductions_globals["__index_size__"] = self._resolve_index_size
                    compiled_fn = self._compile_let_expr(
                        step.value,
                        globals_=deductions_globals,
                    )
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
            effect_set=decl.effects,
        )
        # Posterior-block routing: `over M` programs go to the
        # posterior registry rather than the morphism registry.
        if decl.over_model is not None:
            if not hasattr(self, "_posteriors"):
                self._posteriors = {}
            if decl.over_model not in self._morphisms:
                raise CompileError(
                    f"posterior block 'over {decl.over_model}' references "
                    f"undefined model {decl.over_model!r}",
                    decl.line,
                    decl.col,
                )
            self._posteriors[decl.name] = prog
        else:
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
        elif family in (
            "Exponential",
            "HalfCauchy",
            "HalfNormal",
            "LogNormal",
            "Gamma",
        ):
            return PositiveReals(name=f"_{var_names[0]}", dim=1)
        elif family == "Dirichlet":
            # Inline Dirichlet's simplex dimension:
            #
            # * ``Dirichlet([a_1, …, a_K])`` (parser flattens the
            #   bracketed numeric sequence into K positional literal
            #   floats) → K-simplex.
            # * ``Dirichlet(alpha)`` with a single scalar literal →
            #   simplex dimension comes from the program's declared
            #   codomain (``dim`` for a ContinuousSpace,
            #   ``cardinality`` for a SetObject), defaulting to 2.
            sim_dim: int | None = None
            n_literals = sum(1 for a in args if isinstance(a, (int, float)))
            if n_literals >= 2:
                sim_dim = n_literals
            if sim_dim is None:
                for a in args:
                    if isinstance(a, (list, tuple)) and len(a) > 0:
                        sim_dim = len(a)
                        break
            if sim_dim is None and isinstance(program_codomain, ContinuousSpace):
                sim_dim = getattr(program_codomain, "dim", None)
            if sim_dim is None and isinstance(program_codomain, SetObject):
                # A discrete codomain of cardinality `k` indexes a
                # k-simplex of class probabilities.
                sim_dim = getattr(program_codomain, "cardinality", None)
            if sim_dim is None or sim_dim < 2:
                sim_dim = 2
            return Simplex(name=f"_{var_names[0]}", dim=sim_dim)
        else:
            return Euclidean(name=f"_{var_names[0]}", dim=1)

    def _validate_let_expr_vars(
        self, node: LetExprNode, bound_vars: dict[str, AnySpace | None], step: LetStep
    ) -> None:
        """Validate that all variables in a let expression are bound.

        The validator tolerates references to compiled deductions
        (via ``self._deductions``) and lambdas that bind their own
        parameter — both of these are resolved at runtime by the
        let-expression evaluator's `globals_` channel and its
        lambda-environment extension.

        Free variables that match none of the above are treated as
        *host-data references*: the value is expected to arrive at
        runtime through :func:`quivers.inference.condition`'s data
        dict (e.g. per-row index arrays for hierarchical regression).
        The runtime raises a clear ``KeyError`` if the data dict
        doesn't supply such a name.
        """
        deductions = getattr(self, "_deductions", {})

        # Inner walker carries a set of locally-bound names from
        # surrounding lambdas, so a lambda's `param` is treated as
        # in-scope inside its body.
        def _walk(node, locals_set: set[str]) -> None:
            if isinstance(node, LetExprVar):
                if node.name in bound_vars or node.name in deductions:
                    return
                if node.name in locals_set:
                    return
                # Otherwise the variable is a host-data reference; the
                # trace runtime resolves it against the conditioning
                # data dict at execution time.
                return
            if isinstance(node, LetExprBinOp):
                _walk(node.left, locals_set)
                _walk(node.right, locals_set)
            elif isinstance(node, LetExprUnaryOp):
                _walk(node.operand, locals_set)
            elif isinstance(node, LetExprCall):
                for arg in node.args:
                    _walk(arg, locals_set)
            elif isinstance(node, LetExprList):
                for item in node.items:
                    _walk(item, locals_set)
            elif isinstance(node, LetExprLambda):
                _walk(node.body, locals_set | {node.param})
            elif isinstance(node, LetExprMethodCall):
                _walk(node.receiver, locals_set)
                for arg in node.args:
                    _walk(arg, locals_set)
            elif isinstance(node, LetExprIndex):
                _walk(node.array, locals_set)
                for idx in node.indices:
                    _walk(idx, locals_set)
            elif isinstance(node, LetExprFactor):
                # Each binder's variable is in scope only inside
                # the factor's body / case values; the index type
                # expression itself is a TypeExpr, not a let-expr,
                # so we don't walk it for variable references.
                inner = locals_set | {b.var for b in node.binders}
                if node.body is not None:
                    _walk(node.body, inner)
                for case in node.cases:
                    _walk(case.value, inner)
            # LetExprLiteral, LetExprString carry no variables.

        _walk(node, set())

    @staticmethod
    def _compile_let_expr(
        node: LetExprNode,
        globals_: "dict[str, LetValue] | None" = None,
    ) -> Callable[[dict[str, "LetValue"]], "LetValue"]:
        """Compile a let expression tree into a callable.

        The returned callable takes an environment dict mapping
        names to Python values (tensors, strings, lists, lambdas,
        chart views, structured tuples, ...) and returns the
        expression's value. The let-sublanguage is a small typed
        lambda calculus over heterogeneous values; the runtime
        evaluator preserves autograd through all tensor
        operations.

        Supported node kinds:

        * :class:`LetExprLiteral` — numeric literal → tensor.
        * :class:`LetExprString` — string literal → Python str.
        * :class:`LetExprVar` — variable reference → env lookup.
        * :class:`LetExprBinOp` — arithmetic over tensors.
        * :class:`LetExprUnaryOp` — negation.
        * :class:`LetExprList` — list literal → Python list.
        * :class:`LetExprLambda` — closure over the let environment.
        * :class:`LetExprMethodCall` — dispatch on receiver type.
        * :class:`LetExprCall` — built-in or constructor mode.
        * :class:`LetExprIndex` — tensor gather.
        """
        if isinstance(node, LetExprLiteral):
            val = node.value

            def _literal(env: dict) -> torch.Tensor:
                for v in env.values():
                    if isinstance(v, torch.Tensor):
                        return torch.tensor(val, device=v.device)
                return torch.tensor(val)

            return _literal
        if isinstance(node, LetExprString):
            val = node.value

            def _string(env: dict) -> str:
                return val

            return _string
        if isinstance(node, LetExprVar):
            name = node.name
            globs = globals_ or {}
            constructors = globs.get("__constructors__", frozenset())

            def _var(env: dict):
                if name in env:
                    return env[name]
                if name in constructors:
                    return (name,)
                if name in globs and name != "__constructors__":
                    return globs[name]
                raise CompileError(f"undefined variable {name!r} in let expression")

            return _var
        if isinstance(node, LetExprList):
            item_fns = [
                _ProgramsMixin._compile_let_expr(it, globals_=globals_)
                for it in node.items
            ]

            def _list(env: dict) -> list:
                return [fn(env) for fn in item_fns]

            return _list
        if isinstance(node, LetExprLambda):
            param = node.param
            body_fn = _ProgramsMixin._compile_let_expr(node.body, globals_=globals_)

            def _lambda(env: dict):
                # Returns a Python callable closed over the let-env.
                def _closure(arg):
                    extended = dict(env)
                    extended[param] = arg
                    return body_fn(extended)

                return _closure

            return _lambda
        if isinstance(node, LetExprBinOp):
            left_fn = _ProgramsMixin._compile_let_expr(node.left, globals_=globals_)
            right_fn = _ProgramsMixin._compile_let_expr(node.right, globals_=globals_)
            op = node.op

            def _binop(env: dict) -> torch.Tensor:
                l = left_fn(env)
                r = right_fn(env)
                # Promote scalar / int values to tensors.
                if not isinstance(l, torch.Tensor):
                    l = torch.tensor(float(l))
                if not isinstance(r, torch.Tensor):
                    r = torch.tensor(float(r))
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
            inner_fn = _ProgramsMixin._compile_let_expr(node.operand, globals_=globals_)

            def _neg(env: dict):
                v = inner_fn(env)
                if isinstance(v, torch.Tensor):
                    return -v
                return -v

            return _neg
        if isinstance(node, LetExprMethodCall):
            recv_fn = _ProgramsMixin._compile_let_expr(node.receiver, globals_=globals_)
            method = node.method
            arg_fns = [
                _ProgramsMixin._compile_let_expr(a, globals_=globals_)
                for a in node.args
            ]

            def _method(env: dict):
                receiver = recv_fn(env)
                args = [fn(env) for fn in arg_fns]
                fn = getattr(receiver, method, None)
                if fn is None:
                    raise CompileError(
                        f"object {type(receiver).__name__!r} has no method {method!r}"
                    )
                return fn(*args)

            return _method
        if isinstance(node, LetExprCall):
            func_name = node.func
            arg_fns = [
                _ProgramsMixin._compile_let_expr(a, globals_=globals_)
                for a in node.args
            ]

            # Built-in tensor operations.  Limited to elementwise /
            # shape-preserving primitives; contractions, reductions
            # over named axes, and matrix products go through the
            # typed contraction-declaration surface (see § Operadic
            # contractions in docs/semantics/composition-rules.md).
            _TENSOR_BUILTINS = {
                "sigmoid": lambda a: torch.sigmoid(a),
                "exp": lambda a: torch.exp(a),
                "log": lambda a: torch.log(a),
                "abs": lambda a: torch.abs(a),
                "softplus": lambda a: torch.nn.functional.softplus(a),
                "cumsum": lambda a: torch.cumsum(a, dim=-1),
                "softmax": lambda a: torch.softmax(a, dim=-1),
                "log1p": lambda a: torch.log1p(a),
                "sqrt": lambda a: torch.sqrt(a),
                "neg": lambda a: -a,
            }

            def _call(env: dict):
                # Higher-order combinators come first; they consume
                # raw closure args without eager evaluation of the
                # lambda body.
                if func_name == "length":
                    val = arg_fns[0](env)
                    if isinstance(val, list):
                        return float(len(val))
                    if isinstance(val, torch.Tensor):
                        return float(val.shape[0])
                    if isinstance(val, tuple):
                        return float(len(val))
                    raise CompileError(
                        f"length() does not support {type(val).__name__}"
                    )
                if func_name in ("map", "filter"):
                    coll = arg_fns[0](env)
                    fn = arg_fns[1](env)
                    if func_name == "map":
                        return [fn(x) for x in coll]
                    return [x for x in coll if fn(x)]
                if func_name == "fold":
                    # fold(list, init, accumulator_lambda)
                    coll = arg_fns[0](env)
                    init = arg_fns[1](env)
                    fn = arg_fns[2](env)
                    acc = init
                    for x in coll:
                        # The lambda takes one arg (the current
                        # element); inner closures handle accumulation
                        # via additional lambda nesting:
                        #   fold(xs, 0, x -> acc_so_far -> acc + x)
                        # ... this is awkward. Two-argument folds
                        # work better; we expose `fold` as taking a
                        # lambda whose body is itself a lambda
                        # (curried), invoked here as `fn(x)(acc)`.
                        if callable(fn):
                            step = fn(acc)
                            if callable(step):
                                acc = step(x)
                            else:
                                acc = step
                        else:
                            raise CompileError("fold's accumulator must be a lambda")
                    return acc
                if func_name == "logsumexp_over":
                    # logsumexp_over(list, lambda x -> log_weight_x)
                    coll = arg_fns[0](env)
                    fn = arg_fns[1](env)
                    if not coll:
                        return torch.tensor(-float("inf"))
                    weights = []
                    for x in coll:
                        w = fn(x)
                        if not isinstance(w, torch.Tensor):
                            w = torch.tensor(float(w))
                        weights.append(w)
                    return torch.logsumexp(torch.stack(weights), dim=0)
                if func_name == "logsumexp":
                    # logsumexp(a, b, ...) over an explicit list of args
                    coll = [fn(env) for fn in arg_fns]
                    coll = [
                        torch.tensor(float(c)) if not isinstance(c, torch.Tensor) else c
                        for c in coll
                    ]
                    return torch.logsumexp(torch.stack(coll), dim=0)
                if func_name == "parse":
                    # parse(D, input) — invoke a registered deduction
                    # on an axiom list / input. The first arg is a
                    # let_var naming a compiled DeductionSystem; the
                    # remaining arg is the input to feed it.
                    # The runtime env carries a `__compiler__` key
                    # set by the program runner with the compiled
                    # deductions dict.
                    if len(arg_fns) != 2:
                        raise CompileError(
                            "parse() takes exactly two arguments: "
                            "deduction-name and input"
                        )
                    ded = arg_fns[0](env)
                    inp = arg_fns[1](env)
                    if hasattr(ded, "__call__"):
                        return ded(inp)
                    raise CompileError(
                        f"parse() first arg must be a DeductionSystem, "
                        f"got {type(ded).__name__}"
                    )
                # Standard scalar / tensor builtins.
                if func_name in _TENSOR_BUILTINS:
                    args = [fn(env) for fn in arg_fns]
                    return _TENSOR_BUILTINS[func_name](args[0])
                if func_name == "cholesky_quad_form":
                    args = [fn(env) for fn in arg_fns]
                    # cholesky_quad_form is a tensor builtin; its
                    # operands are guaranteed to be tensors by the
                    # surface contract.  Narrow for the type
                    # checker without runtime cost.
                    L_flat = cast(torch.Tensor, args[0])
                    scale = cast(torch.Tensor, args[1])
                    K = scale.shape[-1]
                    L = L_flat.reshape(*L_flat.shape[:-1], K, K)
                    mask = torch.tril(torch.ones(K, K, device=L.device, dtype=L.dtype))
                    L = L * mask
                    R = L @ L.transpose(-1, -2)
                    D = scale.unsqueeze(-1) * torch.eye(
                        K, device=L.device, dtype=L.dtype
                    )
                    cov = D @ R @ D
                    return cov.reshape(*cov.shape[:-2], K * K)
                # Constructor mode: build a tuple `(func_name, *args)`
                # only when `func_name` is in the user-declared
                # constructor set (passed via `globals_["__constructors__"]`).
                # The free term algebra over named constructor symbols
                # is thus fully under the user's control — no
                # identifier is silently treated as a constructor.
                constructors = (globals_ or {}).get("__constructors__", frozenset())
                if func_name in constructors:
                    args = [fn(env) for fn in arg_fns]
                    return (func_name, *args)
                raise CompileError(
                    f"unknown function {func_name!r} in let expression; "
                    f"declare it as a constructor (e.g., in a deduction's "
                    f"`atoms` block) or use a registered builtin"
                )

            return _call
        if isinstance(node, LetExprFactor):
            # Multi-axis factor: build a finite-domain-indexed
            # tensor by evaluating the body once per tuple of index
            # values.  The dual of `let_index` (LetExprIndex): the
            # left adjoint of indexing in the indexed-family
            # category over FinSet.
            globs = globals_ or {}
            resolver = globs.get("__index_size__")
            if resolver is None:
                raise CompileError(
                    "factor expression compiled without an index-size "
                    "resolver in the let-expression globals; this is a "
                    "compiler integration bug — factor expressions must "
                    "be compiled through `_ProgramsMixin._compile_program`",
                    0,
                    0,
                )
            # Resolve each binder's index type to an integer
            # cardinality at compile time.  Duplicate binder names
            # within a single factor are rejected.
            sizes: list[int] = []
            binder_names: list[str] = []
            for b in node.binders:
                if b.var in binder_names:
                    raise CompileError(
                        f"factor binder name {b.var!r} repeated; each "
                        f"binder must bind a distinct identifier",
                        b.line,
                        b.col,
                    )
                binder_names.append(b.var)
                sizes.append(resolver(b.index))

            if node.cases:
                # Pattern-match form: single-binder, one body per
                # integer label, labels must cover {0, ..., size-1}.
                if len(node.binders) != 1:
                    raise CompileError(
                        f"factor pattern-match form requires exactly one "
                        f"binder; got {len(node.binders)}",
                        node.binders[0].line,
                        node.binders[0].col,
                    )
                size = sizes[0]
                seen_labels: dict[int, "LetFactorCase"] = {}
                for case in node.cases:
                    if not (0 <= case.label < size):
                        raise CompileError(
                            f"factor case label {case.label} out of range "
                            f"[0, {size}); index has cardinality {size}",
                            case.line,
                            case.col,
                        )
                    if case.label in seen_labels:
                        raise CompileError(
                            f"factor case label {case.label} appears more than once",
                            case.line,
                            case.col,
                        )
                    seen_labels[case.label] = case
                missing = sorted(set(range(size)) - set(seen_labels))
                if missing:
                    raise CompileError(
                        f"factor pattern-match must cover every index "
                        f"in [0, {size}); missing labels: {missing!r}",
                        node.binders[0].line,
                        node.binders[0].col,
                    )
                # Compile each case's value once.  Each compiled
                # function takes the current env plus the bound
                # index variable and returns a tensor; the runtime
                # body iterates labels 0..size-1 in order.
                case_fns = [
                    _ProgramsMixin._compile_let_expr(
                        seen_labels[i].value, globals_=globals_
                    )
                    for i in range(size)
                ]
                binder_var = node.binders[0].var

                def _eval_pattern(env: dict) -> torch.Tensor:
                    pieces = []
                    for i, fn in enumerate(case_fns):
                        extended = {**env, binder_var: torch.tensor(i)}
                        pieces.append(cast(torch.Tensor, fn(extended)))
                    return torch.stack(pieces, dim=0)

                return _eval_pattern

            # Uniform body form: a single body expression, evaluated
            # once per tuple of index values from the Cartesian
            # product of the binders.
            if node.body is None:
                raise CompileError(
                    "factor expression has neither a uniform body nor "
                    "a pattern-match block",
                    node.binders[0].line,
                    node.binders[0].col,
                )
            body_fn = _ProgramsMixin._compile_let_expr(node.body, globals_=globals_)
            # Closure-capture the binder names and sizes for the
            # runtime nested loop.
            local_binders = tuple(binder_names)
            local_sizes = tuple(sizes)

            def _eval_uniform(env: dict) -> torch.Tensor:
                # Build a (size_1, ..., size_n, *body_shape) tensor
                # by evaluating body at each index tuple and
                # reshaping.  We iterate in lexicographic order and
                # reshape at the end so torch.stack only needs to
                # see a flat list of body tensors.
                flat: list[torch.Tensor] = []
                for tup in _cartesian_product(*(range(s) for s in local_sizes)):
                    extended = {
                        **env,
                        **{
                            name: torch.tensor(val)
                            for name, val in zip(local_binders, tup)
                        },
                    }
                    flat.append(cast(torch.Tensor, body_fn(extended)))
                stacked = torch.stack(flat, dim=0)
                return stacked.reshape(*local_sizes, *stacked.shape[1:])

            return _eval_uniform
        if isinstance(node, LetExprIndex):
            # Indexed gather along the leading axis of the array.
            # Realises the Kleisli pullback ι^* v = v ∘ ι for a finite
            # fibration ι : N → A and a plate variable v : A → B.
            arr_fn = _ProgramsMixin._compile_let_expr(node.array, globals_=globals_)
            idx_fns = [
                _ProgramsMixin._compile_let_expr(ix, globals_=globals_)
                for ix in node.indices
            ]

            def _index(env: dict) -> torch.Tensor:
                # Indexed-gather always operates on tensor-valued
                # arrays and tensor-valued indices; narrow for the
                # type checker without runtime cost.
                arr = cast(torch.Tensor, arr_fn(env))
                idx_tensors = [cast(torch.Tensor, fn(env)) for fn in idx_fns]
                long_idx = tuple(
                    ix.to(torch.long) if ix.dtype != torch.long else ix
                    for ix in idx_tensors
                )
                return arr[long_idx]

            return _index
        raise CompileError(f"unknown let expression node: {type(node).__name__}")

    def _compile_let(self, decl: LetDecl) -> None:
        """Compile a let-binding with optional where clause.

        The RHS is first classified by surface shape:

        * If it's an expression that denotes a transformation
          (bare reference to a registered trans singleton or
          let-bound trans, a constructor call against the
          transformation catalog, or ``t1 >>> t2``), the binding
          lands in :attr:`_transformations`.
        * Otherwise it's compiled as a morphism expression and
          lands in :attr:`_morphisms`.

        The two namespaces are disjoint: a name cannot be used as
        both a morphism and a transformation in the same module.
        """
        if hasattr(decl, "where") and decl.where:
            for where_decl in decl.where:
                self._compile_let(where_decl)
        if decl.name in self._morphisms or decl.name in self._transformations:
            raise CompileError(f"name {decl.name!r} already bound", decl.line, decl.col)
        if self._is_trans_expr(decl.expr):
            self._transformations[decl.name] = self._compile_trans_expr(decl.expr)
            return
        morph = self._compile_expr(decl.expr)
        self._morphisms[decl.name] = morph

    def _is_trans_expr(self, expr) -> bool:
        """Return True iff ``expr`` denotes a transformation value.

        The classification is *purely structural* — based on the
        expression's surface shape — and is the criterion the
        let-binding logic uses to choose between the morphism and
        transformation namespaces.  An :class:`ExprMorphismCall`
        whose callee is in :attr:`_trans_constructors` is a
        transformation; the same shape with a callee in
        :attr:`_contractions` is a morphism.
        """
        if isinstance(expr, ExprTransCompose):
            return True
        if isinstance(expr, ExprIdent):
            return (
                expr.name in self._trans_singletons
                or expr.name in self._transformations
                or expr.name in self._trans_constructors
            )
        if isinstance(expr, ExprMorphismCall):
            return expr.callee in self._trans_constructors
        return False
