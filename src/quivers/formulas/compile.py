"""Bidirectional :class:`didactic.api.Lens` from :class:`Formula` to a
QVR :class:`~quivers.dsl.ast_nodes.Module` AST.

The compilation from a formula to a QVR program is a panproto-style
*lens* whose complement is the strict subset of :class:`Formula`
fields that are not recoverable from the emitted
:class:`Module` — packaged as :class:`FormulaData`. The forward
direction produces ``(module, formula_data)`` where the structural
fields of the formula (which columns exist, intercept flag, random
effect group / slope structure, response identifier) are encoded as
QVR latent / let / observe shape and the un-encodable fields (per-row
data, original identifier names lost to :func:`_qvr_name`'s
non-alphanumeric → underscore substitution, the ``term`` / ``name``
presentation labels, the original formula string) ride in
:class:`FormulaData`. The backward direction recovers the structure
by calling :func:`_decode_module` on the target and fuses it with the
complement.

The lens never touches strings — the target is a :class:`Module`
that the existing :class:`quivers.dsl.compiler.Compiler` consumes
directly, identical in shape to one produced by
:func:`quivers.dsl.parser.parse`.

Emitted structure (one named scalar coefficient per design-matrix
*column*, matching the brms / lme4 canonical layout in which
``poly(x, 2)`` produces two coefficients ``poly(x, 2)_1`` and
``poly(x, 2)_2``):

* One ``object Resp : N`` declaration per response plate.
* One ``object G : K`` declaration per random-effect grouping
  factor (with ``K`` levels).
* For each fixed-design column ``c``: one scalar latent draw
  inside the program body, ``beta_c <- Normal(0, fixed_prior)``.
  The per-row covariate values for ``c`` flow in as a free
  variable via the host-data channel (``observations[c]``).
* For each random-effect group ``(slope | g)``: a
  :class:`HalfNormal` scale latent plus a per-level plate draw,
  with the per-row contribution as a plate-gather
  ``alpha_g[g_idx]`` (or ``beta_g_slope[g_idx] * slope`` for a
  random slope).  Multiple random slopes per group are emitted
  as independent random-effect terms (the uncorrelated /
  ``(... || g)`` semantics in lme4).
* One ``observe`` step closes the program with the family's
  observation kernel applied to the inverse-link of the linear
  predictor.

GetPut holds for every :class:`Formula` ``f``: ``backward(*forward(f)) == f``.
"""

from __future__ import annotations

from typing import Literal, Mapping

import didactic.api as dx
import numpy as np
import torch

from quivers.dsl.ast_nodes import (
    BindStep,
    ExportDecl,
    ExprIdent,
    LetExprBinOp,
    LetExprCall,
    LetExprIndex,
    LetExprLiteral,
    LetExprNode,
    LetExprVar,
    LetStep,
    Module,
    ObjectDecl,
    ProgramDecl,
    ProgramStep,
    Statement,
    TypeName,
)
from quivers.formulas.family import Family
from quivers.formulas.formula import (
    FixedColumn,
    Formula,
    FormulaData,
    RandomTerm,
    _qvr_name,
)


def _parse_prior_call(text: str) -> tuple[str, tuple[str | float, ...]]:
    """Split a brms-style prior template ``"Family(arg, arg, ...)"``
    into its family name and a tuple of argument tokens.  Numeric
    tokens become floats; identifier tokens stay as strings so they
    can refer to other latents in the emitted program.
    """
    text = text.strip()
    if "(" not in text or not text.endswith(")"):
        raise ValueError(
            f"compile_formula: prior {text!r} is not in the form "
            f"`Family(arg1, arg2, ...)`"
        )
    family, _, rest = text.partition("(")
    body = rest[:-1]
    args: list[str | float] = []
    if body.strip():
        for token in body.split(","):
            token = token.strip()
            try:
                args.append(float(token))
            except ValueError:
                args.append(token)
    return family.strip(), tuple(args)


def _draw(
    var: str,
    family: str,
    args: tuple[str | float, ...],
    *,
    index: TypeName | None = None,
    mode: Literal["sample", "score", "marginal"] = "sample",
) -> BindStep:
    return BindStep(
        vars=(var,),
        morphism=family,
        args=args,
        index=index,
        mode=mode,
    )


def _let(name: str, value: LetExprNode) -> LetStep:
    return LetStep(name=name, value=value)


def _var(name: str) -> LetExprVar:
    return LetExprVar(name=name)


def _add(*terms: LetExprNode) -> LetExprNode:
    if not terms:
        return LetExprLiteral(value=0.0)
    out = terms[0]
    for t in terms[1:]:
        out = LetExprBinOp(op="+", left=out, right=t)
    return out


def _mul(left: LetExprNode, right: LetExprNode) -> LetExprNode:
    return LetExprBinOp(op="*", left=left, right=right)


def _apply_link(eta: LetExprNode, link_name: str) -> LetExprNode:
    if link_name == "identity":
        return eta
    if link_name == "logit":
        return LetExprCall(func="sigmoid", args=(eta,))
    if link_name == "log":
        return LetExprCall(func="exp", args=(eta,))
    if link_name == "softmax":
        return LetExprCall(func="softmax", args=(eta,))
    if link_name == "inverse":
        return LetExprBinOp(op="/", left=LetExprLiteral(value=1.0), right=eta)
    raise ValueError(
        f"compile_formula: unsupported link {link_name!r}; choices are "
        f"identity, logit, log, softmax, inverse"
    )


def _decode_module(module: Module) -> dict:
    """Recover the structural part of a :class:`Formula` from the
    emitted :class:`Module`.

    Returns a dict carrying the fields the lens forward can deterministically
    produce from a :class:`Formula`:

    * ``n_obs`` — the ``Resp`` cardinality.
    * ``group_cardinalities`` — ``{qvr_group_name: K}`` for each
      ``object G : K`` declaration preceding the program block.
    * ``fixed_qvr_names`` — list of ``(qvr_name, is_intercept)`` in
      emission order, recovered from the latent ``intercept`` /
      ``beta_<qvr_name>`` declarations.
    * ``random_terms_qvr`` — list of ``(qvr_group_name, slope_qvr_or_intercept)``
      in emission order, recovered from the
      ``sigma_<g>_<slope>`` + ``alpha_<g>``/``beta_<g>_<slope>`` pattern.
    * ``response_qvr_name`` — the QVR-legal identifier of the
      observe step's target.
    * ``observe_family`` — the family name on the observe step.

    The decoder is intentionally narrow: it knows the canonical
    emission shape of :class:`FormulaToQVRModule` and recognises that
    shape only. It is not a general QVR parser.
    """
    n_obs: int | None = None
    group_cardinalities: dict[str, int] = {}
    program: ProgramDecl | None = None
    for stmt in module.statements:
        if isinstance(stmt, ObjectDecl):
            type_expr = stmt.type_expr
            if not isinstance(type_expr, TypeName):
                continue
            try:
                cardinality = int(type_expr.name)
            except ValueError:
                continue
            if stmt.name == "Resp":
                n_obs = cardinality
            else:
                group_cardinalities[stmt.name] = cardinality
        elif isinstance(stmt, ProgramDecl):
            program = stmt
            break

    if program is None or n_obs is None:
        raise ValueError(
            "_decode_module: module is not a formula-emitted program "
            "(missing Resp object or program declaration)"
        )

    # Walk the program steps in emission order. The lens emits, per
    # the layout in `_build_module`, the following pattern:
    #
    #   * One BindStep per fixed coefficient (`intercept` or
    #     `beta_<qvr_name>`).
    #   * For non-intercepts, a LetStep `<beta_name>_per_row =
    #     beta_name * <qvr_name>` immediately after the BindStep.
    #   * One BindStep `sigma_<g>_<slope>` per random-effect entry,
    #     followed by the plate draw (`alpha_<g>` or
    #     `beta_<g>_<slope>`) and a LetStep `<latent_var>_per_row`.
    #   * Aux-family BindSteps (one per aux param).
    #   * `let eta = ...`, `let mu = ...`.
    #   * A final BindStep with `mode="score"` and `index="Resp"`
    #     bearing the response qvr_name and the observe-family.
    fixed_qvr_names: list[tuple[str, bool]] = []
    random_terms_qvr: list[tuple[str, str]] = []
    response_qvr_name: str = ""
    observe_family: str = ""

    seen_letnames: set[str] = set()
    for step in program.draws:
        if isinstance(step, LetStep):
            seen_letnames.add(step.name)
            continue
        if not isinstance(step, BindStep):
            continue
        if step.mode == "score" and step.index is not None:
            response_qvr_name = step.vars[0]
            observe_family = step.morphism
            continue
        if len(step.vars) != 1:
            continue
        var = step.vars[0]
        # Fixed effects: `intercept` or `beta_<qvr_name>` with no
        # plate index. Random-effect sigmas are also unindexed, but
        # are followed by an indexed plate draw; distinguish by
        # checking the variable name pattern.
        if step.index is None:
            if var == "intercept":
                fixed_qvr_names.append(("", True))
            elif var.startswith("beta_") and not var.startswith("beta_"):
                # unreachable; placeholder to clarify next branch
                pass
            elif var.startswith("sigma_"):
                # random-effect scale; defer to the plate draw that
                # follows, which carries (group, slope) in its name.
                continue
            elif var.startswith("beta_"):
                qvr_name = var.removeprefix("beta_")
                fixed_qvr_names.append((qvr_name, False))
            else:
                # aux family parameter (e.g. `sigma`, `phi`, ...).
                continue
        else:
            # Indexed plate draw: this is the random-effect latent.
            # `alpha_<g>` ↔ intercept slope; `beta_<g>_<slope>` ↔
            # named slope.
            if not isinstance(step.index, TypeName):
                continue
            qgroup = step.index.name
            if var.startswith("alpha_"):
                random_terms_qvr.append((qgroup, "Intercept"))
            elif var.startswith("beta_"):
                # Strip `beta_<qgroup>_` to isolate the slope qvr_name.
                prefix = f"beta_{qgroup}_"
                if var.startswith(prefix):
                    slope_qvr = var.removeprefix(prefix)
                    random_terms_qvr.append((qgroup, slope_qvr))

    return {
        "n_obs": n_obs,
        "group_cardinalities": group_cardinalities,
        "fixed_qvr_names": fixed_qvr_names,
        "random_terms_qvr": random_terms_qvr,
        "response_qvr_name": response_qvr_name,
        "observe_family": observe_family,
    }


class FormulaToQVRModule(dx.Lens[Formula, Module, FormulaData]):
    """Translate a :class:`Formula` to a QVR :class:`Module` AST.

    A typed :class:`didactic.api.Lens` whose complement is a
    :class:`FormulaData` carrier: just the fields of the source
    :class:`Formula` that are *not* recoverable from the emitted
    :class:`Module`. The per-row data arrays, the original (pre-
    :func:`_qvr_name`) identifiers, the per-column ``term`` / ``name``
    presentation labels, and the original formula string travel in
    the complement; everything else (which columns there are, the
    intercept / random-term structure, the family choice) is decoded
    back out of the Module by :func:`_decode_module`.

    Parameters
    ----------
    family : Family
        Response family from :data:`quivers.formulas.families`.
    fixed_prior : str
        Default prior for fixed-effect coefficients, in the surface
        form ``"Family(arg, arg, ...)"``; numeric args become floats,
        identifier args stay as variable references in the emitted
        program.
    random_scale_prior : str
        Default prior for random-effect scale parameters.
    user_priors : Mapping[str, str]
        Per-name prior overrides keyed by the latent's variable
        name in the emitted module.

    Notes
    -----
    GetPut: :meth:`backward` ``(forward(f))`` ``=`` ``f`` for every
    :class:`Formula` ``f``. PutGet holds on pairs ``(t, c)`` for
    which ``t`` is in the image of :meth:`forward` and ``c`` is the
    corresponding :class:`FormulaData`.
    """

    def __init__(
        self,
        family: Family,
        *,
        fixed_prior: str = "Normal(0.0, 5.0)",
        random_scale_prior: str = "HalfNormal(1.0)",
        user_priors: Mapping[str, str] | None = None,
    ) -> None:
        self._family = family
        self._fixed_prior = fixed_prior
        self._random_scale_prior = random_scale_prior
        self._user_priors: Mapping[str, str] = dict(user_priors or {})

    def forward(self, formula: Formula, /) -> tuple[Module, FormulaData]:
        module = self._build_module(formula)
        complement = FormulaData(
            formula=formula.formula,
            response_name=formula.response_name,
            response_values=formula.response_values,
            fixed_column_names={
                col.qvr_name: (col.term, col.name) for col in formula.fixed_columns
            },
            fixed_column_data={
                col.qvr_name: col.data
                for col in formula.fixed_columns
                if not col.is_intercept
            },
            group_original_names={_qvr_name(g): g for g in formula.group_levels.keys()},
            group_levels=dict(formula.group_levels),
            group_indices=dict(formula.group_indices),
        )
        return module, complement

    def backward(self, target: Module, complement: FormulaData, /) -> Formula:
        decoded = _decode_module(target)
        fixed_column_names = complement.fixed_column_names
        fixed_column_data = complement.fixed_column_data

        fixed_columns: list[FixedColumn] = []
        n_obs = decoded["n_obs"]
        ones = np.ones(n_obs, dtype=float)
        for qvr_name, is_intercept in decoded["fixed_qvr_names"]:
            if is_intercept:
                term, name = fixed_column_names.get(
                    "intercept", ("Intercept", "Intercept")
                )
                fixed_columns.append(
                    FixedColumn(
                        term=term,
                        name=name,
                        qvr_name="intercept",
                        is_intercept=True,
                        data=ones,
                    )
                )
            else:
                term, name = fixed_column_names.get(qvr_name, (qvr_name, qvr_name))
                fixed_columns.append(
                    FixedColumn(
                        term=term,
                        name=name,
                        qvr_name=qvr_name,
                        is_intercept=False,
                        data=fixed_column_data[qvr_name],
                    )
                )

        random_terms: list[RandomTerm] = []
        for qgroup, slope_qvr in decoded["random_terms_qvr"]:
            group = complement.group_original_names.get(qgroup, qgroup)
            if slope_qvr == "Intercept":
                random_terms.append(RandomTerm(slope="Intercept", group=group))
            else:
                # Recover the original slope name by inverting via the
                # presentation map when available; otherwise pass the
                # qvr-name through.
                term_name_pairs = {
                    qvr: (term, name)
                    for qvr, (term, name) in fixed_column_names.items()
                }
                _, original_slope = term_name_pairs.get(
                    slope_qvr, (slope_qvr, slope_qvr)
                )
                random_terms.append(RandomTerm(slope=original_slope, group=group))

        return Formula(
            formula=complement.formula,
            response_name=complement.response_name,
            fixed_columns=tuple(fixed_columns),
            random_terms=tuple(random_terms),
            response_values=complement.response_values,
            group_levels=complement.group_levels,
            group_indices=complement.group_indices,
        )

    def fixed_column_observations(self, formula: Formula) -> dict[str, torch.Tensor]:
        """Per-column free-variable bindings for the host-data
        channel.  One entry per non-intercept fixed column, shape
        ``(N,)``.
        """
        obs: dict[str, torch.Tensor] = {}
        for col in formula.fixed_columns:
            if col.is_intercept:
                continue
            obs[col.qvr_name] = torch.as_tensor(col.data.copy(), dtype=torch.float32)
        return obs

    def _build_module(self, formula: Formula) -> Module:
        statements: list[Statement] = []
        n_obs = formula.response_values.shape[0]
        statements.append(
            ObjectDecl(name="Resp", type_expr=TypeName(name=str(int(n_obs))))
        )
        seen_groups: set[str] = set()
        for term in formula.random_terms:
            group = term.group
            qgroup = _qvr_name(group)
            if qgroup in seen_groups:
                continue
            seen_groups.add(qgroup)
            levels = formula.group_levels[group]
            statements.append(
                ObjectDecl(
                    name=qgroup,
                    type_expr=TypeName(name=str(len(levels))),
                )
            )

        program_steps: list[ProgramStep] = []
        linear_terms: list[LetExprNode] = []

        # Fixed effects: one scalar latent per design-matrix column.
        for col in formula.fixed_columns:
            beta_name = "intercept" if col.is_intercept else f"beta_{col.qvr_name}"
            prior_text = self._user_priors.get(beta_name, self._fixed_prior)
            family_name, args = _parse_prior_call(prior_text)
            program_steps.append(_draw(beta_name, family_name, args))
            if col.is_intercept:
                linear_terms.append(_var(beta_name))
            else:
                contrib_name = f"{beta_name}_per_row"
                program_steps.append(
                    _let(contrib_name, _mul(_var(beta_name), _var(col.qvr_name)))
                )
                linear_terms.append(_var(contrib_name))

        # Random effects.  Each (slope | g) entry emits its own scale
        # latent + per-level plate draw + per-row gather; multiple
        # slopes per group are independent (uncorrelated; matches
        # lme4's `(... || g)` semantics).
        for term in formula.random_terms:
            group = term.group
            qgroup = _qvr_name(group)
            slope = term.slope
            qslope = _qvr_name(slope)
            sigma_var = f"sigma_{qgroup}_{qslope}"
            sigma_prior_text = self._user_priors.get(
                sigma_var, self._random_scale_prior
            )
            sf_family, sf_args = _parse_prior_call(sigma_prior_text)
            program_steps.append(_draw(sigma_var, sf_family, sf_args))
            if slope == "Intercept":
                latent_var = f"alpha_{qgroup}"
            else:
                latent_var = f"beta_{qgroup}_{qslope}"
            program_steps.append(
                _draw(
                    latent_var,
                    "Normal",
                    (0.0, sigma_var),
                    index=TypeName(name=qgroup),
                )
            )
            contrib_name = f"{latent_var}_per_row"
            idx_name = f"{qgroup}_idx"
            gather = LetExprIndex(array=_var(latent_var), indices=(_var(idx_name),))
            if slope == "Intercept":
                program_steps.append(_let(contrib_name, gather))
            else:
                program_steps.append(_let(contrib_name, _mul(gather, _var(qslope))))
            linear_terms.append(_var(contrib_name))

        # Auxiliary family parameters.
        for aux in self._family.aux_params:
            prior = self._user_priors.get(aux.name, aux.prior)
            family_name, args = _parse_prior_call(prior)
            program_steps.append(_draw(aux.name, family_name, args))

        if not linear_terms:
            eta_expr: LetExprNode = LetExprLiteral(value=0.0)
        else:
            eta_expr = _add(*linear_terms)
        program_steps.append(_let("eta", eta_expr))
        mu_expr = _apply_link(_var("eta"), self._family.location_link.name)
        program_steps.append(_let("mu", mu_expr))

        obs_args: tuple[str | float, ...] = ("mu",) + tuple(
            self._family.extra_observe_args
        )
        program_steps.append(
            _draw(
                _qvr_name(formula.response_name),
                self._family.observe_family,
                obs_args,
                index=TypeName(name="Resp"),
                mode="score",
            )
        )

        program_decl = ProgramDecl(
            name="model",
            params=None,
            domain=TypeName(name="Resp"),
            codomain=TypeName(name="Resp"),
            draws=tuple(program_steps),
            return_vars=(_qvr_name(formula.response_name),),
        )
        statements.append(program_decl)
        statements.append(ExportDecl(expr=ExprIdent(name="model")))
        return Module(statements=tuple(statements))
