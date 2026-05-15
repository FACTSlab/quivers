"""Bidirectional :class:`didactic.api.Lens` from :class:`Formula` to a
QVR :class:`~quivers.dsl.ast_nodes.Module` AST.

The compilation from a formula to a QVR program is a panproto-style
*lens*: the forward direction projects the formula onto its QVR
encoding as a typed AST; the complement holds the original
:class:`Formula` so the backward direction recovers it verbatim.
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

The lens satisfies the GetPut law trivially (forward returns
``(module, formula)``; backward returns the complement); PutGet
is the identity on ``(module, complement)`` pairs.
"""

from __future__ import annotations

from typing import Literal, Mapping

import didactic.api as dx
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
from quivers.formulas.formula import Formula, _qvr_name


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


class FormulaToQVRModule(dx.Lens[Formula, Module, Formula]):
    """Translate a :class:`Formula` to a QVR :class:`Module` AST.

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
    Forward returns ``(module, formula)``; backward returns the
    complement so :meth:`backward(*forward(f)) == f`.
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

    def forward(self, formula: Formula, /) -> tuple[Module, Formula]:
        return self._build_module(formula), formula

    def backward(self, target: Module, complement: Formula, /) -> Formula:
        return complement

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
