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

Emitted structure (one named scalar coefficient per fixed-effect
term, matching the brms / lme4 canonical layout):

* One ``object Resp : N`` declaration per response plate.
* One ``object G : K`` declaration per random-effect grouping
  factor (with ``K`` levels).
* For each fixed term ``t``: one scalar latent draw inside the
  program body, ``beta_t <- Normal(0, fixed_prior)``.  The
  per-row covariate column for ``t`` flows in as a free variable
  via the host-data channel (``observations[t]``).
* For each random-effect group ``(slope | g)``: a
  :class:`HalfNormal` scale latent plus a per-level plate draw,
  with the per-row contribution as a plate-gather
  ``alpha_g[g_idx]`` (or ``beta_g_slope[g_idx] * slope`` for a
  random slope).
* One ``observe`` step closes the program with the family's
  observation kernel applied to the inverse-link of the linear
  predictor.

Carrying the formula in the complement gives the lens the exact
information panproto's :meth:`backward` needs to satisfy the
GetPut law ``backward(*forward(f)) == f``.  PutGet is the trivial
identity on ``(module, complement)`` pairs.
"""

from __future__ import annotations

from typing import Mapping

import didactic.api as dx
import torch

from typing import Literal

from quivers.dsl.ast_nodes import (
    BindStep,
    ExportDecl,
    ExprIdent,
    LetDecl,
    LetExprBinOp,
    LetExprCall,
    LetExprIndex,
    LetExprLiteral,
    LetExprNode,
    LetExprVar,
    LetStep,
    Module,
    MorphismDecl,
    ObjectDecl,
    ProgramDecl,
    ProgramStep,
    SpaceDecl,
    TypeName,
)
from quivers.formulas.family import Family
from quivers.formulas.formula import Formula


def _parse_prior_call(text: str) -> tuple[str, tuple[str | float, ...]]:
    """Split a brms-style prior string ``"Family(arg, arg, ...)"`` into
    its family name and a tuple of argument tokens.  Numeric tokens
    become floats; identifier tokens stay as strings (so they can
    refer to other latents in the emitted program).
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
        Default prior for fixed-effect coefficients.  Surface form
        ``"Family(arg, arg, ...)"``; numeric args become floats,
        identifier args stay as variable references in the emitted
        program.
    random_scale_prior : str
        Default prior for random-effect scale parameters.
    user_priors : Mapping[str, str]
        Per-name prior overrides keyed by the latent's variable
        name in the emitted module.

    Notes
    -----
    Forward returns ``(module, formula)`` so the complement carries
    the original formula verbatim; backward returns the complement.
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

    def constant_data(self, formula: Formula) -> dict[str, torch.Tensor]:
        """Empty: the term-by-term layout consumes per-column tensors
        through the inference-time observations dict (host-data
        channel), not the compile-time ``data=`` kwarg.
        """
        return {}

    def fixed_column_observations(self, formula: Formula) -> dict[str, torch.Tensor]:
        """Per-column free-variable bindings for the host-data
        channel.  One entry per non-intercept fixed term, shape
        ``(N,)``.
        """
        obs: dict[str, torch.Tensor] = {}
        col_index = {name: i for i, name in enumerate(formula.fixed_term_names)}
        for name in formula.fixed_term_names:
            if name == "Intercept":
                continue
            col = formula.fixed_design[:, col_index[name]]
            obs[_qvr_name(name)] = torch.as_tensor(col, dtype=torch.float32)
        return obs

    def _build_module(self, formula: Formula) -> Module:
        statements: list[
            ObjectDecl | SpaceDecl | MorphismDecl | LetDecl | ProgramDecl | ExportDecl
        ] = []
        n_obs = int(formula.fixed_design.shape[0])
        statements.append(ObjectDecl(name="Resp", type_expr=TypeName(name=str(n_obs))))
        seen_groups: set[str] = set()
        for term in formula.random_terms:
            group = term.group
            if group in seen_groups:
                continue
            seen_groups.add(group)
            levels = formula.group_levels[group]
            statements.append(
                ObjectDecl(
                    name=_qvr_name(group),
                    type_expr=TypeName(name=str(len(levels))),
                )
            )

        program_steps: list[ProgramStep] = []
        linear_terms: list[LetExprNode] = []

        # Fixed effects: one scalar latent per term + a let computing
        # its per-row contribution.  Each non-intercept term's column
        # flows in via the host-data channel as a free variable named
        # `qvr_name(term)`.
        for term in formula.fixed_term_names:
            qname = _qvr_name(term)
            beta_name = f"beta_{qname}" if term != "Intercept" else "intercept"
            prior_text = self._user_priors.get(beta_name, self._fixed_prior)
            family_name, args = _parse_prior_call(prior_text)
            program_steps.append(_draw(beta_name, family_name, args, mode="sample"))
            if term == "Intercept":
                linear_terms.append(_var(beta_name))
            else:
                contrib_name = f"{beta_name}_per_row"
                program_steps.append(
                    _let(contrib_name, _mul(_var(beta_name), _var(qname)))
                )
                linear_terms.append(_var(contrib_name))

        # Random effects.
        for term in formula.random_terms:
            group = term.group
            qgroup = _qvr_name(group)
            slope = term.slope
            sigma_var = f"sigma_{qgroup}_{slope}"
            sigma_prior = self._user_priors.get(sigma_var, self._random_scale_prior)
            sf_family, sf_args = _parse_prior_call(sigma_prior)
            program_steps.append(_draw(sigma_var, sf_family, sf_args, mode="sample"))
            if slope == "Intercept":
                latent_var = f"alpha_{qgroup}"
            else:
                latent_var = f"beta_{qgroup}_{slope}"
            program_steps.append(
                _draw(
                    latent_var,
                    "Normal",
                    (0.0, sigma_var),
                    index=TypeName(name=qgroup),
                    mode="sample",
                )
            )
            contrib_name = f"{latent_var}_per_row"
            idx_name = f"{qgroup}_idx"
            gather = LetExprIndex(array=_var(latent_var), indices=(_var(idx_name),))
            if slope == "Intercept":
                program_steps.append(_let(contrib_name, gather))
            else:
                program_steps.append(
                    _let(contrib_name, _mul(gather, _var(_qvr_name(slope))))
                )
            linear_terms.append(_var(contrib_name))

        # Auxiliary family parameters (sigma, disp, phi, nu, ...).
        for aux in self._family.aux_params:
            prior = self._user_priors.get(aux.name, aux.prior)
            family_name, args = _parse_prior_call(prior)
            program_steps.append(_draw(aux.name, family_name, args, mode="sample"))

        # Linear predictor + inverse link.
        if not linear_terms:
            eta_expr: LetExprNode = LetExprLiteral(value=0.0)
        else:
            eta_expr = _add(*linear_terms)
        program_steps.append(_let("eta", eta_expr))
        mu_expr = _apply_link(_var("eta"), self._family.location_link.name)
        program_steps.append(_let("mu", mu_expr))

        # Response observation step.
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


def _qvr_name(raw: str) -> str:
    """Normalize a column / term name into a legal QVR identifier."""
    cleaned = "".join(c if c.isalnum() or c == "_" else "_" for c in raw)
    if not cleaned or cleaned[0].isdigit():
        cleaned = "_" + cleaned
    return cleaned
