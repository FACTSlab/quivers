"""User-facing :func:`fit` entry point and the
:class:`BayesianFit` result wrapper.

The compilation path is fully AST-driven: the formula lens emits a
:class:`~quivers.dsl.ast_nodes.Module`, the existing
:class:`quivers.dsl.compiler.Compiler` consumes it directly (no
source-string round-trip), and inference runs on the resulting
:class:`~quivers.continuous.programs.MonadicProgram`.  Source text
is generated only when the user requests it via
:func:`formula_to_qvr` or :meth:`BayesianFit.dump_qvr`, in which
case :func:`quivers.dsl.emit.module_to_source` walks the same AST
to produce canonical ``.qvr`` source.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Mapping

import didactic.api as dx
import torch
from narwhals.typing import IntoDataFrame

from quivers.continuous.programs import MonadicProgram
from quivers.dsl import Compiler
from quivers.dsl.emit import module_to_source
from quivers.formulas.compile import FormulaToQVRModule
from quivers.formulas.family import Family, families
from quivers.formulas.formula import Formula, _qvr_name, formula_from_data
from quivers.inference import (
    ELBO,
    HMCKernel,
    MCMC,
    NUTSKernel,
    SVI,
    AutoNormalGuide,
)
from quivers.inference.guides.base import Guide
from quivers.inference.mcmc.driver import MCMCResult


class BayesianFit(dx.Model):
    """A fitted Bayesian regression: the compiled program, the
    parsed formula, the family, the user-supplied data, and the
    posterior samples.

    Attributes
    ----------
    formula : Formula
        Parsed formula IR.
    family : Family
        Response family used at compile time.
    program : MonadicProgram
        The compiled program.
    posterior : MCMCResult or Guide
        :class:`~quivers.inference.MCMCResult` from NUTS / HMC, or
        a :class:`~quivers.inference.guides.base.Guide` from SVI.
    observations : Mapping[str, torch.Tensor]
        Inference-time observations dict (response + per-column
        covariates + per-group plate indices).
    """

    formula: Formula = dx.field(opaque=True)
    family: Family
    program: MonadicProgram = dx.field(opaque=True)
    posterior: MCMCResult | Guide = dx.field(opaque=True)
    observations: Mapping[str, torch.Tensor] = dx.field(
        default_factory=dict, opaque=True
    )
    reparameterize: Literal["centered", "noncentered"] = "noncentered"

    @property
    def qvr_source(self) -> str:
        """Lazily emit the AST-equivalent ``.qvr`` source for display."""
        lens = FormulaToQVRModule(self.family, reparameterize=self.reparameterize)
        module, _ = lens.forward(self.formula)
        return module_to_source(module)

    def dump_qvr(self, path: str | Path) -> Path:
        """Write the AST-equivalent ``.qvr`` source to ``path`` and
        return the resulting :class:`Path`.
        """
        out = Path(path)
        out.write_text(self.qvr_source)
        return out


def fit(
    formula: str,
    *,
    data: IntoDataFrame,
    family: str | Family = "gaussian",
    method: Literal["nuts", "hmc", "svi"] = "nuts",
    num_warmup: int = 500,
    num_samples: int = 1000,
    num_chains: int = 4,
    fixed_prior: str = "Normal(0.0, 5.0)",
    random_scale_prior: str = "HalfNormal(1.0)",
    priors: Mapping[str, str] | None = None,
    guide: type | None = None,
    reparameterize: Literal["centered", "noncentered"] = "noncentered",
    seed: int = 0,
) -> BayesianFit:
    """Compile a brms-style formula, fit it, and return the result.

    See :mod:`quivers.formulas` for surface details.  This entry
    point composes :func:`formula_from_data`, :class:`FormulaToQVRModule`,
    :class:`Compiler`, and the inference layer in one call.
    """
    if isinstance(family, str):
        if family not in families:
            raise ValueError(
                f"fit: unknown family {family!r}; choices are {sorted(families)}"
            )
        family_obj = families[family]
    else:
        family_obj = family

    parsed = formula_from_data(formula, data)
    lens = FormulaToQVRModule(
        family_obj,
        fixed_prior=fixed_prior,
        random_scale_prior=random_scale_prior,
        user_priors=priors,
    )
    module, _ = lens.forward(parsed)
    compiler = Compiler(module)
    program_runtime = compiler.compile()
    morphism = program_runtime.morphism
    if not isinstance(morphism, MonadicProgram):
        raise TypeError(
            f"fit: compiled morphism has type "
            f"{type(morphism).__name__}, expected MonadicProgram"
        )
    program = morphism

    observations: dict[str, torch.Tensor] = {}
    observations.update(lens.fixed_column_observations(parsed))
    response_name = parsed.response_name
    observations[response_name] = torch.as_tensor(
        parsed.response_values.copy(), dtype=torch.float32
    ).reshape(-1)
    for group, codes in parsed.group_indices.items():
        observations[f"{_qvr_name(group)}_idx"] = torch.as_tensor(
            list(codes), dtype=torch.long
        )

    torch.manual_seed(seed)
    if method == "svi":
        posterior = _fit_svi(program, observations, num_samples, guide_cls=guide)
    else:
        posterior = _fit_mcmc(
            program,
            observations,
            sampler=method,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
        )

    return BayesianFit(
        formula=parsed,
        family=family_obj,
        program=program,
        posterior=posterior,
        observations=observations,
        reparameterize=reparameterize,
    )


def formula_to_qvr(
    formula: str,
    *,
    data: IntoDataFrame,
    family: str | Family = "gaussian",
    fixed_prior: str = "Normal(0.0, 5.0)",
    random_scale_prior: str = "HalfNormal(1.0)",
    priors: Mapping[str, str] | None = None,
    reparameterize: Literal["centered", "noncentered"] = "noncentered",
    path: str | Path | None = None,
) -> str:
    """Emit ``.qvr`` source for a brms-style formula without fitting.

    Builds the formula AST → QVR Module via :class:`FormulaToQVRModule`,
    then serialises the module via :func:`quivers.dsl.emit.module_to_source`.
    Optionally writes the result to ``path``.
    """
    if isinstance(family, str):
        if family not in families:
            raise ValueError(
                f"formula_to_qvr: unknown family {family!r}; choices are "
                f"{sorted(families)}"
            )
        family_obj = families[family]
    else:
        family_obj = family
    parsed = formula_from_data(formula, data)
    lens = FormulaToQVRModule(
        family_obj,
        fixed_prior=fixed_prior,
        random_scale_prior=random_scale_prior,
        user_priors=priors,
        reparameterize=reparameterize,
    )
    module, _ = lens.forward(parsed)
    source = module_to_source(module)
    if path is not None:
        Path(path).write_text(source)
    return source


def _fit_mcmc(program, observations, *, sampler, num_warmup, num_samples, num_chains):
    """Run NUTS / HMC on the compiled program."""
    kernel = NUTSKernel() if sampler == "nuts" else HMCKernel()
    mcmc = MCMC(
        kernel=kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
    )
    n_obs = int(observations[next(iter(observations))].shape[0])
    x = torch.zeros(n_obs, 1, dtype=torch.long)
    return mcmc.run(program, x, observations)


def _fit_svi(program, observations, num_steps, *, guide_cls=None):
    """Run an SVI fit + ELBO.

    Default guide is :class:`AutoNormalGuide`: a mean-field
    diagonal-Normal that scales well across model shapes. Mean-field
    is known to underestimate posterior variance components in
    hierarchical / mixed-effects models; for serious analysis of
    those models, use ``method="nuts"``. Users can swap in any other
    ``Guide`` class via ``fit(..., guide=SomeGuide)``; the class is
    constructed with ``(program, observed_names=...)``.
    """
    if guide_cls is None:
        guide_cls = AutoNormalGuide
    guide = guide_cls(program, observed_names=set(observations.keys()))
    optimizer = torch.optim.Adam(
        list(program.parameters()) + list(guide.parameters()),
        lr=1e-2,
    )
    svi = SVI(program, guide, optimizer, ELBO())
    n_obs = int(observations[next(iter(observations))].shape[0])
    x = torch.zeros(n_obs, 1, dtype=torch.long)
    for _ in range(num_steps):
        svi.step(x, observations)
    return guide
