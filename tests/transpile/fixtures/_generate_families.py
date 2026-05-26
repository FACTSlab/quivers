"""Generate ``families/<family>.qvr`` from the live QVR family registry.

Run as a script:
``.venv/bin/python tests/transpile/fixtures/_generate_families.py``.

The generator inspects
[`_get_family_registry`][quivers.dsl.compiler._prelude._get_family_registry]
at import time and emits one minimal `.qvr` per family. Each fixture
uses whichever of the QVR family-application surfaces actually
compiles for that family:

- **Inline-sample** (``sample x <- F(literals)``) for the 11 families
  in [`_FIXED_FACTORIES`][quivers.continuous.inline._FIXED_FACTORIES].
- **Inline-observe with variable arg** for `TruncatedNormal` (its
  inline builder requires at least one variable parameter).
- **Morphism kernel** (``morphism k : Obs -> Obs [role=kernel] ~ F``)
  for families that have a `Conditional<F>` class but no inline
  factory.
- **Hand-written structural prior** for `Horseshoe` (no inline /
  conditional surface; built from HalfCauchy + Normal).
- **Vector / matrix families** (`Dirichlet`, `MultivariateNormal`,
  `Wishart`, `InverseWishart`, `MatrixNormal`, `LowRankMVN`, `GP`,
  `Categorical`, `RelaxedOneHotCategorical`) get hand-written
  fixtures that declare the event dimension via `object` + `over=`.

Re-running is idempotent: existing fixture content is overwritten
only when the regenerated source differs. The script reports which
families were freshly written, which were unchanged, and which need
a hand-edit (printed at the end and exit-coded non-zero).
"""

from __future__ import annotations

import pathlib
import sys

from quivers.dsl.compiler._prelude import _get_family_registry


# Families generated with the inline-sample form. Args are scalar
# literals; values picked to put the prior in a benign region.
_INLINE_SAMPLE: dict[str, str] = {
    "Beta":        "2.0, 2.0",
    "Bernoulli":   "0.5",
    "Exponential": "1.0",
    "Gamma":       "2.0, 1.0",
    "HalfCauchy":  "1.0",
    "HalfNormal":  "1.0",
    "LogNormal":   "0.0, 1.0",
    "LogitNormal": "0.0, 1.0",
    "Normal":      "0.0, 1.0",
    "Uniform":     "0.0, 1.0",
}


# Families generated with morphism-kernel form (Real codomain).
# These have a `Conditional<F>` class but no inline factory.
_MORPHISM_KERNEL: frozenset[str] = frozenset({
    "Cauchy",
    "Chi2",
    "ContinuousBernoulli",
    "FisherSnedecor",
    "Gumbel",
    "InverseGamma",
    "Kumaraswamy",
    "Laplace",
    "Pareto",
    "RelaxedBernoulli",
    "StudentT",
    "Weibull",
    "GeneralizedPareto",
})


# Families requiring the observe-with-variable form.
_OBSERVE_WITH_VAR: frozenset[str] = frozenset({"TruncatedNormal"})


# Vector / matrix families that need hand-written fixtures.
_VECTOR_FAMILIES: frozenset[str] = frozenset({
    "Categorical",
    "Dirichlet",
    "GP",
    "InverseWishart",
    "LowRankMVN",
    "MatrixNormal",
    "MultivariateNormal",
    "RelaxedOneHotCategorical",
    "Wishart",
})


# Families with no inline / conditional kernel surface; the fixture
# is the structural prior built from primitives.
_STRUCTURAL_PRIOR: frozenset[str] = frozenset({"Horseshoe"})


_GENERATED_HEADER = "# Auto-generated fixture exercising the "


def _inline_sample_source(family: str, args: str) -> str:
    return (
        f"{_GENERATED_HEADER}{family} family.\n"
        f"# Regenerate via tests/transpile/fixtures/_generate_families.py.\n"
        f"object Obs : FinSet 8\n"
        f"program {family.lower()}_fixture : Obs -> Obs\n"
        f"    sample theta <- {family}({args})\n"
        f"    return theta\n"
        f"export {family.lower()}_fixture\n"
    )


def _observe_with_var_source(family: str) -> str:
    """For TruncatedNormal: ``observe y <- F(mu, ...)`` with a
    variable ``mu`` so the family's inline-with-variable builder
    fires."""
    return (
        f"{_GENERATED_HEADER}{family} family.\n"
        f"# Uses the inline-with-variable builder (literals-only path is\n"
        f"# unavailable for {family}).\n"
        f"object Obs : FinSet 8\n"
        f"program {family.lower()}_fixture : Obs -> Obs\n"
        f"    sample mu <- Uniform(0.0, 1.0)\n"
        f"    observe y : Obs <- {family}(mu, 0.2, 0.0, 1.0)\n"
        f"    return mu\n"
        f"export {family.lower()}_fixture\n"
    )


def _morphism_kernel_source(family: str) -> str:
    """``morphism k : Obs -> Obs [role=kernel] ~ F(args)``.

    Uses Real codomain since all morphism-kernel-only families take
    values on the full real line or its subset. Carries the
    canonical-args inline on the ``~`` clause so backends that
    require positional family args (WebPPL's `(location, scale)`
    object-literal, Stan's `<lower=0>` constraint detection) have
    them available after morphism resolution.
    """
    args = _MORPHISM_KERNEL_DEFAULT_ARGS.get(family, "")
    args_clause = f"({args})" if args else ""
    return (
        f"{_GENERATED_HEADER}{family} family.\n"
        f"# Family lacks an inline factory; declared via a morphism\n"
        f"# kernel with `~ {family}{args_clause}`.\n"
        f"object Obs : Real 4\n"
        f"morphism {family.lower()}_kernel : Obs -> Obs [role=kernel] ~ {family}{args_clause}\n"
        f"program {family.lower()}_fixture : Obs -> Obs\n"
        f"    sample x <- {family.lower()}_kernel\n"
        f"    return x\n"
        f"export {family.lower()}_fixture\n"
    )


_MORPHISM_KERNEL_DEFAULT_ARGS: dict[str, str] = {
    "Cauchy":              "0.0, 1.0",
    "Chi2":                "3.0",
    "ContinuousBernoulli": "0.5",
    "FisherSnedecor":      "5.0, 5.0",
    "Gumbel":              "0.0, 1.0",
    "InverseGamma":        "3.0, 1.0",
    "Kumaraswamy":         "2.0, 2.0",
    "Laplace":             "0.0, 1.0",
    "Pareto":              "1.0, 2.0",
    "RelaxedBernoulli":    "0.5, 1.0",
    "StudentT":            "5.0, 0.0, 1.0",
    "Weibull":             "1.5, 1.0",
    "GeneralizedPareto":   "0.0, 1.0, 0.5",
}


# Hand-written sources for vector / matrix families. Each one
# declares an event dimension via ``object`` and uses ``[over=Dim]``
# on the family application. Wishart / InverseWishart / MatrixNormal
# / LKJCholesky need TWO event axes; LowRankMVN needs an explicit
# rank parameter.
_VECTOR_SOURCES: dict[str, str] = {
    "Categorical": (
        "# Hand-written Categorical fixture (vector family).\n"
        "object Item : FinSet 8\n"
        "object Comp : FinSet 4\n"
        "program categorical_fixture(probs : Real) : Item -> Item\n"
        "    sample probs <- Dirichlet(1.0) [over=Comp]\n"
        "    marginalize cls : Comp <- Categorical(probs) [over=Item, reduction=logsumexp]\n"
        "        observe r : Item <- Normal(0.0, 1.0) [via=idx]\n"
        "    return probs\n"
        "export categorical_fixture\n"
    ),
    "Dirichlet": (
        "# Hand-written Dirichlet fixture (vector family).\n"
        "object Obs : FinSet 4\n"
        "object Comp : FinSet 3\n"
        "program dirichlet_fixture(alpha : Real) : Obs -> Obs\n"
        "    sample probs <- Dirichlet(alpha) [over=Comp]\n"
        "    return probs\n"
        "export dirichlet_fixture\n"
    ),
    "GP": (
        "# Hand-written GP fixture (function-space family).\n"
        "# The kernel option chooses the covariance family; length_scale\n"
        "# initialises the kernel's hyperparameter (positive, learnable).\n"
        "object Idx : FinSet 8\n"
        "object Obs : Real 1\n"
        "morphism gp_kernel : Idx -> Obs [role=kernel, kernel=rbf, length_scale=1.0] ~ GP\n"
        "program gp_fixture : Idx -> Obs\n"
        "    sample f <- gp_kernel\n"
        "    return f\n"
        "export gp_fixture\n"
    ),
    "InverseWishart": (
        "# Hand-written InverseWishart fixture (matrix family).\n"
        "object Row : FinSet 3\n"
        "object Col : FinSet 3\n"
        "object Obs : Real 9\n"
        "morphism iw_kernel : Obs -> Obs [role=kernel] ~ InverseWishart\n"
        "program inversewishart_fixture : Obs -> Obs\n"
        "    sample sigma <- iw_kernel\n"
        "    return sigma\n"
        "export inversewishart_fixture\n"
    ),
    "LowRankMVN": (
        "# Hand-written LowRankMVN fixture (vector family with low-rank factor).\n"
        "object Dim : FinSet 5\n"
        "object Obs : Real 5\n"
        "morphism lr_kernel : Obs -> Obs [role=kernel, rank=2] ~ LowRankMVN\n"
        "program lowrankmvn_fixture : Obs -> Obs\n"
        "    sample x <- lr_kernel\n"
        "    return x\n"
        "export lowrankmvn_fixture\n"
    ),
    "MatrixNormal": (
        "# Hand-written MatrixNormal fixture (matrix family; Kronecker covariance).\n"
        "# ``over=[Row, Col]`` binds the two event axes to the Kronecker factors.\n"
        "object Row : FinSet 3\n"
        "object Col : FinSet 4\n"
        "morphism mn_kernel : Row * Col -> Row * Col [role=kernel, over=[Row, Col]] ~ MatrixNormal\n"
        "program matrixnormal_fixture : Row * Col -> Row * Col\n"
        "    sample m <- mn_kernel\n"
        "    return m\n"
        "export matrixnormal_fixture\n"
    ),
    "MultivariateNormal": (
        "# Hand-written MultivariateNormal fixture (vector family).\n"
        "object Dim : FinSet 4\n"
        "object Obs : Real 4\n"
        "morphism mvn_kernel : Obs -> Obs [role=kernel] ~ MultivariateNormal\n"
        "program multivariatenormal_fixture : Obs -> Obs\n"
        "    sample x <- mvn_kernel\n"
        "    return x\n"
        "export multivariatenormal_fixture\n"
    ),
    "RelaxedOneHotCategorical": (
        "# Hand-written RelaxedOneHotCategorical fixture "
        "(Gumbel-softmax over a simplex).\n"
        "object Cls : FinSet 4\n"
        "object Obs : Real 4\n"
        "morphism roc_kernel : Obs -> Obs [role=kernel, temperature=0.5] ~ RelaxedOneHotCategorical\n"
        "program relaxedonehotcategorical_fixture : Obs -> Obs\n"
        "    sample z <- roc_kernel\n"
        "    return z\n"
        "export relaxedonehotcategorical_fixture\n"
    ),
    "Wishart": (
        "# Hand-written Wishart fixture (matrix family).\n"
        "object Dim : FinSet 3\n"
        "object Obs : Real 9\n"
        "morphism w_kernel : Obs -> Obs [role=kernel] ~ Wishart\n"
        "program wishart_fixture : Obs -> Obs\n"
        "    sample sigma <- w_kernel\n"
        "    return sigma\n"
        "export wishart_fixture\n"
    ),
}


# Hand-written sources for structural-only families.
_STRUCTURAL_SOURCES: dict[str, str] = {
    "Horseshoe": (
        "# Hand-written Horseshoe fixture (structural prior; no inline\n"
        "# or conditional kernel surface). The horseshoe is built as\n"
        "# tau (global) * lambda_local (per-coord) * z_raw (standard\n"
        "# Normal raw), per Carvalho-Polson-Scott (2010).\n"
        "object Coef : FinSet 4\n"
        "object Obs : FinSet 200\n"
        "program horseshoe_fixture : Obs -> Obs\n"
        "    sample tau <- HalfCauchy(1.0)\n"
        "    sample lambda_local : Coef <- HalfCauchy(1.0)\n"
        "    sample z_raw : Coef <- Normal(0.0, 1.0)\n"
        "    let beta = tau * lambda_local * z_raw\n"
        "    return beta\n"
        "export horseshoe_fixture\n"
    ),
}


def main() -> int:
    registry = _get_family_registry()
    out_dir = pathlib.Path(__file__).resolve().parent / "families"
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    unchanged: list[str] = []
    uncovered: list[str] = []

    for family in sorted(registry):
        path = out_dir / f"{family.lower()}.qvr"
        if family in _INLINE_SAMPLE:
            source = _inline_sample_source(family, _INLINE_SAMPLE[family])
        elif family in _OBSERVE_WITH_VAR:
            source = _observe_with_var_source(family)
        elif family in _MORPHISM_KERNEL:
            source = _morphism_kernel_source(family)
        elif family in _VECTOR_FAMILIES and family in _VECTOR_SOURCES:
            source = _VECTOR_SOURCES[family]
        elif family in _STRUCTURAL_PRIOR and family in _STRUCTURAL_SOURCES:
            source = _STRUCTURAL_SOURCES[family]
        else:
            uncovered.append(family)
            continue

        if path.exists() and path.read_text() == source:
            unchanged.append(family)
            continue
        path.write_text(source)
        written.append(family)

    print(
        f"wrote {len(written)} fixture(s); {len(unchanged)} unchanged"
    )
    if uncovered:
        print(
            f"ERROR: {len(uncovered)} families have no generator entry: "
            f"{uncovered}. Add them to one of the dispatch tables."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
