"""Independent verification of the QVR reference oracle itself.

Every numeric claim the transpile tier makes is a claim *relative to
the QVR oracle*. Theorem 4.1 of
[docs/semantics/transpile-correctness.md](../../docs/semantics/transpile-correctness.md)
asks only that `log p_QVR - log p_backend` be the same constant at
every point, so the backend comparison is invariant to adding a
constant to the oracle and, at a point set where the oracle and the
backend read the *same* wrong value, invariant to far more than that.
The oracle therefore cannot be validated by the thing it validates.

Two independent witnesses cover the pinned registry, and
[`test_every_pinned_example_has_an_independent_witness`][tests.transpile.test_oracle_reference_strength.test_every_pinned_example_has_an_independent_witness]
asserts the cover is total:

1. **Container witness.** An example with at least one live
   `(backend, example)` cell in
   [`test_gallery_numeric_equivalence`][tests.transpile.test_gallery_numeric_equivalence]
   has its oracle re-derived by a foreign runtime (Stan's own
   `log_prob`, NumPyro's `log_joint`, JAGS's node scores) at every
   point of the set.
2. **Raw-torch witness, this module.** Six examples have *no* live
   cell at all: `bnn`, `continuous_hmm`, `linear_gaussian_ssm`,
   `mixture_model`, `parametric_pooling`, `tree_categorical`. Every
   backend either raises a pinned `UnsupportedConstruct` or sits in
   `_SKIP_PROBE_INCOMPATIBLE`. Nothing outside this module ever
   compares their oracle against an independent computation, so this
   module rebuilds each joint from the `.qvr` source and the `.md`
   synthetic-data snippet in raw `torch.distributions` and asserts
   the match **per site** and **per point**.

Per site rather than per joint on purpose. Matching only the total
lets two errors cancel: a prior term inflated by the same amount a
likelihood term is deflated reproduces the joint exactly. The trace
exposes `Trace.sites[name].log_prob`, so the reconstruction is
compared summand by summand, and a site the reconstruction does not
model must score exactly zero in the trace (the deterministic `let`
bindings, whose value is a function of already-scored sites).

Conventions the reconstruction follows, matching the oracle's:

- **Constrained space, no unconstraining Jacobian.** A positive scale
  is scored by its own density at its own value; no `log |dtheta/dz|`
  term is added.
- **Sum, never mean, over a plate.** The joint is a product over the
  plate index, so its log is a sum.
- **Discrete-marginalized latents integrated by `logsumexp`.**
  `mixture_model`'s per-row component assignment is integrated in
  closed form, not sampled.

What the reconstruction *rejects* is measured, not asserted.
[`test_reconstruction_rejects_mutant`][tests.transpile.test_oracle_reference_strength.test_reconstruction_rejects_mutant]
runs a catalogue of named statistical defects through the same code
path (a plate averaged instead of summed, a mixture collapsed to one
component, a location and a log-scale head transposed, a Jacobian
added, a wiring that conditions the emission on the previous state,
a soft sum-to-zero factor dropped) and requires each to move the
joint past a pinned floor. A defect that stops being rejected trips
the floor before it trips the tolerance.

Ten gallery examples ship synthetic data and carry no pin at all,
and this module holds their exemptions to the same standard rather
than to their own prose. The two structural ones must parse to a
module that declares no probabilistic program. The eight sequence
models must be *demonstrably* non-deterministic: each is traced twice
under different global RNG states and required to disagree, so an
example whose composition marginalisation later becomes
deterministic loses its exemption instead of keeping an unpinned
joint forever.
"""

from __future__ import annotations

import math
import pathlib
from collections.abc import Callable

import pytest
import torch
from torch import Tensor
from torch import distributions as td

from quivers.dsl.ast_nodes.declarations import ProgramDecl
from quivers.dsl.parser import parse
from tests.transpile import _equivalence, _gallery_data
from tests.transpile import test_gallery_numeric_equivalence as _gallery_tier
from tests.transpile.probes._protocol import Point
from tests.transpile.probes.qvr import (
    assert_reference_joint_deterministic,
    reference_traces,
)


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SOURCE_DIR = _REPO_ROOT / "docs" / "examples" / "source"

# `ConditionalNormal` clamps its scale at this floor before scoring.
# Reproduced here as a constant rather than imported so the
# reconstruction does not read the value out of the module it is
# checking; a change to the runtime floor must show up as a
# disagreement, not be absorbed silently.
_SCALE_FLOOR = 1e-7


# ---------------------------------------------------------------------
# Point marshalling.
#
# The reconstruction has to score the same numbers the oracle scored,
# so it inflates the wire `Point` the same way. This is deliberately a
# second implementation rather than a call into
# `tests.transpile.probes.qvr.clamping_observations`: the two must
# agree, and a disagreement is a real ambiguity in how a point reaches
# the oracle, not something to paper over by sharing one code path.
# Nothing here computes a density.
# ---------------------------------------------------------------------


def _flat_tensor(
    value: float | int | list[float] | list[int],
) -> Tensor:
    """Inflate one wire entry to the 1-D float32 tensor the oracle
    clamps with."""
    if isinstance(value, (int, float)):
        return torch.tensor([float(value)], dtype=torch.float32)
    return torch.tensor([float(v) for v in value], dtype=torch.float32)


def _clamped_values(
    dataset: _gallery_data.GalleryDataset, point: Point,
) -> dict[str, Tensor]:
    """Every clamped value the oracle sees at `point`, by site name.

    Latents arrive flat (the wire discards their axes and the oracle
    re-derives them); observations arrive pre-shaped through
    [`observations_for_point`][tests.transpile._gallery_data.observations_for_point],
    which is the only channel that preserves a multi-axis observation
    and takes precedence over the flat payload in the oracle too.
    """
    values: dict[str, Tensor] = {}
    for name in dataset.params:
        if name not in point.params:
            raise AssertionError(
                f"latent {name!r} is captured in the dataset but absent "
                f"from the point's params section, so the "
                f"reconstruction has no value to score it at."
            )
        values[name] = _flat_tensor(point.params[name])
    for name, tensor in _gallery_data.observations_for_point(
        dataset, point,
    ).items():
        values[name] = tensor
    return values


# ---------------------------------------------------------------------
# Raw density primitives.
#
# Written out rather than taken from `quivers.continuous.families`,
# which is the code under test. `torch.distributions` supplies every
# family the six examples name; the two hand-written helpers below
# exist only where the runtime's parameterisation (a network head
# emitting a location and a log-scale side by side) has no
# `torch.distributions` counterpart.
# ---------------------------------------------------------------------


def _affine(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    """`x W^T + b`, written as a matmul rather than through
    `torch.nn.functional.linear`."""
    return x @ weight.transpose(-1, -2) + bias


def _normal_head(
    raw: Tensor, event_dim: int, unit_scale: bool = False,
) -> tuple[Tensor, Tensor]:
    """Split a `(..., 2 * event_dim)` parameter row into `(loc, scale)`.

    The runtime's `ConditionalNormal` reads the first `event_dim`
    columns as the location and exponentiates the rest, clamping the
    result at `_SCALE_FLOOR`. `unit_scale` is the mutant path: a head
    whose scale columns are ignored.
    """
    loc = raw[..., :event_dim]
    if unit_scale:
        return loc, torch.ones_like(loc)
    scale = raw[..., event_dim:].exp().clamp(min=_SCALE_FLOOR)
    return loc, scale


def _unknown_variant(example: str, variant: str) -> AssertionError:
    return AssertionError(
        f"{example!r}: reconstruction asked for unknown variant "
        f"{variant!r}. A mutant whose variant name the reconstruction "
        f"does not branch on would silently score the faithful joint "
        f"and be reported as rejected-with-margin-zero; raising here "
        f"names the typo instead."
    )


# ---------------------------------------------------------------------
# Per-example reconstructions.
#
# Each returns the joint as a dict of per-site scalar contributions,
# keyed by the trace's own site names, so the comparison is summand by
# summand. `variant` selects a mutant; the empty string is the
# faithful reconstruction. Mutants live inside the same function as
# the term they corrupt so that the defect reads as a one-line
# difference from the correct term and cannot drift away from it.
# ---------------------------------------------------------------------


def _reconstruct_bnn(
    dataset: _gallery_data.GalleryDataset,
    values: dict[str, Tensor],
    weights: dict[str, Tensor],
    variant: str,
) -> dict[str, Tensor]:
    """`bnn.qvr`: `observe y : Resp <- net(x)`, `net` a heteroscedastic
    `~ Normal` morphism with `[param_source=mlp, hidden_dim=64]`.

    `net`'s weights are drawn at compile time from torch's default
    `nn.Linear` initialiser. They appear nowhere in the `.qvr` text
    and nowhere in the `.md` snippet, so the reconstruction takes the
    four tensors as given and rebuilds everything else: the layer
    order (affine, `tanh`, affine), the `(loc, log_scale)` split of
    the two output columns, the Normal density, and the sum over the
    200-row `Resp` plate. That is the whole of what the oracle
    computes here apart from the numbers in the weight tensors, which
    no independent source can supply.
    """
    del dataset
    prefix = "_step_y._family.param_source.net."
    hidden = _affine(
        values["x"], weights[prefix + "0.weight"], weights[prefix + "0.bias"],
    )
    if variant == "mlp_without_activation":
        activated = hidden
    elif variant in ("", "homoscedastic", "plate_mean"):
        activated = torch.tanh(hidden)
    else:
        raise _unknown_variant("bnn", variant)
    raw = _affine(
        activated,
        weights[prefix + "2.weight"],
        weights[prefix + "2.bias"],
    )
    # `object Target : Real 1`: a one-dimensional response, so the
    # head emits one location column and one log-scale column.
    loc, scale = _normal_head(
        raw, 1, unit_scale=variant == "homoscedastic",
    )
    per_row = td.Normal(loc, scale).log_prob(values["y"])
    if variant == "plate_mean":
        return {"y": per_row.mean()}
    return {"y": per_row.sum()}


def _reconstruct_state_space(
    dataset: _gallery_data.GalleryDataset,
    values: dict[str, Tensor],
    weights: dict[str, Tensor],
    variant: str,
    state_dim: int,
    obs_dim: int,
) -> dict[str, Tensor]:
    """`continuous_hmm.qvr` and `linear_gaussian_ssm.qvr`, which share
    a program shape:

        sample s_new <- transition(<program input>)
        observe o    <- emission(s_new)

    Both morphisms are `~ Normal` Kleisli arrows over a default linear
    parameter source, so each step is a single affine map from its
    conditioning row to a `(loc, log_scale)` pair. As with `bnn` the
    affine weights are compile-time draws with no source spelling and
    are taken as given; the wiring (which row conditions which step),
    the head split, the density, and the per-step sum over the
    32-element scan are rebuilt.
    """
    x_input = dataset.x_input
    if x_input is None:
        raise AssertionError(
            "a state-space reconstruction needs the snippet's "
            "`state_prev` / `x_in` driver matrix, which the dataset "
            "did not carry."
        )
    state = values["s_new"]

    raw_state = _affine(
        x_input,
        weights["_step_s_new.param_source.linear.weight"],
        weights["_step_s_new.param_source.linear.bias"],
    )
    loc_state, scale_state = _normal_head(
        raw_state, state_dim, unit_scale=variant == "unit_transition_scale",
    )
    per_step_state = td.Normal(loc_state, scale_state).log_prob(state)

    if variant == "emission_reads_previous_state":
        emission_input = x_input
    elif variant in (
        "",
        "unit_transition_scale",
        "drop_transition_term",
        "swap_loc_and_log_scale",
        "plate_mean",
    ):
        emission_input = state
    else:
        raise _unknown_variant("state-space", variant)
    raw_obs = _affine(
        emission_input,
        weights["_step_o.param_source.linear.weight"],
        weights["_step_o.param_source.linear.bias"],
    )
    if variant == "swap_loc_and_log_scale":
        loc_obs = raw_obs[..., obs_dim:]
        scale_obs = raw_obs[..., :obs_dim].exp().clamp(min=_SCALE_FLOOR)
    else:
        loc_obs, scale_obs = _normal_head(raw_obs, obs_dim)
    per_step_obs = td.Normal(loc_obs, scale_obs).log_prob(values["o"])

    if variant == "plate_mean":
        return {
            "s_new": per_step_state.sum(-1).mean(),
            "o": per_step_obs.sum(-1).mean(),
        }
    if variant == "drop_transition_term":
        return {"s_new": torch.zeros(()), "o": per_step_obs.sum()}
    return {"s_new": per_step_state.sum(), "o": per_step_obs.sum()}


def _reconstruct_continuous_hmm(
    dataset: _gallery_data.GalleryDataset,
    values: dict[str, Tensor],
    weights: dict[str, Tensor],
    variant: str,
) -> dict[str, Tensor]:
    """`object State : Real 16`, `object Obs : Real 8`."""
    return _reconstruct_state_space(
        dataset, values, weights, variant, state_dim=16, obs_dim=8,
    )


def _reconstruct_linear_gaussian_ssm(
    dataset: _gallery_data.GalleryDataset,
    values: dict[str, Tensor],
    weights: dict[str, Tensor],
    variant: str,
) -> dict[str, Tensor]:
    """`object State : Real 4`, `object Obs : Real 2`; the transition
    additionally reads a `Driver : Real 2` row, which the snippet
    concatenates onto the previous state before the call."""
    return _reconstruct_state_space(
        dataset, values, weights, variant, state_dim=4, obs_dim=2,
    )


def _reconstruct_mixture_model(
    dataset: _gallery_data.GalleryDataset,
    values: dict[str, Tensor],
    weights: dict[str, Tensor],
    variant: str,
) -> dict[str, Tensor]:
    """`mixture_model.qvr`, instantiated at `gmm(alpha=1.0)`:

        sample probs <- Dirichlet(alpha) [over=Component]
        sample mu    : Component <- Normal(0.0, 5.0)
        sample sigma : Component <- HalfNormal(1.0)
        observe r    : Resp <- MixtureNormal(probs, mu, sigma)

    Wholly reconstructible from the source: every family is named,
    every hyper-parameter is a literal, and the one non-elementary
    step is the closed-form marginalisation the `MixtureNormal`
    likelihood performs,

        log p(r_n) = logsumexp_k [ log probs_k + log N(r_n; mu_k, sigma_k) ],

    summed over the `Resp` plate. No component assignment is drawn.
    """
    del weights
    alpha = dataset.scalar_params["alpha"]
    probs, mu, sigma = values["probs"], values["mu"], values["sigma"]
    response = values["r"]

    concentration = torch.full_like(probs, float(alpha))
    if variant == "drop_dirichlet_prior":
        probs_term = torch.zeros(())
    else:
        probs_term = td.Dirichlet(concentration).log_prob(probs).sum()

    mu_term = td.Normal(0.0, 5.0).log_prob(mu).sum()

    sigma_term = td.HalfNormal(1.0).log_prob(sigma).sum()
    if variant == "unconstrain_sigma":
        # Scoring a positive latent in log space adds a
        # `log |d sigma / d log sigma| = log sigma` Jacobian. The
        # oracle scores in the constrained space and adds none.
        sigma_term = sigma_term + sigma.log().sum()

    per_component = td.Normal(
        mu.unsqueeze(0), sigma.unsqueeze(0),
    ).log_prob(response.unsqueeze(-1))
    weighted = probs.log().unsqueeze(0) + per_component
    if variant == "first_component_only":
        per_row = weighted[..., 0]
    elif variant in (
        "", "drop_dirichlet_prior", "unconstrain_sigma", "likelihood_mean",
    ):
        per_row = torch.logsumexp(weighted, dim=-1)
    else:
        raise _unknown_variant("mixture_model", variant)
    response_term = (
        per_row.mean() if variant == "likelihood_mean" else per_row.sum()
    )

    return {
        "probs": probs_term,
        "mu": mu_term,
        "sigma": sigma_term,
        "r": response_term,
    }


def _reconstruct_parametric_pooling(
    dataset: _gallery_data.GalleryDataset,
    values: dict[str, Tensor],
    weights: dict[str, Tensor],
    variant: str,
) -> dict[str, Tensor]:
    """`parametric_pooling.qvr`, exporting `pooled_tight`:

        sample z : K <- Normal(0.0, 1.0)      (inside school_effects)
        let effect = spread * z               (spread = 0.6)
        sample sigma <- LogNormal(0.0, 0.5)
        let total_effect = sum(theta)
        score centering = -50.0 * total_effect * total_effect
        observe y : School <- Normal(theta, sigma)

    Wholly reconstructible. The `score` step is a log-density summand
    in its own right, not a normalised family: it enters the joint as
    its own value. `theta` and `total_effect` are deterministic `let`
    bindings and score zero.
    """
    del dataset, weights
    z, sigma, y = values["theta$z"], values["sigma"], values["y"]
    # `sample theta <- school_effects(0.6, School)` instantiates the
    # template's `spread` at 0.6.
    spread = 1.0 if variant == "ignore_spread" else 0.6
    theta = spread * z

    z_term = td.Normal(0.0, 1.0).log_prob(z).sum()

    sigma_term = td.LogNormal(0.0, 0.5).log_prob(sigma).sum()
    if variant == "unconstrain_sigma":
        sigma_term = sigma_term + sigma.log().sum()

    total_effect = theta.sum()
    if variant == "drop_centering_score":
        centering = torch.zeros(())
    elif variant in ("", "ignore_spread", "unconstrain_sigma", "plate_mean"):
        centering = -50.0 * total_effect * total_effect
    else:
        raise _unknown_variant("parametric_pooling", variant)

    per_school = td.Normal(theta, sigma).log_prob(y)
    y_term = per_school.mean() if variant == "plate_mean" else per_school.sum()

    return {
        "theta$z": z_term,
        "sigma": sigma_term,
        "centering": centering,
        "y": y_term,
    }


def _reconstruct_tree_categorical(
    dataset: _gallery_data.GalleryDataset,
    values: dict[str, Tensor],
    weights: dict[str, Tensor],
    variant: str,
) -> dict[str, Tensor]:
    """`tree_categorical.qvr`:

        sample p_root, p_left, p_right <- Beta(1.0, 1.0)
        let leaf_log = factor cls : Class in { 0 -> log(1 - p_root) + log(1 - p_left), ... }
        sample sigma_v <- HalfNormal(1.0)
        sample delta : Verb <- Normal(0.0, sigma_v)
        sample mu : Class <- Normal(0.0, 1.0)
        let cell_score = factor v : Verb, cls : Class in delta[v] + mu[cls] + leaf_log[cls]
        let cell0 = cell_score[0, 0]
        observe y : Resp <- Normal(cell0, 0.5)

    Wholly reconstructible. Only cell `(0, 0)` of the rank-2 factor
    reaches the likelihood, so the reconstruction evaluates that one
    cell: `delta[0] + mu[0] + leaf_log[0]`, with
    `leaf_log[0] = log(1 - p_root) + log(1 - p_left)` from the case
    table's `0 ->` arm. The three `Beta(1, 1)` sites score exactly
    zero, which
    [`test_zero_scoring_sites_are_zero_by_identity`][tests.transpile.test_oracle_reference_strength.test_zero_scoring_sites_are_zero_by_identity]
    asserts is an identity rather than a coincidence.

    `y` carries a single observation against `object Resp : FinSet
    200`; the reconstruction scores the tensor the snippet actually
    produced rather than the declared cardinality, matching the
    oracle. The mismatch is the `tree_categorical` blocker recorded in
    `_SKIP_PROBE_INCOMPATIBLE`.
    """
    del dataset, weights
    p_root, p_left, p_right = (
        values["p_root"], values["p_left"], values["p_right"],
    )
    sigma_v, delta, mu, y = (
        values["sigma_v"], values["delta"], values["mu"], values["y"],
    )
    unit = td.Beta(1.0, 1.0)

    if variant == "delta_unit_scale":
        delta_scale = torch.ones_like(sigma_v)
    elif variant in ("", "wrong_leaf_branch", "drop_leaf_offset"):
        delta_scale = sigma_v
    else:
        raise _unknown_variant("tree_categorical", variant)

    if variant == "wrong_leaf_branch":
        leaf_zero = torch.log(p_root) + torch.log(p_left)
    else:
        leaf_zero = torch.log(1.0 - p_root) + torch.log(1.0 - p_left)
    cell_zero = delta[0] + mu[0]
    if variant != "drop_leaf_offset":
        cell_zero = cell_zero + leaf_zero

    # `y` is summed, never averaged. The fixture happens to ship a
    # single response, so a mean / sum confusion would be invisible
    # here; the plate-averaging defect is covered on the examples whose
    # plate actually has width (`bnn`, `linear_gaussian_ssm`,
    # `mixture_model`, `parametric_pooling`).
    y_term = td.Normal(cell_zero, 0.5).log_prob(y).sum()

    return {
        "p_root": unit.log_prob(p_root).sum(),
        "p_left": unit.log_prob(p_left).sum(),
        "p_right": unit.log_prob(p_right).sum(),
        "sigma_v": td.HalfNormal(1.0).log_prob(sigma_v).sum(),
        "delta": td.Normal(
            torch.zeros_like(delta), delta_scale,
        ).log_prob(delta).sum(),
        "mu": td.Normal(0.0, 1.0).log_prob(mu).sum(),
        "y": y_term,
    }


_Builder = Callable[
    [
        _gallery_data.GalleryDataset,
        dict[str, Tensor],
        dict[str, Tensor],
        str,
    ],
    dict[str, Tensor],
]
"""Signature every reconstruction shares: `(dataset, clamped values,
compiled weights, variant) -> per-site log-density terms`."""


class _Reconstruction:
    """One example's raw-torch reconstruction and what it takes as given.

    A plain class rather than a `dx.Model`: it holds a callable and is
    never serialised, compared structurally, or round-tripped through
    didactic's encode / decode.
    """

    def __init__(
        self,
        build: _Builder,
        opaque_weights: tuple[str, ...],
        opaque_reason: str,
    ) -> None:
        self.build = build
        self.opaque_weights = opaque_weights
        self.opaque_reason = opaque_reason


_RECONSTRUCTIONS: dict[str, _Reconstruction] = {
    "bnn": _Reconstruction(
        _reconstruct_bnn,
        (
            "_step_y._family.param_source.net.0.weight",
            "_step_y._family.param_source.net.0.bias",
            "_step_y._family.param_source.net.2.weight",
            "_step_y._family.param_source.net.2.bias",
        ),
        "`net`'s MLP weights are drawn by torch's default `nn.Linear` "
        "initialiser when the module compiles. They are named in "
        "neither the `.qvr` source nor the `.md` snippet, so no "
        "independent source can supply their values.",
    ),
    "continuous_hmm": _Reconstruction(
        _reconstruct_continuous_hmm,
        (
            "_step_s_new.param_source.linear.weight",
            "_step_s_new.param_source.linear.bias",
            "_step_o.param_source.linear.weight",
            "_step_o.param_source.linear.bias",
        ),
        "`transition` and `emission` take the default linear "
        "parameter source, whose affine weights are compile-time "
        "draws with no spelling in the source.",
    ),
    "linear_gaussian_ssm": _Reconstruction(
        _reconstruct_linear_gaussian_ssm,
        (
            "_step_s_new.param_source.linear.weight",
            "_step_s_new.param_source.linear.bias",
            "_step_o.param_source.linear.weight",
            "_step_o.param_source.linear.bias",
        ),
        "`transition_cell` and `emission` take the default linear "
        "parameter source, whose affine weights are compile-time "
        "draws with no spelling in the source.",
    ),
    "mixture_model": _Reconstruction(_reconstruct_mixture_model, (), ""),
    "parametric_pooling": _Reconstruction(
        _reconstruct_parametric_pooling, (), "",
    ),
    "tree_categorical": _Reconstruction(
        _reconstruct_tree_categorical, (), "",
    ),
}


class _Mutant:
    """A named statistical defect the reconstruction must reject.

    A plain class rather than a `dx.Model`, for the same reason as
    `_Reconstruction`.
    """

    def __init__(
        self, example: str, variant: str, defect: str, floor: float,
    ) -> None:
        self.example = example
        self.variant = variant
        self.defect = defect
        self.floor = floor

    @property
    def ident(self) -> str:
        return f"{self.example}:{self.variant}"


_MUTANTS: tuple[_Mutant, ...] = (
    _Mutant(
        "bnn", "mlp_without_activation",
        "the hidden layer's `tanh` is dropped, collapsing the network "
        "to a single affine map: the defect class of a renderer that "
        "emits a linear head for a nonlinear parameter source.",
        5000.0,
    ),
    _Mutant(
        "bnn", "homoscedastic",
        "the head's log-scale columns are ignored and the response is "
        "scored at unit scale, dropping the heteroscedasticity the "
        "model exists to express.",
        80.0,
    ),
    _Mutant(
        "bnn", "plate_mean",
        "the 200-row `Resp` plate is averaged instead of summed.",
        200.0,
    ),
    _Mutant(
        "continuous_hmm", "drop_transition_term",
        "the `sample s_new <- transition` prior term is dropped and "
        "only the emission likelihood is scored.",
        200.0,
    ),
    _Mutant(
        "continuous_hmm", "emission_reads_previous_state",
        "`emission` is conditioned on the program input rather than "
        "on the freshly-drawn `s_new`, an off-by-one in the scan's "
        "wiring that leaves every shape valid.",
        2.0,
    ),
    _Mutant(
        "continuous_hmm", "unit_transition_scale",
        "the transition head's log-scale columns are ignored.",
        10.0,
    ),
    _Mutant(
        "linear_gaussian_ssm", "swap_loc_and_log_scale",
        "the emission head's location and log-scale columns are "
        "transposed, which every shape check still accepts.",
        3.0,
    ),
    _Mutant(
        "linear_gaussian_ssm", "drop_transition_term",
        "the `sample s_new <- transition_cell` prior term is dropped "
        "and only the emission likelihood is scored.",
        60.0,
    ),
    _Mutant(
        "linear_gaussian_ssm", "unit_transition_scale",
        "the transition head's log-scale columns are ignored, so the "
        "process noise is scored at unit scale instead of at the "
        "kernel's own.",
        2.0,
    ),
    _Mutant(
        "linear_gaussian_ssm", "plate_mean",
        "the 32-step scan is averaged instead of summed.",
        100.0,
    ),
    _Mutant(
        "mixture_model", "first_component_only",
        "the closed-form marginalisation is replaced by component 0 "
        "alone, so the `logsumexp` over the discrete latent is lost.",
        1000.0,
    ),
    _Mutant(
        "mixture_model", "drop_dirichlet_prior",
        "the `Dirichlet(alpha)` prior on the mixing weights is "
        "dropped.",
        0.3,
    ),
    _Mutant(
        "mixture_model", "unconstrain_sigma",
        "the `HalfNormal` scale is scored in log space, adding the "
        "`log sigma` Jacobian the constrained-space convention "
        "forbids.",
        1.0,
    ),
    _Mutant(
        "mixture_model", "likelihood_mean",
        "the 100-row `Resp` plate is averaged instead of summed.",
        100.0,
    ),
    _Mutant(
        "parametric_pooling", "drop_centering_score",
        "the soft sum-to-zero `score` factor is dropped. Its value is "
        "~0 at the ground truth, where the snippet centres the group "
        "effects exactly, so this mutant is invisible at point 0 and "
        "only the perturbed points reject it.",
        5.0,
    ),
    _Mutant(
        "parametric_pooling", "ignore_spread",
        "the template's `spread = 0.6` scaling is dropped, so the "
        "non-centred parameterisation collapses to `theta = z`.",
        10.0,
    ),
    _Mutant(
        "parametric_pooling", "unconstrain_sigma",
        "the `LogNormal` observation scale is scored in log space, "
        "adding a `log sigma` Jacobian.",
        0.5,
    ),
    _Mutant(
        "parametric_pooling", "plate_mean",
        "the 8-school plate is averaged instead of summed.",
        2.0,
    ),
    _Mutant(
        "tree_categorical", "wrong_leaf_branch",
        "leaf 0 of the case table reads the `p_root` / `p_left` arm "
        "instead of its complement, the classic tree-traversal "
        "polarity error.",
        0.4,
    ),
    _Mutant(
        "tree_categorical", "delta_unit_scale",
        "the per-verb effects are scored at unit scale instead of at "
        "the sampled `sigma_v`, severing the hierarchy.",
        1.5,
    ),
    _Mutant(
        "tree_categorical", "drop_leaf_offset",
        "the tree-structured leaf log-probability is dropped from the "
        "score cell, leaving `delta[0] + mu[0]`.",
        4.0,
    ),
)


# ---------------------------------------------------------------------
# Oracle fixtures, memoised per example.
#
# Every test below scores the same six points of the same six
# examples; tracing them once per test would multiply the module's
# cost by the number of tests without adding a single independent
# observation.
# ---------------------------------------------------------------------


class _Fixture:
    """One example's dataset, point set, compiled weights, and the
    oracle's per-point joint and per-site summands.

    A plain class rather than a `dx.Model`: it holds torch tensors, a
    compiled
    [`GalleryDataset`][tests.transpile._gallery_data.GalleryDataset],
    and a list of [`Point`][tests.transpile.probes._protocol.Point]s,
    none of which survive didactic's encode / decode round trip.
    """

    def __init__(
        self,
        dataset: _gallery_data.GalleryDataset,
        points: list[Point],
        weights: dict[str, Tensor],
        joints: list[float],
        sites: list[dict[str, float]],
    ) -> None:
        self.dataset = dataset
        self.points = points
        self.weights = weights
        self.joints = joints
        self.sites = sites


_FIXTURES: dict[str, _Fixture] = {}


def _fixture(example: str) -> _Fixture:
    """Build (or return the memoised) oracle fixture for `example`."""
    cached = _FIXTURES.get(example)
    if cached is not None:
        return cached

    source_path = _SOURCE_DIR / f"{example}.qvr"
    dataset = _gallery_data.load_gallery_data(source_path)
    if dataset is None:
        raise AssertionError(
            f"{example!r}: `load_gallery_data` returned None, so this "
            f"module has no point set to reconstruct against. Either "
            f"the `.md` synthetic-data block broke, or the "
            f"reconstruction registry names an example that never had "
            f"one."
        )
    monadic = dataset.monadic
    if monadic is None:
        raise AssertionError(
            f"{example!r}: the synthetic-data block bound no compiled "
            f"`MonadicProgram`, so there is no oracle to check."
        )

    points = _gallery_data.points_from_dataset(dataset)
    weights = {
        name: tensor.detach()
        for name, tensor in monadic.state_dict().items()
    }
    joints: list[float] = []
    sites: list[dict[str, float]] = []
    for point in points:
        traces = reference_traces(
            monadic,
            point,
            x_input=dataset.x_input,
            observations=_gallery_data.observations_for_point(
                dataset, point,
            ),
        )
        # A reconstruction compared against a redrawn "joint" would be
        # comparing against a sample, and the disagreement would read
        # as a defect in whichever side happened to be checked second.
        # `reference_traces` already ran the program under two distinct
        # global RNG states; requiring them to agree bit for bit costs
        # nothing here and rules that reading out.
        assert_reference_joint_deterministic(traces, example)
        tr = traces[0]
        joint = tr.log_joint
        if joint is None:
            raise AssertionError(
                f"{example!r}: the trace returned no `log_joint`."
            )
        joints.append(float(joint.sum().item()))
        sites.append({
            name: float(site.log_prob.sum().item())
            for name, site in tr.sites.items()
            if site.log_prob is not None
        })

    fixture = _Fixture(dataset, points, weights, joints, sites)
    _FIXTURES[example] = fixture
    return fixture


def _reconstruct(
    example: str, index: int, variant: str = "",
) -> dict[str, float]:
    """Per-site reconstruction of `example` at point `index`."""
    fixture = _fixture(example)
    values = _clamped_values(fixture.dataset, fixture.points[index])
    terms = _RECONSTRUCTIONS[example].build(
        fixture.dataset, values, fixture.weights, variant,
    )
    return {name: float(term.item()) for name, term in terms.items()}


def _cells() -> list[tuple[str, int]]:
    """Every `(example, point index)` pair the reconstruction covers."""
    n_points = len(_gallery_data.perturbation_labels())
    return [
        (example, index)
        for example in sorted(_RECONSTRUCTIONS)
        for index in range(n_points)
    ]


def _cell_id(cell: tuple[str, int]) -> str:
    example, index = cell
    return f"{example}-pt{index}"


def _live_backend_cells(example: str) -> list[str]:
    """Backends whose container actually scores `example`.

    A backend is live for an example when the gallery tier neither
    pins its transpile as a raise nor parks the cell in a skip
    registry: exactly the cells that reach
    `assert_log_density_match` and therefore re-derive the oracle in a
    foreign runtime.
    """
    return sorted(
        backend
        for backend in _gallery_tier._BACKENDS_WITH_IMAGES
        if (backend, example) not in _gallery_tier._EXPECTED_TRANSPILE_RAISES
        and (backend, example) not in _gallery_tier._SKIP_PROBE_INCOMPATIBLE
        and example not in _gallery_tier._SKIP_DATASET_LOAD_FAILED
        and example not in _gallery_tier._SKIP_QVR_INCOMPATIBLE
    )


# ---------------------------------------------------------------------
# Coverage of the pinned registry.
# ---------------------------------------------------------------------


def test_reconstruction_registry_is_well_formed() -> None:
    """Every reconstruction names a pinned example, and every tensor it
    takes as given exists in that example's compiled module.

    The second half is what keeps `opaque_weights` honest. A
    reconstruction that quietly reached for a fifth weight tensor, or
    that kept naming a tensor the runtime no longer builds, would
    still run; the declared list would then understate what the check
    assumes and overstate what it verifies.
    """
    pinned = set(_gallery_tier._QVR_REFERENCE_JOINT)
    unknown = sorted(set(_RECONSTRUCTIONS) - pinned)
    assert not unknown, (
        f"{unknown!r} carry a raw-torch reconstruction but no "
        f"`_QVR_REFERENCE_JOINT` entry, so the reconstruction verifies "
        f"a value nothing else asserts. Pin the example or drop the "
        f"reconstruction."
    )

    for example in sorted(_RECONSTRUCTIONS):
        entry = _RECONSTRUCTIONS[example]
        available = set(_fixture(example).weights)
        missing = sorted(set(entry.opaque_weights) - available)
        assert not missing, (
            f"{example!r}: the reconstruction declares {missing!r} as "
            f"compile-time weights it takes as given, but the "
            f"compiled module has no such entries. The runtime's "
            f"parameter layout moved under the reconstruction."
        )
        if entry.opaque_weights:
            assert entry.opaque_reason.strip(), (
                f"{example!r}: takes {len(entry.opaque_weights)} "
                f"tensor(s) as given without saying why they cannot "
                f"be derived independently. An undeclared assumption "
                f"is the part of the joint nothing checks."
            )
        else:
            assert not entry.opaque_reason, (
                f"{example!r}: declares no opaque weights but carries "
                f"a reason for them. Drop the stale prose."
            )


def test_every_pinned_example_has_an_independent_witness() -> None:
    """No pinned reference rests on the oracle's own word.

    Each entry of `_QVR_REFERENCE_JOINT` must be re-derived either by
    a live backend container or by a reconstruction in this module.
    The failing direction that matters is an example losing its last
    live cell: its pin then has no witness at all, and without this
    test that loss is invisible, because the gallery tier turns the
    cell into a `pytest.skip` and stays green.

    The second assertion guards the opposite drift. If every
    reconstructed example acquired a live backend cell, this module
    would be duplicating the containers rather than covering the gap
    they leave, and its coverage claim would need re-aiming at
    whatever the zero-cell set had become.
    """
    pinned = sorted(_gallery_tier._QVR_REFERENCE_JOINT)
    unwitnessed = [
        example
        for example in pinned
        if not _live_backend_cells(example)
        and example not in _RECONSTRUCTIONS
    ]
    assert not unwitnessed, (
        f"{unwitnessed!r} carry a pinned oracle joint that nothing "
        f"outside the oracle reproduces: every backend cell is a "
        f"pinned raise or a skip, and there is no raw-torch "
        f"reconstruction here. Theorem 4.1's constant-spread quotient "
        f"cannot see a point-independent oracle error, so for these "
        f"examples the pin is a transcript of whatever the oracle "
        f"printed. Add a reconstruction to `_RECONSTRUCTIONS`, or fix "
        f"the defect blocking a backend cell."
    )

    idle = sorted(
        example
        for example in _RECONSTRUCTIONS
        if not _live_backend_cells(example)
    )
    assert idle, (
        f"every reconstructed example now has a live backend cell, so "
        f"this module is asserting nothing a container does not "
        f"already assert. That is good news about the backends and "
        f"bad news about this test: re-target the reconstructions at "
        f"whatever the current zero-cell set is, or the coverage "
        f"claim in the module docstring is stale. Reconstructed: "
        f"{sorted(_RECONSTRUCTIONS)!r}."
    )


# ---------------------------------------------------------------------
# The reconstruction itself.
# ---------------------------------------------------------------------


@pytest.mark.parametrize("cell", _cells(), ids=_cell_id)
def test_reconstruction_matches_the_oracle_per_site(
    cell: tuple[str, int],
) -> None:
    """The raw-torch reconstruction reproduces the oracle summand by
    summand, at this point.

    Per site, not per joint. A joint-only comparison is satisfied by
    any pair of compensating errors: a prior term inflated by exactly
    what a likelihood term loses reproduces the total and hides both.
    Comparing each `Trace.sites[name].log_prob` against its
    independently-derived counterpart removes that degree of freedom.

    A site the reconstruction does not model must score exactly zero
    in the trace. Those are the deterministic `let` bindings, whose
    values are functions of already-scored sites and which contribute
    no density; requiring the zero rather than ignoring the site keeps
    a newly-scoring step from slipping past unmodelled.
    """
    example, index = cell
    fixture = _fixture(example)
    labels = _gallery_data.perturbation_labels(len(fixture.points))
    terms = _reconstruct(example, index)
    oracle_sites = fixture.sites[index]

    unknown = sorted(set(terms) - set(oracle_sites))
    assert not unknown, (
        f"{example!r} point {index} ({labels[index]}): the "
        f"reconstruction scores {unknown!r}, which the trace does not "
        f"record as sites. The reconstruction is modelling a step the "
        f"program does not have."
    )

    unmodelled = sorted(
        name
        for name, value in oracle_sites.items()
        if name not in terms and value != 0.0
    )
    assert not unmodelled, (
        f"{example!r} point {index} ({labels[index]}): the oracle "
        f"scores {unmodelled!r} with a non-zero log-density and the "
        f"reconstruction does not model them, so that part of the "
        f"joint is verified by nothing. Add the term."
    )

    for name in sorted(terms):
        expected = oracle_sites[name]
        measured = terms[name]
        assert math.isfinite(measured), (
            f"{example!r} point {index} ({labels[index]}) site "
            f"{name!r}: the reconstruction is {measured!r}."
        )
        atol = _gallery_tier.reference_pin_atol(expected)
        assert abs(measured - expected) <= atol, (
            f"{example!r} point {index} ({labels[index]}) site "
            f"{name!r}: oracle {expected!r} against independent "
            f"reconstruction {measured!r}, a gap of "
            f"{abs(measured - expected):.6g} nats past the "
            f"{atol:.6g} round-off budget. One of the two computes a "
            f"different density. Re-derive the term from the `.qvr` "
            f"source before touching either side; the tolerance is "
            f"the equivalence floor and does not move."
        )

    total = sum(terms.values())
    joint = fixture.joints[index]
    atol = _gallery_tier.reference_pin_atol(joint)
    assert abs(total - joint) <= atol, (
        f"{example!r} point {index} ({labels[index]}): every site "
        f"agreed but the joints do not ({total!r} against {joint!r}). "
        f"The trace is summing something its per-site log-densities do "
        f"not account for."
    )


@pytest.mark.parametrize(
    "example", sorted(_RECONSTRUCTIONS), ids=lambda name: name,
)
def test_reconstruction_matches_the_pinned_reference(
    example: str,
) -> None:
    """The pinned `_QVR_REFERENCE_JOINT` row is re-derived from raw
    `torch.distributions` at every point.

    This is the test that makes the pin reproducible. Before it, the
    registry's claim of independent verification rested on a one-time
    manual act with nothing in the tree that re-ran it, so a pinned
    number and the reconstruction that once justified it could drift
    apart silently. The numbers are now re-derived on every run.
    """
    reference = _gallery_tier._QVR_REFERENCE_JOINT[example]
    fixture = _fixture(example)
    labels = _gallery_data.perturbation_labels(len(fixture.points))
    assert len(reference) == len(fixture.points), (
        f"{example!r}: {len(reference)} pinned value(s) against "
        f"{len(fixture.points)} point(s)."
    )

    for index, expected in enumerate(reference):
        total = sum(_reconstruct(example, index).values())
        atol = _gallery_tier.reference_pin_atol(expected)
        assert abs(total - expected) <= atol, (
            f"{example!r} point {index} ({labels[index]}): pinned "
            f"reference {expected!r} against independent raw-torch "
            f"reconstruction {total!r}, a gap of "
            f"{abs(total - expected):.6g} nats past the {atol:.6g} "
            f"round-off budget. The pinned number is wrong, or the "
            f"program changed. Do not re-pin from the oracle's output "
            f"alone: the reconstruction is the only thing here that "
            f"can tell those two cases apart."
        )


@pytest.mark.parametrize(
    "example", sorted(_RECONSTRUCTIONS), ids=lambda name: name,
)
def test_zero_scoring_sites_are_zero_by_identity(example: str) -> None:
    """A site that contributes exactly zero does so because its family
    is flat, not because this fixture's value happens to sit at a zero.

    The distinction matters for
    [`test_dropping_any_scored_site_is_rejected`][tests.transpile.test_oracle_reference_strength.test_dropping_any_scored_site_is_rejected],
    which cannot detect the removal of a zero term. `tree_categorical`
    draws three `Beta(1, 1)` splits, whose log-density is
    `-log B(1, 1) = 0` for **every** value in `(0, 1)`; dropping such a
    term changes nothing because the term is nothing. Re-checking the
    density across the unit interval turns that from an observation
    about this fixture into a statement about the family, so the
    undetectable case is accounted for rather than merely unnoticed.
    """
    fixture = _fixture(example)
    zero_sites = sorted({
        name
        for index in range(len(fixture.points))
        for name, value in _reconstruct(example, index).items()
        if value == 0.0
    })
    for index in range(len(fixture.points)):
        terms = _reconstruct(example, index)
        for name in zero_sites:
            assert terms[name] == 0.0, (
                f"{example!r} site {name!r} scores zero at some points "
                f"and {terms[name]!r} at point {index}. A term that is "
                f"zero only sometimes is a term whose value the "
                f"fixture is not exercising, not a flat family."
            )
            assert fixture.sites[index][name] == 0.0, (
                f"{example!r} site {name!r}: the reconstruction scores "
                f"zero and the oracle scores "
                f"{fixture.sites[index][name]!r} at point {index}."
            )

    if example != "tree_categorical":
        assert not zero_sites, (
            f"{example!r}: {zero_sites!r} contribute exactly zero to "
            f"the joint at every point, so dropping them from the "
            f"reconstruction would be undetectable. Only "
            f"`tree_categorical`'s `Beta(1, 1)` splits are flat by "
            f"identity; state why these are, or the check is weaker "
            f"than its site count suggests."
        )
        return

    assert zero_sites == ["p_left", "p_right", "p_root"], (
        f"tree_categorical: expected exactly the three `Beta(1, 1)` "
        f"splits to score zero; got {zero_sites!r}."
    )
    grid = torch.linspace(0.05, 0.95, 19)
    flat = td.Beta(1.0, 1.0).log_prob(grid)
    assert torch.equal(flat, torch.zeros_like(flat)), (
        f"`Beta(1, 1)` is not flat across (0, 1) under this torch "
        f"build ({flat.tolist()!r}), so the three splits scoring zero "
        f"is a coincidence of the fixture's values rather than an "
        f"identity, and dropping one of them would be an undetected "
        f"defect."
    )


_SMALLEST_SCORED_SITE = 0.35
"""Floor on the smallest non-flat per-site contribution any
reconstruction makes at any point, in nats.

Each site is taken at its largest contribution across the six points,
and the floor bounds the smallest of those peaks over every site of
every reconstruction. Measured: 0.3827 nats, `tree_categorical`'s
`sigma_v` under `HalfNormal(1)`. The pin tolerance at that example's
magnitude is 7.63e-06, so even the least visible site the
reconstruction checks sits about 50000 times above the noise the
comparison tolerates. The floor sits just under the measurement so
that a term shrinking toward the tolerance trips here first."""


@pytest.mark.parametrize(
    "example", sorted(_RECONSTRUCTIONS), ids=lambda name: name,
)
def test_dropping_any_scored_site_is_rejected(example: str) -> None:
    """Removing any single non-flat term moves the joint past the pin
    tolerance at some point of the set.

    The mechanical half of the sensitivity argument, and the one that
    scales with the registry: whatever sites an example grows, each
    one has to carry weight the comparison can see somewhere. At
    *some* point rather than at every point, because a term can
    legitimately vanish where the fixture sits at its zero:
    `parametric_pooling`'s soft sum-to-zero `score` is 7.1e-13 at the
    ground truth, where the snippet centres the group effects exactly,
    and 10.9 nats once the latents move. Demanding visibility at every
    point would make that site look undetectable and force it out of
    the check; demanding it somewhere is the property the per-point
    pin actually delivers.

    The measured minimum is reported so a term drifting toward zero
    surfaces as a shrinking margin rather than as a silent loss of
    coverage.
    """
    fixture = _fixture(example)
    labels = _gallery_data.perturbation_labels(len(fixture.points))
    per_point = [_reconstruct(example, index) for index in range(len(fixture.points))]
    atols = [
        _gallery_tier.reference_pin_atol(sum(terms.values()))
        for terms in per_point
    ]

    names = sorted({name for terms in per_point for name in terms})
    tightest = math.inf
    tightest_at = ""
    for name in names:
        contributions = [abs(terms[name]) for terms in per_point]
        best = max(contributions)
        if best == 0.0:
            continue
        best_index = contributions.index(best)
        assert best > atols[best_index], (
            f"{example!r}: dropping site {name!r} shifts the joint by "
            f"at most {best:.6g} nats across the whole point set, "
            f"inside the {atols[best_index]:.6g} pin tolerance. That "
            f"site is invisible to the reference pin, so nothing here "
            f"would catch the oracle forgetting it."
        )
        if best < tightest:
            tightest = best
            tightest_at = (
                f"{name!r} at point {best_index} ({labels[best_index]})"
            )

    assert tightest is not math.inf, (
        f"{example!r}: every reconstructed term is exactly zero at "
        f"every point, so the comparison asserts nothing about the "
        f"joint."
    )
    assert tightest >= _SMALLEST_SCORED_SITE, (
        f"{example!r}: the least visible non-flat site now peaks at "
        f"{tightest:.6g} nats ({tightest_at}), below the pinned floor "
        f"{_SMALLEST_SCORED_SITE:.6g}. A shrinking term is a check "
        f"losing its grip; find out why the site stopped carrying "
        f"weight rather than lowering the floor."
    )


# ---------------------------------------------------------------------
# What the reconstruction rejects.
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "mutant", _MUTANTS, ids=lambda m: m.ident,
)
def test_reconstruction_rejects_mutant(mutant: _Mutant) -> None:
    """Each catalogued defect moves the joint past the pin tolerance
    and past its own pinned floor.

    Rejection alone would be a weak claim: a comparison that failed on
    everything would satisfy it, and
    [`test_reconstruction_matches_the_oracle_per_site`][tests.transpile.test_oracle_reference_strength.test_reconstruction_matches_the_oracle_per_site]
    is the other side of that argument, accepting the faithful
    reconstruction at the same tolerance. The floor is what makes this
    a decay alarm: a mutant whose margin collapsed because the fixture
    stopped exercising the term would still clear
    `margin > atol` for a long while, and the floor trips first.

    A defect is required to be visible at **some** point, not at every
    point. `parametric_pooling`'s dropped `score` factor is exactly
    zero at the ground truth, where the snippet centres the group
    effects; only the perturbed points reject it, which is precisely
    why the pin runs at every point.
    """
    fixture = _fixture(mutant.example)
    labels = _gallery_data.perturbation_labels(len(fixture.points))
    margins: list[float] = []
    for index in range(len(fixture.points)):
        total = sum(
            _reconstruct(mutant.example, index, mutant.variant).values()
        )
        assert math.isfinite(total), (
            f"{mutant.ident}: the mutant scores {total!r} at point "
            f"{index} ({labels[index]}). A mutant that leaves the "
            f"support is a different failure from one that denotes a "
            f"different measure, and is not what this catalogue "
            f"measures."
        )
        margins.append(abs(total - fixture.joints[index]))

    best = max(margins)
    best_index = margins.index(best)
    atol = _gallery_tier.reference_pin_atol(fixture.joints[best_index])
    assert best > atol, (
        f"{mutant.ident}: {mutant.defect} The mutant's largest "
        f"deviation from the oracle across the point set is "
        f"{best:.6g} nats, inside the {atol:.6g} pin tolerance, so "
        f"this defect would pass the reference pin unnoticed."
    )
    assert best >= mutant.floor, (
        f"{mutant.ident}: {mutant.defect} Rejected with margin "
        f"{best:.6g} nats at point {best_index} "
        f"({labels[best_index]}), below the pinned floor "
        f"{mutant.floor:.6g}. The margin shrank, which means the "
        f"fixture exercises this defect less than it used to. Fix the "
        f"fixture; never lower the floor to restore green."
    )


_TIGHTEST_MUTANT_MARGIN_IN_TOLERANCES = 4000.0
"""Floor on the *tightest* rejection in the mutant catalogue, measured
in multiples of the reference pin tolerance at the rejecting point.

Measured: 5678, `mixture_model`'s dropped `Dirichlet(1, 1, 1)` prior,
which shifts the joint by exactly `log Gamma(3) = 0.6931` nats against
a 1.221e-04 tolerance. The next-narrowest is 9122
(`continuous_hmm`'s emission conditioned on the wrong state) and the
catalogue runs out to 4.9e07 (`bnn`'s MLP stripped of its `tanh`).
Declaring the tightest one is what keeps the catalogue from decaying
into a set of enormous, uninformative deviations while the small,
realistic defect quietly stops being covered."""


def test_tightest_mutant_rejection_is_declared_and_holds() -> None:
    """The catalogue's narrowest rejection stays above its declared
    floor.

    The per-mutant floors bound each defect against its own past
    measurement. This bounds the *catalogue*: it names the single
    weakest link and fails when that link weakens, which is the number
    a reader should be quoted when asking how strong the reconstruction
    check is. Reporting the strongest rejection instead would be
    meaningless, since any check that fires at all fires hardest
    somewhere.
    """
    ratios: list[tuple[float, str]] = []
    for mutant in _MUTANTS:
        fixture = _fixture(mutant.example)
        best = 0.0
        best_index = 0
        for index in range(len(fixture.points)):
            total = sum(
                _reconstruct(mutant.example, index, mutant.variant).values()
            )
            margin = abs(total - fixture.joints[index])
            if margin > best:
                best = margin
                best_index = index
        atol = _gallery_tier.reference_pin_atol(fixture.joints[best_index])
        ratios.append((best / atol, mutant.ident))

    ratios.sort()
    tightest, ident = ratios[0]
    assert tightest >= _TIGHTEST_MUTANT_MARGIN_IN_TOLERANCES, (
        f"the catalogue's narrowest rejection is now {ident} at "
        f"{tightest:.0f} tolerances, below the declared floor "
        f"{_TIGHTEST_MUTANT_MARGIN_IN_TOLERANCES:.0f}. The check has "
        f"lost grip on its smallest covered defect. Find out why that "
        f"defect moved the joint less than it used to; do not restate "
        f"the floor."
    )


def test_mutant_catalogue_covers_every_reconstruction() -> None:
    """Every reconstruction carries at least three catalogued defects,
    and every catalogued defect names a live reconstruction.

    Without the first half, adding an example to `_RECONSTRUCTIONS`
    would extend the coverage claim without extending the evidence for
    it: the new reconstruction would be asserted correct and never
    shown to be able to fail.
    """
    covered: dict[str, list[str]] = {}
    for mutant in _MUTANTS:
        covered.setdefault(mutant.example, []).append(mutant.variant)

    unknown = sorted(set(covered) - set(_RECONSTRUCTIONS))
    assert not unknown, (
        f"{unknown!r} appear in the mutant catalogue but have no "
        f"reconstruction to mutate."
    )

    thin = sorted(
        example
        for example in _RECONSTRUCTIONS
        if len(covered.get(example, [])) < 3
    )
    assert not thin, (
        f"{thin!r} carry fewer than three catalogued defects, so the "
        f"claim that their reconstruction can fail rests on almost "
        f"nothing. Add mutants covering the terms the example "
        f"actually has: a dropped prior, a mis-parameterised family, "
        f"a plate averaged instead of summed."
    )

    for example, variants in sorted(covered.items()):
        duplicates = sorted(
            variant for variant in set(variants)
            if variants.count(variant) > 1
        )
        assert not duplicates, (
            f"{example!r}: mutant variant(s) {duplicates!r} registered "
            f"more than once, so the catalogue's size overstates the "
            f"number of distinct defects it covers."
        )
        for variant in sorted(set(variants)):
            assert variant, (
                f"{example!r}: a mutant registered the empty variant, "
                f"which is the faithful reconstruction. It would be "
                f"'rejected' with margin zero."
            )


@pytest.mark.parametrize(
    "example", sorted(_RECONSTRUCTIONS), ids=lambda name: name,
)
def test_reconstruction_rejects_an_unknown_variant(example: str) -> None:
    """A variant name the reconstruction does not branch on raises.

    Without this, a typo in the catalogue would score the *faithful*
    reconstruction, and the rejection test would report the mutant as
    failing with margin zero rather than as unrecognised. Raising
    turns a mistyped defect into a named error instead of a confusing
    numeric one.
    """
    with pytest.raises(AssertionError, match="unknown variant"):
        _reconstruct(example, 0, "no_such_defect")


@pytest.mark.parametrize(
    "example", sorted(_RECONSTRUCTIONS), ids=lambda name: name,
)
def test_pin_comparison_boundary_is_the_tolerance(example: str) -> None:
    """The accept / reject boundary of the reference pin sits exactly
    at `reference_pin_atol`, in both directions.

    The tolerance is only a real constraint if a deviation just above
    it fails and one just below it passes. Asserting both pins the
    boundary as a property rather than leaving it to be inferred from
    a comparison operator, so a later `<=` quietly becoming a relative
    band is caught here.
    """
    fixture = _fixture(example)
    for index, joint in enumerate(fixture.joints):
        atol = _gallery_tier.reference_pin_atol(joint)
        assert atol > 0.0, (
            f"{example!r} point {index}: pin tolerance is {atol!r}, so "
            f"the comparison demands bit equality of two independently "
            f"ordered float32 accumulations."
        )
        inside = joint + 0.5 * atol
        outside = joint + 2.0 * atol
        assert abs(inside - joint) <= atol, (
            f"{example!r} point {index}: a deviation of half the "
            f"tolerance was rejected, so the pin is stricter than it "
            f"claims and round-off will make it flaky."
        )
        assert not abs(outside - joint) <= atol, (
            f"{example!r} point {index}: a deviation of twice the "
            f"tolerance was accepted, so the comparison is not the "
            f"absolute band it is documented to be."
        )


def test_reference_pin_is_never_looser_than_the_equivalence_check() -> None:
    """`reference_pin_atol` is bounded by the tolerance the backend
    comparison runs at, everywhere in the pinned registry.

    This is the property that makes the pin a defence rather than a
    formality. A constant oracle error is invisible to Theorem 4.1's
    quotient, so the pin is the only check that can see it; a pin
    looser than the equivalence tolerance would let an error through
    that is large enough to matter to every backend comparison
    downstream of it.
    """
    ceiling = _equivalence.adaptive_atol(n_obs=0)
    loose: list[str] = []
    for example, values in sorted(
        _gallery_tier._QVR_REFERENCE_JOINT.items()
    ):
        for index, value in enumerate(values):
            atol = _gallery_tier.reference_pin_atol(value)
            if atol > ceiling:
                loose.append(f"{example}[{index}]={atol:.3e}")
    assert not loose, (
        f"{loose!r} are pinned at a tolerance looser than the "
        f"{ceiling:.3e} floor `reference_pin_atol` promises. The pin "
        f"underwrites the equivalence check and cannot be slacker "
        f"than it."
    )


# ---------------------------------------------------------------------
# The exemptions.
# ---------------------------------------------------------------------


def _exempt_by(registry: frozenset[str]) -> list[str]:
    return sorted(set(_gallery_tier._REFERENCE_PIN_EXEMPT) & registry)


def test_reference_pin_exemptions_are_each_covered_by_a_check() -> None:
    """Every exemption from the reference pin falls to one of the two
    checks below, and none escapes both.

    `_REFERENCE_PIN_EXEMPT` is the only way an example that ships
    synthetic data can carry no pin, so it is the registry a future
    hole would grow in. Partitioning it, and asserting the partition
    is total, means an exemption of a third kind cannot be added
    without also adding the evidence for it.
    """
    exempt = set(_gallery_tier._REFERENCE_PIN_EXEMPT)
    structural = set(_exempt_by(_gallery_tier._SKIP_DATASET_LOAD_FAILED))
    nondeterministic = set(_exempt_by(_gallery_tier._SKIP_QVR_INCOMPATIBLE))

    unchecked = sorted(exempt - structural - nondeterministic)
    assert not unchecked, (
        f"{unchecked!r} claim exemption from the reference pin without "
        f"falling into either checked category: no probabilistic "
        f"program at all, or a demonstrably non-deterministic oracle. "
        f"An exemption of a new kind needs a test that establishes it, "
        f"not a comment that asserts it."
    )
    overlap = sorted(structural & nondeterministic)
    assert not overlap, (
        f"{overlap!r} are exempt on both grounds at once. An example "
        f"either has no program or has a non-deterministic one; "
        f"claiming both means one of the two skip registries is stale."
    )
    assert structural and nondeterministic, (
        f"one of the two exemption categories is empty (structural="
        f"{sorted(structural)!r}, nondeterministic="
        f"{sorted(nondeterministic)!r}), so one of the checks below "
        f"parametrizes over nothing and passes vacuously."
    )


@pytest.mark.parametrize(
    "example",
    _exempt_by(_gallery_tier._SKIP_DATASET_LOAD_FAILED),
    ids=lambda name: name,
)
def test_structurally_exempt_examples_declare_no_probabilistic_program(
    example: str,
) -> None:
    """An example exempt on structural grounds really does declare no
    probabilistic program.

    `pmf` and `tensor_contraction` export a `define`d composition
    morphism, which denotes a linear map rather than a measure, so
    there is no joint log-density a pin could hold. The claim is
    checked against the parsed module rather than against the prose:
    a `ProgramDecl` appearing in either file means the example gained
    a program and lost its exemption.
    """
    source = (_SOURCE_DIR / f"{example}.qvr").read_text(encoding="utf-8")
    module = parse(source)
    programs = [
        statement.name
        for statement in module.statements
        if isinstance(statement, ProgramDecl)
    ]
    assert not programs, (
        f"{example!r} is exempt from the reference pin on the grounds "
        f"that it declares no probabilistic program, but it declares "
        f"program(s) {programs!r}. Either the example gained a joint "
        f"and needs a `_QVR_REFERENCE_JOINT` row, or the exemption "
        f"names the wrong file."
    )
    assert _gallery_data.load_gallery_data(
        _SOURCE_DIR / f"{example}.qvr",
    ) is None, (
        f"{example!r} declares no program yet `load_gallery_data` "
        f"built a dataset for it, so something does score. Re-derive "
        f"the exemption."
    )


@pytest.mark.parametrize(
    "example",
    _exempt_by(_gallery_tier._SKIP_QVR_INCOMPATIBLE),
    ids=lambda name: name,
)
def test_nondeterministically_exempt_examples_really_redraw(
    example: str,
) -> None:
    """An example exempt because its oracle is non-deterministic really
    is non-deterministic.

    These carry a `SampledComposition` latent whose internal states
    the oracle marginalises by importance sampling, redrawing on every
    call, so the "joint" is a draw from an estimator and no value
    exists for a pin to hold. That is a strong claim, and left as
    prose it would outlive the gap it describes: an example whose
    marginalisation later became deterministic would keep an exemption
    it no longer needs, and its joint would go unpinned forever.

    Tracing the ground-truth point twice under different global RNG
    states settles it. A deterministic joint is invariant to that
    state by definition, so a disagreement is proof of the redraw and
    agreement is proof the exemption is stale.
    """
    dataset = _gallery_data.load_gallery_data(
        _SOURCE_DIR / f"{example}.qvr",
    )
    assert dataset is not None, (
        f"{example!r}: exempt on non-determinism grounds but its "
        f"synthetic-data block does not load, which is the *other* "
        f"exemption. Move the row."
    )
    monadic = dataset.monadic
    assert monadic is not None, (
        f"{example!r}: the synthetic-data block bound no compiled "
        f"program, so nothing can be traced."
    )

    traces = reference_traces(
        monadic,
        _gallery_data.point_from_dataset(dataset),
        x_input=dataset.x_input,
        observations=dataset.observations,
    )
    joints: list[float] = []
    for tr in traces:
        joint = tr.log_joint
        assert joint is not None, (
            f"{example!r}: the trace returned no `log_joint`."
        )
        joints.append(float(joint.sum().item()))

    assert len(set(joints)) > 1, (
        f"{example!r} is exempt from the reference pin because its "
        f"oracle joint is redrawn on every call, but tracing it under "
        f"two distinct RNG states produced the same value "
        f"{joints[0]!r} twice. The composition marginalisation has "
        f"become deterministic, so the exemption is stale: derive the "
        f"joint, pin it at every point, and move the row out of "
        f"`_REFERENCE_PIN_EXEMPT`."
    )
