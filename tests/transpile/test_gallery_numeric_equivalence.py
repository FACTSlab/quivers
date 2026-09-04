"""Tier-4 numeric equivalence on the documentation gallery.

For every `docs/examples/source/<example>.qvr` that ships a `###
Generating synthetic data` block in its sibling `.md`, this test:

1. Extracts and executes the data-gen snippet to recover an
   `observations` dict plus every captured `true_*` ground-truth
   parameter value.
2. Builds a multi-point
   [`Point`][tests.transpile.probes._protocol.Point] set via
   [`points_from_dataset`][tests.transpile._gallery_data.points_from_dataset]:
   the ground truth first, then deterministic in-support
   perturbations of the latents, of the observed data, and of both.
3. Runs the in-process
   [`QvrProbe`][tests.transpile.probes.qvr.QvrProbe] to compute
   `log p_QVR(θ, y) = sum_i log f_i(...)` at every point.
4. For every backend whose Docker image is locally built, runs the
   target's native log-density probe inside the container over the
   same point set and asserts constant-spread equivalence
   (`max_i | δ_i − mean δ | < atol`) per Theorem 4.1 of
   [docs/semantics/transpile-correctness.md](../../docs/semantics/transpile-correctness.md).

The point set is what gives the constant-spread assertion its teeth.
Evaluated at one point the spread is identically zero, so the check
passes whatever the backend computed. Varying the latents catches a
mis-scored prior; varying the *data* catches a backend that drops a
data-dependent term while keeping a stable offset as the latents move
(a Stan `~` sampling statement discards data-only summands, for
instance). Both must vary before the assertion means anything.

Each cell resolves to one of three pre-declared outcomes:

1. `(backend, example) in _EXPECTED_TRANSPILE_RAISES`: the
   pipeline MUST `pytest.raises(UnsupportedConstruct)` with the
   pinned kind-prefix.
2. Cell falls in one of the three `_SKIP_*` registries: a known
   environmental gap (missing data block, QVR-probe incompatibility,
   backend probe script lacking shape registration for arbitrary
   gallery datasets). `pytest.skip` with the diagnostic.
3. Neither: the pipeline MUST emit non-empty bytes, the QVR probe
   MUST evaluate to a finite log-density, and the in-container
   probe MUST return a vector whose constant-spread offset from
   the QVR reference is below the equivalence tolerance.

When neither Docker images nor backend Python runtimes are
available (the typical local-dev state), the test still exercises
the QVR-side trace: every gallery example whose `.md` ships data
gets a verified `log p_QVR` value, which is the strongest in-process
correctness signal we can produce without target runtimes.

The constant-spread contract is blind by construction to a
*point-independent* error in the reference itself: adding the same
constant to every `log p_QVR` leaves every difference `δ_i` shifted
by that constant and every spread unchanged. The oracle therefore
carries a second, independent obligation, and this module holds it:
[`_QVR_REFERENCE_JOINT`][tests.transpile.test_gallery_numeric_equivalence._QVR_REFERENCE_JOINT]
pins the reference joint at **every** point of the set, the pin is
**mandatory** for every example that scores a joint (an example
without one is an assertion failure, never a silent pass), and the
pin tolerance is
[`reference_pin_atol`][tests.transpile.test_gallery_numeric_equivalence.reference_pin_atol],
which is never looser than the equivalence tolerance it underwrites.
The pinned numbers themselves are re-derived from raw
`torch.distributions` by
[`test_oracle_reference_strength`][tests.transpile.test_oracle_reference_strength]
for every example the backend tier cannot reach.
"""

from __future__ import annotations

import math
import pathlib

import pytest
import torch

from quivers.dsl.parser import parse
from quivers.transpile import UnsupportedConstruct, transpile
from tests.transpile import _docker, _equivalence, _gallery_data
from tests.transpile.probes.qvr import QvrProbe


_BACKENDS_WITH_IMAGES = {
    "stan": ("panproto-test-stan", "stan", "stan.py"),
    "numpyro": ("panproto-test-numpyro", "py", "numpyro.py"),
    "pyro": ("panproto-test-pyro", "py", "pyro.py"),
    "pymc": ("panproto-test-pymc", "py", "pymc.py"),
    "edward2": ("panproto-test-edward2", "py", "edward2.py"),
    "turing": ("panproto-test-julia", "jl", "turing.jl"),
    "gen": ("panproto-test-julia", "jl", "gen.jl"),
    "webppl": ("panproto-test-node", "js", "webppl.py"),
    "jags": ("panproto-test-jags", "jags", "jags.py"),
    "bugs": ("panproto-test-bugs", "bugs", "jags.py"),
}


def _gallery_cells() -> list[pathlib.Path]:
    return _gallery_data.gallery_examples_with_data()


# ----------------------------------------------------------------------
# Pre-declared cell outcomes.
#
# Population workflow when a previously-skipped infrastructure gap
# closes (probe script gains shape registration; QVR trace grows a
# primitive): drop the cell from the corresponding registry, re-run,
# pass.
#
# Population workflow when a renderer regression introduces a
# transpile gap: pin the cell in `_EXPECTED_TRANSPILE_RAISES` with
# the new kind-prefix; the test then makes that raise a contract
# instead of a silent xfail.
# ----------------------------------------------------------------------

# Cells where the pipeline raises `UnsupportedConstruct` and the
# raise is the documented support boundary. `pytest.raises` asserts
# the kind-prefix; a closed gap surfaces as a test failure and a
# regression (different kind) surfaces as a mismatch. Each group
# carries a one-line reason for the boundary.
_EXPECTED_TRANSPILE_RAISES: dict[tuple[str, str], str] = {
    # mixture_model names the `MixtureNormal` likelihood, which the
    # BUGS lower family registry has no target name for. Every other
    # backend resolves it and its cell is live.
    ("bugs", "mixture_model"): "family:MixtureNormal",
    # parametric_pooling samples the `school_effects` sub-program
    # (program-as-distribution); no backend resolves it to a target
    # family.
    ("bugs", "parametric_pooling"): "family:school_effects",
    ("edward2", "parametric_pooling"): "family:school_effects",
    ("gen", "parametric_pooling"): "family:school_effects",
    ("jags", "parametric_pooling"): "family:school_effects",
    ("numpyro", "parametric_pooling"): "family:school_effects",
    ("pymc", "parametric_pooling"): "family:school_effects",
    ("pyro", "parametric_pooling"): "family:school_effects",
    ("stan", "parametric_pooling"): "family:school_effects",
    ("turing", "parametric_pooling"): "family:school_effects",
    ("webppl", "parametric_pooling"): "family:school_effects",
    # pmf / tensor_contraction carry `composition` (and `contraction`)
    # declarations; no PPL backend has a surface for them. Both
    # examples still score a joint, because their `.md` snippets wrap
    # the compiled composition in a `MonadicProgram`, so both carry a
    # `_QVR_REFERENCE_JOINT` row. Every one of their cells being a
    # pinned raise means no container re-derives that row, and the
    # witness therefore has to come from the raw-`torch.distributions`
    # side in `test_oracle_reference_strength.py`.
    ("bugs", "pmf"): "composition_decl",
    ("bugs", "tensor_contraction"): "composition_decl",
    ("edward2", "pmf"): "composition_decl",
    ("edward2", "tensor_contraction"): "composition_decl",
    ("gen", "pmf"): "composition_decl",
    ("gen", "tensor_contraction"): "composition_decl",
    ("jags", "pmf"): "composition_decl",
    ("jags", "tensor_contraction"): "composition_decl",
    ("numpyro", "pmf"): "composition_decl",
    ("numpyro", "tensor_contraction"): "composition_decl",
    ("pymc", "pmf"): "composition_decl",
    ("pymc", "tensor_contraction"): "composition_decl",
    ("pyro", "pmf"): "composition_decl",
    ("pyro", "tensor_contraction"): "composition_decl",
    ("stan", "pmf"): "composition_decl",
    ("stan", "tensor_contraction"): "composition_decl",
    ("turing", "pmf"): "composition_decl",
    ("turing", "tensor_contraction"): "composition_decl",
    ("webppl", "pmf"): "composition_decl",
    ("webppl", "tensor_contraction"): "composition_decl",
    # zip_regression names `ContinuousBernoulli` and
    # kumaraswamy_bounded_outcome names `Kumaraswamy`, and neither
    # family has a JAGS or a BUGS target name. The two engines part
    # company on both: the JAGS renderer writes each density out in
    # `log` and `pow` alone and adds it through the zeros trick, so
    # its cells are live and score the model up to the lift the trick
    # pays. The BUGS renderer carries no such path and its cells stay
    # a raise.
    ("bugs", "kumaraswamy_bounded_outcome"): "family:Kumaraswamy",
    ("bugs", "zip_regression"): "family:",
    # beta_binomial_ab_test observes `BetaBinomial`. Every other
    # backend either has the family natively or reaches it through
    # the closed-form marginal: JAGS writes that marginal into the
    # joint with the zeros trick, whose Poisson carrier has to be
    # bound to data, and JAGS binds it in the `data { ... }` block
    # the emitted source opens. The BUGS language has no such block,
    # so the carrier has nowhere to be declared inside a
    # self-contained model file, and
    # `renderers/bugs.py::_NO_BUGS_DISTRIBUTION` records the family
    # as unreachable rather than emitting source that references an
    # undeclared node. The boundary is the BUGS *language*, not the
    # engine: the `panproto-test-bugs` image runs JAGS, which is why
    # the sibling `jags` cell is live and scores the same model.
    ("bugs", "beta_binomial_ab_test"): (
        "family:BetaBinomial:no-bugs-distribution"
    ),
    # gru_lm and lstm_lm build their gate arguments with `+` between
    # two `Hidden`-wide operands. The BUGS model language has no
    # elementwise vector arithmetic: only the contracted product
    # lowers, to `inprod(a, b)`. Both engines therefore reject the
    # `let` before any site is emitted, which is a statement about the
    # target language rather than about the oracle. Both examples sit
    # in `_SKIP_QVR_INCOMPATIBLE`, and the test consults the raises
    # first, so pinning these four turns four cells that asserted
    # nothing into four that assert the boundary. The remaining eight
    # targets render both examples and stay carried by that registry.
}

# bnn's `net` morphism draws its mean from an `mlp` param source. The
# network weights are model-internal, absent from both the wire form
# and the sample sites, so no backend can reconstruct the mean; the
# transpiler raises on every backend. The same holds for every model
# below: each hides part of its structure in a `param_source` network
# rather than writing it as a program whose steps are declared sites.
# `scan(cell)` denotes one draw per sequence position over
# intermediate states the program never names, over a sequence axis
# that is not a declared object. The expansion pass refuses before any
# renderer runs, so the kind is the same on every target, and pinning
# it asserts the boundary where a skip asserted nothing.
for _scan_model in (
    "bidirectional_rnn_lm",
    "gru_lm",
    "lstm_lm",
    "vanilla_rnn_lm",
):
    for _scan_backend in _BACKENDS_WITH_IMAGES:
        _EXPECTED_TRANSPILE_RAISES[(_scan_backend, _scan_model)] = (
            "scan:no-lowering"
        )

for _neural_model in (
    "bnn",
    "deep_markov",
    "seq2seq",
    "transformer_lm",
    "vae",
):
    for _neural_backend in _BACKENDS_WITH_IMAGES:
        _EXPECTED_TRANSPILE_RAISES[(_neural_backend, _neural_model)] = (
            "param-source:mlp"
        )

# Gallery examples whose `.md` synthetic-data snippet fails to
# `exec` (raises at runtime, or never sets the `observations`
# dict). Numeric evaluation has no point set, so the cell skips.
#
# The registry is empty: every gallery example's snippet executes and
# leaves an `observations` dict behind, including the two structural
# algebra examples (`pmf`, `tensor_contraction`), whose snippets wrap
# the compiled composition in a `MonadicProgram` with entrywise
# `Normal(0, 1)` priors over each declared arrow and a
# `Normal(score, 0.5)` likelihood over the contraction. Their joints
# are pinned like every other example's. Keeping the registry (rather
# than deleting it) keeps the skip path a named, testable branch, so a
# snippet that stops executing is recorded here rather than turning
# into a bare load failure.
_SKIP_DATASET_LOAD_FAILED: frozenset[str] = frozenset()

# Gallery examples the in-process [`QvrProbe`][tests.transpile.probes.qvr.QvrProbe]
# cannot score to a deterministic, correct joint.
#
# Each of these carries a `sample h <- backbone` latent whose backbone
# is a `SampledComposition` over continuous intermediate objects
# (RNN/LSTM/GRU scan cells, attention and feed-forward Kleisli chains).
# The per-timestep gates and per-layer composition latents are internal
# to that composition and are never surfaced as trace sites, so no
# point entry can clamp them and `assert_all_latents_clamped` has
# nothing to look at. What the oracle reports for such a program is
# therefore decided entirely by how the composition performs the
# integral, and on that question the eight split two ways. Both grounds
# are measured rather than asserted, and
# `test_oracle_reference_strength.py::test_composition_exemption_grounds_partition_the_registry`
# requires them to cover this registry exactly once:
#
# 1. For `bidirectional_rnn_lm`, `deep_markov`, `seq2seq`,
#    `transformer_lm`, and `vae` the joint is a property of the
#    quadrature rule rather than of the model, so it is the output of
#    an estimator and there is no value a pin could hold. This is a
#    measurement, not an inference from the shape of the program:
#    forcing every `SampledComposition.n_intermediate` in the compiled
#    module from its default 100 down to 23 and re-scoring the same
#    six points moves the joint by 1.04 nats (`seq2seq`), 1.45
#    (`deep_markov`), 12.97 (`vae`), 1.36e04 (`transformer_lm`) and
#    9.17e05 (`bidirectional_rnn_lm`) at the point each moves most.
#    Two rules integrating the same kernel against the same data would
#    return the same density; these return different numbers, and the
#    smallest of them is already thousands of times the pin tolerance
#    at its magnitude.
#    `test_oracle_reference_strength.py::test_quadrature_exempt_examples_have_a_rule_dependent_joint`
#    holds each example to a floor on that ratio under a 37-node rule,
#    so a quadrature converging toward the tolerance, which is the
#    event that would make these pinnable, surfaces as a shrinking
#    margin rather than as a silent change of grounds.
# 2. For `gru_lm`, `lstm_lm`, and `vanilla_rnn_lm` the composite site
#    scores identically zero, so the reported number is the emission
#    likelihood alone and omits the `~ Normal` recurrent transition
#    density the source declares at every step. That flat site is also
#    plate-shaped where the emission site is scalar, so the sum
#    broadcasts and the joint carries the emission likelihood once per
#    scored row: measured exactly 32 times at all six points of all
#    three,
#    `test_oracle_reference_strength.py::test_flat_latent_exempt_examples_inflate_their_emission_per_row`.
#    The excess is `(rows - 1)` times a likelihood the perturbation
#    schedule moves, so it is not the additive constant Theorem 4.1's
#    quotient absorbs; it varies across the point set by 3.2e06
#    equivalence tolerances (`gru_lm`, `lstm_lm`) and 2.7e08
#    (`vanilla_rnn_lm`). Dropping these three rows would therefore
#    produce 30 failing cells rather than 30 recovered ones.
#
# The joints themselves are bitwise stable across global RNG seeds,
# which `test_oracle_determinism.py::test_composition_marginalised_models_are_bitwise_deterministic`
# measures at every point of the set; stability is what makes the
# quadrature probe meaningful, and is not on its own enough to pin
# against. It is emphatically not enough to lift the exemption: a
# joint that reproducibly counts its likelihood 32 times is
# deterministic and wrong, which is why both grounds above are
# measured against the density rather than against the generator.
# Pinning these needs the composition's integral made a rule whose
# value has converged, and its inner latents exposed as sites.
_SKIP_QVR_INCOMPATIBLE: frozenset[str] = frozenset({
    "bidirectional_rnn_lm",
    "gru_lm",
    "lstm_lm",
    "transformer_lm",
    "vanilla_rnn_lm",
})

# ----------------------------------------------------------------------
# The reference pin.
#
# Theorem 4.1's quotient by an additive constant is what makes the
# backend comparison robust to differing base measures, and it is also
# what makes it blind: an oracle that is wrong by the same amount at
# every point produces exactly the spread of a correct one. Nothing on
# the backend side can see such an error, so the reference carries its
# own pin, and the pin has to hold everywhere the comparison does.
#
# Three properties, each asserted rather than assumed:
#
# 1. **Total.** Every gallery example that scores a joint has an entry
#    here or a justified row in `_REFERENCE_PIN_EXEMPT`, and
#    `test_gallery_reference_pin_registry_is_total` fails when the two
#    registries stop covering the gallery exactly. A deleted row, or a
#    new example, is a failure rather than a silently weaker suite.
# 2. **Per point.** An entry pins the joint at every point of
#    `points_from_dataset`, in schedule order, not only at the ground
#    truth. A single-point pin cannot see a *data-dependent* oracle
#    error, and neither can the constant-spread check when the same
#    error rides on both sides, so between them that class would be
#    invisible.
# 3. **Tight.** The comparison runs at `reference_pin_atol`, which is
#    never looser than the equivalence tolerance it underwrites.
#
# Every value here is reproduced from an independent computation on
# each run: an example with live backend cells is re-derived by those
# containers, and every example without one is re-derived from raw
# `torch.distributions` in `test_oracle_reference_strength.py`.
# `test_oracle_reference_strength.py::test_every_pinned_example_has_an_independent_witness`
# asserts that split covers the registry with nothing left over.
#
# Update an entry only after re-deriving the joint independently; a
# drift here is either an oracle regression or a deliberate
# ground-truth change, and the two are told apart by the raw-torch
# reconstruction, not by the number's plausibility.
# ----------------------------------------------------------------------
_QVR_REFERENCE_JOINT: dict[str, tuple[float, ...]] = {
    "ar1": (
        -24.030813217163086,
        -121.17807006835938,
        -44.41229248046875,
        -37.35548400878906,
        -44.10717010498047,
        -38.02667236328125,
    ),
    "bayesian_regression": (
        -54.560791015625,
        -67.99243927001953,
        -81.71551513671875,
        -117.940673828125,
        -164.50860595703125,
        -83.58819580078125,
    ),
    "beta_binomial_ab_test": (
        -70.50210571289062,
        -82.40196228027344,
        -70.82966613769531,
        -76.28248596191406,
        -71.93524932861328,
        -75.79243469238281,
    ),
    "beta_regression": (
        26.09986114501953,
        12.179271697998047,
        21.649688720703125,
        4.467376708984375,
        20.713706970214844,
        16.414939880371094,
    ),
    "bnn": (
        -412.09527587890625,
        -412.09527587890625,
        -433.6674499511719,
        -431.0032653808594,
        -412.09527587890625,
        -416.7749938964844,
    ),
    "ccg": (
        -38.62127685546875,
        -38.8963623046875,
        -38.4906005859375,
        -43.69182586669922,
        -41.17519760131836,
        -40.177955627441406,
    ),
    "changepoint": (
        -131.08474731445312,
        -155.9451446533203,
        -133.2001953125,
        -145.42013549804688,
        -188.2016143798828,
        -135.9104766845703,
    ),
    "continuous_hmm": (
        -21.395313262939453,
        -21.497692108154297,
        -21.478761672973633,
        -22.007896423339844,
        -22.696565628051758,
        -21.73342514038086,
    ),
    "custom_rules": (
        -38.62127685546875,
        -38.8963623046875,
        -38.4906005859375,
        -43.69182586669922,
        -41.17519760131836,
        -40.177955627441406,
    ),
    "factor_analysis": (
        -132.61660766601562,
        -214.58114624023438,
        -162.3575897216797,
        -226.3590087890625,
        -187.85678100585938,
        -168.95579528808594,
    ),
    "gamma_regression": (
        -69.07034301757812,
        -80.96452331542969,
        -76.0668716430664,
        -93.28099822998047,
        -72.47667694091797,
        -76.60382080078125,
    ),
    "half_student_t_hierarchical": (
        -35.37989044189453,
        -75.25849914550781,
        -62.39704132080078,
        -98.7879409790039,
        -56.45060729980469,
        -83.33959197998047,
    ),
    "hmm": (
        266.3363342285156,
        266.090087890625,
        266.0382080078125,
        266.6028137207031,
        266.080810546875,
        266.5773620605469,
    ),
    "horseshoe_regression": (
        -64.92990112304688,
        -85.11199951171875,
        -82.23753356933594,
        -145.4669647216797,
        -135.35128784179688,
        -84.14229583740234,
    ),
    "irt_2pl": (
        -69.23272705078125,
        -76.04976654052734,
        -85.58287811279297,
        -78.32916259765625,
        -71.45870971679688,
        -80.12108612060547,
    ),
    "kumaraswamy_bounded_outcome": (
        20.225561141967773,
        1.1973302364349365,
        21.390592575073242,
        15.331405639648438,
        6.32975959777832,
        15.959518432617188,
    ),
    "lda": (
        -18.173583984375,
        -18.84844970703125,
        -444.3291015625,
        -437.18743896484375,
        -20.41473388671875,
        -442.911376953125,
    ),
    "linear_gaussian_ssm": (
        -6.176448822021484,
        -6.644833564758301,
        -6.308502197265625,
        -6.5248260498046875,
        -6.4006500244140625,
        -6.19073486328125,
    ),
    "logistic_noise_regression": (
        -80.29877471923828,
        -94.24589538574219,
        -92.63084411621094,
        -86.77716827392578,
        -107.02436065673828,
        -84.29177856445312,
    ),
    "mixture_model": (
        -189.3568115234375,
        -197.78488159179688,
        -197.35861206054688,
        -214.3572998046875,
        -197.0948486328125,
        -196.84445190429688,
    ),
    "multimodal_tlg": (
        -38.62127685546875,
        -38.8963623046875,
        -38.4906005859375,
        -43.69182586669922,
        -41.17519760131836,
        -40.177955627441406,
    ),
    "negbin_regression": (
        -203.68475341796875,
        -225.95242309570312,
        -270.5653381347656,
        -302.58197021484375,
        -221.177734375,
        -268.622314453125,
    ),
    "parametric_pooling": (
        -16.053003311157227,
        -21.250812530517578,
        -15.52226448059082,
        -19.705469131469727,
        -26.930932998657227,
        -18.749011993408203,
    ),
    "pcfg": (
        -23.014850616455078,
        -26.371784210205078,
        -25.505828857421875,
        -23.08839225769043,
        -22.200294494628906,
        -27.879863739013672,
    ),
    "pmcfg": (
        -38.62127685546875,
        -38.8963623046875,
        -38.4906005859375,
        -43.69182586669922,
        -41.17519760131836,
        -40.177955627441406,
    ),
    "pmf": (
        -96.90896606445312,
        -128.39248657226562,
        -98.03665161132812,
        -123.15997314453125,
        -137.48460388183594,
        -100.93830871582031,
    ),
    "ppca": (
        -67.73426818847656,
        -231.67868041992188,
        -105.8104476928711,
        -274.6122741699219,
        -188.58070373535156,
        -141.64019775390625,
    ),
    "quantifier_scope": (
        -38.62127685546875,
        -38.8963623046875,
        -38.4906005859375,
        -43.69182586669922,
        -41.17519760131836,
        -40.177955627441406,
    ),
    "stochastic_volatility": (
        -307.8003845214844,
        -497.64605712890625,
        -374.3330078125,
        -449.9366760253906,
        -448.69561767578125,
        -387.8696594238281,
    ),
    "survival_weibull": (
        -28.67302131652832,
        -39.528465270996094,
        -37.15167236328125,
        -47.73482894897461,
        -234.78810119628906,
        -36.604496002197266,
    ),
    "tensor_contraction": (
        -52.925621032714844,
        -64.15614318847656,
        -52.4114875793457,
        -61.599327087402344,
        -67.39108276367188,
        -52.22042465209961,
    ),
    "tree_categorical": (
        -155.84384155273438,
        -278.3675842285156,
        -167.4322967529297,
        -167.1120147705078,
        -157.07876586914062,
        -171.585205078125,
    ),
    "type_logical": (
        -38.62127685546875,
        -38.8963623046875,
        -38.4906005859375,
        -43.69182586669922,
        -41.17519760131836,
        -40.177955627441406,
    ),
    "zip_regression": (
        -651.6888427734375,
        -664.6071166992188,
        -755.0150146484375,
        -759.4091186523438,
        -665.1041259765625,
        -747.2142333984375,
    ),
    "deep_markov": (
        -70.9559326171875,
        -71.23564910888672,
        -70.98413848876953,
        -70.98515319824219,
        -70.94621276855469,
        -71.1627197265625,
    ),
    "seq2seq": (
        -15812.892578125,
        -15826.595703125,
        -15821.7587890625,
        -15847.22265625,
        -15836.48046875,
        -15819.671875,
    ),
    "vae": (
        -41.807464599609375,
        -42.399044036865234,
        -42.460121154785156,
        -42.049835205078125,
        -42.313289642333984,
        -42.44348907470703,
    ),
}

# Gallery examples that carry synthetic data but score no joint the
# pin could hold, each with the reason. An entry is an assertion, not
# an escape hatch: `test_gallery_reference_pin_registry_is_total`
# requires the reason to name a registry that independently agrees the
# example has no reference, so an exemption cannot outlive the gap it
# describes.
_REFERENCE_PIN_EXEMPT: dict[str, str] = {
    # Sequence models carrying a `SampledComposition` latent. Every one
    # of them scores a bitwise-stable joint, so non-determinism is not
    # the ground and never was the one that mattered: what disqualifies
    # the number is that it is not the model's density. Five report the
    # output of a quadrature rule, which moves by 1.04 to 9.17e05 nats
    # when the rule's node count changes; three report their emission
    # likelihood inflated once per scored row, because the composite
    # site carries no density at all. Each sits in
    # `_SKIP_QVR_INCOMPATIBLE`, whose comment carries the full
    # diagnosis and names the test that measures each ground.
    "bidirectional_rnn_lm": "oracle joint is a quadrature output, not a density",
    "gru_lm": "oracle joint omits the recurrent density and inflates the emission",
    "lstm_lm": "oracle joint omits the recurrent density and inflates the emission",
    "transformer_lm": "oracle joint is a quadrature output, not a density",
    "vanilla_rnn_lm": "oracle joint omits the recurrent density and inflates the emission",
}

_REFERENCE_PIN_ULP_BUDGET = 8
"""Round-off budget for
[`reference_pin_atol`][tests.transpile.test_gallery_numeric_equivalence.reference_pin_atol],
in units of the float32 grid spacing at the pinned magnitude.

The reference joint reaches the harness as
``float(trace.log_joint.sum().item())``: the exact float64 widening of
a float32 accumulator. Two evaluations of the *same* density therefore
agree exactly whenever they accumulate in the same order, and differ
only in the low bits of that accumulator when they do not (a different
reduction kernel, SIMD width, or BLAS). The budget is calibrated
against a direct measurement of that re-association effect rather than
against a guess: the raw-`torch.distributions` reconstructions in
`test_oracle_reference_strength.py` sum the same per-site terms in a
different order and in different groupings, and across all six models
and all six points their largest disagreement with the trace is
**one** float32 ULP (`continuous_hmm`, 6.10e-05 at magnitude 736;
`linear_gaussian_ssm`, 1.53e-05 at magnitude 219). Eight ULPs is three
bits of headroom above the measured worst case.

This is a headroom figure, not a necessity: the oracle is bit-exact
run to run and across `torch.set_num_threads`, both measured. A
failure at this budget is a real change in the density, not noise."""


def _float32_ulp(value: float) -> float:
    """Spacing of the float32 grid at `value`.

    `math.frexp` returns ``(m, e)`` with ``|value| = m * 2**e`` and
    ``m`` in ``[0.5, 1)``, so the leading significand bit has weight
    ``2**(e - 1)`` and the trailing bit of a 24-bit significand has
    weight ``2**(e - 24)``.
    """
    magnitude = abs(value)
    if not math.isfinite(magnitude):
        raise ValueError(
            f"_float32_ulp needs a finite value; got {value!r}. A "
            f"non-finite reference is a broken pin, not a tolerance "
            f"question."
        )
    if magnitude == 0.0:
        return math.ldexp(1.0, -24)
    _, exponent = math.frexp(magnitude)
    return math.ldexp(1.0, exponent - 24)


def reference_pin_atol(reference: float) -> float:
    """Absolute tolerance for the reference pin at `reference`.

    Two bounds, and the tighter one wins.

    The first is
    [`_REFERENCE_PIN_ULP_BUDGET`][tests.transpile.test_gallery_numeric_equivalence._REFERENCE_PIN_ULP_BUDGET]
    ULPs of the float32 grid at the pinned magnitude, floored at the
    same budget taken at magnitude 1. The floor keeps a joint that
    happens to land near zero from demanding bit equality: such a
    joint is still a sum of order-one terms, and the grid at 1 is the
    finest resolution those terms carry.

    The second is the equivalence tolerance from
    [`adaptive_atol`][tests.transpile._equivalence.adaptive_atol] at
    its floor, which is the tolerance
    [`assert_log_density_match`][tests.transpile._equivalence.assert_log_density_match]
    holds the backends to. Taking the minimum is the whole point of
    the function: a constant oracle error is invisible on the backend
    side, so the pin is the only defence against it, and a defence
    looser than the check it underwrites defends nothing. It also ties
    the two together in code, so the pin cannot be left behind if the
    equivalence floor ever moves.

    Across the 192 pinned values the ULP bound binds for every one.
    The largest pinned magnitudes are `continuous_hmm`'s 717 to 736,
    where eight float32 ULPs are 4.88e-04, still inside the 5e-04
    equivalence floor, so the loosest pin in the registry is
    4.88e-04. The tightest is 9.54e-07, the magnitude-1 floor, which
    binds wherever a joint lands below 2
    (`kumaraswamy_bounded_outcome` at its first latents point,
    magnitude 1.20). Measured at the ground-truth point of every
    registry entry, the band is between 1173 times (`hmm`) and 3190
    times (`survival_weibull`) tighter than the
    `1e-3 * |reference| + 2e-2` relative band it replaces.
    """
    ulp_bound = _REFERENCE_PIN_ULP_BUDGET * max(
        _float32_ulp(reference), _float32_ulp(1.0),
    )
    return min(_equivalence.adaptive_atol(n_obs=0), ulp_bound)

# Gallery examples that genuinely carry no perturbable observation, so
# their point set moves the latents alone. Each entry states why the
# example's observed data cannot move, and
# `test_gallery_multipoint_set_is_in_support_and_varies` asserts the
# claim: an example listed here must have a frozen data section, and an
# example not listed here must have a data section that moves.
#
# Populate an entry only for a model whose every observation is an
# index-valued covariate or a degenerate single-valued response, never
# to quiet a data section the harness merely fails to perturb. A frozen
# data section makes the constant-spread check blind to a backend that
# drops a data-dependent term, which is the single failure mode the
# multi-point set exists to catch.
_NO_PERTURBABLE_OBSERVATION: dict[str, str] = {}

# (backend, example) cells whose measured log-density cannot yet be
# compared against the QVR reference, each grouped under the single
# defect that blocks it. Every entry has been run against its
# container and carries the error the run actually produced, so the
# comment names a live blocker rather than a historical one.
#
# Closure path: fix the named defect in its owning file, re-measure
# the cell with the six-point set, drop it from this registry. A cell
# whose spread is constant to within `adaptive_atol` belongs outside
# the registry; nothing else does.
_SKIP_PROBE_INCOMPATIBLE: frozenset[tuple[str, str]] = frozenset({
    # The scan-bearing sequence models are absent on every target:
    # their transpile raises before a probe could run, and the raise
    # is pinned in `_EXPECTED_TRANSPILE_RAISES`, which asserts
    # something a skip never could.
    # continuous_hmm / linear_gaussian_ssm: the emission is faithful
    # and the fixture cannot feed it. Two blockers, measured
    # separately, and the second is the one that decides these rows.
    #
    # 1. A Kleisli morphism declared `~ Family` with no
    #    `[param_source=...]` takes the default linear source, whose
    #    affine weights are drawn when the module compiles. The
    #    lowering emits them as inputs named
    #    `<morphism>_param_weight` / `<morphism>_param_bias`, and
    #    nothing supplies them: the snippet does not know the names
    #    and the trace does not clamp them. This one is mechanical.
    #    Deriving the tensors from the compiled program (the site's
    #    `_step_<site>.param_source.linear.*`, keyed by the morphism
    #    each site draws from) and adding them to `Point.data` clears
    #    it, and was measured to: with the weights wired, Stan stops
    #    reporting them and moves on to the next name.
    #
    # 2. It moves on to `state`, and that one is not plumbing. Both
    #    programs are declared without a plate
    #    (`program generative_step : State -> State`, one step, no
    #    `observe o : Resp`), so the emitted program is single-row and
    #    declares `vector[16] state`. The fixture evaluates the same
    #    program batched over 32 rows, which is what the in-process
    #    trace broadcasts to and what the pinned reference and the
    #    reconstruction in `test_oracle_reference_strength` both
    #    score. Supplying `x_input` cannot bridge that: a `(32, 16)`
    #    matrix against a declared `(16)` is a rank mismatch, not a
    #    missing argument.
    #
    # So the cells stay until the two sides agree on how many rows a
    # plate-less program scores: either the examples declare the plate
    # their data has, or the fixture scores one row and the pin and
    # reconstruction move with it. Choosing between those is a
    # question about what a plate-less program means, not a defect in
    # the emission, which is why this is a registry entry and not a
    # renderer fix.
    #   numpyro / pyro: model() missing 'state' (and 'driver')
    #   stan: dims declared=(16), dims found=() for `state`
    #   pymc: ShapeError, actual 2 != expected 1
    #   edward2: cannot convert None to a Tensor
    #   turing / gen: no method matching model(::Matrix{Float64})
    #   jags / bugs: dimension mismatch in subset expression of `o`
    #   webppl: Parameter "mu" should be of type "real"
    # tree_categorical on Stan alone. The example binds
    # `let cell0 = cell_score[0, 0]`, a literal index into the rank-2
    # score table, and the Stan renderer carries that subscript
    # through unchanged: the emitted transformed-parameters block
    # holds `cell0[m_Resp] = cell_score[0,0];`. Stan indexes from 1,
    # so cmdstan rejects the program at `log_prob` time with
    # "index 0 out of range; expecting index to be between 1 and 12",
    # and no point of the set is ever scored. The renderer's literal
    # index path in `src/quivers/transpile/renderers/stan.py` owns
    # the off-by-one; every other 1-based target already rebases the
    # same expression, which is why JAGS and BUGS score it.
    #
    # The other nine cells are live and pass. Measured spread against
    # the QVR reference over the six-point set: numpyro / pymc /
    # turing / gen / jags / bugs / webppl 7.49e-05 (the float64
    # backends agree with each other far below that and inherit the
    # same offset from the float32 reference), pyro 2.91e-05,
    # edward2 1.78e-05, all under the 5e-04 constant-spread floor.
    ('stan', 'tree_categorical'),
    # hmm: the axis-role derivation now ranks `sample initial_row :
    # State <- Dirichlet(1.0) [over=State]` as one simplex rather than
    # a batch of them, so the typed backends that reported the rank
    # clash score the model: edward2 reproduces the reference exactly
    # (spread 0.0) and pymc to 1.16e-05, both out of this registry
    # beside numpyro and pyro. These five carry the blockers that
    # survive, each the error its own container returned on the
    # six-point set.
    #   gen: `Gen.assess` requires every traced address to be
    #     constrained, so the marginalized `state` surfaces as
    #     `KeyError: key :state not found` before anything is scored.
    #     The same missing log-weight primitive blocks gen/lda.
    #   jags: `RendererBase.explicit_latent_scope` lowers the
    #     marginalize to a live `IRSample(state)` and drops the
    #     reduction, so the engine rejects the model at
    #     `console.update` with `Error in node state / Cannot
    #     normalize density` rather than integrating it out. The
    #     `bugs` cell is live and failing on the same engine error
    #     from its own renderer's emission, so it stays a visible
    #     failure rather than a sixth row here.
    #   stan: the program now compiles and scores every point, and
    #     disagrees with the reference by a spread of 2.19 nats over
    #     the six-point set, four thousand times the 5e-04 floor. The
    #     offset is not constant, so the emitted measure differs from
    #     the reference rather than differing by a base measure.
    #   turing: the renderer hands `Categorical` a row of the
    #     `emission_rows` matrix, and Distributions.jl rejects the
    #     `Vector{Float64}` where its `SubArray`-parameterised
    #     constructor was resolved (`MethodError: Cannot convert an
    #     object of type Vector{Float64} to an object of type
    #     SubArray{...}`).
    # The `webppl` cell is out of this registry: its `Categorical`
    # emission now carries the support WebPPL requires beside the
    # probabilities, and it scores the reference.
    ('gen', 'hmm'),
    ('jags', 'hmm'),
    ('stan', 'hmm'),
    ('turing', 'hmm'),
    # lda: the five backends that integrate the topic latent
    # correctly are out of this registry; these four each carry a
    # distinct blocker.
    #   gen: `Gen.assess` requires every traced address to be
    #     constrained and the `@gen` DSL has no log-weight primitive,
    #     so the marginalized `z` surfaces as KeyError (:z, 1).
    #   jags: `RendererBase.explicit_latent_scope` lowers the
    #     marginalize to a live `IRSample(z)` and drops the
    #     reduction, so the emitted measure lives on a strictly
    #     larger space; the engine now rejects the model outright
    #     with `Error in node z[6] / Cannot normalize density` rather
    #     than scoring it. The zeros trick `jags.py::_emit_score`
    #     already uses is the closure. The `bugs` cell is live and
    #     failing on the same engine error from its own renderer's
    #     emission, so it stays a visible failure rather than a sixth
    #     row here.
    #   turing: the gathered per-word topic weights index a scalar,
    #     raising BoundsError at index [2].
    #   webppl: the Dirichlet concentration reaches WebPPL as a plain
    #     JS array rather than a vector.
    ('gen', 'lda'),
    ('jags', 'lda'),
    ('turing', 'lda'),
    ('webppl', 'lda'),
    # zip_regression: the backends whose Poisson uses an `xlogy` form
    # integrate the zero-inflation indicator correctly and are out of
    # this registry; these three do not.
    #   numpyro: its `Poisson` computes log(rate) * value directly,
    #     so the z = 0 atom's rate of exactly 0 yields nan at every
    #     y == 0 observation. The emitted expression is faithful; the
    #     deficiency is upstream in numpyro.
    #   stan: `_is_continuous_support` routes `ContinuousBernoulli`
    #     to the continuous marginalization, declaring a live
    #     400-dim `z` parameter the point payload cannot fill
    #     (dims declared=(400), found=()), where the QVR compiler
    #     enumerates the hard support {0, 1}.
    #   gen: same missing log-weight primitive as gen/lda; the
    #     reduced density has no address to ride on, and
    #     `Gen.logpdf(Gen.poisson, 0, 0.0)` is NaN where the
    #     reference scores the point mass exactly.
    ('gen', 'zip_regression'),
    ('numpyro', 'zip_regression'),
    ('stan', 'zip_regression'),
    # webppl/stochastic_volatility: the latent trajectory emits as
    # `mapIndexed(..., repeat(200, 0))` and WebPPL's `repeat`
    # requires its second argument to be a function. The probe's
    # `mapIndexed` lift is otherwise ready for this cell.
    ('webppl', 'stochastic_volatility'),
})


def test_gallery_reference_pin_registry_is_total() -> None:
    """Every gallery example is either pinned or justifiably exempt,
    and neither registry may carry a row the gallery does not.

    This is the test that makes the pin a guarantee rather than an
    opt-in. The failure mode it exists to prevent is silent: a
    `dict.get` that returns `None` and a guard that skips the
    assertion turn a deleted row, or a newly-added example, into a
    check that passes while asserting nothing about the value it was
    written to protect. Requiring the two registries to *partition*
    the gallery makes both directions loud. Deleting a row fails here.
    Adding an example without deriving its reference fails here.
    Retiring an example without dropping its row fails here too, so a
    stale pin cannot sit in the registry looking like coverage.

    The exemption side is checked against the registries that
    independently agree the example has no reference, not taken on its
    word: an exempt example must appear in `_SKIP_DATASET_LOAD_FAILED`
    (no dataset at all) or in `_SKIP_QVR_INCOMPATIBLE` (no
    deterministic joint). An exemption whose gap has closed therefore
    fails rather than quietly suppressing a pin the example could now
    carry.
    """
    gallery = {example.stem for example in _gallery_cells()}
    pinned = set(_QVR_REFERENCE_JOINT)
    exempt = set(_REFERENCE_PIN_EXEMPT)

    overlap = sorted(pinned & exempt)
    assert not overlap, (
        f"{overlap!r} are both pinned and exempt. An example carries a "
        f"reference or it does not; the two registries must be "
        f"disjoint so a reader can tell which examples are actually "
        f"guarded."
    )

    unpinned = sorted(gallery - pinned - exempt)
    assert not unpinned, (
        f"{unpinned!r} ship synthetic data but appear in neither "
        f"`_QVR_REFERENCE_JOINT` nor `_REFERENCE_PIN_EXEMPT`, so "
        f"nothing asserts their oracle joint is right. Derive the "
        f"reference from raw `torch.distributions` and pin every "
        f"point, or record why the example has no reference."
    )

    stale = sorted((pinned | exempt) - gallery)
    assert not stale, (
        f"{stale!r} are registered but are no longer gallery examples "
        f"with synthetic data. Drop the rows: a pin on an example that "
        f"no longer runs reads as coverage and is not."
    )

    expected_points = len(_gallery_data.perturbation_labels())
    for stem, values in sorted(_QVR_REFERENCE_JOINT.items()):
        assert len(values) == expected_points, (
            f"{stem!r}: pinned at {len(values)} point(s) against a "
            f"{expected_points}-point set. The pin has to cover every "
            f"point the equivalence check evaluates, or a "
            f"data-dependent oracle error hides in the unpinned tail."
        )
        for index, value in enumerate(values):
            assert math.isfinite(value), (
                f"{stem!r}: pinned value at point {index} is "
                f"{value!r}. A non-finite pin asserts nothing: every "
                f"comparison against it is `nan`, which no `<=` "
                f"rejects."
            )

    for stem, reason in sorted(_REFERENCE_PIN_EXEMPT.items()):
        assert reason.strip(), (
            f"{stem!r}: exempt from the reference pin with an empty "
            f"reason. An exemption without a stated cause is an "
            f"unexplained hole in the guarantee."
        )
        assert (
            stem in _SKIP_DATASET_LOAD_FAILED
            or stem in _SKIP_QVR_INCOMPATIBLE
        ), (
            f"{stem!r}: claims exemption from the reference pin "
            f"({reason}) but is in neither `_SKIP_DATASET_LOAD_FAILED` "
            f"nor `_SKIP_QVR_INCOMPATIBLE`, so the gallery tier does "
            f"score a joint for it. Either the gap closed and the row "
            f"belongs in `_QVR_REFERENCE_JOINT`, or the skip registry "
            f"is missing the entry that justifies the exemption."
        )


@pytest.mark.parametrize(
    "example", _gallery_cells(), ids=lambda p: p.stem
)
def test_gallery_qvr_reference_pin_holds_at_every_point(
    example: pathlib.Path,
) -> None:
    """The QVR reference reproduces its pinned value at **every**
    point of the set, not only at the ground truth.

    A ground-truth-only pin and the constant-spread check have
    complementary blind spots that overlap exactly on the class of
    error that matters most here. The pin at point 0 sees a constant
    offset but says nothing about points 1..5. The spread check sees a
    varying offset but is invariant to a constant one. An oracle error
    that is *zero at the ground truth and constant across the
    perturbed points* therefore passes both: the pin holds where it
    looks, and the spread is unchanged because the same wrong value
    feeds both sides of every difference. Pinning per point removes
    that overlap, because a per-point pin is violated by any error
    that moves at all.

    The concrete shape of that error class in this codebase is a
    data-dependent term the oracle drops. The dropped term is zero at
    the ground truth for a fixture whose ground truth sits at the
    term's zero (a centred residual, a sum-to-zero score, a
    log-normaliser that cancels at the generating parameters) and
    non-zero once the data moves.
    """
    if example.stem in _SKIP_DATASET_LOAD_FAILED:
        pytest.skip(
            f"{example.stem!r}: synthetic-data snippet in the `.md` "
            f"file fails to load; populate / drop from "
            f"`_SKIP_DATASET_LOAD_FAILED`."
        )
    if example.stem in _SKIP_QVR_INCOMPATIBLE:
        pytest.skip(
            f"{example.stem!r}: in-process QVR trace cannot evaluate "
            f"this program; populate / drop from `_SKIP_QVR_INCOMPATIBLE`."
        )

    reference = _QVR_REFERENCE_JOINT.get(example.stem)
    assert reference is not None, (
        f"{example.stem!r}: reaches the multi-point scoring path with "
        f"no `_QVR_REFERENCE_JOINT` entry. See "
        f"`test_gallery_reference_pin_registry_is_total`."
    )

    dataset = _gallery_data.load_gallery_data(example)
    assert dataset is not None, (
        f"{example.stem!r}: `load_gallery_data` returned None even "
        f"though the example was not in `_SKIP_DATASET_LOAD_FAILED`."
    )

    points = _gallery_data.points_from_dataset(dataset)
    labels = _gallery_data.perturbation_labels(len(points))
    assert len(points) == len(reference), (
        f"{example.stem!r}: {len(points)} point(s) against "
        f"{len(reference)} pinned value(s). The point schedule moved "
        f"under the pin; re-derive every value before re-pinning."
    )

    probe = QvrProbe()
    scratch = pathlib.Path("/tmp") / f"qvr_gallery_pin_{example.stem}"
    scratch.mkdir(exist_ok=True, parents=True)
    source = example.read_bytes()
    for index, point in enumerate(points):
        measured = probe.evaluate(
            source,
            example.stem,
            [point],
            scratch=scratch,
            monadic=dataset.monadic,
            x_input=dataset.x_input,
            observations=_gallery_data.observations_for_point(
                dataset, point,
            ),
        ).log_densities[0]
        expected = reference[index]
        atol = reference_pin_atol(expected)
        assert abs(measured - expected) <= atol, (
            f"{example.stem!r} point {index} ({labels[index]}): QVR "
            f"joint {measured!r} against pinned reference "
            f"{expected!r}, a gap of {abs(measured - expected):.6g} "
            f"nats past the {atol:.6g} round-off budget. The oracle's "
            f"density changed at this point. Re-derive it from raw "
            f"`torch.distributions` before touching the pin; widening "
            f"the tolerance would restore exactly the blindness the "
            f"per-point pin exists to remove."
        )


@pytest.mark.parametrize(
    "example", _gallery_cells(), ids=lambda p: p.stem
)
def test_gallery_qvr_logdensity_finite(example: pathlib.Path) -> None:
    """The QVR-side log-density at the ground-truth (θ_true, y)
    point evaluates to a finite real number.

    This is the always-on Tier-4 invariant: regardless of backend
    Docker availability, every gallery example with synthetic data
    must produce a well-defined joint log-density under the
    reference QVR trace. A non-finite value (`-inf`, `nan`)
    indicates that either (a) the data-gen snippet's ground-truth
    parameters fall outside the model's support, or (b) the QVR
    program has a structural defect that the trace surfaces only
    on real data.
    """
    if example.stem in _SKIP_DATASET_LOAD_FAILED:
        pytest.skip(
            f"{example.stem!r}: synthetic-data snippet in the `.md` "
            f"file fails to load; populate / drop from "
            f"`_SKIP_DATASET_LOAD_FAILED`."
        )
    if example.stem in _SKIP_QVR_INCOMPATIBLE:
        pytest.skip(
            f"{example.stem!r}: in-process QVR trace cannot evaluate "
            f"this program; populate / drop from `_SKIP_QVR_INCOMPATIBLE`."
        )

    dataset = _gallery_data.load_gallery_data(example)
    assert dataset is not None, (
        f"{example.stem!r}: `load_gallery_data` returned None even "
        f"though the example was not in `_SKIP_DATASET_LOAD_FAILED`. "
        f"Add the example's stem to that registry, or fix the "
        f"`.md` snippet so it produces an `observations` dict."
    )

    point = _gallery_data.point_from_dataset(dataset)
    probe = QvrProbe()
    source = example.read_bytes()
    scratch = pathlib.Path("/tmp") / f"qvr_gallery_{example.stem}"
    scratch.mkdir(exist_ok=True, parents=True)
    result = probe.evaluate(
        source,
        example.stem,
        [point],
        scratch=scratch,
        monadic=dataset.monadic,
        x_input=dataset.x_input,
        observations=dataset.observations,
    )
    assert len(result.log_densities) == 1, (
        f"{example.stem!r}: expected exactly one log-density, got "
        f"{result.log_densities!r}"
    )
    lp = result.log_densities[0]
    assert math.isfinite(lp), (
        f"{example.stem!r}: QVR log p(θ_true, y) is non-finite ({lp!r}); "
        f"observations={list(dataset.observations)} params={list(dataset.params)}"
    )

    # Finiteness alone is a weak oracle check: a measure bug can return a
    # finite-but-wrong joint (an unclamped latent resampling, a
    # marginalized site double-counted, a plate broadcast inflating the
    # sum). The reference pin is what turns it into a correctness
    # check, and it is mandatory: an example that reaches this line has
    # a joint, so a missing entry is a hole in the guarantee and fails
    # here rather than reverting to the finiteness-only check.
    reference = _QVR_REFERENCE_JOINT.get(example.stem)
    assert reference is not None, (
        f"{example.stem!r}: scores a finite QVR joint ({lp!r}) but has "
        f"no `_QVR_REFERENCE_JOINT` entry, so nothing checks that the "
        f"value is *right*. Theorem 4.1's constant-spread quotient "
        f"cannot see a point-independent oracle error, which makes "
        f"this pin the only thing that can. Re-derive the joint from "
        f"raw `torch.distributions` (see "
        f"`test_oracle_reference_strength.py`), pin every point of "
        f"the set, and add the row. If this example genuinely has no "
        f"reference, record the reason in `_REFERENCE_PIN_EXEMPT` "
        f"instead."
    )
    atol = reference_pin_atol(reference[0])
    assert abs(lp - reference[0]) <= atol, (
        f"{example.stem!r}: QVR joint {lp!r} drifted from its "
        f"independently-verified reference {reference[0]!r} by "
        f"{abs(lp - reference[0]):.6g} nats, past the "
        f"{atol:.6g} round-off budget. Either the oracle regressed or "
        f"the ground-truth point changed; re-derive the joint "
        f"independently before updating the `_QVR_REFERENCE_JOINT` "
        f"entry, and never widen the tolerance to absorb the drift."
    )


@pytest.mark.parametrize(
    "example", _gallery_cells(), ids=lambda p: p.stem
)
def test_gallery_multipoint_set_is_in_support_and_varies(
    example: pathlib.Path,
) -> None:
    """The multi-point set stays in support and actually moves.

    Three properties make
    [`assert_log_density_match`][tests.transpile._equivalence.assert_log_density_match]
    a real test rather than a tautology, and each is asserted here so
    a regression surfaces without needing a Docker image:

    1. Every point scores a **finite** QVR joint. A perturbation that
       steps outside the support sends both evaluators to `-inf`, and
       two `-inf` values differ by `nan` rather than by a constant, so
       the comparison would be meaningless.
    2. The joint **varies** across the set. A point set that collapses
       to repeats of the ground truth restores the single-point
       vacuity the multi-point check exists to remove: the spread of a
       constant difference sequence is zero whatever the backend
       computed. Variation in the joint is the observable form of
       "the latents or the data really moved", and it holds for every
       gallery shape, including examples that capture no latents (only
       the data moves) and examples whose data is entirely
       integer-valued covariates (only the latents move).
    3. The **observed data** varies across the set. This is strictly
       stronger than (2) and it is the property the constant-spread
       contract actually needs: a latents-only perturbation moves the
       joint while leaving every observation at ground truth, so (2)
       passes on a point set whose data section is byte-identical
       throughout. Against such a set a backend that drops a
       data-dependent term keeps a perfectly constant offset and the
       equivalence assertion is vacuous. An example whose data
       genuinely cannot move states so in `_NO_PERTURBABLE_OBSERVATION`
       and has the frozen data section asserted rather than assumed.
    """
    if example.stem in _SKIP_DATASET_LOAD_FAILED:
        pytest.skip(
            f"{example.stem!r}: synthetic-data snippet in the `.md` "
            f"file fails to load; populate / drop from "
            f"`_SKIP_DATASET_LOAD_FAILED`."
        )
    if example.stem in _SKIP_QVR_INCOMPATIBLE:
        pytest.skip(
            f"{example.stem!r}: in-process QVR trace cannot evaluate "
            f"this program; populate / drop from `_SKIP_QVR_INCOMPATIBLE`."
        )

    dataset = _gallery_data.load_gallery_data(example)
    assert dataset is not None, (
        f"{example.stem!r}: `load_gallery_data` returned None even "
        f"though the example was not in `_SKIP_DATASET_LOAD_FAILED`."
    )

    points = _gallery_data.points_from_dataset(dataset)
    labels = _gallery_data.perturbation_labels(len(points))
    assert len(points) >= 2, (
        f"{example.stem!r}: {len(points)} point(s); the constant-spread "
        f"contract needs at least two to be testable."
    )

    probe = QvrProbe()
    scratch = pathlib.Path("/tmp") / f"qvr_gallery_points_{example.stem}"
    scratch.mkdir(exist_ok=True, parents=True)
    source = example.read_bytes()
    lps: list[float] = []
    for point in points:
        lps.extend(
            probe.evaluate(
                source,
                example.stem,
                [point],
                scratch=scratch,
                monadic=dataset.monadic,
                x_input=dataset.x_input,
                observations=_gallery_data.observations_for_point(
                    dataset, point,
                ),
            ).log_densities
        )

    for index, lp in enumerate(lps):
        assert math.isfinite(lp), (
            f"{example.stem!r}: point {index} ({labels[index]}) scores a "
            f"non-finite QVR joint ({lp!r}), so it left the model's "
            f"support. The perturbation for one of its sites does not "
            f"respect the site's declared constraint."
        )

    assert len({round(lp, 6) for lp in lps}) > 1, (
        f"{example.stem!r}: every point scores the same QVR joint "
        f"({lps[0]!r}), so no site moved and the constant-spread check "
        f"would pass unconditionally. Either the dataset captured "
        f"nothing perturbable, or a constraint this example needs is "
        f"missing from `_perturb_by_support`."
    )

    observed = _gallery_data.observed_data_names(dataset)
    moved = _gallery_data.varying_observation_names(dataset, points)
    frozen_reason = _NO_PERTURBABLE_OBSERVATION.get(example.stem)
    if frozen_reason is None:
        assert moved, (
            f"{example.stem!r}: the observed data is byte-identical at "
            f"every point ({sorted(observed)} all frozen), so the "
            f"constant-spread check for this example is a "
            f"latents-only test: a backend that drops a "
            f"data-dependent term would keep a constant offset and "
            f"pass. Give the data section a support the perturber can "
            f"step in (an observed count moves inside its attested "
            f"range, a simplex-valued observation renormalises, a "
            f"bounded one moves in its own space), or, if this "
            f"example truly has no perturbable observation, record "
            f"the reason in `_NO_PERTURBABLE_OBSERVATION`."
        )
    else:
        assert not moved, (
            f"{example.stem!r}: `_NO_PERTURBABLE_OBSERVATION` claims "
            f"this example has no perturbable observation "
            f"({frozen_reason}), but {sorted(moved)} moved across the "
            f"point set. Drop the entry: the data section is testable "
            f"and the claim is stale."
        )


@pytest.mark.parametrize(
    "example", _gallery_cells(), ids=lambda p: p.stem
)
@pytest.mark.parametrize("backend", sorted(_BACKENDS_WITH_IMAGES))
def test_gallery_backend_logdensity_matches_qvr(
    example: pathlib.Path, backend: str
) -> None:
    """Constant-spread equivalence between the QVR reference and the
    backend's native log-density at the ground-truth (θ_true, y)
    point set for `example`."""
    image, ext, script_name = _BACKENDS_WITH_IMAGES[backend]
    if not _docker.docker_available():
        raise RuntimeError(
            "docker daemon not reachable; the session-scope "
            "`_ensure_docker_environment` autouse fixture should have "
            "started it"
        )
    if not _docker.image_available(image):
        raise RuntimeError(
            f"docker image {image!r} not available; the session-scope "
            f"`_ensure_docker_environment` autouse fixture should have "
            f"built it"
        )

    cell = (backend, example.stem)

    expected_raise = _EXPECTED_TRANSPILE_RAISES.get(cell)
    if expected_raise is not None:
        source = example.read_bytes()
        with pytest.raises(UnsupportedConstruct) as exc_info:
            transpile(parse(source.decode("utf-8")), target=backend)
        kinds = exc_info.value.kinds
        assert any(k.startswith(expected_raise) for k in kinds), (
            f"{backend!r} on {example.stem!r}: expected raise with "
            f"kind prefix {expected_raise!r}, got kinds={kinds!r}. "
            f"Either the renderer changed (update the entry in "
            f"`_EXPECTED_TRANSPILE_RAISES`) or a different gap fired."
        )
        return

    if example.stem in _SKIP_DATASET_LOAD_FAILED:
        pytest.skip(
            f"{example.stem!r}: synthetic-data snippet in the `.md` "
            f"file fails to load; populate / drop from "
            f"`_SKIP_DATASET_LOAD_FAILED`."
        )
    if example.stem in _SKIP_QVR_INCOMPATIBLE:
        pytest.skip(
            f"{example.stem!r}: in-process QVR trace cannot evaluate "
            f"this program; populate / drop from `_SKIP_QVR_INCOMPATIBLE`."
        )
    if cell in _SKIP_PROBE_INCOMPATIBLE:
        pytest.skip(
            f"{backend!r} on {example.stem!r}: in-container probe "
            f"script has no shape registration for this example's "
            f"dataset; populate / drop from `_SKIP_PROBE_INCOMPATIBLE`."
        )

    dataset = _gallery_data.load_gallery_data(example)
    assert dataset is not None, (
        f"{example.stem!r}: `load_gallery_data` returned None even "
        f"though the example was not in `_SKIP_DATASET_LOAD_FAILED`. "
        f"Add the example's stem to that registry, or fix the "
        f"`.md` snippet so it produces an `observations` dict."
    )

    source = example.read_bytes()
    emitted = transpile(parse(source.decode("utf-8")), target=backend)

    points = _gallery_data.points_from_dataset(dataset)
    labels = _gallery_data.perturbation_labels(len(points))
    qvr_probe = QvrProbe()
    # Isolate the bind-mounted scratch per (backend, example) so a
    # probe never mounts a source or helper file another backend's run
    # left behind in a shared directory.
    scratch = pathlib.Path("/tmp") / f"qvr_gallery_eq_{example.stem}_{backend}"
    scratch.mkdir(exist_ok=True, parents=True)
    # One probe call per point: the probe's `observations` keyword
    # overrides the flat per-point payload (it is the only channel that
    # preserves multi-axis shapes), so a perturbed point needs its own
    # pre-shaped observation dict. Passing the dataset's ground-truth
    # observations once for the whole set would score the QVR side at
    # the unperturbed data while the container scored the perturbed
    # data.
    qvr_lps: list[float] = []
    for point in points:
        qvr_lps.extend(
            qvr_probe.evaluate(
                source,
                example.stem,
                [point],
                scratch=scratch,
                monadic=dataset.monadic,
                x_input=dataset.x_input,
                observations=_gallery_data.observations_for_point(
                    dataset, point,
                ),
            ).log_densities
        )

    script_path = (
        pathlib.Path(__file__).parent / "probes" / "_scripts" / script_name
    )
    raw_result = _docker.run_probe(
        image=image,
        script=script_path,
        source=emitted,
        source_ext=ext,
        points=[
            {"params": point.params, "data": point.data}
            for point in points
        ],
        scratch=scratch,
        shapes=_shapes_from_dataset(dataset),
        dtypes=_dtypes_from_dataset(dataset),
    )

    backend_lps = [float(x) for x in raw_result["log_densities"]]

    _equivalence.assert_log_density_match(
        qvr_lps,
        backend_lps,
        context=f"{backend}@{example.stem}",
        labels=labels,
        min_points=2,
    )


def _shapes_from_dataset(
    dataset: _gallery_data.GalleryDataset,
) -> dict[str, list[int]]:
    """Per-name shape table the per-backend probe scripts read to
    rebuild nested arrays from the row-major flat lists in `Point`.

    Scalar tensors (zero-dim) map to ``[]`` rather than ``[1]``;
    `point_from_dataset` collapses length-1 lists to bare scalars
    and `_reshape.reshape_value` reads the empty shape as a no-op
    after the dtype cast.
    """
    def _emitted_shape(value: torch.Tensor) -> list[int]:
        """The shape the emitted program declares for this name.

        A plate-less program is emitted for one row, so the fixture's
        length-one batch axis has no counterpart in the declaration:
        Stan writes `vector[16] state`, not `array[1] vector[16]`.
        The flat list is identical, so dropping the axis here is what
        lets the probe rebuild it against what the program declares.
        """
        shape = list(value.shape)
        if dataset.single_row and len(shape) > 1 and shape[0] == 1:
            return shape[1:]
        return shape

    shapes: dict[str, list[int]] = {}
    for k, v in dataset.observations.items():
        shapes[k] = _emitted_shape(v)
    for k, v in dataset.params.items():
        shapes[k] = _emitted_shape(v)
    # A compiled parameter map reaches the container in the data
    # section, so the probe needs its shape to rebuild the matrix from
    # the flat row-major list it travels as.
    for k, v in dataset.param_wires.items():
        shapes[k] = _emitted_shape(v)
    # A scalar type-parameter reaches the container as a bare float in
    # the point's data section; the empty shape casts it through the
    # dtype table without rebuilding a nested container around it.
    for name in dataset.scalar_params:
        shapes[name] = []
    return shapes


def _dtypes_from_dataset(
    dataset: _gallery_data.GalleryDataset,
) -> dict[str, str]:
    """Per-name dtype tag (``"int"`` or ``"float"``) so backends that
    distinguish integer and real declarations (Stan, JAGS, BUGS, PyMC)
    get the right native type after reshape.

    Resolution order, most authoritative first:

    1. **The declared family.** When the name answers to a stochastic
       site of the compiled program, its
       [`site_supports`][tests.transpile._gallery_data.site_supports]
       constraint decides: an integer-supported site (a Poisson /
       Binomial / Categorical / Bernoulli observation) is ``"int"``
       and a continuous-supported one is ``"float"``, whatever its
       ground-truth value happens to look like. This is what keeps a
       continuous latent whose ground truth is integer-valued (a
       horseshoe global scale pinned at ``tau = 1.0``, unit local
       scales) out of the ``int`` bucket: the in-container reshape
       casts every ``int``-tagged leaf through Python's `int`, which
       truncates the value and makes the backend's density piecewise
       constant in integer buckets rather than equal to the QVR
       reference.
    2. **The declared torch dtype**, for a name with no site: an
       integer-kinded covariate (a plate subscript such as
       ``coef_idx``) is ``"int"``.
    3. **The value domain**, for a float-kinded covariate: an
       all-integer-valued one is ``"int"``. This last rule is the only
       place the value heuristic survives, and it applies to
       covariates alone: an observation's kind now comes from its
       family instead.
    """
    integer_dtypes = (
        torch.int8, torch.int16, torch.int32, torch.int64,
        torch.uint8, torch.bool,
    )
    supports = _gallery_data.site_supports(dataset)
    out: dict[str, str] = {}
    for section in (dataset.observations, dataset.params):
        for name, tensor in section.items():
            support = supports.get(name)
            if support is not None:
                out[name] = (
                    "int"
                    if _gallery_data.is_discrete_support(support)
                    else "float"
                )
            elif tensor.dtype in integer_dtypes:
                out[name] = "int"
            elif tensor.numel() > 0 and torch.equal(
                tensor, tensor.round(),
            ):
                out[name] = "int"
            else:
                out[name] = "float"
    # A scalar type-parameter is declared `Real` in the program header
    # and rendered as a real input (`real alpha;` in Stan's data block,
    # `rep_vector(alpha, 3)` as a Dirichlet concentration), so it is
    # float-tagged whatever its instantiated value looks like: an
    # integer cast of `alpha = 1.0` would still parse and still be
    # wrong the moment the snippet instantiates at a fractional value.
    for name in dataset.scalar_params:
        out[name] = "float"
    # A compiled parameter map is an affine weight table: real on
    # every backend, and never a site whose family could say
    # otherwise.
    for name in dataset.param_wires:
        out[name] = "float"

    return out
