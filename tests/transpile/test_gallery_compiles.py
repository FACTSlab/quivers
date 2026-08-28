"""Run every QVR example in `docs/examples/source/` through every
backend's target compiler / parser, asserting the transpiled bytes
are syntactically valid in the target language.

This is the gallery-level extension of
[`test_external_syntax.py`][tests.transpile.test_external_syntax]:
that test only exercises the canonical Beta-Bernoulli fixture; this
one drives every example in the documentation gallery through every
backend whose syntax-check tool is on PATH.

Each (backend, example) cell either:

- emits non-empty bytes that the target syntax check accepts, or
- raises `UnsupportedConstruct` from a cell registered in
  `_EXPECTED_UNSUPPORTED` (with the expected kind-prefix).

A cell missing from the registry that nevertheless raises is a
regression. A registered cell that no longer raises is a closed
gap — drop the entry.

The four-tier verification hierarchy:

1. Walker structural assertions ([`test_structural.py`][tests.transpile.test_structural]).
2. Mapping composition laws ([`test_lens_laws.py`][tests.transpile.test_lens_laws]).
3. Target compiler acceptance (this test for the gallery; [`test_external_syntax.py`][tests.transpile.test_external_syntax] for the canonical fixture).
4. Measure equivalence in Docker
   ([`test_numeric_equivalence.py`][tests.transpile.test_numeric_equivalence]).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from quivers.dsl.parser import parse
from quivers.transpile import UnsupportedConstruct, transpile


_GALLERY = Path(__file__).resolve().parents[2] / "docs" / "examples" / "source"


def _gallery_examples() -> list[Path]:
    return sorted(_GALLERY.glob("*.qvr"))


# Per-backend external syntax checker: (binary, argv-builder, stdin?).
# Backends with no canonical lint-only tool are skipped (Church has
# no standard interpreter we can lint against; Gen and Turing share
# Julia's `Meta.parse`).
_SYNTAX_CHECKS: dict[str, tuple[str, list[str], bool]] = {
    "stan":    ("stanc",  ["stanc", "--info", "-"], True),
    "numpyro": ("python", ["python", "-c",
                           "import ast, sys; ast.parse(sys.stdin.read())"], True),
    "pyro":    ("python", ["python", "-c",
                           "import ast, sys; ast.parse(sys.stdin.read())"], True),
    "pymc":    ("python", ["python", "-c",
                           "import ast, sys; ast.parse(sys.stdin.read())"], True),
    "edward2": ("python", ["python", "-c",
                           "import ast, sys; ast.parse(sys.stdin.read())"], True),
    "webppl":  ("node",   ["node", "--check", "/dev/stdin"], True),
    "turing":  ("julia",  ["julia", "--startup-file=no", "--quiet", "-e",
                           "src = read(stdin, String); Meta.parseall(src)"], True),
    "gen":     ("julia",  ["julia", "--startup-file=no", "--quiet", "-e",
                           "src = read(stdin, String); Meta.parseall(src)"], True),
}


# Cells where the gallery example exercises a construct outside the
# backend's support tier. Each entry pins the expected kind-prefix
# of the raise, so a closed gap surfaces as a test failure
# (`pytest.raises` matches no entry) and a regression surfaces as a
# different-kind raise.
#
# Five boundary classes are represented:
#
# 1. Structural / categorical declarations (`schema`, `bundle`,
#    `composition`, `contraction`, `encoder`/`decoder`/`loss`/
#    `signature`). No PPL backend has a surface for these, so the
#    examples that carry them fail to lower on every target:
#    schema_chart_parser (schema + bundle), pmf (composition),
#    tensor_contraction (composition + contraction), and
#    term_autoencoder (encoder / decoder / loss / signature).
# 2. Lower-pass family resolution. parametric_pooling samples the
#    `school_effects` sub-program (program-as-distribution), which
#    resolves to no target family on any backend.
# 3. Method-call let-expressions, which Stan cannot render (it has no
#    method-dispatch syntax), gapping the montague_nli Stan cell.
#
# The `sum` builtin is deliberately absent from this registry. It
# lowers to each target's own sum-axis reduction (`jnp.sum(...,
# axis=-1)`, `torch.sum(..., dim=-1)`, `pymc.math.sum(..., axis=-1)`,
# `tf.reduce_sum(..., axis=-1)`), so the factor_analysis and ppca
# Python cells transpile rather than raise. Those eight emissions match
# the QVR joint to within 3e-5 across the gallery point set, so the gap
# is closed rather than merely silenced.
#
# Key: (backend, example-stem). Value: kind-prefix the raised
# `UnsupportedConstruct.kinds` must match.
_EXPECTED_UNSUPPORTED: dict[tuple[str, str], str] = {
    # 1. Structural / categorical declarations (all backends).
    ("edward2", "schema_chart_parser"): "bundle_decl",
    ("gen", "schema_chart_parser"): "bundle_decl",
    ("numpyro", "schema_chart_parser"): "bundle_decl",
    ("pymc", "schema_chart_parser"): "bundle_decl",
    ("pyro", "schema_chart_parser"): "bundle_decl",
    ("stan", "schema_chart_parser"): "bundle_decl",
    ("turing", "schema_chart_parser"): "bundle_decl",
    ("webppl", "schema_chart_parser"): "bundle_decl",
    ("edward2", "pmf"): "composition_decl",
    ("gen", "pmf"): "composition_decl",
    ("numpyro", "pmf"): "composition_decl",
    ("pymc", "pmf"): "composition_decl",
    ("pyro", "pmf"): "composition_decl",
    ("stan", "pmf"): "composition_decl",
    ("turing", "pmf"): "composition_decl",
    ("webppl", "pmf"): "composition_decl",
    ("edward2", "tensor_contraction"): "composition_decl",
    ("gen", "tensor_contraction"): "composition_decl",
    ("numpyro", "tensor_contraction"): "composition_decl",
    ("pymc", "tensor_contraction"): "composition_decl",
    ("pyro", "tensor_contraction"): "composition_decl",
    ("stan", "tensor_contraction"): "composition_decl",
    ("turing", "tensor_contraction"): "composition_decl",
    ("webppl", "tensor_contraction"): "composition_decl",
    ("edward2", "term_autoencoder"): "signature_decl",
    ("gen", "term_autoencoder"): "signature_decl",
    ("numpyro", "term_autoencoder"): "signature_decl",
    ("pymc", "term_autoencoder"): "signature_decl",
    ("pyro", "term_autoencoder"): "signature_decl",
    ("stan", "term_autoencoder"): "signature_decl",
    ("turing", "term_autoencoder"): "signature_decl",
    ("webppl", "term_autoencoder"): "signature_decl",
    # 2. Lower-pass family resolution (all backends).
    ("edward2", "parametric_pooling"): "family:school_effects",
    ("gen", "parametric_pooling"): "family:school_effects",
    ("numpyro", "parametric_pooling"): "family:school_effects",
    ("pymc", "parametric_pooling"): "family:school_effects",
    ("pyro", "parametric_pooling"): "family:school_effects",
    ("stan", "parametric_pooling"): "family:school_effects",
    ("turing", "parametric_pooling"): "family:school_effects",
    ("webppl", "parametric_pooling"): "family:school_effects",
    # 3. Method-call let-expressions have no Stan rendering.
    ("stan", "montague_nli"): "let-expr:LetExprMethodCall",
}

# 4. Neural morphisms (`param_source=mlp`) compute their mean with a
#    network whose weights are model-internal: they appear in neither
#    the wire form nor the sample sites, so no backend can reconstruct
#    the mean. The transpiler raises rather than emit a meanless
#    observation. Every syntax-check backend is affected.
for _neural_example in (
    "bnn",
    "bidirectional_rnn_lm",
    "deep_markov",
    "seq2seq",
    "transformer_lm",
    "vae",
):
    for _syntax_backend in _SYNTAX_CHECKS:
        _EXPECTED_UNSUPPORTED[(_syntax_backend, _neural_example)] = (
            "param-source:mlp"
        )

# 5. Linear parameter maps (`param_source=linear`). A Kleisli
#    morphism declared between objects of *different* width, as in
#    continuous_hmm's `emission : State -> Obs` (Real 16 to Real 8)
#    and linear_gaussian_ssm's `emission : State -> Obs` (Real 4 to
#    Real 2), carries a map from its domain to the family's parameter
#    heads on its codomain. The runtime realises it as a
#    [`LinearSource`][quivers.continuous.param_source.LinearSource]:
#    continuous_hmm's emission holds a 16-to-16 weight (8 `loc` heads
#    and 8 `scale` heads over `Obs`), linear_gaussian_ssm's a 4-to-4
#    weight (2 heads each). Those weights are drawn when the module
#    compiles, so they appear in neither the QVR text nor any sample
#    site, and a target has nothing to reconstruct them from.
#
#    The declared morphism's parameter map therefore does not reach
#    the targets, and the transpile raises rather than emitting a
#    program that computes a different measure. The only emission
#    available without the map is the one
#    [`assert_no_dropped_param_map`][quivers.transpile.renderers._base.assert_no_dropped_param_map]
#    exists to reject: for continuous_hmm it binds the 16-wide `s_new`
#    straight into the 8-wide `Obs` site (`normal_lpdf(o[m_Obs] |
#    s_new, 1)` in Stan, `Normal(loc=s_new, scale=1)` under
#    `plate("Obs", 8)` in NumPyro), dropping the emission map
#    entirely, substituting a unit scale for the learned one, and
#    scoring a measure on a space of the wrong dimension.
#
#    This is a real gap, not an inherent limit of the targets. Closing
#    it means threading the ParamSource through the renderers, so that
#    a declared morphism's weights and bias are emitted as data (or as
#    explicit sampled weights plus a deterministic forward pass) and
#    the site scores against the map's output. Until they are, both
#    examples raise on every syntax-check backend.
for _linear_param_map_example in (
    "continuous_hmm",
    "linear_gaussian_ssm",
):
    for _syntax_backend in _SYNTAX_CHECKS:
        _EXPECTED_UNSUPPORTED[(_syntax_backend, _linear_param_map_example)] = (
            "param-source:linear"
        )


@pytest.mark.parametrize(
    "example", _gallery_examples(), ids=lambda p: p.stem
)
@pytest.mark.parametrize("backend", sorted(_SYNTAX_CHECKS))
def test_gallery_example_compiles(example: Path, backend: str) -> None:
    """Transpile a gallery example to `backend` and run its target
    compiler / parser as a syntax check."""
    binary, argv, _uses_stdin = _SYNTAX_CHECKS[backend]
    if shutil.which(binary) is None:
        pytest.skip(
            f"{binary!r} not on PATH; install it in the local toolchain "
            f"or add the install step to CI"
        )

    source = example.read_text()
    cell = (backend, example.stem)
    expected_unsupported = _EXPECTED_UNSUPPORTED.get(cell)
    if expected_unsupported is not None:
        with pytest.raises(UnsupportedConstruct) as exc_info:
            transpile(parse(source), target=backend)
        kinds = exc_info.value.kinds
        assert any(
            k.startswith(expected_unsupported) for k in kinds
        ), (
            f"{backend!r} on {example.name}: expected raise with "
            f"kind prefix {expected_unsupported!r}, got {kinds!r}. "
            f"Either the renderer changed (update the entry in "
            f"`_EXPECTED_UNSUPPORTED`) or a different gap fired "
            f"(fix the renderer)."
        )
        return

    emitted = transpile(parse(source), target=backend)

    completed = subprocess.run(
        argv,
        input=emitted,
        capture_output=True,
        timeout=60.0,
    )
    assert completed.returncode == 0, (
        f"{backend!r} compiler rejected {example.name}: "
        f"stdout={completed.stdout.decode('utf-8', errors='replace')!r} "
        f"stderr={completed.stderr.decode('utf-8', errors='replace')!r}\n"
        f"emitted source:\n{emitted.decode('utf-8', errors='replace')}"
    )
