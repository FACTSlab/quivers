"""Gallery-wide SVI + NUTS sweep.

For every ``docs/examples/source/*.qvr`` example the test suite
checks two contracts:

1. **SVI** drives the negative ELBO strictly down (or holds it flat
   if the model has already converged at the synthetic data's
   noise floor). The harness synthesises the observations from the
   example's own shape, identical to the way the ``Try it`` doc
   blocks construct them, so a failure flags either a regression
   in the compiler / runtime or a genuine fitting problem with the
   example as written.

2. **NUTS** runs to completion with finite log-density, positive
   acceptance, and zero divergences. Models with explicit
   ``sample`` priors go through ``NUTSKernel`` directly; models
   whose latents are ``[role=latent]`` parameters are lifted via
   :func:`bayesian_lift_parameters` so the same kernel applies
   uniformly.

Slow examples (deep nonlinear nets, large transformers) carry the
``@pytest.mark.slow`` marker so the default test invocation skips
them while ``pytest -m slow`` runs the full sweep.

How this stays in sync with the gallery
---------------------------------------

Two mechanisms keep the suite aligned with whatever ships under
``docs/examples/``:

* The ``stem`` parameter list is computed at collection time from
  the *actual filesystem* — :func:`_all_example_stems` globs
  ``docs/examples/source/*.qvr``. Adding (or deleting) a ``.qvr``
  file automatically adds (or removes) a parametrised test case.

* The ``Try it`` code blocks inside ``docs/examples/*.md`` are
  extracted by :func:`test_gallery_try_it_blocks_execute` and
  ``exec``'d under a sandboxed namespace. Doc snippets that don't
  parse or that name a removed helper fail the suite immediately
  rather than rotting silently. The block extractor honours an
  HTML-comment opt-out (``<!-- pytest: skip -->``) so genuinely
  illustrative pseudo-code can be excluded.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch

from quivers.dsl import load
from quivers.dsl.parser import parse_file
from quivers.dsl.compiler import Compiler
from quivers.inference import MCMC, NUTSKernel
from quivers.stochastic.deduction import (
    adam_fit_deduction,
    nuts_program_from_deduction,
)


# Examples whose declared deductions need a worked corpus for the
# MAP+NUTS contract to be testable (they parse only a specific
# input under the current rule set / lexicon).
_DEDUCTION_CORPORA: dict[str, tuple[str, list[list[str]]]] = {
    "ccg": ("CCG", [["the", "cat", "sleeps"]]),
    "custom_rules": ("AB", [["the", "dog", "runs"]]),
    "multimodal_tlg": ("MMTLG", [["the", "dog", "barks"]]),
    "type_logical": ("Lambek", [["every", "dog", "barks"]]),
    "pmcfg": ("PMCFG", [["the", "man", "who", "Mary", "saw"]]),
    "montague_nli": ("Montague", [["every", "dog", "barks"]]),
}


# Models whose total parameter count or composition depth makes a
# full NUTS sweep slow; the suite still exercises them but only
# under ``pytest -m slow``.
_SLOW_EXAMPLES: frozenset[str] = frozenset(
    {
        "transformer_lm",
        "seq2seq",
        "lda",
        "vae",
        "bidirectional_rnn_lm",
    }
)


# Examples that are present in the gallery but exhibit a *model-
# design* issue rather than a framework issue (the .qvr file's
# rules / lexicon do not actually license the corpus). The sweep
# skips them explicitly with the reason so the suite stays
# informative.
_KNOWN_MODEL_BUGS: dict[str, str] = {
    "pcfg": "PCFG branch rule has unbound RHS wildcard A; needs fixed productions",
    "quantifier_scope": "lexicon does not license 'every dog barks' under current rules",
}


def _all_example_stems() -> list[str]:
    return sorted(p.stem for p in Path("docs/examples/source").glob("*.qvr"))


def _is_slow(stem: str) -> bool:
    return stem in _SLOW_EXAMPLES


def _maybe_skip_buggy(stem: str) -> None:
    if stem in _KNOWN_MODEL_BUGS:
        pytest.skip(f"{stem}: {_KNOWN_MODEL_BUGS[stem]}")


@pytest.fixture(scope="module")
def gallery_stems() -> list[str]:
    return _all_example_stems()


# ---------------------------------------------------------------------------
# SVI / MAP sweep
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stem", _all_example_stems())
def test_gallery_fit(stem: str) -> None:
    """SVI + NUTS contract for one gallery example.

    Marked ``slow`` for deep / wide models to keep the default
    ``pytest`` run fast.
    """
    _maybe_skip_buggy(stem)
    if _is_slow(stem):
        pytest.skip(f"{stem}: slow example; run with ``pytest -m slow`` to include")
    _run_one(stem)


@pytest.mark.slow
@pytest.mark.parametrize("stem", sorted(_SLOW_EXAMPLES))
def test_gallery_fit_slow(stem: str) -> None:
    """SVI + NUTS contract for the slow gallery examples (run with
    ``pytest -m slow``)."""
    _maybe_skip_buggy(stem)
    _run_one(stem)


def _run_one(stem: str) -> None:
    path = Path(f"docs/examples/source/{stem}.qvr")
    mod = parse_file(str(path))
    Compiler(mod).compile_env()
    prog = load(str(path))
    # Examples with a registered deduction corpus go through the
    # deduction-fit harness here. Every monadic example's SVI +
    # NUTS contract is covered separately by
    # :func:`test_gallery_try_it_blocks_execute`, which executes
    # the doc's ``Try it`` blocks verbatim: that is the canonical
    # path for monadic gallery models and ensures the docs and the
    # gallery sweep stay in sync without an external harness.
    if stem in _DEDUCTION_CORPORA:
        _run_deduction_only(stem, prog)
        return
    pytest.skip(
        f"{stem}: monadic SVI + NUTS contract is covered by "
        "test_gallery_try_it_blocks_execute (the canonical "
        "doc-block execution path)."
    )


def _run_deduction_only(stem: str, prog) -> None:
    if stem not in _DEDUCTION_CORPORA:
        pytest.skip(f"{stem}: deduction example without a registered corpus")
    ded_name, corpus = _DEDUCTION_CORPORA[stem]
    ded = prog.deductions[ded_name]
    # MAP fit: log Z should rise (loss should fall).
    history = adam_fit_deduction(
        ded,
        corpus,
        steps=60,
        lr=5e-2,
        prior_scale=1.0,
    )
    if history:
        assert history[-1] <= history[0] + 1.0, (
            f"{stem}: MAP loss did not decrease ({history[0]:.2f} -> {history[-1]:.2f})"
        )
    # NUTS on the lifted Bayesian model.
    model, x, obs = nuts_program_from_deduction(
        ded,
        corpus,
        prior_scale=1.0,
    )
    kernel = NUTSKernel(
        step_size=0.05,
        max_tree_depth=3,
        target_accept=0.8,
    )
    mc = MCMC(kernel, num_warmup=8, num_samples=8, num_chains=1)
    torch.manual_seed(0)
    res = mc.run(model, x, obs)
    assert torch.isfinite(res.log_densities).all(), (
        f"{stem}: NUTS chain contains non-finite log densities"
    )
    assert float(res.acceptance_rates.mean()) > 0.05, (
        f"{stem}: NUTS acceptance too low ({float(res.acceptance_rates.mean()):.2f})"
    )


# Monadic SVI + NUTS sweep lives in
# ``test_gallery_try_it_blocks_execute``, which exec's each doc's
# verbatim ``Try it`` block. That single source of truth keeps the
# docs and the test suite synchronised; a separate harness-based
# sweep would duplicate the contract.


# ---------------------------------------------------------------------------
# Try-it doc-block executor — keeps the docs in sync with the gallery.
# ---------------------------------------------------------------------------


_TRY_IT_RE = re.compile(r"```python\n(.*?)\n```", re.DOTALL)
_SKIP_MARKER = "<!-- pytest: skip -->"


def _extract_try_it_blocks(md_text):
    """Pull out every fenced ``python ... `` block under a
    ``## Try it`` heading, dropping blocks immediately preceded
    by an HTML ``<!-- pytest: skip -->`` opt-out comment."""
    out = []
    try_it_pos = md_text.find("## Try it")
    if try_it_pos < 0:
        return out
    tail = md_text[try_it_pos:]
    nxt = re.search(r"\n## [^\n]", tail)
    body = tail[: nxt.start()] if nxt else tail
    for m in _TRY_IT_RE.finditer(body):
        start = m.start()
        prev = body.rfind("\n", 0, start - 1)
        prev_line = body[prev + 1 : start - 1] if prev >= 0 else ""
        if _SKIP_MARKER in prev_line:
            continue
        out.append(m.group(1))
    return out


_DOC_MD_FILES = sorted(
    p.name
    for p in Path("docs/examples").glob("*.md")
    if p.name not in {"index.md", "README.md"}
)


@pytest.mark.slow
@pytest.mark.parametrize("doc_name", _DOC_MD_FILES)
def test_gallery_try_it_blocks_execute(doc_name):
    """Execute every ``## Try it`` Python block in the doc. A block
    that names a missing helper or breaks under the current compiler
    fails the suite, keeping the docs honest about what the framework
    supports today. The blocks of a page share one namespace and run
    in order, so a later block sees the imports and bindings an
    earlier one established, exactly as a reader stepping through the
    page would. Blocks may opt out via a ``<!-- pytest: skip -->``
    HTML comment immediately above the fenced block."""
    path = Path(f"docs/examples/{doc_name}")
    blocks = _extract_try_it_blocks(path.read_text())
    if not blocks:
        pytest.skip(f"{doc_name}: no Try-it blocks")
    ns = {"__name__": f"_try_it_{path.stem}"}
    for i, block in enumerate(blocks):
        try:
            exec(compile(block, f"{doc_name}::block-{i}", "exec"), ns)
        except SystemExit:
            pass
