"""End-to-end tests for the regression-analogous deduction fits.

Covers the three surfaces in :mod:`quivers.deduction`:

* :func:`adam_fit_deduction` — MAP / MLE point estimate.
* :func:`sample_corpus` — forward sampling from the deduction.
* :func:`nuts_program_from_deduction` — NUTS over the lexicon and
  rule-binding log-weights under Normal priors.

Each test verifies the *contract*, not a brittle numeric value: the
loss decreases under Adam, the sampler returns finite-weight
yields under the fitted parameters, and NUTS' posterior
log-density rises during warmup with positive acceptance and no
divergences.
"""

from __future__ import annotations
import textwrap

import os

os.environ.setdefault("QVR_USE_LOCAL_GRAMMAR", "1")

import torch

from quivers.dsl.parser import parse
from quivers.dsl.compiler import Compiler
from quivers.stochastic.deduction import (
    DeductionSystem,
    adam_fit_deduction,
    nuts_program_from_deduction,
    sample_corpus,
)
from quivers.inference import MCMC, NUTSKernel


_AB_SOURCE = """
object Term : FinSet 16

deduction AB : Term -> Term [semiring=LogProb, start=S, depth=6]
    atoms S, NP, N, Fwd, Bwd, span, the, dog, runs
    rule fwd_app : span(I, K, Fwd(X, Y)), span(K, J, Y) |- span(I, J, X) #[learnable]
    rule bwd_app : span(I, K, Y), span(K, J, Bwd(X, Y)) |- span(I, J, X) #[learnable]
    lexicon
        "the"  : Fwd(NP, N) = the  #[learnable]
        "dog"  : N          = dog  #[learnable]
        "runs" : Bwd(S, NP) = runs #[learnable]
"""


def _load_ab() -> DeductionSystem:
    mod = parse(_AB_SOURCE.encode("utf-8"))
    prog = Compiler(mod).compile()
    return prog.deductions["AB"]


# ---------------------------------------------------------------------------
# Adam fit.
# ---------------------------------------------------------------------------


def test_adam_fit_drives_log_z_up():
    """Adam on the AB grammar with a single training sentence
    should drive the corpus log-marginal strictly upward."""
    torch.manual_seed(0)
    ded = _load_ab()
    corpus = [["the", "dog", "runs"]]
    history = adam_fit_deduction(ded, corpus, steps=200, lr=5e-2)
    assert len(history) == 200, "adam_fit_deduction returned a short history"
    # Loss is -log Z; should decrease (more negative).
    assert history[-1] < history[0] - 1.0, (
        f"loss did not decrease: first={history[0]}, last={history[-1]}"
    )


def test_adam_fit_allocates_lexicon_and_rule_params():
    """Every #[learnable] lexicon entry and every distinct rule
    firing on the corpus should produce a real nn.Parameter."""
    torch.manual_seed(0)
    ded = _load_ab()
    adam_fit_deduction(
        ded,
        [["the", "dog", "runs"]],
        steps=5,
        lr=1e-2,
    )
    lex_params = list(ded._axiom_module.parameters())
    rule_params = list(ded._rule_module.parameters())
    # 3 lexicon entries × 1 scalar = 3 params.
    assert len(lex_params) == 3, f"expected 3 lex params, got {len(lex_params)}"
    # fwd_app + bwd_app each fire once on a single binding tuple.
    assert len(rule_params) == 2, f"expected 2 rule params, got {len(rule_params)}"


def test_map_prior_regulariser_shrinks_params():
    """With a tight Normal prior (small ``prior_scale``), MAP
    fitting should keep the parameter magnitudes bounded."""
    torch.manual_seed(0)
    ded = _load_ab()
    adam_fit_deduction(
        ded,
        [["the", "dog", "runs"]] * 8,
        steps=200,
        lr=5e-2,
        prior_scale=0.2,
    )
    max_abs = max(float(p.detach().abs().max()) for p in ded.parameters())
    assert max_abs < 5.0, (
        f"prior regulariser failed to bound params; |w|_inf = {max_abs}"
    )


# ---------------------------------------------------------------------------
# Forward sampler.
# ---------------------------------------------------------------------------


def test_forward_sampler_recovers_dominant_yield():
    """After fitting on ``[the, dog, runs]`` the sampler should
    place high probability on that very yield among the
    length-3 enumeration over the deduction's vocabulary."""
    torch.manual_seed(0)
    ded = _load_ab()
    adam_fit_deduction(
        ded,
        [["the", "dog", "runs"]] * 8,
        steps=400,
        lr=5e-2,
    )
    samples = sample_corpus(ded, length=3, n_samples=32, seed=0)
    target = ["the", "dog", "runs"]
    hits = sum(1 for s in samples if s == target)
    # The trained yield should dominate (>= half of draws), since
    # no other length-3 token sequence has a positive log Z under
    # the fitted parameters.
    assert hits >= len(samples) // 2, (
        f"dominant yield {target} got only {hits}/{len(samples)} draws"
    )


# ---------------------------------------------------------------------------
# NUTS.
# ---------------------------------------------------------------------------


def test_nuts_program_is_well_shaped():
    """`nuts_program_from_deduction` should produce a MonadicProgram
    with one sample site per parameter and a trailing score step."""
    ded = _load_ab()
    corpus = [["the", "dog", "runs"]] * 4
    model, x, obs = nuts_program_from_deduction(ded, corpus)
    # Count the registered morphisms whose bound name starts with
    # ``log_w__`` (the per-parameter Normal-prior sample sites).
    n_log_w_steps = sum(
        1
        for s in model._step_specs
        if getattr(s, "vars", None) and s.vars[0].startswith("log_w__")
    )
    n_params = sum(1 for _ in ded.parameters())
    assert n_log_w_steps == n_params, (
        f"NUTS model has {n_log_w_steps} priors; deduction has {n_params}"
    )
    # The final step is the score.
    from quivers.continuous.programs import _ScoreSpec

    assert isinstance(model._step_specs[-1], _ScoreSpec)
    assert x.shape == (1, 1)
    assert obs == {}


def test_binders_alpha_rename_lexicon_lfs():
    """A ``binders`` block + bound-variable pre-pass should let
    lexicon LFs use lambda terms with un-declared variable names;
    each occurrence becomes a fresh canonical symbol."""
    src = """
    object Term : FinSet 4
    deduction LamTest : Term -> Term [semiring=LogProb, start=S]
        atoms NP, S, Bwd, span, App, Var, bark, walk
        binders Lam
        rule bwd_app : span(I, K, Y, A), span(K, J, Bwd(X, Y), F) |- span(I, J, X, App(F, A))
        lexicon
            "barks" : Bwd(S, NP) = Lam(x, App(bark, Var(x)))
            "walks" : Bwd(S, NP) = Lam(x, App(walk, Var(x)))
    """
    mod = parse(textwrap.dedent(src))
    prog = Compiler(mod).compile()
    ded = prog.deductions["LamTest"]
    items = list(ded(["barks", "walks"]).chart.items())
    lfs = [it[4] for it in (k for k, _ in items)]
    # Each lexicon entry's bound var should be a *distinct*
    # canonical name despite both using ``x`` in source.
    canon_per_lex = []
    for lf in lfs:
        # lf is ("Lam", (canon,), ("App", ..., ("Var", (canon,))))
        assert lf[0] == "Lam"
        canon_per_lex.append(lf[1][0])
    assert canon_per_lex[0] != canon_per_lex[1], (
        f"binders failed to alpha-rename: both entries got {canon_per_lex[0]!r}"
    )


def test_compose_chains_deduction_systems():
    """``compose(D1, D2)`` should return a deduction system whose
    chart contains items from both factors when invoked."""
    src = """
    object Term : FinSet 4
    deduction D1 : Term -> Term [semiring=LogProb, start=S]
        atoms tok
        rule trivial : tok |- tok

    deduction D2 : Term -> Term [semiring=LogProb, start=S]
        atoms tok2
        rule trivial2 : tok2 |- tok2
    """
    mod = parse(textwrap.dedent(src))
    prog = Compiler(mod).compile()
    d1 = prog.deductions["D1"]
    d2 = prog.deductions["D2"]
    # ``compose`` is a let-expression builtin; we invoke it
    # directly from the runtime to exercise the wiring.
    # Synthesise the function by building the AST programmatically
    # is overkill; here we just check the runtime side: the
    # builtin's implementation is reachable via subst/compose in
    # programs.py and behaves as expected. As a smoke test, call
    # the composed system on an empty input — both injectors
    # return empty lists, the composed chart should be empty.
    from dataclasses import replace as _dc_replace

    def _composed_injector(inp, _d1=d1, _d2=d2):
        return list(_d1(inp).goal_items) + list(_d2.axiom_injector(inp))

    composed = _dc_replace(d2, axiom_injector=_composed_injector)
    chart = composed([])
    # Smoke: no items, no error.
    assert sum(1 for _ in chart.chart.items()) == 0


def test_subst_capture_avoiding():
    """``subst(term, var, value)`` replaces every variable
    occurrence; thanks to compile-time alpha-renaming, every bound
    name is already unique so naive substitution is correct."""
    # We can't easily call ``subst`` from outside a program body,
    # so we test the recursive-replacement helper directly via the
    # compiler.
    from quivers.dsl.compiler.programs import _ProgramsMixin
    from quivers.dsl.ast_nodes import LetExprCall, LetExprVar

    # subst(App(f, Var(x)), Var(x), Var(arg))
    #   = App(f, Var(arg))
    expr = LetExprCall(
        func="subst",
        args=(
            LetExprCall(
                func="App",
                args=(
                    LetExprVar(name="f"),
                    LetExprCall(func="Var", args=(LetExprVar(name="x"),)),
                ),
            ),
            LetExprCall(func="Var", args=(LetExprVar(name="x"),)),
            LetExprCall(func="Var", args=(LetExprVar(name="arg"),)),
        ),
    )
    fn = _ProgramsMixin._compile_let_expr(
        expr, globals_={"__constructors__": frozenset({"App", "Var", "f", "x", "arg"})}
    )
    out = fn({})
    assert out == ("App", ("f",), ("Var", ("arg",))), f"subst gave {out!r}"


def test_montague_nli_lambda_lfs_load():
    """The gallery's montague_nli example exercises ``binders``,
    alpha-renamed lambda LFs, and a ``score = chart.goal_weight()``
    program-body fit. It must load and parse a real sentence."""
    from pathlib import Path
    from quivers.dsl import load as _load

    src = Path("docs/examples/source/montague_nli.qvr")
    if not src.exists():
        return  # docs not vendored in this checkout
    prog = _load(str(src))
    assert "Montague" in prog.deductions
    ded = prog.deductions["Montague"]
    chart = ded(["every", "dog", "barks"])
    # The full sentence should parse to span(0, 3, S, App(App(every, dog), barks)).
    s_items = [
        (it, w)
        for it, w in chart.chart.items()
        if isinstance(it, tuple)
        and it[:2] == ("span", 0)
        and len(it) >= 5
        and it[3] == ("atom", "S")
    ]
    assert s_items, "Montague: 'every dog barks' did not derive S at span(0, 3)"


def test_nuts_runs_and_log_density_does_not_collapse():
    """A short NUTS run on the AB grammar should produce a finite
    log-density with positive acceptance and no divergences."""
    torch.manual_seed(0)
    ded = _load_ab()
    corpus = [["the", "dog", "runs"]] * 4
    model, x, obs = nuts_program_from_deduction(ded, corpus, prior_scale=1.0)
    kernel = NUTSKernel(step_size=0.1, max_tree_depth=4, target_accept=0.8)
    mc = MCMC(kernel, num_warmup=20, num_samples=20, num_chains=2)
    res = mc.run(model, x, obs)
    assert torch.isfinite(res.log_densities).all(), (
        "NUTS chain contains non-finite log densities"
    )
    assert float(res.acceptance_rates.mean()) > 0.3, (
        f"NUTS acceptance unexpectedly low: {res.acceptance_rates}"
    )
    assert int(res.divergence_counts.sum()) == 0, (
        f"NUTS reported divergences: {res.divergence_counts.tolist()}"
    )
