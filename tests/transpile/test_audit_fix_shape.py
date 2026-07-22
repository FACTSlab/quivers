"""Regression tests for audit-confirmed cross-backend shape / plate
defects in the QVR transpiler.

Each test transpiles a gallery model and asserts the emitted program
now carries the axis roles the QVR source declared. Assertions run on
the lowered IR where the defect is a lost axis role, and on the emitted
text where the defect is a missing broadcast or gather.

Covered defects:

* `marginalize-over-axis-dropped-numpyro-pymc`: flattening a
  `MarginalizeStep` into a `SampleStep` before lowering discarded the
  step's `[over=...]` grouping axes and promoted its `: T` support
  index to a batch axis, so the latent plated over the wrong object.
* `stan-continuous-marginalize-index-plate-dropped`: the let-plate
  propagation fixpoint never descended into a marginalize scope, so a
  `let` reading the plated latent stayed scalar and dragged every
  upstream `let` down with it.
* `numpyro-jags-scalar-dirichlet-concentration-not-broadcast`: a free
  scalar data input in a vector-expected argument slot was emitted
  verbatim, which is a rank error at run time.
"""

from __future__ import annotations

from pathlib import Path

from quivers.dsl.parser import parse
from quivers.transpile import transpile
from quivers.transpile._expand_composites import expand_composite_lets
from quivers.transpile.ir import IRDeterministic, IRMarginalize, IRNode
from quivers.transpile.lower import Lower

_GALLERY = Path(__file__).resolve().parents[2] / "docs" / "examples" / "source"


def _emit(name: str, target: str) -> str:
    """Transpile the named gallery model to `target` and decode."""
    source = (_GALLERY / f"{name}.qvr").read_text()
    return transpile(parse(source), target=target).decode()


def _lower(name: str) -> tuple[IRNode, ...]:
    """Lower the named gallery model and return its IR body."""
    source = (_GALLERY / f"{name}.qvr").read_text()
    return Lower().forward(parse(source)).body


def _nospace(text: str) -> str:
    """Drop spaces so assertions ignore the formatter's comma spacing."""
    return text.replace(" ", "")


def _find_marginalize(body: tuple[IRNode, ...]) -> IRMarginalize:
    """Return the single `IRMarginalize` in `body`."""
    found = [n for n in body if isinstance(n, IRMarginalize)]
    assert len(found) == 1, f"expected one marginalize, got {len(found)}"
    return found[0]


def _find_let(body: tuple[IRNode, ...], name: str) -> IRDeterministic:
    """Return the `IRDeterministic` bound to `name` in `body`."""
    found = [
        n for n in body if isinstance(n, IRDeterministic) and n.name == name
    ]
    assert len(found) == 1, f"expected one let named {name!r}"
    return found[0]


# ---------------------------------------------------------------------------
# marginalize-over-axis-dropped-numpyro-pymc
# ---------------------------------------------------------------------------


def test_marginalize_survives_composite_expansion_for_every_target() -> None:
    """`expand_composite_lets` must not rewrite a `MarginalizeStep`.

    A `SampleStep` has no slot for the marginalize distinction between
    the `[over=...]` grouping axes and the `: T` support index, so the
    rewrite could only preserve one of the two.
    """
    source = (_GALLERY / "lda.qvr").read_text()
    module = parse(source)
    for target in ("numpyro", "pymc", "pyro", "stan", "turing"):
        expanded = expand_composite_lets(module, target=target)
        marginalize = _find_marginalize(Lower().forward(expanded).body)
        assert marginalize.latent == "z"
        assert [d.name for d in marginalize.plate.batch_dims] == ["Doc"]
        assert marginalize.plate.event_dims == ()


def test_lda_marginalize_batch_axis_is_the_over_axis() -> None:
    """`marginalize z : Topic <- ... [over=Doc]` plates over `Doc`.

    `Topic` names the enumerated support the reduction sums over, not a
    replication axis, so it contributes no plate dim.
    """
    marginalize = _find_marginalize(_lower("lda"))
    batch = marginalize.plate.batch_dims
    assert [d.name for d in batch] == ["Doc"]
    assert [d.size for d in batch] == [20]


def test_lda_numpyro_plates_latent_over_documents() -> None:
    """NumPyro emits the LDA topic latent under a 20-document plate."""
    emitted = _emit("lda", "numpyro")
    assert _nospace('numpyro.plate("Doc_z",20)') in _nospace(emitted)
    assert _nospace('numpyro.plate("Topic",3)') in _nospace(emitted)
    # The `Topic`-sized plate belongs to `phi`, never to `z`.
    z_line = next(ln for ln in emitted.splitlines() if 'sample("z"' in ln)
    assert "Categorical" in z_line


def test_lda_numpyro_gathers_latent_through_the_via_fibration() -> None:
    """`observe w ... [via=word_idx]` gathers `z` at the word index.

    `z` carries the 20-document batch shape while `w` carries the
    200-word one, so `phi[z]` alone cannot broadcast under the word
    plate.
    """
    emitted = _emit("lda", "numpyro")
    assert "phi[z[word_idx]]" in _nospace(emitted)


def test_lda_pymc_dims_name_the_document_axis() -> None:
    """PyMC gives the LDA topic latent `dims=("Doc",)`."""
    emitted = _nospace(_emit("lda", "pymc"))
    assert 'pymc.Categorical("z",p=theta,dims=("Doc",))' in emitted


def test_lda_pyro_plates_latent_over_documents() -> None:
    """Pyro plates the LDA topic latent over documents, not topics."""
    emitted = _nospace(_emit("lda", "pyro"))
    assert 'pyro.plate("Doc_z",20)' in emitted


def test_lda_stan_enumerates_topics_per_document() -> None:
    """Stan keeps the accumulator per document and sums over topics."""
    emitted = _nospace(_emit("lda", "stan"))
    assert "lps_z=rep_array(rep_vector(0,3),20)" in emitted
    assert "log_sum_exp(lps_z[g_Doc])" in emitted


# ---------------------------------------------------------------------------
# stan-continuous-marginalize-index-plate-dropped
# ---------------------------------------------------------------------------


def test_zip_marginalize_latent_is_per_response() -> None:
    """A non-enumerable marginalize latent replicates over its index.

    `ContinuousBernoulli` has no finite support to sum over, so
    `marginalize z : Resp` means one latent value per response.
    """
    marginalize = _find_marginalize(_lower("zip_regression"))
    batch = marginalize.plate.batch_dims
    assert [d.name for d in batch] == ["Resp"]
    assert [d.size for d in batch] == [400]


def test_zip_scope_let_inherits_the_latent_plate() -> None:
    """`let gated_rate = z * rate` inside the scope is per-response."""
    marginalize = _find_marginalize(_lower("zip_regression"))
    gated = _find_let(marginalize.scope, "gated_rate")
    assert [d.name for d in gated.plate.batch_dims] == ["Resp"]


def test_zip_scope_let_propagates_the_plate_upstream() -> None:
    """The scope let drags every `let` it reads up to its own plate.

    `gated_rate` reads `rate`, which reads `ar` and `br`; all three sit
    outside the marginalize scope and all three are per-response.
    """
    body = _lower("zip_regression")
    for name in ("ar", "br", "rate"):
        node = _find_let(body, name)
        assert [d.name for d in node.plate.batch_dims] == ["Resp"], name


def test_zip_stan_declares_gated_rate_as_a_response_array() -> None:
    """Stan declares every ZIP transformed parameter over 400 responses."""
    emitted = _nospace(_emit("zip_regression", "stan"))
    for name in ("ar", "br", "rate", "gated_rate"):
        assert f"array[400]real{name};" in emitted, name
    assert "gated_rate[m_Resp]=z[m_Resp]*rate[m_Resp];" in emitted


def test_zip_stan_indexes_the_rate_inside_the_response_loop() -> None:
    """The Poisson likelihood reads the per-response gated rate."""
    emitted = _nospace(_emit("zip_regression", "stan"))
    assert "y[m_Resp]~poisson(gated_rate[m_Resp]);" in emitted


# ---------------------------------------------------------------------------
# numpyro-jags-scalar-dirichlet-concentration-not-broadcast
# ---------------------------------------------------------------------------


_SCALAR_DIRICHLET_SRC = """
object K : FinSet 4

program dirichlet_scalar_conc : K -> K
    sample p <- Dirichlet(a) [over=K]
    return p

export dirichlet_scalar_conc
"""


def test_numpyro_broadcasts_a_free_scalar_dirichlet_concentration() -> None:
    """A free scalar input in a vector slot is broadcast to the event axis.

    `Dirichlet(concentration=a)` with a scalar `a` is a rank error; the
    QVR `[over=K]` clause says the concentration is constant across the
    four-atom event axis.
    """
    emitted = _nospace(
        transpile(parse(_SCALAR_DIRICHLET_SRC), target="numpyro").decode()
    )
    assert "Dirichlet(concentration=jnp.full((4,),a))" in emitted


def test_numpyro_broadcasts_a_scalar_program_parameter() -> None:
    """The same broadcast fires for a declared `Real` program parameter."""
    emitted = _nospace(_emit("lda", "numpyro"))
    assert "Dirichlet(concentration=jnp.full((3,),alpha))" in emitted
    assert "Dirichlet(concentration=jnp.full((200,),beta))" in emitted


def test_numpyro_leaves_an_already_vector_argument_alone() -> None:
    """A let-bound / sampled vector argument is emitted verbatim.

    `Categorical(probs=theta)` reads a simplex-valued sample, so no
    broadcast wrapper may be injected around it.
    """
    emitted = _nospace(_emit("lda", "numpyro"))
    assert "Categorical(probs=theta)" in emitted
