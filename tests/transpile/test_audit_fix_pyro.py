"""Regression tests for the Pyro-backend transpile audit fixes.

Each test transpiles a model exercising one family and asserts the
emitted Pyro source now carries the corrected distribution call. The
audit-confirmed defects these guard:

- `lkj-missing-dimension-arg`: `LKJCholesky` lowered to
  `pyro.distributions.LKJCorrCholesky(eta)` without the mandatory
  matrix-dimension leading argument, so `eta` bound to `d` and the
  concentration was dropped. The dimension is now prepended from the
  sample's event axis: `LKJCorrCholesky(4, eta)`.
- `logitnormal-nonexistent-pyro`: `LogitNormal` emitted as
  `pyro.distributions.LogitNormal`, which Pyro does not ship
  (AttributeError). It is now a grafted runtime helper class realising
  the sigmoid-transform of a Normal, called by its bare name.
- `halfstudentt-nonexistent-pyro`: `HalfStudentT` emitted as
  `pyro.distributions.HalfStudentT`, which Pyro does not ship. It is
  now a grafted runtime helper class realising the folded StudentT.
- `matrixnormal-inversewishart-nonexistent-pyro` (MatrixNormal half):
  `MatrixNormal` emitted as `pyro.distributions.MatrixNormal`, which
  Pyro does not ship, and its covariance arguments were broadcast to
  the full `loc` shape `(3, 4)` rather than the per-factor `(3, 3)` /
  `(4, 4)`. It is now a grafted runtime helper class with per-factor
  covariance shapes derived from the structured-lowering axis indices.

Assertions run on the emitted source text (fast, deterministic).
"""

from __future__ import annotations

import pathlib

import pytest

from quivers.dsl.parser import parse
from quivers.transpile import transpile


_FIXTURES = (
    pathlib.Path(__file__).parent / "fixtures" / "families"
)


def _emit(source: str) -> tuple[str, str]:
    """Transpile `source` to Pyro; return the decoded text and a
    whitespace-stripped variant for spacing-robust substring checks."""
    text = transpile(parse(source), target="pyro").decode("utf-8")
    stripped = "".join(text.split())
    return text, stripped


_LKJ_SOURCE = """
object Dim : FinSet 4

program correlation_model : Dim -> Dim
    sample eta <- HalfNormal(2.0)
    sample chol : Dim <- LKJCholesky(eta)
    return chol

export correlation_model
"""


_LOGITNORMAL_SOURCE = """
object Obs : FinSet 8

program logitnormal_model : Obs -> Obs
    sample theta <- LogitNormal(0.0, 1.0)
    return theta

export logitnormal_model
"""


_HALFSTUDENTT_SOURCE = """
object Resp : FinSet 1

program half_student_t_model : Resp -> Resp
    sample sigma <- HalfStudentT(3.0, 1.0)
    return sigma

export half_student_t_model
"""


def test_lkj_prepends_matrix_dimension() -> None:
    """`LKJCholesky` over a `FinSet 4` event axis emits
    `LKJCorrCholesky(4, eta)`, dimension first, concentration second.
    """
    text, stripped = _emit(_LKJ_SOURCE)
    assert "LKJCorrCholesky" in text
    # Dimension prepended ahead of the concentration reference.
    assert "LKJCorrCholesky(4,eta)" in stripped, text
    # The pre-fix single-arg form must not survive.
    assert "LKJCorrCholesky(eta)" not in stripped, text


def test_logitnormal_grafts_helper_not_pyro_class() -> None:
    """`LogitNormal` emits a grafted helper class plus a bare-name
    call, never the nonexistent `pyro.distributions.LogitNormal`."""
    text, stripped = _emit(_LOGITNORMAL_SOURCE)
    # The runtime helper class is grafted onto the module.
    assert "class LogitNormal(" in text
    # Called by its bare name with (loc, scale) in order.
    assert "LogitNormal(0,1)" in stripped, text
    # The undefined pyro attribute must not be emitted.
    assert "pyro.distributions.LogitNormal" not in text


def test_halfstudentt_grafts_helper_not_pyro_class() -> None:
    """`HalfStudentT` emits a grafted helper class plus a bare-name
    call, never the nonexistent `pyro.distributions.HalfStudentT`."""
    text, stripped = _emit(_HALFSTUDENTT_SOURCE)
    assert "class HalfStudentT(" in text
    # Called by its bare name with (df, scale) in order.
    assert "HalfStudentT(3,1)" in stripped, text
    assert "pyro.distributions.HalfStudentT" not in text


def test_matrixnormal_grafts_helper_with_per_factor_covariance_shapes() -> None:
    """`MatrixNormal` over `Row(3) x Col(4)` emits a grafted helper
    class with the row covariance broadcast to `(3, 3)`, the column
    covariance to `(4, 4)`, and the mean to `(3, 4)` (not every
    covariance to the `loc` shape), never the nonexistent
    `pyro.distributions.MatrixNormal`."""
    source = (_FIXTURES / "matrixnormal.qvr").read_text()
    text, stripped = _emit(source)
    assert "class MatrixNormal(" in text
    assert "pyro.distributions.MatrixNormal" not in text
    # Mean fills the full (3, 4) matrix.
    assert "torch.full((3,4,),m_loc)" in stripped, text
    # Row covariance is the (Row, Row) = (3, 3) factor.
    assert "torch.full((3,3,),m_row_covariance)" in stripped, text
    # Column covariance is the (Col, Col) = (4, 4) factor.
    assert "torch.full((4,4,),m_col_covariance)" in stripped, text
    # The pre-fix bug filled the covariances to the (3, 4) loc shape.
    assert "torch.full((3,4,),m_row_covariance)" not in stripped, text
    assert "torch.full((3,4,),m_col_covariance)" not in stripped, text


@pytest.mark.parametrize(
    "source",
    [_LOGITNORMAL_SOURCE, _HALFSTUDENTT_SOURCE, _LKJ_SOURCE],
    ids=["logitnormal", "halfstudentt", "lkj"],
)
def test_emitted_model_executes(source: str) -> None:
    """The emitted Pyro model runs end-to-end under a `{pyro, torch}`
    namespace: the grafted helpers and the LKJ dimension arg are not
    merely well-formed text but a runnable program."""
    pyro = pytest.importorskip("pyro")
    torch = pytest.importorskip("torch")
    text, _ = _emit(source)
    namespace: dict[str, object] = {"pyro": pyro, "torch": torch}
    exec(text, namespace)  # noqa: S102 - emitted source under test
    model = namespace["model"]
    assert callable(model)
    with pyro.poutine.trace():
        model()
