"""Static atom counts for categorical marginalization."""

from __future__ import annotations

from quivers.transpile.ir import (
    CSIntegerInterval,
    DimStatic,
    IRArgBroadcast,
    IRArgList,
    IRArgNumber,
    IRArgRef,
    IRMarginalize,
    Plate,
)
from quivers.transpile.renderers._python_helpers import marginal_support_size


def _marginal(argument: IRArgRef | IRArgList | IRArgBroadcast) -> IRMarginalize:
    """A categorical marginal whose family constraint does not encode K."""

    return IRMarginalize(
        latent="z",
        family="Categorical",
        args=(argument,),
        arg_names=("probs",),
        constraint=CSIntegerInterval(lower=0, upper=1),
        plate=Plate(event_dims=(), batch_dims=()),
        reduction="logsumexp",
        scope=(),
    )


def test_computed_probability_tensor_uses_its_trailing_batch_extent() -> None:
    """A QIEC call or factor records result axes as batch dimensions."""

    node = _marginal(IRArgRef(name="probs"))
    plates = {
        "probs": Plate(
            event_dims=(),
            batch_dims=(
                DimStatic(size=4, name="Group"),
                DimStatic(size=6, name="Class"),
            ),
        )
    }
    assert marginal_support_size(node, name_plates=plates) == 6


def test_sampled_simplex_uses_its_trailing_event_extent() -> None:
    """A sampled Dirichlet continues to expose its class event axis."""

    node = _marginal(IRArgRef(name="probs"))
    plates = {
        "probs": Plate(event_dims=(DimStatic(size=5, name="Class"),), batch_dims=())
    }
    assert marginal_support_size(node, name_plates=plates) == 5


def test_an_indexed_reference_drops_consumed_leading_axes() -> None:
    node = _marginal(IRArgRef(name="probs", indices=(IRArgNumber(value=0.0),)))
    plates = {
        "probs": Plate(
            event_dims=(),
            batch_dims=(
                DimStatic(size=4, name="Group"),
                DimStatic(size=3, name="Class"),
            ),
        )
    }
    assert marginal_support_size(node, name_plates=plates) == 3


def test_literal_and_broadcast_vectors_state_their_extent_directly() -> None:
    literal = _marginal(
        IRArgList(
            elements=(
                IRArgNumber(value=0.2),
                IRArgNumber(value=0.3),
                IRArgNumber(value=0.5),
            )
        )
    )
    assert marginal_support_size(literal, name_plates={}) == 3

    broadcast = _marginal(
        IRArgBroadcast(value=IRArgNumber(value=1.0), target_shape=(2, 7))
    )
    assert marginal_support_size(broadcast, name_plates={}) == 7
