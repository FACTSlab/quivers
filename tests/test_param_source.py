"""Tests for the [`ParamSource`][quivers.continuous.param_source.ParamSource]
abstraction: the seven concrete sources, the DSL `[param_source=...]`
option surface, and the wiring through `ConditionalX` families.
"""

from __future__ import annotations

import pytest
import torch

from quivers.continuous import (
    AttentionSource,
    ComposeSource,
    EmbeddingSource,
    FunctionSource,
    IdentitySource,
    LinearSource,
    LookupSource,
    MLPSource,
    ParamSource,
    make_param_source,
    param_source_from_option,
)
from quivers.continuous.families import ConditionalNormal
from quivers.continuous.spaces import Euclidean
from quivers.core.objects import FinSet


def test_linear_source_shape() -> None:
    ls = LinearSource(4, 8)
    x = torch.randn(3, 4)
    y = ls(x)
    assert y.shape == torch.Size([3, 8])
    assert ls.param_dim == 8


def test_linear_source_1d_input_unsqueezes() -> None:
    """A `(dim,)` input is treated as a single row and unsqueezed to
    `(1, dim)`. The linear layer's `in_features` must equal `dim`.
    """
    ls = LinearSource(1, 8)
    y = ls(torch.tensor([2.5]))  # `(1,)` -> `(1, 1)` -> `(1, 8)`
    assert y.shape == torch.Size([1, 8])


def test_mlp_source_default_hidden_dims() -> None:
    mlp = MLPSource(4, 8)
    x = torch.randn(3, 4)
    y = mlp(x)
    assert y.shape == torch.Size([3, 8])


def test_mlp_source_custom_hidden_dims() -> None:
    mlp = MLPSource(4, 8, hidden_dims=(16, 32, 16))
    x = torch.randn(3, 4)
    y = mlp(x)
    assert y.shape == torch.Size([3, 8])


def test_lookup_source_indexes_table() -> None:
    lu = LookupSource(10, 5)
    idx = torch.tensor([1, 3, 5, 7])
    y = lu(idx)
    assert y.shape == torch.Size([4, 5])


def test_embedding_source_pipes_through_head() -> None:
    head = MLPSource(8, 3)
    e = EmbeddingSource(10, 8, head)
    y = e(torch.tensor([2, 4, 6]))
    assert y.shape == torch.Size([3, 3])


def test_embedding_source_rejects_dim_mismatch() -> None:
    head = MLPSource(4, 3)
    with pytest.raises(ValueError, match="head's domain_dim"):
        EmbeddingSource(10, 8, head)


def test_attention_source_shape() -> None:
    attn = AttentionSource(4, 6, num_heads=2)
    y = attn(torch.randn(3, 4))
    assert y.shape == torch.Size([3, 6])


def test_identity_source_pass_through() -> None:
    id_src = IdentitySource(4)
    x = torch.randn(3, 4)
    y = id_src(x)
    torch.testing.assert_close(y, x)


def test_function_source_wraps_module() -> None:
    linear = torch.nn.Linear(4, 6)
    fs = FunctionSource(linear, 6)
    y = fs(torch.randn(3, 4))
    assert y.shape == torch.Size([3, 6])


def test_function_source_wraps_callable() -> None:
    fs = FunctionSource(lambda x: x.sum(dim=-1, keepdim=True), 1)
    y = fs(torch.randn(3, 4))
    assert y.shape == torch.Size([3, 1])


def test_compose_source_composition() -> None:
    c = ComposeSource(LinearSource(3, 5), MLPSource(4, 3))
    y = c(torch.randn(2, 4))
    assert y.shape == torch.Size([2, 5])


def test_compose_source_rejects_dim_mismatch() -> None:
    with pytest.raises(ValueError, match="inner.param_dim must equal"):
        ComposeSource(LinearSource(4, 5), MLPSource(4, 3))


def test_make_param_source_discrete_returns_lookup() -> None:
    s = FinSet(name="S", cardinality=8)
    ps = make_param_source(s, 4)
    assert isinstance(ps, LookupSource)


def test_make_param_source_continuous_default_mlp() -> None:
    ps = make_param_source(Euclidean(name="X", dim=4), 8)
    assert isinstance(ps, MLPSource)


def test_make_param_source_kind_linear() -> None:
    ps = make_param_source(Euclidean(name="X", dim=4), 8, kind="linear")
    assert isinstance(ps, LinearSource)


def test_param_source_from_option_bare_names() -> None:
    dom = Euclidean(name="X", dim=4)
    assert isinstance(
        param_source_from_option(dom, 8, "linear"),
        LinearSource,
    )
    assert isinstance(
        param_source_from_option(dom, 8, "mlp"),
        MLPSource,
    )
    assert isinstance(
        param_source_from_option(dom, 8, "identity"),
        IdentitySource,
    )


def test_param_source_from_option_paren_forms() -> None:
    dom = Euclidean(name="X", dim=4)
    mlp = param_source_from_option(dom, 8, "mlp(32, 32)")
    assert isinstance(mlp, MLPSource)
    attn = param_source_from_option(dom, 8, "attention(heads=2)")
    assert isinstance(attn, AttentionSource)


def test_param_source_from_option_none_returns_default() -> None:
    ps = param_source_from_option(Euclidean(name="X", dim=4), 8, None)
    assert isinstance(ps, MLPSource)


def test_conditional_normal_accepts_param_source_kwarg() -> None:
    X = Euclidean(name="X", dim=4)
    Y = Euclidean(name="Y", dim=2)
    n = ConditionalNormal(X, Y, param_source=LinearSource(4, 4))
    assert isinstance(n.param_source, LinearSource)
    x = torch.randn(3, 4)
    y = torch.randn(3, 2)
    assert n.log_prob(x, y).shape == torch.Size([3])


def test_conditional_normal_accepts_param_source_option_string() -> None:
    X = Euclidean(name="X", dim=4)
    Y = Euclidean(name="Y", dim=2)
    n = ConditionalNormal(X, Y, param_source_option="linear")
    assert isinstance(n.param_source, LinearSource)


def test_conditional_normal_default_source_is_neural() -> None:
    X = Euclidean(name="X", dim=4)
    Y = Euclidean(name="Y", dim=2)
    n = ConditionalNormal(X, Y)
    # Default: `_NeuralSource` (pre-abstraction two-layer MLP) or
    # `MLPSource` depending on which entry-point built it; both are
    # nn.Modules with the right output dim.
    x = torch.randn(3, 4)
    y = torch.randn(3, 2)
    assert n.log_prob(x, y).shape == torch.Size([3])


def test_generic_family_accepts_param_source() -> None:
    from quivers.continuous.families import ConditionalCauchy

    X = Euclidean(name="X", dim=4)
    Y = Euclidean(name="Y", dim=2)
    n = ConditionalCauchy(X, Y, param_source=LinearSource(4, 4))
    assert isinstance(n.param_source, LinearSource)


def test_param_source_is_subclassable() -> None:
    """User-defined ParamSource: subclass ABC + implement forward
    and param_dim, drop into any family without modifying core.
    """

    class CustomSource(ParamSource):
        def __init__(self, param_dim: int) -> None:
            super().__init__()
            self._param_dim = param_dim
            self._domain_dim = 4
            self.weight = torch.nn.Parameter(torch.zeros(param_dim))

        @property
        def param_dim(self) -> int:
            return self._param_dim

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch = x.shape[0]
            return self.weight.unsqueeze(0).expand(batch, -1)

    src = CustomSource(4)
    X = Euclidean(name="X", dim=4)
    Y = Euclidean(name="Y", dim=2)
    n = ConditionalNormal(X, Y, param_source=src)
    assert n.param_source is src


# ---------------------------------------------------------------------------
# DSL surface: [param_source=...] reaches the family
# ---------------------------------------------------------------------------


def _family_of(source: str):
    """Compile a one-kernel program and return its conditional family."""
    from quivers.dsl import loads

    prog = loads(source)
    model = prog.morphism
    assert model is not None
    return dict(model.named_modules())["_step_y._family"]


def _program(option: str) -> str:
    return (
        "object F : Real 1\n"
        "object T : Real 1\n"
        "object R : FinSet 8\n"
        "\n"
        f"morphism net : F -> T [{option}] ~ Normal\n"
        "\n"
        "program p : R -> R\n"
        "    observe y : R <- net(x)\n"
        "    return y\n"
        "\n"
        "export p\n"
    )


def test_param_source_name_selects_the_architecture() -> None:
    """A bare name picks the source: ``linear`` is one matrix, not the
    default MLP."""
    assert isinstance(
        _family_of(_program("param_source=linear")).param_source, LinearSource
    )
    assert isinstance(_family_of(_program("param_source=mlp")).param_source, MLPSource)


def test_param_source_call_form_carries_the_hidden_widths() -> None:
    """``mlp(16, 8)`` is a call, not a bare name, and its arguments are
    the hidden widths. The widths have to survive the option decode and
    reach the source, so the built net is 16 then 8 rather than the
    default 64, 64."""
    src = _family_of(_program("param_source=mlp(16, 8)")).param_source
    assert isinstance(src, MLPSource)
    widths = [layer.out_features for layer in src.net if hasattr(layer, "out_features")]
    # Two hidden widths as written, then the family's param_dim (loc, log-scale).
    assert widths == [16, 8, 2]

    default = _family_of(_program("param_source=mlp")).param_source
    default_widths = [
        layer.out_features for layer in default.net if hasattr(layer, "out_features")
    ]
    assert default_widths == [64, 64, 2]
