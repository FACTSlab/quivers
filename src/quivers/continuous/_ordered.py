"""Ordered-categorical distribution families that PyTorch does not
ship natively (`OrderedLogistic`, `OrderedProbit`).

The DSL surfaces these as inline call sites:

    observe y : Resp <- OrderedLogistic(eta, cutpoints)

with ``eta`` a real predictor (one row per observation) and
``cutpoints`` either a globally shared sorted vector of length
``K - 1`` or a per-row sorted matrix of shape ``(batch, K - 1)``.
Per-row cutpoints land in the ordinal-mixed-model setting where
every participant has distinct thresholds; the distribution's
broadcast rules handle both shapes uniformly.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.distributions import constraints as _constraints
from torch.distributions.distribution import Distribution


class OrderedLogistic(Distribution):
    """Ordered-logit distribution over ``K = cutpoints.shape[-1] + 1``
    ordered categories indexed ``0 .. K - 1``.

    With a real predictor ``eta`` and a sorted cutpoint vector
    ``c = (c_0 < c_1 < ... < c_{K-2})``:

    * ``P(Y = 0)       = sigmoid(c_0 - eta)``
    * ``P(Y = k)       = sigmoid(c_k - eta) - sigmoid(c_{k-1} - eta)``
                          for ``0 < k < K - 1``
    * ``P(Y = K - 1)   = 1 - sigmoid(c_{K-2} - eta)``

    Cutpoint broadcasting:

    * Shared cutpoints ``c`` of shape ``(K - 1,)`` apply uniformly to
      every row.
    * Per-row cutpoints of shape ``(batch, K - 1)`` apply distinct
      thresholds per observation (the ordinal-mixed-model case).
    * Arbitrary leading batch shape is supported; the last axis is
      always the cutpoint axis.

    Reference: [McCullagh 1980](https://doi.org/10.1111/j.2517-6161.1980.tb01109.x).
    """

    arg_constraints = {
        "predictor": _constraints.real,
        "cutpoints": _constraints.real_vector,
    }
    has_rsample = False

    def __init__(
        self,
        predictor: Tensor,
        cutpoints: Tensor,
        validate_args: bool | None = None,
    ) -> None:
        if cutpoints.dim() == 0:
            raise ValueError(
                "OrderedLogistic: `cutpoints` must have at least one "
                f"dimension carrying the K-1 thresholds, got "
                f"shape={tuple(cutpoints.shape)}"
            )
        if cutpoints.shape[-1] < 1:
            raise ValueError(
                "OrderedLogistic: `cutpoints` last dimension must "
                f"have size >= 1 (K >= 2 categories), got "
                f"shape={tuple(cutpoints.shape)}"
            )
        self.predictor = predictor
        self.cutpoints = cutpoints
        self._num_categories = int(cutpoints.shape[-1]) + 1
        batch_shape = torch.broadcast_shapes(predictor.shape, cutpoints.shape[:-1])
        super().__init__(
            batch_shape=batch_shape,
            event_shape=torch.Size(()),
            validate_args=validate_args,
        )

    @property
    def num_categories(self) -> int:
        return self._num_categories

    @_constraints.dependent_property
    def support(self):
        return _constraints.integer_interval(0, self._num_categories - 1)

    @property
    def mean(self) -> Tensor:
        weights = torch.arange(
            self._num_categories,
            device=self.predictor.device,
            dtype=self.predictor.dtype,
        )
        probs = self._category_probs()
        return (probs * weights).sum(dim=-1)

    @property
    def mode(self) -> Tensor:
        return self._category_probs().argmax(dim=-1)

    def log_prob(self, value: Tensor) -> Tensor:
        if self._validate_args:
            self._validate_sample(value)
        probs = self._category_probs()
        idx = value.long().unsqueeze(-1)
        idx = idx.expand(*probs.shape[:-1], 1)
        selected = probs.gather(-1, idx).squeeze(-1)
        return selected.clamp_min(torch.finfo(selected.dtype).tiny).log()

    def sample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        # `torch.distributions.Distribution.sample` accepts any
        # `Sequence[int]` (tuple, list, or `torch.Size`); coerce
        # so callers writing ``.sample((200,))`` work uniformly.
        sample_shape = torch.Size(sample_shape)
        probs = self._category_probs()
        flat = probs.reshape(-1, self._num_categories)
        draws = torch.multinomial(
            flat,
            num_samples=max(1, sample_shape.numel()) if sample_shape else 1,
            replacement=True,
        )
        if not sample_shape:
            return draws[..., 0].reshape(self.batch_shape)
        return draws.t().reshape(*sample_shape, *self.batch_shape)

    def _category_probs(self) -> Tensor:
        """Return ``(*batch_shape, num_categories)`` probabilities."""
        eta = self.predictor.unsqueeze(-1)
        cdf = torch.sigmoid(self.cutpoints - eta)
        zero = torch.zeros_like(cdf[..., :1])
        one = torch.ones_like(cdf[..., :1])
        padded = torch.cat([zero, cdf, one], dim=-1)
        return padded[..., 1:] - padded[..., :-1]


__all__ = ["OrderedLogistic"]
