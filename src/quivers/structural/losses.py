"""Loss attachment registry.

Holds the table of weighted scalar losses declared in a compiled
module, keyed by attachment site, so the training driver can
evaluate the right ones at the right point in the training step.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal

import torch


type LossBody = Callable[[Mapping[str, "TrainEnv"]], torch.Tensor]
"""A loss term: maps the training environment to a scalar loss."""

type LossWeight = Callable[[Mapping[str, "TrainEnv"]], torch.Tensor]
"""A loss weight: maps the training environment to a scalar multiplier."""


type TrainEnv = torch.nn.Module | torch.Tensor | int | float | str | list | tuple | dict
"""A value of the training environment a loss body reads.

The environment holds compiled encoders, decoders, and deductions
(each an ``nn.Module``), input tensors, structured-term
observations, and raw scalar or tensor targets.
"""


type AttachmentKind = Literal[
    "global",
    "program",
    "deduction",
    "encoder",
    "decoder",
    "rule",
    "chart",
]
"""The kind of construct a registered loss is attached to."""


@dataclass
class LossEntry:
    """One registered loss.

    Attributes
    ----------
    name : str
        Diagnostic identifier (the DSL name).
    body : callable
        Computes the scalar loss given a training-step environment.
    weight : callable | None
        Computes a scalar multiplier given the same environment, or
        ``None`` for an implicit weight of 1.
    attachment_kind : str
        Where this loss fires (see `AttachmentKind`).
    target : str | None
        Name of the attachment target (program / deduction /
        encoder / decoder / chart / rule).
    rule_deduction : str | None
        For ``rule``-attached losses, the deduction the rule lives in.
    """

    name: str
    body: LossBody
    weight: LossWeight | None = None
    attachment_kind: AttachmentKind = "global"
    target: str | None = None
    rule_deduction: str | None = None


@dataclass
class LossRegistry:
    """All losses declared in a compiled module.

    Attributes
    ----------
    entries : list[LossEntry]
        The registered losses in registration order.
    """

    entries: list[LossEntry] = field(default_factory=list)

    def add(self, entry: LossEntry) -> None:
        """Register a loss.

        Parameters
        ----------
        entry : LossEntry
            The loss to append to the registry.
        """
        self.entries.append(entry)

    def by_attachment(
        self,
        kind: AttachmentKind,
        target: str | None = None,
    ) -> list[LossEntry]:
        """Return the losses attached to a kind of construct.

        Parameters
        ----------
        kind : AttachmentKind
            The attachment kind to select.
        target : str or None
            When given, only losses attached to the target of this
            name are returned.

        Returns
        -------
        list[LossEntry]
            The matching losses in registration order.
        """
        return [
            e
            for e in self.entries
            if e.attachment_kind == kind and (target is None or e.target == target)
        ]

    def evaluate(
        self,
        env: Mapping[str, TrainEnv] | None = None,
    ) -> torch.Tensor:
        """Sum all registered losses, weighted, under ``env``.

        Parameters
        ----------
        env : Mapping[str, TrainEnv] or None
            The training-step environment passed to every loss body
            and weight; empty when omitted.

        Returns
        -------
        torch.Tensor
            The scalar weighted sum of the losses.
        """
        return self._weighted_sum(self.entries, env or {})

    def evaluate_on(
        self,
        kind: AttachmentKind,
        target: str | None = None,
        env: Mapping[str, TrainEnv] | None = None,
        rule_deduction: str | None = None,
    ) -> torch.Tensor:
        """Sum only the losses whose attachment matches the filter.

        ``kind`` selects the attachment kind; ``target`` filters by
        attachment target (the program / deduction / encoder /
        decoder / rule name); ``rule_deduction`` further narrows the
        ``"rule"`` kind to a specific enclosing deduction.

        Parameters
        ----------
        kind : AttachmentKind
            The attachment kind to select.
        target : str or None
            When given, the attachment target the losses must name.
        env : Mapping[str, TrainEnv] or None
            The training-step environment passed to every loss body
            and weight; empty when omitted.
        rule_deduction : str or None
            When given, the enclosing deduction a ``"rule"`` loss must
            name.

        Returns
        -------
        torch.Tensor
            The scalar weighted sum of the matching losses.
        """
        matching = []
        for e in self.entries:
            if e.attachment_kind != kind:
                continue
            if target is not None and e.target != target:
                continue
            if rule_deduction is not None and e.rule_deduction != rule_deduction:
                continue
            matching.append(e)
        return self._weighted_sum(matching, env or {})

    def _weighted_sum(
        self,
        entries: list[LossEntry],
        env: Mapping[str, TrainEnv],
    ) -> torch.Tensor:
        total = torch.zeros(())
        for e in entries:
            val = e.body(env)
            if not isinstance(val, torch.Tensor):
                val = torch.tensor(float(val))
            if e.weight is not None:
                w = e.weight(env)
                if not isinstance(w, torch.Tensor):
                    w = torch.tensor(float(w))
                val = val * w
            total = total + val
        return total


__all__ = [
    "LossBody",
    "LossWeight",
    "TrainEnv",
    "AttachmentKind",
    "LossEntry",
    "LossRegistry",
]
