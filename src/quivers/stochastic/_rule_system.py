"""Core RuleSystem dataclass for composable chart parsing rules.

This private module exists to break circular imports between
``biclosed.py`` (primitive categorical operations) and ``rules.py``
(grammar-specific rule combinations).

Each rule may carry an optional initial log-weight. When registered
in a ``ChartParser``, these become learnable ``nn.Parameter`` values
that are added to rule scores during chart filling, implementing a
weighted deductive system.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RuleSystem:
    """A composable set of structural rules for chart parsing.

    Parameters
    ----------
    binary_rules : tuple of (int, int, int)
        Each triple is (result_idx, left_idx, right_idx).
    unary_rules : tuple of (int, int)
        Each pair is (result_idx, input_idx).
    n_categories : int
        Total number of categories in the system.
    description : str
        Human-readable label (e.g., "CCG", "Lambek(L)").
    binary_weights : tuple of float or None
        Initial log-weights for binary rules. If None, all rules
        are unweighted (weight 0 in log-space). Length must match
        ``binary_rules`` if provided.
    unary_weights : tuple of float or None
        Initial log-weights for unary rules. If None, all rules
        are unweighted. Length must match ``unary_rules`` if provided.
    """

    binary_rules: tuple[tuple[int, int, int], ...] = ()
    unary_rules: tuple[tuple[int, int], ...] = ()
    n_categories: int = 0
    description: str = ""
    binary_weights: tuple[float, ...] | None = None
    unary_weights: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.binary_weights is not None:
            if len(self.binary_weights) != len(self.binary_rules):
                raise ValueError(
                    f"binary_weights length ({len(self.binary_weights)}) "
                    f"must match binary_rules ({len(self.binary_rules)})"
                )

        if self.unary_weights is not None:
            if len(self.unary_weights) != len(self.unary_rules):
                raise ValueError(
                    f"unary_weights length ({len(self.unary_weights)}) "
                    f"must match unary_rules ({len(self.unary_rules)})"
                )

    @property
    def n_binary(self) -> int:
        """Number of binary rules."""
        return len(self.binary_rules)

    @property
    def n_unary(self) -> int:
        """Number of unary rules."""
        return len(self.unary_rules)

    @property
    def has_weights(self) -> bool:
        """Whether any rules carry explicit weights."""
        return (
            self.binary_weights is not None
            or self.unary_weights is not None
        )

    def binary_tensors(
        self, device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return binary rules as (results, lefts, rights) index tensors.

        Returns
        -------
        tuple of torch.Tensor
            Three long tensors of shape ``(n_binary,)``.
        """
        if not self.binary_rules:
            empty = torch.zeros(0, dtype=torch.long, device=device)
            return empty, empty, empty

        results, lefts, rights = zip(*self.binary_rules)

        return (
            torch.tensor(results, dtype=torch.long, device=device),
            torch.tensor(lefts, dtype=torch.long, device=device),
            torch.tensor(rights, dtype=torch.long, device=device),
        )

    def binary_weight_tensor(
        self, device: torch.device | None = None,
    ) -> torch.Tensor:
        """Return initial binary rule weights as a float tensor.

        Returns
        -------
        torch.Tensor
            Shape ``(n_binary,)``. Zeros if no explicit weights.
        """
        if self.binary_weights is not None:
            return torch.tensor(
                self.binary_weights, dtype=torch.float, device=device,
            )

        return torch.zeros(
            self.n_binary, dtype=torch.float, device=device,
        )

    def unary_tensors(
        self, device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return unary rules as (results, inputs) index tensors.

        Returns
        -------
        tuple of torch.Tensor or None
            Two long tensors of shape ``(n_unary,)`` or None if no
            unary rules.
        """
        if not self.unary_rules:
            return None

        results, inputs = zip(*self.unary_rules)

        return (
            torch.tensor(results, dtype=torch.long, device=device),
            torch.tensor(inputs, dtype=torch.long, device=device),
        )

    def unary_weight_tensor(
        self, device: torch.device | None = None,
    ) -> torch.Tensor:
        """Return initial unary rule weights as a float tensor.

        Returns
        -------
        torch.Tensor
            Shape ``(n_unary,)``. Zeros if no explicit weights.
        """
        if self.unary_weights is not None:
            return torch.tensor(
                self.unary_weights, dtype=torch.float, device=device,
            )

        return torch.zeros(
            self.n_unary, dtype=torch.float, device=device,
        )

    def __add__(self, other: RuleSystem) -> RuleSystem:
        """Merge two rule systems (deduplicated union).

        Parameters
        ----------
        other : RuleSystem
            The rule system to merge with.

        Returns
        -------
        RuleSystem
            A new rule system with the union of both rule sets.

        Raises
        ------
        ValueError
            If the rule systems have different category counts.
        """
        if self.n_categories != other.n_categories:
            raise ValueError(
                f"cannot merge rule systems with different category "
                f"counts: {self.n_categories} vs {other.n_categories}"
            )

        # deduplicated merge for binary rules, preserving weights
        seen_binary: dict[tuple[int, int, int], float] = {}

        for i, rule in enumerate(self.binary_rules):
            w = self.binary_weights[i] if self.binary_weights else 0.0
            seen_binary[rule] = w

        for i, rule in enumerate(other.binary_rules):
            if rule not in seen_binary:
                w = other.binary_weights[i] if other.binary_weights else 0.0
                seen_binary[rule] = w

        # deduplicated merge for unary rules, preserving weights
        seen_unary: dict[tuple[int, int], float] = {}

        for i, rule in enumerate(self.unary_rules):
            w = self.unary_weights[i] if self.unary_weights else 0.0
            seen_unary[rule] = w

        for i, rule in enumerate(other.unary_rules):
            if rule not in seen_unary:
                w = other.unary_weights[i] if other.unary_weights else 0.0
                seen_unary[rule] = w

        binary_rules = tuple(seen_binary.keys())
        unary_rules = tuple(seen_unary.keys())

        # only carry weights if either side had explicit weights
        has_binary_w = (
            self.binary_weights is not None
            or other.binary_weights is not None
        )
        has_unary_w = (
            self.unary_weights is not None
            or other.unary_weights is not None
        )

        binary_weights = (
            tuple(seen_binary.values()) if has_binary_w else None
        )
        unary_weights = (
            tuple(seen_unary.values()) if has_unary_w else None
        )

        desc_parts = []

        if self.description:
            desc_parts.append(self.description)

        if other.description:
            desc_parts.append(other.description)

        return RuleSystem(
            binary_rules=binary_rules,
            unary_rules=unary_rules,
            n_categories=self.n_categories,
            description=" + ".join(desc_parts) if desc_parts else "",
            binary_weights=binary_weights,
            unary_weights=unary_weights,
        )

    def __repr__(self) -> str:
        label = f" ({self.description})" if self.description else ""
        weighted = " weighted" if self.has_weights else ""
        return (
            f"RuleSystem(binary={self.n_binary}, "
            f"unary={self.n_unary}, "
            f"categories={self.n_categories}"
            f"{weighted}{label})"
        )
