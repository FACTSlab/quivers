"""Analytic marginalisation of a conjugate pair as a reparameterisation."""

from __future__ import annotations

import torch

from quivers.effects.conjugate import conjugate_solver, family_of
from quivers.effects.reparam.base import Reparam, SiteRequest


class ConjugateReparam(Reparam):
    """Score a child site under the conjugate marginal of its parent.

    Applied to the child site of a conjugate pair, the strategy answers
    the site with the run's observation for it, or a draw from the
    child's own sampleable when there is none, and scores it under the
    closed-form marginal that integrates the parent out, read from the
    parent's parameters as the orchestrator saw them. The parent's own
    density is left in the joint; collapse the parent with
    `quivers.effects.collapse` when the marginal joint is wanted.

    Parameters
    ----------
    parent : str
        The parent site's name.
    parent_family : str
        The parent's family.
    child_family : str
        The child's family.

    Raises
    ------
    KeyError
        If the pair has no registered marginal.
    """

    def __init__(self, parent: str, parent_family: str, child_family: str) -> None:
        self.parent = parent
        self.parent_family = parent_family
        self.child_family = child_family
        self.solver = conjugate_solver(parent_family, child_family)

    def apply(self, site: SiteRequest) -> tuple[torch.Tensor, torch.Tensor]:
        """Rewrite the child site.

        Parameters
        ----------
        site : SiteRequest
            The child site.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The value and its marginal log density.

        Raises
        ------
        KeyError
            If the orchestrator has not seen the parent site.
        ValueError
            If the parent or the child is not of the declared family.
        """
        try:
            parent = site.seen[self.parent]
        except KeyError as error:
            raise KeyError(
                f"ConjugateReparam: parent site {self.parent!r} was not seen "
                f"before child {site.name!r}"
            ) from error
        if family_of(parent) != self.parent_family:
            raise ValueError(
                f"ConjugateReparam: parent {self.parent!r} draws from "
                f"{family_of(parent)!r}, not {self.parent_family!r}"
            )
        if family_of(site.sampleable) != self.child_family:
            raise ValueError(
                f"ConjugateReparam: child {site.name!r} draws from "
                f"{family_of(site.sampleable)!r}, not {self.child_family!r}"
            )
        if site.given is not None:
            value = site.given
        elif site.observation is not None:
            value = site.observation
        else:
            value = site.sampleable.rsample()
        parent_value = site.sampleable.input
        return value, self.solver(
            parent.parameters(), parent_value, site.sampleable, value
        )


__all__ = ["ConjugateReparam"]
