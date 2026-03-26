"""Backward compatibility shim. Use quivers.stochastic.giry instead."""

from __future__ import annotations

from quivers.stochastic.giry import GiryMonad, FinStoch

__all__ = ["GiryMonad", "FinStoch"]
