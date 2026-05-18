"""Top-level Module model: the root of every parsed .qvr file."""

import didactic.api as dx

from quivers.dsl.ast_nodes.declarations import Statement

class Module(dx.Model):
    """A complete .qvr program (sequence of statements)."""

    statements: tuple[Statement, ...] = dx.field(default_factory=tuple)

__all__ = ["Module"]
