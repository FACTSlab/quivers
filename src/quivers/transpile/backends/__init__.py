"""Backends for [`quivers.transpile`][quivers.transpile].

Each backend module registers itself via
[`@dx.codegen.emitter("qvr-<name>")`][didactic.codegen.emitter] on
import, so the public surface is just the side-effect of importing
this package.
"""

from __future__ import annotations

__all__: list[str] = []
