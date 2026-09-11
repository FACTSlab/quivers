"""Backends for [`quivers.transpile`][quivers.transpile].

Each backend module registers itself via
[`@dx.codegen.emitter("qvr-<name>")`][didactic.codegen.emitter] on
import. Importing this package performs the registration.
"""

from __future__ import annotations

__all__: list[str] = []
