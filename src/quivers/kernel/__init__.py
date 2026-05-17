"""Jupyter kernel for QVR.

Two console-script entry points:

- ``qvr kernel install`` registers the kernelspec.
- ``qvr kernel run`` is invoked by Jupyter once a notebook starts.

The kernel reuses :class:`quivers.cli.repl_session.ReplSession` so
notebook cells behave exactly like the REPL: leading ``:`` runs a meta
command; bare cells parse first as statements (appended to the
session's module) and fall back to type-printing.
"""

from quivers.kernel.quivers_kernel import QuiversKernel

__all__ = ["QuiversKernel"]
