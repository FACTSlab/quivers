"""Jupyter kernel for QVR.

Two console-script entry points:

- ``qvr kernel install`` registers the kernelspec.
- ``qvr kernel run`` is invoked by Jupyter once a notebook starts.

The kernel reuses [`ReplSession`][quivers.cli.ReplSession] so
notebook cells behave exactly like the REPL: leading ``:`` runs a meta
command; bare cells parse first as statements (appended to the
session's module) and fall back to type-printing.
"""

from quivers.kernel.install import KERNELSPEC, install_kernelspec
from quivers.kernel.quivers_kernel import QuiversKernel

__all__ = ["KERNELSPEC", "QuiversKernel", "install_kernelspec"]
