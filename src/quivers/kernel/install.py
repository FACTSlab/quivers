"""Kernelspec installation for the QVR Jupyter kernel.

``qvr kernel install`` writes a ``kernel.json`` whose argv is
``["python", "-m", "quivers.kernel", "_run_f", "{connection_file}"]``;
``qvr kernel run`` and the harness entry point invoke
`QuiversKernel` through ipykernel's bootstrap.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path


KERNELSPEC = {
    "argv": [
        sys.executable,
        "-m",
        "quivers.kernel",
        "_run_f",
        "{connection_file}",
    ],
    "display_name": "quivers",
    "language": "qvr",
    "interrupt_mode": "signal",
    "metadata": {
        "debugger": False,
    },
}
"""The ``kernel.json`` document `install_kernelspec` registers: it launches the
kernel with the running interpreter and names the language ``qvr``."""


def install_kernelspec(*, user: bool = True, prefix: str | None = None) -> Path:
    """Register the QVR kernelspec with Jupyter.

    Parameters
    ----------
    user : bool
        Install into the per-user kernel directory rather than the
        system one.
    prefix : str or None
        An explicit Jupyter prefix to install under, as
        ``PREFIX/share/jupyter/kernels``; ``None`` uses Jupyter's
        default location.

    Returns
    -------
    Path
        The directory the kernelspec was installed into.

    Raises
    ------
    ImportError
        When ``jupyter_client`` is not installed.
    """
    from jupyter_client.kernelspec import KernelSpecManager

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "kernel.json").write_text(
            json.dumps(KERNELSPEC, indent=2), encoding="utf-8"
        )
        mgr = KernelSpecManager()
        dest = mgr.install_kernel_spec(
            str(tmp_path),
            kernel_name="quivers",
            user=user,
            prefix=prefix,
        )
    return Path(dest)


def main(args: argparse.Namespace) -> int:
    if args.kernel_cmd == "install":
        try:
            path = install_kernelspec(user=args.user, prefix=args.prefix)
        except ImportError as e:
            sys.stderr.write(
                f"jupyter not installed ({e}); install with "
                "`pip install 'quivers[repl]'`.\n"
            )
            return 2
        sys.stdout.write(f"installed kernelspec to {path}\n")
        return 0
    if args.kernel_cmd == "run":
        from ipykernel.kernelapp import IPKernelApp

        from quivers.kernel.quivers_kernel import QuiversKernel

        IPKernelApp.launch_instance(kernel_class=QuiversKernel)
        return 0
    if args.kernel_cmd == "_run_f":
        # Called by Jupyter with a connection file path.
        from ipykernel.kernelapp import IPKernelApp

        from quivers.kernel.quivers_kernel import QuiversKernel

        IPKernelApp.launch_instance(
            kernel_class=QuiversKernel,
            argv=["-f", args.connection_file],
        )
        return 0
    sys.stderr.write("unknown kernel command\n")
    return 2


def _entry() -> int:
    """Console-script entry point for ``qvr-kernel``."""
    parser = argparse.ArgumentParser(prog="qvr-kernel")
    sub = parser.add_subparsers(dest="kernel_cmd", required=True)
    install_p = sub.add_parser("install")
    install_p.add_argument("--user", action="store_true")
    install_p.add_argument("--prefix", default=None)
    sub.add_parser("run")
    run_f = sub.add_parser("_run_f")
    run_f.add_argument("connection_file")
    args = parser.parse_args()
    return main(args)


__all__ = ["KERNELSPEC", "install_kernelspec"]
