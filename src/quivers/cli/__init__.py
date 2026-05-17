"""Command-line entry points for quivers.

The :func:`main` function is registered as the ``qvr`` console script
in ``pyproject.toml``.

Subcommands:

- ``qvr check FILES...`` — parse + compile every supplied ``.qvr``
  file, emitting structured diagnostics. Exits 0 on full success,
  non-zero when any file produces an error.

Output format: human-readable by default, structured JSON when
``--json`` is supplied. Each diagnostic carries:

- ``file``: source path,
- ``line``, ``col``: 1-indexed source location,
- ``severity``: ``"error"``, ``"warning"``, or ``"note"``,
- ``code``: stable diagnostic code (``parse``, ``compile``,
  ``effect_constraint``, ``residuated_constraint``),
- ``message``: human-readable description.
"""

from quivers.cli.check import main as check_main


def main() -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(prog="qvr", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    check = sub.add_parser(
        "check",
        help="Parse + compile .qvr files and report diagnostics.",
    )
    check.add_argument("files", nargs="+", help="Paths to .qvr files.")
    check.add_argument(
        "--json",
        action="store_true",
        help="Emit structured JSON diagnostics on stdout.",
    )

    repl = sub.add_parser(
        "repl",
        help="Start an interactive type-exploration REPL.",
    )
    repl.add_argument(
        "file",
        nargs="?",
        default=None,
        help="Optional .qvr file to :load on startup.",
    )
    repl.add_argument(
        "--plain",
        action="store_true",
        help="Use the prompt_toolkit single-line front end instead of the TUI.",
    )

    lsp = sub.add_parser(
        "lsp",
        help="Run the QVR Language Server (LSP 3.17 over stdio).",
    )
    lsp.add_argument(
        "--tcp",
        type=int,
        default=None,
        metavar="PORT",
        help="Bind to TCP port instead of stdio.",
    )

    kernel = sub.add_parser(
        "kernel",
        help="Jupyter kernel: install kernelspec or run a kernel.",
    )
    kernel_sub = kernel.add_subparsers(dest="kernel_cmd", required=True)
    kernel_install = kernel_sub.add_parser(
        "install", help="Register the `quivers` Jupyter kernelspec."
    )
    kernel_install.add_argument(
        "--user",
        action="store_true",
        help="Install to the user kernel directory instead of system.",
    )
    kernel_install.add_argument(
        "--prefix",
        default=None,
        help="Install kernelspec under PREFIX/share/jupyter/kernels.",
    )
    kernel_sub.add_parser(
        "run", help="Run as a Jupyter kernel (invoked by Jupyter itself)."
    )
    kernel_run_f = kernel_sub.add_parser(
        "_run_f",
        help=argparse.SUPPRESS,
    )
    kernel_run_f.add_argument("connection_file")

    args = parser.parse_args()
    if args.cmd == "check":
        return check_main(args.files, json_output=args.json)
    if args.cmd == "repl":
        from quivers.cli.repl import main as repl_main

        return repl_main(args)
    if args.cmd == "lsp":
        from quivers.cli.lsp import main as lsp_main

        return lsp_main(args)
    if args.cmd == "kernel":
        from quivers.kernel.install import main as kernel_main

        return kernel_main(args)
    parser.print_help(sys.stderr)
    return 2


__all__ = ["main"]
