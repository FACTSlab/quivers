"""Running generated WebPPL programs through the `webppl` compiler.

The WebPPL target's runtime is written in WebPPL's own functional subset,
so a generated program is exercised by the compiler that defines the
language rather than by Node alone: WebPPL rejects assignment, loops,
and `try`, and calls a function held in an object field outside its
transform, none of which Node would notice.
"""

from __future__ import annotations

import pathlib
import subprocess

from tests.transpile._tools import require_tool

#: The `webppl` command line.
WEBPPL_EXECUTABLE = require_tool("webppl")

#: WebPPL compiles a program through a recursive transform whose depth
#: follows the program's size, and the grafted runtime is large enough
#: to overflow Node's default stack; the interpreter itself is plain
#: Node, so the limit is lifted without changing any semantics.
_NODE_STACK_SIZE = "40000"


def run_webppl(
    script: pathlib.Path, *, check: bool = True
) -> subprocess.CompletedProcess[str]:
    """Compile and run one WebPPL program.

    Parameters
    ----------
    script : pathlib.Path
        The program.
    check : bool
        Whether a nonzero exit raises.

    Returns
    -------
    subprocess.CompletedProcess[str]
        The completed run, with its standard output and error.

    Raises
    ------
    subprocess.CalledProcessError
        If ``check`` is set and the program exits nonzero.
    """
    completed = subprocess.run(
        [
            "node",
            f"--stack-size={_NODE_STACK_SIZE}",
            WEBPPL_EXECUTABLE,
            str(script),
        ],
        check=check,
        capture_output=True,
        text=True,
        timeout=600,
    )
    # The command line prints the program's own final value after
    # whatever the program displayed; a program ending in a `display`
    # call has none, so that trailing line is not part of its output.
    stdout = completed.stdout
    if stdout.endswith("undefined\n"):
        stdout = stdout[: -len("undefined\n")]
    return subprocess.CompletedProcess(
        completed.args, completed.returncode, stdout, completed.stderr
    )
