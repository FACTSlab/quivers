"""The external tools the transpile tests run emitted programs through.

A test that needs a language's own compiler or interpreter names it
here, and a missing tool is a failure that says what to install rather
than a skipped test: the fast CI job installs every tool named below,
and a local checkout is expected to match it.
"""

from __future__ import annotations

import shutil

#: How to install each tool, by the first name it is looked up under.
_INSTALL: dict[str, str] = {
    "scheme": "Chez Scheme (`brew install chezscheme` or `apt-get install chezscheme`)",
    "julia": "Julia (`brew install julia` or the julia-actions/setup-julia action)",
    "webppl": "WebPPL (`npm install -g webppl`)",
    "cc": "a C compiler (Xcode command line tools or `apt-get install build-essential`)",
    "jags": "JAGS (`brew install jags` or `apt-get install jags`)",
    "stanc": "stanc (the stanc3 release binary on PATH)",
    "node": "Node.js (`brew install node` or the actions/setup-node action)",
}


def require_tool(*names: str) -> str:
    """The executable of a tool, looked up under the names it goes by.

    Parameters
    ----------
    *names : str
        The command names to try, in order.

    Returns
    -------
    str
        The path of the first name found on ``PATH``.

    Raises
    ------
    RuntimeError
        If none of the names is on ``PATH``, naming what to install.
    """
    for name in names:
        executable = shutil.which(name)
        if executable is not None:
            return executable
    hint = _INSTALL.get(names[0], names[0])
    raise RuntimeError(
        f"the transpile tests need {hint}; none of {', '.join(names)} is on PATH"
    )


def require_scheme() -> str:
    """The Chez Scheme executable.

    Returns
    -------
    str
        The path of the interpreter, under whichever name it is installed.

    Raises
    ------
    RuntimeError
        If no Chez Scheme interpreter is on ``PATH``.
    """
    return require_tool("scheme", "chez", "petite", "chezscheme")


__all__ = ["require_scheme", "require_tool"]
