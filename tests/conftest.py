"""Pytest configuration: the ``slow`` marker + ``--runslow`` flag.

Recovery and benchmark tests are gated behind ``pytest.mark.slow`` so
the everyday test run stays fast.  Run them explicitly via either::

    pytest -m slow tests/
    pytest --runslow tests/

The marker is declared in ``pyproject.toml`` under
``[tool.pytest.ini_options]``; the default ``addopts`` deselects
slow tests.  ``--runslow`` overrides the deselection by clearing the
marker filter for the current invocation.
"""

from __future__ import annotations

import os

# Activate the local-grammar override before any quivers import so the
# whole test run parses against the in-tree grammar at `grammars/qvr/`
# rather than whatever `panproto-grammars-all` ships for `qvr`. The
# override compiles the committed `parser.c` on demand, so a working C
# compiler is the only requirement.
os.environ.setdefault("QVR_USE_LOCAL_GRAMMAR", "1")


def pytest_addoption(parser):
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run tests marked `slow` in addition to the default suite",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # Strip the `not slow` deselector so slow tests are collected.
        config.option.markexpr = ""
