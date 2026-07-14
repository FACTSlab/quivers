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
