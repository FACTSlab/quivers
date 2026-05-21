"""Pytest configuration: register the ``slow`` marker used by the
gallery sweep so ``pytest -m 'not slow'`` (the default) skips
long-running examples while ``pytest -m slow`` runs the full set."""


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: long-running gallery example (run with ``pytest -m slow``)",
    )
