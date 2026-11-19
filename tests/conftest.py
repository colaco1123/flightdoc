import os

import pytest

DATA = os.path.join(os.path.dirname(__file__), "data")


def data(*parts):
    path = os.path.join(DATA, *parts)
    if not os.path.exists(path):
        pytest.skip(f"fixture {path} missing")
    return path


@pytest.fixture(scope="session")
def diagnose_fixture():
    """Diagnose a fixture once and share the result across tests."""
    from flightdoc import analyze
    cache = {}

    def run(*parts):
        key = os.path.join(*parts)
        if key not in cache:
            cache[key] = analyze(data(*parts))
        return cache[key]
    return run
