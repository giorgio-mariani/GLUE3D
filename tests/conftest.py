import pytest

JUDGE_MARKER = "test_judge"


def pytest_addoption(parser):
    parser.addoption(f"--{JUDGE_MARKER}", action="store_true", default=False, help="run LLM-as-Judge tests")


def pytest_configure(config):
    config.addinivalue_line("markers", f"{JUDGE_MARKER}: mark test to execute LLM-as-Judge tests")


def pytest_collection_modifyitems(config, items):
    if config.getoption(f"--{JUDGE_MARKER}"):
        # --runslow given in cli: do not skip slow tests
        return

    skip_slow = pytest.mark.skip(reason=f"need --{JUDGE_MARKER} option to run")
    for item in items:
        if JUDGE_MARKER in item.keywords:
            item.add_marker(skip_slow)
