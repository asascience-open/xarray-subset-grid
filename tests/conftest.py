# conftest: some configuration for the tests

import pytest


def pytest_addoption(parser):
    # put a @pytest.mark.online decorator on tests that require net access
    parser.addoption(
        "--online",
        action="store_true",  # what is this?
        default=False,
        help="run tests that access AWS resources - have to be online",
    )


def pytest_configure(config):
    # register an additional marker
    config.addinivalue_line(
        "markers",
        "online: mark test to run only when online (using AWS resources)"
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--online"):
        # --online not given in cli: skip tests that require online access
        skip_online = pytest.mark.skip(reason="need --online option to run")
        for item in items:
            if "online" in item.keywords:
                item.add_marker(skip_online)


# # example from docs
# def pytest_runtest_setup(item):
#     envnames = [mark.args[0] for mark in item.iter_markers(name="env")]
#     if envnames:
#         if item.config.getoption("-E") not in envnames:
#             pytest.skip(f"test requires env in {envnames!r}")
