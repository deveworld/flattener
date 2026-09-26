"""Keep gradient and training tests opt-in, including their setup fixtures."""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-training",
        action="store_true",
        default=False,
        help="Explicitly allow tests marked training (may perform CPU training).",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-training"):
        return
    skip_training = pytest.mark.skip(reason="training requires explicit --run-training")
    for item in items:
        if item.get_closest_marker("training") is not None:
            item.add_marker(skip_training)
