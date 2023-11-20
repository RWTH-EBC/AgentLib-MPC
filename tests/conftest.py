from pathlib import Path
import sys

from _pytest.mark import Mark

sys.path.append(str((Path(__file__)).parent))

import pytest
from agentlib.utils import custom_injection

# order pytests to do tests marked by slow last
empty_mark = Mark("", [], {})


def by_slow_marker(item):
    return item.get_closest_marker("slow", default=empty_mark)


def pytest_collection_modifyitems(items):
    items.sort(key=by_slow_marker, reverse=False)


@pytest.fixture
def model_type():
    file = Path(Path(__file__).parent, "fixtures//casadi_test_model.py")
    return {"file": file, "class_name": "MyCasadiModel"}


@pytest.fixture
def example_casadi_model(model_type):
    custom_cls = custom_injection(config=model_type)
    return custom_cls()
