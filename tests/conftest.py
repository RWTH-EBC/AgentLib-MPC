from pathlib import Path
import pytest
from agentlib.utils import custom_injection


@pytest.fixture
def model_type():
    file = Path(Path(__file__).parent, "fixtures//casadi_test_model.py")
    return {"file": file, "class_name": "MyCasadiModel"}


@pytest.fixture
def example_casadi_model(model_type):
    custom_cls = custom_injection(config=model_type)
    return custom_cls()
