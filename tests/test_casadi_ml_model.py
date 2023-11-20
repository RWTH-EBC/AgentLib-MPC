"""Module for tests of casadi_ml_model.py"""
import pytest

from agentlib_mpc.models.casadi_model import CasadiInput, CasadiOutput
from agentlib_mpc.models.casadi_ml_model import CasadiMLModelConfig, CasadiMLModel


class CasadiMLTestConfig(CasadiMLModelConfig):
    inputs: list[CasadiInput] = [CasadiInput(name="x", value=1)]
    outputs: list[CasadiOutput] = [CasadiOutput(name="y")]


class CasadiMLTestModel(CasadiMLModel):
    config: CasadiMLTestConfig

    def setup_system(self):
        pass


# todo run this when we have non-recursive ml models
@pytest.mark.skip
def test_casadi_ml_model(example_serialized_ml_model):
    test_model = CasadiMLTestModel(ml_model_sources=[example_serialized_ml_model])
    test_model.do_step(t_start=0)
