import pydantic
from agentlib.core import AgentVariable, AgentVariables
from agentlib.core.errors import ConfigurationError
from agentlib.modules.simulation.simulator import SimulatorConfig, Simulator
from pydantic_core.core_schema import FieldValidationInfo

from agentlib_mpc.models.casadi_ml_model import CasadiMLModel
from agentlib_mpc.models.serialized_ml_model import SerializedMLModel
from pydantic import field_validator


class MLModelSimulatorConfig(SimulatorConfig):
    serialized_ml_models: AgentVariables = []

    @field_validator("t_sample")
    @classmethod
    def check_t_sample(cls, t_sample, info: FieldValidationInfo):
        """Check if t_sample is smaller than stop-start time"""
        if not "model" in info.data.keys():
            raise ConfigurationError(
                f"There is an Error in the model. Most likely it is raised in the "
                f"'check_model' method of the SimulatorConfig class of the core Agentlib. "
                f"Please check your model for any mistakes."
            )
        dt = info.data["model"].dt
        if t_sample % dt != 0:
            raise ConfigurationError(
                f"Sampling Time of Simulator must be multiple of MLModel time step. Current"
                f" MLModel time step is {dt} and chosen sampling time is {t_sample}."
            )
        return t_sample


class MLModelSimulator(Simulator):
    config: MLModelSimulatorConfig
    model: CasadiMLModel

    def _callback_update_model_input(self, inp: AgentVariable, name: str):
        """Set given model input value to the model"""
        self.logger.debug("Updating model input %s=%s", inp.name, inp.value)
        self.model.set_with_timestamp(
            name=name, value=inp.value, timestamp=inp.timestamp
        )

    def register_callbacks(self):
        for ml_model_var in self.config.serialized_ml_models:
            self.agent.data_broker.register_callback(
                callback=self._update_ml_model_callback,
                alias=ml_model_var.alias,
                source=ml_model_var.source,
                name=ml_model_var.name,
            )

    def _update_ml_model_callback(self, variable: AgentVariable, name: str):
        """Updates the MLModels of the underlying model."""
        try:
            ml_model = SerializedMLModel.load_serialized_model_from_string(
                variable.value
            )
        except pydantic.ValidationError:
            self.logger.error(
                f"Callback 'update_ml_model' got activated for variable {name} , but the "
                f"received AgentVariable did not contain a valid MLModel. Got "
                f"{variable.value} of type '{type(variable.value)} instead."
            )
            return
        try:
            self.model.update_ml_models(ml_model, time=self.env.now)
            self.logger.info(f"Successfully updated MLModel for variable {name}.")
        except ConfigurationError as e:
            self.logger.error(
                f"Tried to update the MLModels, but new MLModels do not have matching 'dt'. "
                f"Error message from model: '{e}'."
            )
