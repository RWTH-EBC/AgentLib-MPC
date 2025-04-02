"""Holds the class for full featured MPCs."""
from typing import Dict, Union

import numpy as np
import pandas as pd
from agentlib.core import AgentVariable

from agentlib_mpc.data_structures import mpc_datamodels
from pydantic import Field, field_validator, FieldValidationInfo
from rapidfuzz import process, fuzz

from agentlib_mpc.modules.mpc import BaseMPCConfig, BaseMPC


class MPCConfig(BaseMPCConfig):
    """
    Pydantic data model for MPC configuration parser
    """

    r_del_u: dict[str, float] = Field(
        default={},
        description="Weights that are applied to the change in control variables.",
    )
    enable_deactivate_mpc: bool = Field(
        default=False,
        description="If true, the MPC module uses an AgentVariable `active` which"
        "other modules may change to disable the MPC operation "
        "temporarily",
    )
    active: AgentVariable = Field(
        default=AgentVariable(
            name="active",
            description="MPC is active",
            type="bool",
            value=True,
            shared=False,
        ),
        description="Variable used to activate or deactivate the MPC operation",
    )
    control_values_when_deactivated: Dict[str, Union[float, bool, int]] = Field(
        default={},
        description="When the MPC is deactivated, send the controls with these values"
        "specified as `{control_name: control_value}`."
        "In case of variables to send which are not listed as model variables,"
        "a plain AgentVariable is send. This may be used to deactivate "
        "supervisory control in a simulation / real PLC.",
    )
    outputs_to_send_when_activated: Dict[str, Union[float, bool, int]] = Field(
        default={},
        description="When the MPC is activated, send the controls with these values"
        "specified as `{control_name: control_value}`."
        "In case of variables to send which are not listed as model variables,"
        "a plain AgentVariable is send. This may be used to deactivate "
        "supervisory control in a simulation / real PLC.",
    )

    @field_validator("r_del_u")
    def check_r_del_u_in_controls(
        cls, r_del_u: dict[str, float], info: FieldValidationInfo
    ):
        """Ensures r_del_u is only set for control variables."""
        controls = {ctrl.name for ctrl in info.data["controls"]}
        for name in r_del_u:
            if name in controls:
                # everything is fine
                continue

            # raise error
            matches = process.extract(query=name, choices=controls, scorer=fuzz.WRatio)
            matches = [m[0] for m in matches]
            raise ValueError(
                f"Tried to specify control change weight for {name}. However, "
                f"{name} is not in the set of control variables. Did you mean one "
                f"of these? {', '.join(matches)}"
            )
        return r_del_u


class MPC(BaseMPC):
    """
    A model predictive controller.
    More info to follow.
    """

    config: MPCConfig

    def _init_optimization(self):
        super()._init_optimization()
        self._lags_dict_seconds = self.optimization_backend.get_lags_per_variable()

        history = {}
        # create a dict to keep track of all values for lagged variables timestamped
        for v in self._lags_dict_seconds:
            var = self.get(v)
            history[v] = {}
            # store scalar values as initial if they exist
            if isinstance(var.value, (float, int)):
                timestamp = var.timestamp or self.env.time
                value = var.value
            elif var.value is None:
                self.logger.info(
                    "Initializing history for variable %s, but no value was available."
                    " Interpolating between bounds or setting to zero."
                )
                timestamp = self.env.time
                value = var.value or np.nan_to_num(
                    (var.ub + var.lb) / 2, posinf=1000, neginf=1000
                )
            else:
                # in this case it should probably be a series, which we can take as is
                continue
            history[v][timestamp] = value
        self.history: dict[str, dict[float, float]] = history
        self.register_callbacks_for_lagged_variables()

    def check_if_mpc_step_should_be_skipped(self):
        """Checks if mpc steps should be skipped based on external activation flag."""
        if not self.config.enable_deactivate_mpc:
            return False
        active = self.get("active")
        if active.value:
            for output, value in self.config.outputs_to_send_when_activated.items():
                if output in self.config.outputs:
                    self.set(output, value)
                else:
                    self.agent.data_broker.send_variable(
                        AgentVariable(
                            name=output,
                            value=value,
                            source=self.source,
                            shared=True
                        )
                    )
            return False
        source = str(active.source)
        if source == "None_None":
            source = "unknown (not specified in config)"
        if not self.config.control_values_when_deactivated:
            self.logger.info("MPC was deactivated by source %s", source)
            return True
        self.logger.info(
            "MPC was deactivated by source %s, sending control_values_when_deactivated %s",
            source,
            self.config.control_values_when_deactivated,
        )
        for control_name, value in self.config.control_values_when_deactivated.items():
            if control_name in self.config.controls:
                self.set(control_name, value)
            else:
                self.agent.data_broker.send_variable(
                    AgentVariable(
                        name=control_name,
                        value=value,
                        source=self.source,
                        shared=True
                    )
                )
        return True

    def do_step(self):
        if self.check_if_mpc_step_should_be_skipped():
            return
        super().do_step()
        self._remove_old_values_from_history()

    def _remove_old_values_from_history(self):
        """Clears the history of all entries that are older than current time minus
        horizon length."""
        # iterate over all variables which save lag
        for var_name, lag_in_seconds in self._lags_dict_seconds.items():
            var_history = self.history[var_name]

            # iterate over all saved values and delete them, if they are too old
            for timestamp in list(var_history):
                if timestamp < (self.env.time - lag_in_seconds):
                    var_history.pop(timestamp)

    def _callback_hist_vars(self, variable: AgentVariable, name: str):
        """Adds received measured inputs to the past trajectory."""
        # if variables are intentionally sent as series, we don't need to store them
        # ourselves
        # only store scalar values
        if isinstance(variable.value, (float, int)):
            self.history[name][variable.timestamp] = variable.value

    def register_callbacks_for_lagged_variables(self):
        """Registers callbacks which listen to the variables which have to be saved as
        time series. These callbacks save the values in the history for use in the
        optimization."""

        for lagged_input in self._lags_dict_seconds:
            var = self.get(lagged_input)
            self.agent.data_broker.register_callback(
                alias=var.alias,
                source=var.source,
                callback=self._callback_hist_vars,
                name=var.name,
            )

    def _after_config_update(self):
        self._internal_variables = self._create_internal_variables()
        super()._after_config_update()

    def _setup_var_ref(self) -> mpc_datamodels.VariableReferenceT:
        return mpc_datamodels.FullVariableReference.from_config(self.config)

    def collect_variables_for_optimization(
        self, var_ref: mpc_datamodels.VariableReference = None
    ) -> dict[str, AgentVariable]:
        """Gets all variables noted in the var ref and puts them in a flat
        dictionary."""
        if var_ref is None:
            var_ref = self.var_ref

        # config variables
        variables = {v: self.get(v) for v in var_ref.all_variables()}

        # history variables
        for hist_var in self._lags_dict_seconds:
            past_values = self.history[hist_var]
            if not past_values:
                # if the history of a variable is empty, fallback to the scalar value
                continue

            # create copy to not mess up scalar value of original variable in case
            # fallback is needed
            updated_var = variables[hist_var].copy(
                update={"value": pd.Series(past_values)}
            )
            variables[hist_var] = updated_var

        return {**variables, **self._internal_variables}

        # class AgVarDropin:
        #     ub: float
        #     lb: float
        #     value: Union[float, list, pd.Series]
        #     interpolation_method: InterpolationMethod

    def _create_internal_variables(self) -> dict[str, AgentVariable]:
        """Creates a reference of all internal variables that are used for the MPC,
        but not shared as AgentVariables.

        Currently, this includes:
           - Weights for control change (r_del_u)
        """
        r_del_u: dict[str, mpc_datamodels.MPCVariable] = {}
        for control in self.config.controls:
            r_del_u_name = mpc_datamodels.r_del_u_convention(control.name)
            var = mpc_datamodels.MPCVariable(name=r_del_u_name)
            r_del_u[r_del_u_name] = var
            if control.name in self.config.r_del_u:
                var.value = self.config.r_del_u[control.name]
            else:
                var.value = 0

        return r_del_u
