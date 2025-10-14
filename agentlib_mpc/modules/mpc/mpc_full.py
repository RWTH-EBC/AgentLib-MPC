"""Holds the class for full featured MPCs."""

from typing import Dict, Union, Optional

import agentlib
import numpy as np
import pandas as pd
from agentlib.core import AgentVariable
from agentlib.core.errors import OptionalDependencyError

from agentlib_mpc.data_structures import mpc_datamodels
from pydantic import Field, field_validator, FieldValidationInfo
from rapidfuzz import process, fuzz

from agentlib_mpc.modules.mpc.mpc import BaseMPCConfig, BaseMPC
from agentlib_mpc.modules.mpc.skippable_mixin import (
    SkippableMixinConfig,
    SkippableMixin,
)


class MPCConfig(BaseMPCConfig, SkippableMixinConfig):
    """
    Pydantic data model for MPC configuration parser
    """

    r_del_u: dict[str, float] = Field(
        default={},
        description="Weights that are applied to the change in control variables.",
    )

    enable_state_fallback: bool = Field(
        default=False,
        description="Enable fallback to predicted states when measurements are too old",
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


class MPC(BaseMPC, SkippableMixin):
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

        # Initialize predicted states storage for fallback functionality
        if self.config.enable_state_fallback:
            self.predicted_states: dict[str, dict[float, float]] = {}
            for state_name in self.var_ref.states:
                self.predicted_states[state_name] = {}

    def do_step(self):
        if self.check_if_should_be_skipped():
            return
        super().do_step()
        self._remove_old_values_from_history()
        if self.config.enable_state_fallback:
            self._remove_old_predicted_states()

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

    def _remove_old_predicted_states(self):
        """Clears predicted states that are older than current time minus horizon length."""
        max_age = self.config.prediction_horizon * self.config.time_step
        cutoff_time = self.env.time - max_age

        for state_name in self.predicted_states:
            state_predictions = self.predicted_states[state_name]
            for timestamp in list(state_predictions):
                if timestamp < cutoff_time:
                    state_predictions.pop(timestamp)

    def _store_predicted_states(self, solution: mpc_datamodels.Results):
        """Store predicted state trajectories for fallback functionality."""
        if not self.config.enable_state_fallback:
            return

        df = solution.df
        current_time = self.env.time

        for state_name in self.var_ref.states:
            state_trajectory = df["variable"][state_name]
            # Convert index (which starts at 0) to actual timestamps
            for idx, value in state_trajectory.items():
                prediction_time = current_time + idx
                self.predicted_states[state_name][prediction_time] = value

    def _get_predicted_state_value(
        self, state_name: str, target_time: float
    ) -> Optional[float]:
        """Get predicted state value at target_time using linear interpolation if needed."""
        state_predictions = self.predicted_states[state_name]

        if not state_predictions:
            return None

        timestamps = sorted(state_predictions.keys())

        # Exact match
        if target_time in state_predictions:
            return state_predictions[target_time]

        # Find surrounding timestamps for interpolation
        before = None
        after = None

        for timestamp in timestamps:
            if timestamp <= target_time:
                before = timestamp
            elif timestamp > target_time and after is None:
                after = timestamp
                break

        # Linear interpolation between before and after
        if before is not None and after is not None:
            t1, t2 = before, after
            v1, v2 = state_predictions[t1], state_predictions[t2]
            # Linear interpolation
            alpha = (target_time - t1) / (t2 - t1)
            return v1 + alpha * (v2 - v1)

        # Use nearest value if we can't interpolate
        if before is not None:
            return state_predictions[before]
        elif after is not None:
            return state_predictions[after]

        return None

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

        # Apply state fallback if enabled
        if self.config.enable_state_fallback:
            self._apply_state_fallback(variables)

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

    def _apply_state_fallback(self, variables: dict[str, AgentVariable]):
        """Apply fallback logic for state variables with old measurements."""
        current_time = self.env.time
        fallback_threshold = self.config.time_step + 0.1

        for state_name in self.var_ref.states:
            if state_name not in variables:
                continue

            state_var = variables[state_name]
            measurement_time = state_var.timestamp

            # Check if measurement is too old
            if measurement_time is None:
                # probably default value, we don't have fallback anyway
                measurement_age = 0
            else:
                measurement_age = current_time - measurement_time

            if measurement_age > fallback_threshold:
                # Try to use predicted value
                predicted_value = self._get_predicted_state_value(
                    state_name, current_time
                )

                if predicted_value is not None:
                    self.logger.info(
                        f"Using predicted fallback value for state '{state_name}'. "
                        f"Measurement age: {measurement_age:.2f}s > threshold: {fallback_threshold:.2f}s"
                    )
                    # Create updated variable with predicted value
                    variables[state_name] = state_var.copy(
                        update={"value": predicted_value, "timestamp": current_time}
                    )
                else:
                    self.logger.info(
                        f"State '{state_name}' measurement is old ({measurement_age:.2f}s) "
                        f"but no predicted value available for fallback"
                    )

    def do_step(self):
        """
        Performs an MPC step.
        """
        if self.check_if_should_be_skipped():
            return

        # Call parent do_step which handles the optimization
        super().do_step()

        # Clean up old data
        self._remove_old_values_from_history()
        if self.config.enable_state_fallback:
            self._remove_old_predicted_states()

    def set_actuation(self, solution: mpc_datamodels.Results):
        """Takes the solution from optimization backend and sends the first
        step to AgentVariables."""
        # Store predicted states for fallback before setting actuation
        if self.config.enable_state_fallback:
            self._store_predicted_states(solution)

        # Call parent method to handle actuation
        super().set_actuation(solution)

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

    @classmethod
    def visualize_results(
        cls,
        results_data: pd.DataFrame,
        module_id: str,
        agent_id: str,
        convert_to: str = "hours",
        step: bool = False,
        use_datetime: bool = False,
        max_predictions: int = 1000,
    ):
        """
        Create visualization components for MPC results to be displayed in MAS dashboard.

        Args:
            results_data: DataFrame with MPC results data
            module_id: ID of the MPC module
            agent_id: ID of the agent containing this module
            convert_to: Time unit for plotting ("seconds", "minutes", "hours", "days")
            step: Whether to use step plots
            use_datetime: Whether to interpret timestamps as datetime
            max_predictions: Maximum number of predictions to show

        Returns:
            Dash HTML Div containing the visualization components
        """
        try:
            from dash import html, dcc
            import plotly.graph_objects as go
            from agentlib_mpc.utils.plotting.interactive import get_port
            from agentlib_mpc.utils.plotting.mpc import interpolate_colors
            from agentlib_mpc.utils.plotting.basic import EBCColors
            from agentlib_mpc.utils import TIME_CONVERSION
        except ImportError as e:
            raise OptionalDependencyError(
                dependency_name="interactive",
                dependency_install="plotly, dash",
                used_object="MPC visualization",
            ) from e

        if results_data is None or results_data.empty:
            return html.Div(
                [
                    html.H4(f"MPC Results - {module_id}"),
                    html.P("No data available for visualization."),
                ]
            )

        # Import the plotting functions from mpc_dashboard
        from agentlib_mpc.utils.plotting.mpc_dashboard import (
            make_components,
            reduce_triple_index,
            detect_index_type,
        )

        try:
            # Process the data similar to mpc_dashboard
            data = results_data.copy()

            # Reduce triple index to double index if needed
            if isinstance(data.index, pd.MultiIndex) and len(data.index.levels) > 2:
                data = reduce_triple_index(data)

            # Detect index type
            is_multi_index, detected_use_datetime = detect_index_type(data)

            # Normalize time if needed
            if is_multi_index and not detected_use_datetime:
                first_time = data.index.levels[0][0]
                data.index = data.index.set_levels(
                    data.index.levels[0] - first_time, level=0
                )

            # Check for stats data (look for companion stats in the same results)
            stats = None
            # Note: stats would need to be passed separately or found in a predictable way

            # Create the dashboard components using existing MPC dashboard logic
            components_div = make_components(
                data=data,
                convert_to=convert_to,
                stats=stats,
                use_datetime=detected_use_datetime or use_datetime,
                step=step,
            )

            # Wrap with a header indicating this is MPC data
            return html.Div(
                [
                    html.H4(f"MPC Results - Agent: {agent_id}, Module: {module_id}"),
                    components_div,
                ]
            )

        except Exception as e:
            # Return error information for debugging
            return html.Div(
                [
                    html.H4(f"MPC Visualization Error - {module_id}"),
                    html.P(f"Error processing MPC results: {str(e)}"),
                    html.Details(
                        [
                            html.Summary("Data Info"),
                            html.P(
                                f"Data shape: {results_data.shape if results_data is not None else 'None'}"
                            ),
                            html.P(f"Data type: {type(results_data)}"),
                            html.P(
                                f"Index type: {type(results_data.index) if results_data is not None else 'None'}"
                            ),
                            html.P(
                                f"Columns: {list(results_data.columns) if results_data is not None else 'None'}"
                            ),
                        ]
                    ),
                ]
            )
