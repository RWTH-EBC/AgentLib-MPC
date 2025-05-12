import os
import logging
from typing import Dict, Optional, List, Literal
from dataclasses import dataclass

import casadi as ca
import numpy as np
import pydantic
from agentlib.core.errors import OptionalDependencyError
from pydantic_core.core_schema import FieldValidationInfo

from agentlib_mpc.data_structures import mpc_datamodels
from agentlib_mpc.data_structures.mpc_datamodels import (
    MPCVariable,
    MINLPVariableReference,
    stats_path,
    cia_relaxed_results_path,
)
from agentlib_mpc.optimization_backends.casadi_.core.casadi_backend import (
    CasadiBackendConfig,
)
from agentlib_mpc.optimization_backends.casadi_.core.discretization import Results
from agentlib_mpc.optimization_backends.casadi_.minlp import CasADiMINLPBackend

try:
    import pycombina
except ImportError:
    raise OptionalDependencyError(
        used_object="Pycombina",
        dependency_install=".\ after cloning pycombina. Instructions: "
        "https://pycombina.readthedocs.io/en/latest/install.html#",
    )

logger = logging.getLogger(__name__)


@dataclass
class ModeState:
    """Tracks the state of each binary mode"""

    active: bool
    last_switch_time: float
    time_in_current_state: float


class PreventModeConstraintConfig(pydantic.BaseModel):
    """Configuration for preventing a mode unless its relaxed value meets a threshold."""

    mode_name: str = pydantic.Field(
        description="Name of the binary control variable to potentially prevent."
    )
    threshold: float = pydantic.Field(
        description="Relaxed value threshold to compare against."
    )
    condition: Literal["<=", ">="] = pydantic.Field(
        description="Condition to trigger prevention (e.g., '<=' means prevent if relaxed value is below threshold)."
    )
    interval_index: int = pydantic.Field(
        default=0,
        description="Time interval index (0-based) to check the relaxed value in.",
    )


class CasadiCIABackendConfig(CasadiBackendConfig):
    min_up_times: Optional[Dict[str, float]] = pydantic.Field(
        default=None,
        description="Minimum time (in seconds) a binary control must remain ON after switching ON. "
        "Provided as a dictionary mapping control names to times.",
    )
    min_down_times: Optional[Dict[str, float]] = pydantic.Field(
        default=None,
        description="Minimum time (in seconds) a binary control must remain OFF after switching OFF. "
        "Provided as a dictionary mapping control names to times.",
    )
    # max_switches: Optional[Dict[str, int]] = pydantic.Field( # Example if needed later
    #     default=None,
    #     description="Maximum number of switches allowed per control over the horizon."
    # )
    initial_mode_states: Optional[Dict[str, bool]] = pydantic.Field(
        default=None,
        description="Initial state (True=ON, False=OFF) for binary controls before the first solve. "
        "Defaults to OFF if not specified for a control.",
    )
    prevent_mode_constraints: Optional[
        List[PreventModeConstraintConfig]
    ] = pydantic.Field(
        default=None,
        description="List of constraints to prevent certain modes unless their relaxed value meets a threshold.",
    )

    @pydantic.field_validator("overwrite_result_file")
    @classmethod
    def check_overwrite(cls, overwrite_result_file: bool, info: FieldValidationInfo):
        """Checks, whether the overwrite results sttings are valid, and deletes
        existing result files if applicable."""
        res_file = info.data.get("results_file")
        relaxed_res_file = cia_relaxed_results_path(res_file)
        if res_file and info.data["save_results"]:
            if overwrite_result_file:
                try:
                    os.remove(res_file)
                    os.remove(mpc_datamodels.stats_path(res_file))
                    os.remove(relaxed_res_file)
                    os.remove(mpc_datamodels.stats_path(relaxed_res_file))
                except FileNotFoundError:
                    pass
            else:
                if os.path.isfile(info.data["results_file"]):
                    raise FileExistsError(
                        f"Results file {res_file} already exists and will not be "
                        f"overwritten automatically. Set 'overwrite_result_file' to "
                        f"True to enable automatic overwrite it."
                    )
        return overwrite_result_file


class CasADiCIABackend(CasADiMINLPBackend):
    """
    Class doing optimization with the CIA decomposition algorithm.
    """

    # system_type = CasadiMINLPSystem
    # discretization_types = {DiscretizationMethod.collocation: DirectCollocation}
    # system: CasadiMINLPSystem
    var_ref: MINLPVariableReference
    config_type = CasadiCIABackendConfig
    config: CasadiCIABackendConfig  # Add type hint for config

    def __init__(self, config: dict):
        super().__init__(config)
        self._created_rel_file: bool = False  # flag if we checked the rel file location
        # State tracking attributes
        self.mode_states: Dict[str, ModeState] = {}
        self.last_solve_time: Optional[float] = None
        self._mode_name_to_idx: Dict[str, int] = {}
        self._state_initialized: bool = False

    def _initialize_state_tracking(self):
        """Initializes the mode state tracking based on config and variable references."""

        self._mode_name_to_idx = {
            name: i for i, name in enumerate(self.var_ref.binary_controls)
        }
        initial_states = self.config.initial_mode_states or {}

        self.mode_states = {
            name: ModeState(
                active=initial_states.get(name, False),
                last_switch_time=0.0,  # Assuming simulation starts at t=0
                time_in_current_state=float("inf")
                if initial_states.get(name, False)
                else 0.0,  # Assume it's been in initial state "forever"
            )
            for name in self.var_ref.binary_controls
        }
        self._state_initialized = True
        logger.debug(f"Initialized mode states: {self.mode_states}")

    def update_mode_states(
        self, current_time: float, binary_solution_array: np.ndarray
    ):
        """Updates mode states based on the latest binary solution."""
        if self.last_solve_time is None:
            # First solve after initialization, just record time
            self.last_solve_time = current_time
            # Correct initial time_in_current_state if it wasn't infinite
            for state in self.mode_states.values():
                if not state.active:
                    state.time_in_current_state = current_time - state.last_switch_time
            logger.debug(
                f"First solve at {current_time}, initial states updated: {self.mode_states}"
            )
            return

        dt = current_time - self.last_solve_time
        if dt < 0:
            # raise hard for debugging
            raise RuntimeError
        elif dt == 0:
            raise RuntimeError

        for mode_idx, mode_name in enumerate(self.var_ref.binary_controls):
            mode_state = self.mode_states[mode_name]
            # Use the first time step's solution value
            new_is_active = binary_solution_array[mode_idx, 0] > 0.5

            if new_is_active == mode_state.active:
                mode_state.time_in_current_state += dt
            else:
                # State changed
                mode_state.time_in_current_state = (
                    dt  # Time in new state starts with this step
                )
                mode_state.last_switch_time = current_time
                mode_state.active = new_is_active
                logger.debug(
                    f"Mode '{mode_name}' switched to {new_is_active} at {current_time}"
                )

        self.last_solve_time = current_time
        logger.debug(f"Updated mode states at {current_time}: {self.mode_states}")

    def _get_current_mode_constraints(
        self, current_time: float
    ) -> tuple[Optional[str], Optional[float]]:
        """
        Determines if any mode needs to be forced ON due to min_up_time constraints.

        Returns:
            tuple[Optional[str], Optional[float]]: (mode_name, remaining_required_up_time)
                                                  or (None, None) if no forcing needed.
        """
        if not self._state_initialized or self.config.min_up_times is None:
            return None, None

        forced_mode_name = None
        max_remaining_time = 0.0

        for mode_name, mode_state in self.mode_states.items():
            min_up_time = self.config.min_up_times.get(mode_name)
            if mode_state.active and min_up_time is not None and min_up_time > 0:
                # Use time_in_current_state which accurately reflects time since last switch ON
                time_active = mode_state.time_in_current_state
                if time_active < min_up_time:
                    remaining_time = min_up_time - time_active
                    # We need to force the one with the *longest* remaining time
                    # as pycombina likely can only handle one forced mode via set_valid_controls
                    if remaining_time > max_remaining_time:
                        max_remaining_time = remaining_time
                        forced_mode_name = mode_name
                        logger.debug(
                            f"Mode '{mode_name}' requires remaining uptime: {remaining_time:.2f}s"
                        )

        if forced_mode_name:
            logger.info(
                f"Forcing mode '{forced_mode_name}' ON for {max_remaining_time:.2f}s due to min_up_time."
            )
            return forced_mode_name, max_remaining_time
        else:
            return None, None

    def solve(self, now: float, current_vars: dict[str, MPCVariable]) -> Results:
        # collect and format inputs
        mpc_inputs = self._get_current_mpc_inputs(agent_variables=current_vars, now=now)

        # solve NLP with relaxed binaries
        relaxed_results = self.discretization.solve(mpc_inputs)

        relaxed_binary_array = self.make_binary_array(full_results=relaxed_results)
        # Pass current time to do_pycombina for state-based constraints
        binary_array = self.do_pycombina(b_rel=relaxed_binary_array, current_time=now)

        mpc_inputs_new = self.constrain_binary_inputs(
            mpc_inputs_old=mpc_inputs,
            binary_array=binary_array,
        )
        # solve NLP with fixed binaries
        full_results_final = self.discretization.solve(mpc_inputs_new)

        # Update mode states after the final solve
        self.update_mode_states(now, binary_array)

        self.save_rel_result_df(relaxed_results, now=now)
        self.save_result_df(full_results_final, now=now)

        return full_results_final

    def make_binary_array(self, full_results: Results):
        """
        get the binary control variables for input of pycombina and their control vector indexes
        """

        b_rel = [full_results[var] for var in self.var_ref.binary_controls]
        b_rel_np = np.vstack(b_rel)

        # clip binary values within tolerance
        tolerance = 1e-5
        bin_array = b_rel_np
        bin_array = np.where(
            (-tolerance < bin_array) & (bin_array < 0),
            0,
            np.where((1 < bin_array) & (bin_array < 1 + tolerance), 1, bin_array),
        )

        # add additional row to fulfill pycombinas Special Ordered Sets of
        # type 1 condition
        if len(bin_array) == 1:
            ones = np.full(bin_array.shape[1], 1, dtype=float)
            diff = ones - np.sum(bin_array, axis=0)
            diff[diff < 0] = 0
            bin_array = np.vstack([bin_array, diff])

        return bin_array

    def do_pycombina(self, b_rel: np.ndarray, current_time: float):
        """
        Solves the binary approximation problem using pycombina, applying
        configured constraints.

        Args:
            b_rel: The relaxed binary solution array (n_modes x n_steps).
            current_time: The current simulation time (used for state-based constraints).

        Returns:
            np.ndarray: The integer binary solution array (n_modes x n_steps).
        """
        # --- 1. Initialization and Input Validation ---
        if not self._state_initialized:
            self._initialize_state_tracking()
            # Re-check after initialization attempt
            if not self._state_initialized:
                logger.error(
                    "Failed to initialize state tracking. Cannot apply state-based constraints."
                )
                # Fallback: Solve without state-based constraints? Or raise error?
                # For now, proceed but log error. Constraints depending on state won't work.

        if np.min(b_rel) < 0:
            logger.warning(
                f"Clipping relaxed binary input elements < 0 (min value: {np.min(b_rel):.3f})."
            )
            b_rel = np.clip(b_rel, a_min=0, a_max=None)
        if np.max(b_rel) > 1:
            logger.warning(
                f"Clipping relaxed binary input elements > 1 (max value: {np.max(b_rel):.3f})."
            )
            b_rel = np.clip(b_rel, a_min=None, a_max=1)

        # --- 2. Setup PyCombina BinApprox ---
        # Get time grid for pycombina (needs endpoint)
        time_grid = self.discretization.grid(
            self.system.binary_controls
        )  # List of time points
        if not time_grid:
            raise ValueError("Could not obtain time grid for pycombina.")
        time_grid_np = np.array(time_grid)
        # Ensure grid has endpoint for pycombina intervals
        if len(time_grid_np) > 1:
            # Estimate last interval duration from previous one
            last_dt = time_grid_np[-1] - time_grid_np[-2]
        elif self.config.discretization_options.time_step:
            last_dt = self.config.discretization_options.time_step
        else:
            # Fallback if only one point and no time_step config
            logger.warning(
                "Cannot determine last time step duration for pycombina grid endpoint. Using fallback of 1.0"
            )
            last_dt = 1.0
        time_grid_with_endpoint = np.append(time_grid_np, time_grid_np[-1] + last_dt)

        binapprox = pycombina.BinApprox(
            t=time_grid_with_endpoint,
            b_rel=b_rel,
        )
        n_modes = binapprox.n_c
        n_intervals = binapprox.n_t

        # --- 3. Apply Configured Constraints to BinApprox ---

        # Min Up/Down Times (Per-Variable)
        if self.config.min_up_times:
            min_up_array = np.zeros(n_modes)
            for name, time_val in self.config.min_up_times.items():
                if name in self._mode_name_to_idx:
                    min_up_array[self._mode_name_to_idx[name]] = time_val
                else:
                    logger.warning(
                        f"Mode name '{name}' in min_up_times config not found in binary controls."
                    )
            logger.debug(f"Applying min_up_times: {min_up_array}")
            binapprox.set_min_up_times(min_up_array)

        if self.config.min_down_times:
            min_down_array = np.zeros(n_modes)
            for name, time_val in self.config.min_down_times.items():
                if name in self._mode_name_to_idx:
                    min_down_array[self._mode_name_to_idx[name]] = time_val
                else:
                    logger.warning(
                        f"Mode name '{name}' in min_down_times config not found in binary controls."
                    )
            logger.debug(f"Applying min_down_times: {min_down_array}")
            binapprox.set_min_down_times(min_down_array)

        # Prevent Mode Constraints (Applied FIRST)
        if self.config.prevent_mode_constraints:
            for pc in self.config.prevent_mode_constraints:
                if pc.mode_name not in self._mode_name_to_idx:
                    logger.warning(
                        f"Mode name '{pc.mode_name}' in prevent_mode_constraints not found."
                    )
                    continue
                mode_idx = self._mode_name_to_idx[pc.mode_name]

                if not (0 <= pc.interval_index < n_intervals):
                    logger.warning(
                        f"Invalid interval_index {pc.interval_index} for prevent_mode_constraint on '{pc.mode_name}'. Max index is {n_intervals-1}."
                    )
                    continue

                relaxed_value = binapprox.b_rel[mode_idx, pc.interval_index]
                prevent = False
                if pc.condition == "<=" and relaxed_value <= pc.threshold:
                    prevent = True
                elif pc.condition == ">=" and relaxed_value >= pc.threshold:
                    prevent = True  # Should this also prevent? Or only force ON? Let's assume prevent OFF.

                if prevent:
                    interval_start = binapprox.t[pc.interval_index]
                    interval_end = binapprox.t[pc.interval_index + 1]
                    dt_interval = (interval_start, interval_end)
                    valid_controls = np.ones(n_modes, dtype=int)
                    valid_controls[mode_idx] = 0  # Force OFF
                    logger.info(
                        f"Applying prevent constraint: Forcing mode '{pc.mode_name}' (idx {mode_idx}) "
                        f"to OFF in interval {pc.interval_index} ({dt_interval}) because "
                        f"relaxed value {relaxed_value:.3f} {pc.condition} {pc.threshold:.3f}."
                    )
                    binapprox.set_valid_controls_for_interval(
                        dt=dt_interval, b_bin_valid=valid_controls
                    )

        # State-Based Min Up-Time Forcing (Applied AFTER prevent constraints)
        if self._state_initialized:  # Only apply if state tracking is working
            forced_mode_name, remaining_time = self._get_current_mode_constraints(
                current_time
            )
            if forced_mode_name is not None and remaining_time > 0:
                forced_mode_idx = self._mode_name_to_idx[forced_mode_name]
                force_until = current_time + remaining_time
                valid_controls = np.zeros(n_modes, dtype=int)  # Force others OFF
                valid_controls[forced_mode_idx] = 1  # Force this one ON
                logger.info(
                    f"Applying state constraint: Forcing mode '{forced_mode_name}' (idx {forced_mode_idx}) "
                    f"to ON for interval ({current_time:.2f}, {force_until:.2f}) "
                    f"due to remaining min_up_time ({remaining_time:.2f}s)."
                )
                # This might overwrite a 'prevent' constraint if they conflict, which is intended.
                binapprox.set_valid_controls_for_interval(
                    dt=(current_time, force_until), b_bin_valid=valid_controls
                )

        # Initial State for PyCombina
        if self._state_initialized:
            b_bin_pre = np.zeros(n_modes)
            for name, state in self.mode_states.items():
                if state.active:
                    b_bin_pre[self._mode_name_to_idx[name]] = 1
            logger.debug(f"Setting b_bin_pre: {b_bin_pre}")
            binapprox.set_b_bin_pre(b_bin_pre)

        # --- 4. Solve Binary Approximation ---
        bnb = pycombina.CombinaBnB(binapprox)
        # TODO: Make solver options configurable?
        bnb.solve(
            use_warm_start=False,
            max_cpu_time=15,
            verbosity=0,
        )
        b_bin = binapprox.b_bin

        # if there is only one mode, we created a dummy mode which we remove now
        if len(self.var_ref.binary_controls) == 1:
            b_bin = b_bin[0, :].reshape(1, -1)

        return b_bin

    def constrain_binary_inputs(
        self,
        mpc_inputs_old: Dict[str, ca.DM],
        binary_array: np.ndarray,
    ) -> dict[str, ca.DM]:
        """

        Args:
            mpc_inputs_old:
            binary_array:

        Returns:

        """

        mpc_inputs_new = mpc_inputs_old.copy()
        name = self.system.binary_controls.name
        mpc_inputs_new[f"lb_{name}"] = binary_array
        mpc_inputs_new[f"ub_{name}"] = binary_array
        return mpc_inputs_new

    def save_rel_result_df(
        self,
        results: Results,
        now: float = 0,
    ):
        """
        Save the results of `solve` for relaxed MINLP into a dataframe at each time step.

        Example results dataframe:

        value_type               variable              ...     lower
        variable                      T_0   T_0_slack  ... T_0_slack mDot_0
        time_step                                      ...
        2         0.000000     298.160000         NaN  ...       NaN    NaN
                  101.431499   297.540944 -149.465942  ...      -inf    0.0
                  450.000000   295.779780 -147.704779  ...      -inf    0.0
                  798.568501   294.720770 -146.645769  ...      -inf    0.0
        Args:
            results:
            now:

        Returns:

        """
        res_file = cia_relaxed_results_path(self.config.results_file)
        if not self.rel_results_file_exists():
            results.write_columns(res_file)
            results.write_stats_columns(stats_path(res_file))

        df = results.df
        df.index = list(map(lambda x: str((now, x)), df.index))
        df.to_csv(res_file, mode="a", header=False)

        with open(stats_path(res_file), "a") as f:
            f.writelines(results.stats_line(str(now)))

    def rel_results_file_exists(self) -> bool:
        """Checks if the relaxed results file already exists, and if not, creates it with
        headers."""
        if self._created_rel_file:
            return True

        res_file = cia_relaxed_results_path(self.config.results_file)

        if res_file.is_file():
            # todo, this case is weird, as it is the mistake-append
            self._created_rel_file = True
            return True

        # we only check the file location once to save system calls
        res_file.parent.mkdir(parents=True, exist_ok=True)
        self._created_rel_file = True
        return False
