import logging
from math import inf, isclose
from typing import Union, Optional

# Assuming agentlib components are available
from agentlib.core import Agent, AgentVariable
from agentlib.core.errors import ConfigurationError
from agentlib.modules.controller import SISOController, SISOControllerConfig
from agentlib.modules.controller.pid import PIDConfig, PID
from pydantic import Field, field_validator

from agentlib_mpc.data_structures import mpc_datamodels


class FallbackPIDConfig(PIDConfig):
    """Config for FallbackPID: Adds the MPC active flag."""

    mpc_active_flag: AgentVariable = Field(
        default=AgentVariable(
            name=mpc_datamodels.MPC_FLAG_ACTIVE, type="bool", value=True
        ),
        description="Boolean variable indicating if MPC is active (True=MPC active, PID inactive).",
    )


class FallbackPID(PID):
    """
    PID controller active only when the MPC (indicated by mpc_active_flag) is inactive.
    Simplified error handling. Assumes configuration and data are valid.
    Resets integral state and timing upon activation/deactivation.
    """

    config: FallbackPIDConfig
    _mpc_was_active: Optional[bool] = None  # Track previous MPC state

    def __init__(self, *, config: FallbackPIDConfig, agent: Agent):
        super().__init__(config=config, agent=agent)
        # Initialize tracker, actual state checked in first callback
        self._mpc_was_active: Optional[bool] = None
        self.logger.info(
            f"FallbackPID initialized. Monitoring MPC flag '{self.config.mpc_active_flag.name}'."
        )

    def _siso_callback(self, inp: AgentVariable, name: str):
        """Handles input, checks MPC status, runs PID if MPC is inactive."""

        # 1. Get current MPC status (assume variable exists and is bool)
        mpc_flag_var = self.get(self.config.mpc_active_flag.name)
        mpc_is_active = bool(mpc_flag_var.value)  # Assume value is not None

        # 2. Check for state transitions and reset states
        if self._mpc_was_active is None:
            # First run: just store the state
            self._mpc_was_active = mpc_is_active
            self.logger.info(
                f"First run detected. Initial MPC state: {mpc_is_active}. Fallback PID active: {not mpc_is_active}"
            )
            if not mpc_is_active:  # If starting active (MPC inactive)
                self.last_time = inp.timestamp  # Set time correctly for first step
                self.integral = 0.0
                self.e_last = 0.0

        elif mpc_is_active != self._mpc_was_active:
            if mpc_is_active:
                # Transition: Fallback PID -> INACTIVE (MPC became Active)
                self.logger.info(
                    f"MPC flag '{mpc_flag_var.name}' became True. Deactivating FallbackPID."
                )
                self.integral = 0.0  # Reset integral
                self.e_last = 0.0  # Reset last error
            else:
                # Transition: Fallback PID -> ACTIVE (MPC became Inactive)
                self.logger.info(
                    f"MPC flag '{mpc_flag_var.name}' became False. Activating FallbackPID."
                )
                # Reset time to current input to avoid large dt spike on first step
                self.last_time = inp.timestamp
                # Integral and e_last should be 0 from deactivation, but reset again just in case
                self.integral = 0.0
                self.e_last = 0.0
            self._mpc_was_active = mpc_is_active  # Update tracked state

        # 3. Execute PID logic only if MPC is inactive
        if not mpc_is_active:
            self.logger.debug(
                f"MPC inactive. Running FallbackPID step for input {name}={inp.value}."
            )
            # Call the generator's send method, executing do_step
            out_val = self._step.send(inp)

            if out_val is not None:
                out_name = self.config.output.name
                self.logger.debug("Sending FallbackPID output %s=%s", out_name, out_val)
                self.set(name=out_name, value=out_val)
            else:
                self.logger.warning(
                    "FallbackPID do_step returned None (likely due to small t_sample). No output sent."
                )
        else:
            # MPC is active, PID is dormant
            pass
