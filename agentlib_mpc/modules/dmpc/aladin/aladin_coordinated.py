from collections import defaultdict
from typing import Optional, List

import numpy as np
from agentlib import Agent, AgentVariable

from agentlib_mpc.data_structures.mpc_datamodels import MPCVariable
from agentlib_mpc.modules.dmpc.admm.admm_coordinated import (
    CoordinatedADMM,
    CoordinatedADMMConfig,
)
from agentlib_mpc.data_structures import coordinator_datatypes as cdt
import agentlib_mpc.modules.dmpc.aladin.aladin_datatypes as ald
from agentlib_mpc.optimization_backends.casadi_.aladin import CasADiALADINBackend


class CoordinatedALADINConfig(CoordinatedADMMConfig):
    ...


class CoordinatedALADIN(CoordinatedADMM):
    """
    Module to implement an ALADIN agent, which is guided by a coordinator.
    Only optimizes based on callbacks.
    """

    config: CoordinatedALADINConfig
    optimization_backend: CasADiALADINBackend

    def __init__(self, *, config: dict, agent: Agent):
        super().__init__(config=config, agent=agent)

    def process(self):

        # send registration request to coordinator
        timeout = self.config.registration_interval

        while True:
            if not self._registered_coordinator:
                self.set(cdt.REGISTRATION_A2C, "")
            yield self.env.timeout(timeout)

    def registration_callback(self, variable: AgentVariable):
        """callback for registration"""
        if self._registered_coordinator:
            # ignore if registration has already been done
            return

        self.logger.debug(
            f"receiving {variable.name}={variable.value} from {variable.source}"
        )
        # global parameters to define optimisation problem
        options = cdt.ParametersC2A.from_json(variable.value)
        if not options.receiver_agent_id == self.source.agent_id:
            return
        self._set_mpc_parameters(options=options)
        current_vars = defaultdict(lambda: MPCVariable(name="", value=0))
        current_vars.update(self.collect_variables_for_optimization())
        initial_guess, coup_vars_n = self.optimization_backend.get_aladin_registration(
            current_vars=current_vars, now=self.env.time
        )
        coup_vars_alias = {self.get(n).alias: v for n, v in coup_vars_n.items()}
        answer = ald.RegistrationA2C(
            local_solution=initial_guess,
            coup_vars=coup_vars_alias,
        )

        self._registered_coordinator = variable.source
        self.set(cdt.REGISTRATION_A2C, answer.to_json())

    def init_iteration_callback(self, variable: AgentVariable):
        """
        Callback that processes the coordinators 'startIteration' flag.
        Args:
            variable:

        """
        # value is True on start
        if variable.value:
            # custom function which can be overloaded to do stuff before a step
            self.pre_computation_hook()

            # prepare optimization
            self._start_optimization_at = self.env.time
            self.get_new_measurement()
            new_trajectory = self.shift_opt_variable()

            self.set(cdt.START_ITERATION_A2C, new_trajectory)
            self.logger.debug("Sent 'StartIteration' True.")

        # value is False on convergence/iteration limit
        else:
            self._finish_optimization()

    def shift_opt_variable(self) -> Optional[List[float]]:
        return self.optimization_backend.shift_opt_var()

    @property
    def penalty_factor_var(self) -> MPCVariable:
        return MPCVariable(
            name=ald.LOCAL_PENALTY_FACTOR, value=self.config.penalty_factor
        )

    def get_new_measurement(self):
        """
        Retrieve new measurement from relevant sensors
        Returns:

        """
        opt_inputs = self.collect_variables_for_optimization()
        opt_inputs[ald.LOCAL_PENALTY_FACTOR] = self.penalty_factor_var
        self._optimization_inputs = opt_inputs

    def optimize(self, variable: AgentVariable):
        """
        Performs the optimization given the mean trajectories and multipliers from the
        coordinator.
        Replies with the local optimal trajectories.
        Returns:

        """
        # unpack message
        updates = ald.CoordinatorToAgent.from_json(variable.value)
        if not updates.target == self.source.agent_id:
            return
        self.logger.debug("Received update from Coordinator.")

        # load mpc inputs and current coupling inputs of this iteration
        opt_inputs = self._optimization_inputs.copy()

        # add the coupling inputs of this iteration to the other mpc inputs
        for alias, multiplier in updates.lam.items():
            coup_in = self._alias_to_input_names[alias]
            opt_inputs[coup_in.lam] = MPCVariable(name=coup_in.lam, value=multiplier)

        opt_inputs[ald.LOCAL_PENALTY_FACTOR].value = updates.rho
        # perform optimization
        self.optimization_backend.set_global_variable(updates.z)
        self._result = self.optimization_backend.solve(
            now=self._start_optimization_at, current_vars=opt_inputs
        )
        opt_return = self.optimization_backend.get_sensitivities()
        self.logger.debug("Sent optimal solution.")
        self.set(name=cdt.OPTIMIZATION_A2C, value=opt_return.to_json())


def regularizeH(H, opts):
    # Eigenvalue decomposition of the Hessian
    V, D = np.linalg.eig(np.asarray(H))
    e = np.diag(D)

    # Check if regularization is needed
    didReg = False
    if np.min(e) < 0:
        didReg = True

    # Flip the eigenvalues
    e = np.abs(e)

    # Modify zero and too large eigenvalues (regularization)
    reg = opts.regParam
    e[e <= reg] = reg  # 1e-4

    # Regularization for small stepsize
    Hreg = V @ np.diag(e) @ V.T

    return np.real(Hreg), didReg
