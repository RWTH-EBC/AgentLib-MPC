import itertools
import time
from dataclasses import asdict
from typing import Dict, Tuple, Set, Optional

import casadi as ca
import numpy as np
from scipy import sparse
from agentlib import Agent, AgentVariable, Source
from pydantic import Field

from agentlib_mpc.modules.dmpc.coordinator import Coordinator, CoordinatorConfig
from agentlib_mpc.data_structures import coordinator_datatypes as cdt
import agentlib_mpc.modules.dmpc.aladin.aladin_datatypes as ald


class ALADINCoordinatorConfig(CoordinatorConfig):
    """Hold the config for ALADINCoordinator"""

    penalty_factor: float = Field(
        title="penalty_factor",
        default=10,
        description="Penalty factor of the ALADIN algorithm. Should be equal "
        "for all agents.",
    )
    iter_max: int = Field(
        title="iter_max",
        default=20,
        description="Maximum number of ALADIN iterations before termination of control "
        "step.",
    )
    qp_penalty: float = Field(
        default=100, description="Mu parameter of the ALADIN algorithm."
    )
    qp_step_size: float = Field(
        default=1,
        ge=0,
        le=1,
        description="Constant step size used for multiplier and primal variables "
        "update.",
    )
    activation_margin: float = Field(
        default=0.001, gt=0, description="Threshold for active set detection."
    )
    regularization_parameter: float = Field(default=0, ge=0)
    solve_stats_file: str = Field(
        default="aladin_stats.csv",
        description="File name for the solve stats.",
    )


class ALADINCoordinator(Coordinator):
    """Coordinator implementation for the ALADIN algorithm"""

    config: ALADINCoordinatorConfig

    def __init__(self, *, config: dict, agent: Agent):
        super().__init__(config=config, agent=agent)
        self.agent_dict: Dict[Source, ald.AgentDictEntry] = {}
        self.current_healthy_agents: Optional[Set[str]] = None

        # ALADIN-specific parameters
        self.penalty_parameter: float = self.config.penalty_factor
        self.qp_penalty: float = self.config.qp_penalty
        self.lambda_: np.ndarray = None
        self.lambda_old: np.ndarray = None
        self.alias_to_parent: Dict[str, str] = {}

    def _initial_registration(self, variable: AgentVariable):
        """Handles initial registration of an agent with ALADIN-specific entry type."""
        if not (variable.source in self.agent_dict):
            entry = ald.AgentDictEntry(  # Use ALADIN-specific type
                name=variable.source,
                status=cdt.AgentStatus.pending,
            )
            self.agent_dict[variable.source] = entry
            self._send_parameters_to_agent(variable)
            self.logger.info(
                f"Coordinator got request from agent {variable.source} and set to 'pending'."
            )
        elif self.agent_dict[variable.source].status is cdt.AgentStatus.pending:
            self.register_agent(variable=variable)

    def _realtime_step(self):
        """Implement one step of the ALADIN algorithm in realtime mode."""
        # ------------------
        # start iteration
        # ------------------
        self.status = cdt.CoordinatorStatus.init_iterations
        self.start_algorithm_at = self.env.time
        self._performance_counter = time.perf_counter()
        # maybe this will hold information instead of "True"
        self.set(cdt.START_ITERATION_C2A, True)
        # check for all_finished here
        time.sleep(self.config.wait_time_on_start_iters)
        if not list(self._agents_with_status(status=cdt.AgentStatus.ready)):
            self.logger.info(f"No Agents available at time {self.env.now}.")
            return  # if no agents registered return early

        # Create couplings on first iteration or if agents change
        self.create_couplings()

        # ------------------
        # iteration loop
        # ------------------
        iteration: int = 0
        for iteration in range(1, self.config.iter_max + 1):
            # ------------------
            # optimization
            # ------------------
            # send
            self.status = cdt.CoordinatorStatus.optimization
            # set all agents to busy
            self.trigger_optimizations()

            # check for all finished here
            self._wait_for_ready()

            # ------------------
            # perform update steps
            # ------------------
            self.status = cdt.CoordinatorStatus.updating
            AQP, bQP, HQP, gQP = self.setup_qp_params()
            self._solve_coordination_qp(AQP, bQP, HQP, gQP)
            self.compute_al_step()

            # ------------------
            # check convergence
            # ------------------
            converged = self._check_convergence(iteration)
            if converged:
                self.logger.info("Converged within %s iterations. ", iteration)
                break
        else:
            self.logger.warning(
                "Did not converge within the maximum number of iterations " "%s. ",
                self.config.iter_max,
            )
        self._wrap_up_algorithm(iterations=iteration)
        self.set(cdt.START_ITERATION_C2A, False)  # this signals the finish

    def _fast_process(self):
        """Process function for use in fast-as-possible simulations."""
        yield self._wait_non_rt()

        while True:
            # ------------------
            # start iteration
            # ------------------
            self.status = cdt.CoordinatorStatus.init_iterations
            self.start_algorithm_at = self.env.time
            self._performance_counter = time.perf_counter()
            self.set(cdt.START_ITERATION_C2A, True)
            yield self._wait_non_rt()
            if not list(self._agents_with_status(status=cdt.AgentStatus.ready)):
                self.logger.info(f"No Agents available at time {self.env.now}.")
                communication_time = self.env.time - self.start_algorithm_at
                yield self.env.timeout(self.config.sampling_time - communication_time)
                continue  # if no agents registered return early

            self.create_couplings()

            # ------------------
            # iteration loop
            # ------------------
            iter_ = 0
            for iter_ in range(1, self.config.iter_max + 1):
                # parallel steps optimization
                self.status = cdt.CoordinatorStatus.optimization
                self.trigger_optimizations()
                yield self._wait_non_rt()
                self._wait_for_ready()

                # coordinator update
                self.status = cdt.CoordinatorStatus.updating
                AQP, bQP, HQP, gQP = self.setup_qp_params()
                self._solve_coordination_qp(AQP, bQP, HQP, gQP)
                self.compute_al_step()

                # check convergence
                converged = self._check_convergence(iter_)
                if converged:
                    self.logger.info("Converged within %s iterations. ", iter_)
                    break
            else:
                self.logger.warning(
                    "Did not converge within the maximum number of iterations " "%s. ",
                    self.config.iter_max,
                )
            self._wrap_up_algorithm(iterations=iter_)
            self.set(cdt.START_ITERATION_C2A, False)  # this signals the finish
            self.status = cdt.CoordinatorStatus.sleeping
            time_spent_on_communication = self.env.time - self.start_algorithm_at
            yield self.env.timeout(
                self.config.sampling_time - time_spent_on_communication
            )

    def init_iteration_callback(self, variable: AgentVariable):
        """Process an agent's InitIteration confirmation."""
        ag_dict_entry = self._confirm_init_iteration(variable)
        if ag_dict_entry is None:
            return

        # if there is a list value, that means we get a shifted trajectory
        if isinstance(variable.value, list):
            ag_dict_entry.local_solution = variable.value
            ag_dict_entry.local_target = variable.value
        ag_dict_entry.status = cdt.AgentStatus.ready
        self.received_variable.set()

    def _send_parameters_to_agent(self, variable: AgentVariable):
        """Send global parameters to an agent after registration request."""
        aladin_parameters = ald.ALADINParameters(
            receiver_agent_id=variable.source.agent_id,
            prediction_horizon=self.config.prediction_horizon,
            time_step=self.config.time_step,
            penalty_factor=self.config.penalty_factor,
        )

        message = cdt.RegistrationMessage(
            agent_id=variable.source.agent_id, opts=aladin_parameters.to_dict()
        )
        self.set(cdt.REGISTRATION_C2A, asdict(message))

    def register_agent(self, variable: AgentVariable):
        """Register an agent after it sends its initial data."""
        reg_msg: ald.RegistrationA2C = ald.RegistrationA2C.from_json(variable.value)
        src = variable.source
        ag_dict_entry = self.agent_dict[variable.source]
        ag_dict_entry.opt_var_length = len(reg_msg.local_solution)
        ag_dict_entry.local_solution = reg_msg.local_solution
        ag_dict_entry.local_target = np.array(reg_msg.local_solution)
        ag_dict_entry.local_update = np.full_like(reg_msg.local_solution, 0)
        ag_dict_entry.coup_vars = {
            k: np.array(v, dtype=int) for k, v in reg_msg.coup_vars.items()
        }

        # loop over coupling variables of this agent
        for alias, traj in ag_dict_entry.coup_vars.items():
            ag_dict_entry.multipliers[alias] = np.full_like(traj, 0, dtype=int)

        # set agent from pending to standby
        ag_dict_entry.status = cdt.AgentStatus.standby
        self.logger.info(
            f"Coordinator successfully registered agent {variable.source}."
        )

    def trigger_optimizations(self):
        """Trigger optimizations for all agents with ready status."""
        # create an iterator for all agents which are ready for this round
        active_agents = self._active_agents()

        # aggregate and send trajectories per agent
        for source, agent in active_agents.items():
            # package all coupling inputs needed for an agent
            coordi_to_agent = ald.CoordinatorToAgent(
                z=agent.local_target.ravel(),
                lam=agent.multipliers,
                target=source.agent_id,
                rho=self.penalty_parameter,
            )

            self.logger.debug("Sending to %s with source %s", agent.name, source)
            self.logger.debug("Set %s to busy.", agent.name)

            # send values
            agent.status = cdt.AgentStatus.busy
            self.set(cdt.OPTIMIZATION_C2A, coordi_to_agent.to_json())

    def optim_results_callback(self, variable: AgentVariable):
        """Process the results from a local optimization."""
        local_result = ald.AgentToCoordinator.from_json(variable.value)
        agent = self.agent_dict[variable.source]
        agent.hessian = local_result.H
        agent.local_solution = local_result.x
        agent.jacobian = local_result.J
        agent.gradient = local_result.g
        agent.status = cdt.AgentStatus.ready
        self.received_variable.set()

    def _active_agents(self) -> Dict[Source, ald.AgentDictEntry]:
        """Get a dictionary of active agents (with ready status)."""
        active_agents = {
            key: ag
            for key, ag in self.agent_dict.items()
            if ag.status is cdt.AgentStatus.ready
        }
        if self.current_healthy_agents is not None and (
            set(active_agents) != self.current_healthy_agents
        ):
            raise NotImplementedError(
                "There was a change in active Agents, this is currently not supported"
            )
        return active_agents

    def create_couplings(self):
        """Create coupling matrices for all agents."""
        # Check agents that are currently registered and on standby
        active_agents = self._active_agents()
        # if agents are the same as last time, we do not need to remake couplings
        if set(active_agents) == self.current_healthy_agents:
            return

        # if we got here, we need to redo the couplings and remember the healthy agents
        self.current_healthy_agents = set(active_agents)
        global_var_len: int = 0

        # assign parent / child on coupling variables
        for source, entry in active_agents.items():
            for alias in entry.coup_vars:
                if alias not in self.alias_to_parent:
                    self.alias_to_parent[alias] = source
                    entry.coup_vars_parent.append(alias)
                else:
                    entry.coup_vars_child.append(alias)

        # build the coupling matrices
        for source, child_entry in active_agents.items():
            child_entry: ald.AgentDictEntry
            for alias in child_entry.coup_vars_child:
                parent_src = self.alias_to_parent[alias]
                parent_entry = active_agents[parent_src]
                parent_indices = parent_entry.coup_vars[alias]
                child_indices = child_entry.coup_vars[alias]

                for i, _ in enumerate(parent_indices):
                    parent_entry.sparse_builder.add(
                        value=-1,
                        column=parent_indices[i],
                        row=global_var_len,
                    )
                    child_entry.sparse_builder.add(
                        value=1,
                        column=child_indices[i],
                        row=global_var_len,
                    )
                    global_var_len += 1
        for entry in active_agents.values():
            entry.finalize_matrix(global_var_len)
        self.lambda_ = np.zeros((global_var_len, 1), dtype=float)
        self.lambda_old = self.lambda_

    def _solve_coordination_qp(
        self,
        AQP: sparse.csc_matrix,
        bQP: np.ndarray,
        HQP: sparse.csc_matrix,
        gQP: np.ndarray,
    ):
        """Solve the coordination QP problem."""

        def debug_show_matrices():
            A = AQP.toarray()
            b = HQP.toarray()
            return A, b

        Anp, Hnp = debug_show_matrices()

        # casadi
        cqp = {"a": ca.DM(AQP).sparsity(), "h": ca.DM(HQP).sparsity()}
        solver = ca.conic("QPSolver", "osqp", cqp, {"error_on_fail": False})
        solution = solver(
            h=ca.DM(HQP), g=ca.DM(gQP), a=ca.DM(AQP), lba=ca.DM(bQP), uba=ca.DM(bQP)
        )

        from pprint import pprint

        stats = solver.stats()
        pprint(stats)

        primal_solution = solution["x"].toarray()
        dual_solution = solution["lam_a"]

        self.lambda_old = self.lambda_
        # lambda is set here to full step. We keep old lambda, so we can do partial
        # step later
        self.lambda_ = dual_solution[: len(self.lambda_)].toarray()

        # split solution to agents
        index = 0
        for source, entry in self._active_agents().items():
            end = index + entry.opt_var_length
            entry.local_update = primal_solution[index:end]
            index = end

        return primal_solution

    def setup_qp_params(
        self,
    ) -> Tuple[sparse.csr_matrix, np.ndarray, sparse.csr_matrix, np.ndarray]:
        """Create the parameters for the coordination QP."""
        active_agents = self._active_agents().values()
        # Construct A matrix
        # build sparse only when needed
        full_coupling_matrix: sparse.csr_matrix = sparse.hstack(
            [a.coupling_matrix for a in active_agents]
        ).tocsr()
        _debug_full_coupling_matrix = full_coupling_matrix.toarray()
        len_global_var = full_coupling_matrix.shape[0]

        # Compute rhsQP vector
        full_local_sol = np.array(
            list(itertools.chain.from_iterable(a.local_solution for a in active_agents))
        )
        qp_right_hand_side = -full_coupling_matrix @ full_local_sol

        # Construct HQP matrix
        hessians = [a.hessian for a in active_agents]
        qp_pen = self.qp_penalty / 2
        hessians.append(sparse.diags([qp_pen], shape=(len_global_var, len_global_var)))
        HQP = sparse.block_diag(hessians, format="csc")

        # Construct JacCon matrix
        jac_con = sparse.block_diag([a.jacobian for a in active_agents], format="csc")

        # Check condition number of JacCon
        if np.linalg.cond(jac_con.toarray()) > 1e8:
            print("Condition number of constraints is greater than 1e8")

        # Construct AQP matrix
        number_of_active_constraints = jac_con.shape[0]
        AQP = sparse.vstack(
            [
                sparse.hstack([full_coupling_matrix, -np.eye(len_global_var)]),
                sparse.hstack(
                    [jac_con, np.zeros((number_of_active_constraints, len_global_var))]
                ),
            ]
        ).tocsc()

        # Construct bQP vector
        bQP = np.concatenate(
            [qp_right_hand_side, np.zeros((number_of_active_constraints, 1))]
        )

        # Construct gQP vector
        gradients = list(itertools.chain(a.gradient for a in active_agents))
        gradients.append(self.lambda_)
        gQP = np.concatenate(gradients)

        # Debug info
        _debug_AQP_dense = AQP.toarray()
        _debug_HQP_dense = HQP.toarray()

        return AQP, bQP, HQP, gQP

    def compute_al_step(self):
        """Compute the ALADIN step."""
        alpha = self.config.qp_step_size
        self.lambda_ = self.lambda_old + alpha * (self.lambda_ - self.lambda_old)
        for source, entry in self._active_agents().items():
            entry.local_target = (
                entry.local_target
                + (entry.local_solution - entry.local_target)
                + alpha * entry.local_update
            )
            full_multiplier = (self.lambda_.T @ entry.coupling_matrix).ravel()
            for alias, indices in entry.coup_vars.items():
                entry.multipliers[alias] = full_multiplier[indices]

    def _check_convergence(self, iteration) -> bool:
        """Check if the algorithm has converged."""
        # Implement ALADIN-specific convergence check
        # For now, using a simple implementation based on the norm of updates
        active_agents = self._active_agents()

        # Compute primal and dual residuals
        primal_residual = 0
        dual_residual = 0

        for source, entry in active_agents.items():
            # Primal residual: norm of updates
            primal_update = np.linalg.norm(entry.local_update)
            primal_residual += primal_update**2

            # Dual residual: norm of multiplier changes
            if self.lambda_old is not None:
                dual_update = np.linalg.norm(self.lambda_ - self.lambda_old)
                dual_residual += dual_update**2

        primal_residual = np.sqrt(primal_residual)
        dual_residual = np.sqrt(dual_residual) if dual_residual > 0 else 0

        # Track residuals for stats
        self._primal_residuals_tracker.append(primal_residual)
        self._dual_residuals_tracker.append(dual_residual)
        self._performance_tracker.append(
            time.perf_counter() - self._performance_counter
        )

        self.logger.debug(
            "Finished iteration %s . \n Primal residual: %s \n Dual residual: %s",
            iteration,
            primal_residual,
            dual_residual,
        )

        if iteration % self.config.save_iter_interval == 0:
            self._save_stats(iterations=iteration)

        # Check convergence against tolerance
        return (
            primal_residual < self.config.abs_tol
            and dual_residual < self.config.abs_tol
        )

    def _save_stats(self, iterations: int) -> None:
        """Save iteration statistics to a file."""
        data_dict = {
            "primal_residual": self._primal_residuals_tracker,
            "dual_residual": self._dual_residuals_tracker,
            "wall_time": self._performance_tracker,
        }
        super()._save_stats(iterations=iterations, data_dict=data_dict)

    def _wrap_up_algorithm(self, iterations):
        """Clean up at the end of an algorithm execution."""
        self._save_stats(iterations=iterations)
