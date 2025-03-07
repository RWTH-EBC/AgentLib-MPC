import itertools
import json
import time
from dataclasses import asdict
from pathlib import Path
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

        # Enable debugging if save_solve_stats is True
        if self.config.save_solve_stats:
            debug_dir = str(Path(self.config.solve_stats_file).parent / "debug")
            self.enable_detailed_debugging(debug_dir)

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

        stats = solver.stats()
        if hasattr(self, "debug_logger"):
            self.debug_logger.debug(f"QP solver stats: {stats}")

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

        # Log detailed QP solution info if debugging is enabled
        if hasattr(self, "debug_enabled") and self.debug_enabled:
            current_iteration = len(self._primal_residuals_tracker) + 1
            self.log_qp_solution(
                current_iteration, AQP, bQP, HQP, gQP, primal_solution, dual_solution
            )

        return primal_solution

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

        # Log detailed convergence info if debugging is enabled
        if hasattr(self, "debug_enabled") and self.debug_enabled:
            self.log_convergence_metrics(iteration, primal_residual, dual_residual)
            self.log_agent_updates(iteration)

        if iteration % self.config.save_iter_interval == 0:
            self._save_stats(iterations=iteration)

        # Check convergence against tolerance
        return (
            primal_residual < self.config.abs_tol
            and dual_residual < self.config.abs_tol
        )

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

    def enable_detailed_debugging(self, log_dir: str = "debug_logs"):
        """Enable detailed debugging output for ALADIN algorithm."""

        self.debug_log_dir = Path(log_dir)
        self.debug_log_dir.mkdir(exist_ok=True, parents=True)
        self.debug_enabled = True
        self.iteration_debug_data = []

        # Setup a specialized debug logger
        import logging

        debug_logger = logging.getLogger(f"{self.__class__.__name__}_debug")
        debug_logger.setLevel(logging.DEBUG)
        file_handler = logging.FileHandler(self.debug_log_dir / "aladin_debug.log")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        debug_logger.addHandler(file_handler)
        self.debug_logger = debug_logger

        self.debug_logger.info("Detailed debugging enabled for ALADIN")

    def log_qp_solution(
        self, iteration, AQP, bQP, HQP, gQP, primal_solution, dual_solution
    ):
        """Log detailed information about QP solution."""
        if not hasattr(self, "debug_enabled") or not self.debug_enabled:
            return

        import numpy as np
        import matplotlib.pyplot as plt

        # Create iteration directory
        iter_dir = self.debug_log_dir / f"iteration_{iteration}"
        iter_dir.mkdir(exist_ok=True)

        # Log matrices to files (summary information due to size)
        matrices_info = {
            "AQP_shape": AQP.shape,
            "AQP_nnz": AQP.nnz,
            "AQP_density": AQP.nnz / (AQP.shape[0] * AQP.shape[1]),
            "HQP_shape": HQP.shape,
            "HQP_nnz": HQP.nnz,
            "HQP_density": HQP.nnz / (HQP.shape[0] * HQP.shape[1]),
            "bQP_shape": bQP.shape,
            "gQP_shape": gQP.shape,
        }

        with open(iter_dir / "matrices_info.json", "w") as f:
            import json

            json.dump(matrices_info, f, indent=2)

        # Log solution vectors
        dual_solution = dual_solution.toarray()
        np.save(iter_dir / "primal_solution.npy", primal_solution)
        np.save(iter_dir / "dual_solution.npy", dual_solution)

        # Create visualizations
        fig, ax = plt.subplots(2, 1, figsize=(12, 10))
        ax[0].plot(primal_solution)
        ax[0].set_title(f"Primal solution vector (iteration {iteration})")
        ax[0].set_xlabel("Variable index")
        ax[0].set_ylabel("Value")

        if dual_solution is not None and len(dual_solution) > 0:
            ax[1].plot(
                dual_solution.toarray()
                if hasattr(dual_solution, "toarray")
                else dual_solution
            )
            ax[1].set_title(f"Dual solution vector (iteration {iteration})")
            ax[1].set_xlabel("Constraint index")
            ax[1].set_ylabel("Value")

        plt.tight_layout()
        plt.savefig(iter_dir / "qp_solution.png")
        plt.close()

        # Log solution statistics
        solution_stats = {
            "primal_solution_norm": float(np.linalg.norm(primal_solution)),
            "primal_solution_min": float(np.min(primal_solution)),
            "primal_solution_max": float(np.max(primal_solution)),
            "dual_solution_norm": float(
                np.linalg.norm(
                    dual_solution.toarray()
                    if hasattr(dual_solution, "toarray")
                    else dual_solution
                )
            ),
        }

        with open(iter_dir / "solution_stats.json", "w") as f:
            json.dump(solution_stats, f, indent=2)

        self.debug_logger.info(f"QP solution logged for iteration {iteration}")

    def log_agent_updates(self, iteration):
        """Log detailed information about agent updates."""
        if not hasattr(self, "debug_enabled") or not self.debug_enabled:
            return

        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd

        # Create iteration directory
        iter_dir = self.debug_log_dir / f"iteration_{iteration}"
        iter_dir.mkdir(exist_ok=True)

        # Collect information about each agent
        agent_data = {}
        for source, agent in self._active_agents().items():
            agent_id = f"{source.agent_id}"
            agent_data[agent_id] = {
                "local_solution_norm": float(np.linalg.norm(agent.local_solution)),
                "local_update_norm": float(np.linalg.norm(agent.local_update)),
                "local_target_norm": float(np.linalg.norm(agent.local_target)),
                "hessian_condition": float(
                    np.linalg.cond(
                        agent.hessian.toarray()
                        if hasattr(agent.hessian, "toarray")
                        else agent.hessian
                    )
                ),
                "jacobian_shape": list(agent.jacobian.shape)
                if hasattr(agent.jacobian, "shape")
                else None,
                "gradient_norm": float(np.linalg.norm(agent.gradient)),
            }

            # Save agent data as arrays
            agent_dir = iter_dir / agent_id
            agent_dir.mkdir(exist_ok=True)
            np.save(agent_dir / "local_solution.npy", agent.local_solution)
            np.save(agent_dir / "local_update.npy", agent.local_update)
            np.save(agent_dir / "local_target.npy", agent.local_target)

            # Create visualization of agent trajectories
            fig, ax = plt.subplots(3, 1, figsize=(12, 15))
            ax[0].plot(agent.local_solution)
            ax[0].set_title(f"Agent {agent_id} local solution (iteration {iteration})")
            ax[0].set_xlabel("Variable index")
            ax[0].set_ylabel("Value")

            ax[1].plot(agent.local_update)
            ax[1].set_title(f"Agent {agent_id} local update (iteration {iteration})")
            ax[1].set_xlabel("Variable index")
            ax[1].set_ylabel("Value")

            ax[2].plot(agent.local_target)
            ax[2].set_title(f"Agent {agent_id} local target (iteration {iteration})")
            ax[2].set_xlabel("Variable index")
            ax[2].set_ylabel("Value")

            plt.tight_layout()
            plt.savefig(agent_dir / "trajectories.png")
            plt.close()

            # Log coupling variables specifically
            coupling_data = {}
            for alias, indices in agent.coup_vars.items():
                values = agent.local_solution[indices]
                multipliers = agent.multipliers.get(alias)
                coupling_data[alias] = {
                    "indices": indices.tolist(),
                    "values": values.tolist(),
                    "multipliers": multipliers.tolist()
                    if multipliers is not None
                    else None,
                    "values_norm": float(np.linalg.norm(values)),
                    "multipliers_norm": float(np.linalg.norm(multipliers))
                    if multipliers is not None
                    else None,
                }

                # Create coupling visualization
                fig, ax = plt.subplots(2, 1, figsize=(10, 8))
                ax[0].plot(values)
                ax[0].set_title(
                    f"Coupling {alias} values (agent {agent_id}, iteration {iteration})"
                )
                ax[0].set_xlabel("Time step")
                ax[0].set_ylabel("Value")

                if multipliers is not None:
                    ax[1].plot(multipliers)
                    ax[1].set_title(
                        f"Coupling {alias} multipliers (agent {agent_id}, iteration {iteration})"
                    )
                    ax[1].set_xlabel("Time step")
                    ax[1].set_ylabel("Value")

                plt.tight_layout()
                plt.savefig(agent_dir / f"coupling_{alias}.png")
                plt.close()

            with open(agent_dir / "coupling_data.json", "w") as f:
                import json

                json.dump(coupling_data, f, indent=2, cls=NumpyEncoder)

        with open(iter_dir / "agent_data.json", "w") as f:
            import json

            json.dump(agent_data, f, indent=2)

        self.debug_logger.info(f"Agent updates logged for iteration {iteration}")

    def log_convergence_metrics(self, iteration, primal_residual, dual_residual):
        """Log detailed convergence metrics."""
        if not hasattr(self, "debug_enabled") or not self.debug_enabled:
            return

        # Add to iteration data
        self.iteration_debug_data.append(
            {
                "iteration": iteration,
                "timestamp": time.time(),
                "primal_residual": float(primal_residual),
                "dual_residual": float(dual_residual),
                "penalty_parameter": float(self.penalty_parameter),
                "qp_penalty": float(self.qp_penalty),
            }
        )

        # Save to CSV
        import pandas as pd

        df = pd.DataFrame(self.iteration_debug_data)
        df.to_csv(self.debug_log_dir / "convergence_metrics.csv", index=False)

        # Create convergence plots
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(2, 1, figsize=(10, 8))
        ax[0].semilogy(df["iteration"], df["primal_residual"])
        ax[0].set_title("Primal residual")
        ax[0].set_xlabel("Iteration")
        ax[0].set_ylabel("Residual (log scale)")
        ax[0].grid(True)

        ax[1].semilogy(df["iteration"], df["dual_residual"])
        ax[1].set_title("Dual residual")
        ax[1].set_xlabel("Iteration")
        ax[1].set_ylabel("Residual (log scale)")
        ax[1].grid(True)

        plt.tight_layout()
        plt.savefig(self.debug_log_dir / "convergence.png")
        plt.close()

        self.debug_logger.info(f"Convergence metrics logged for iteration {iteration}")

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


# Helper class for JSON serialization of numpy arrays
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return json.JSONEncoder.default(self, obj)
