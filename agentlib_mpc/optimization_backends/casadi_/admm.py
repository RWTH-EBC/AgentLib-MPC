import casadi as ca
import pandas as pd

from agentlib_mpc.data_structures.casadi_utils import DiscretizationMethod, Integrators
from agentlib_mpc.data_structures.mpc_datamodels import stats_path
from agentlib_mpc.data_structures.result_writers import HDF5ResultWriter
from agentlib_mpc.models.casadi_model import CasadiModel, CasadiInput, CasadiParameter
from agentlib_mpc.data_structures import admm_datatypes
from agentlib_mpc.optimization_backends.casadi_.core.VariableGroup import (
    OptimizationVariable,
    OptimizationParameter,
)
from agentlib_mpc.optimization_backends.casadi_.basic import (
    DirectCollocation,
    MultipleShooting,
    CasADiBaseBackend,
)
from agentlib_mpc.optimization_backends.backend import ADMMBackend
from agentlib_mpc.optimization_backends.casadi_.core.discretization import Results
from agentlib_mpc.optimization_backends.casadi_.full import FullSystem


class CasadiADMMSystem(FullSystem):
    local_couplings: OptimizationVariable
    global_couplings: OptimizationParameter
    multipliers: OptimizationParameter
    local_exchange: OptimizationVariable
    exchange_diff: OptimizationParameter
    exchange_multipliers: OptimizationParameter
    penalty_factor: OptimizationParameter

    def initialize(self, model: CasadiModel, var_ref: admm_datatypes.VariableReference):
        super().initialize(model=model, var_ref=var_ref)

        coup_names = [c.name for c in var_ref.couplings]
        exchange_names = [c.name for c in var_ref.exchange]
        pure_outs = [
            m for m in model.outputs if m.name not in coup_names + exchange_names
        ]
        self.outputs = OptimizationVariable.declare(
            denotation="y",
            variables=pure_outs,
            ref_list=var_ref.outputs,
        )

        self.local_couplings = OptimizationVariable.declare(
            denotation="local_couplings",
            variables=[model.get(name) for name in coup_names],
            ref_list=coup_names,
        )
        couplings_global = [coup.mean for coup in var_ref.couplings]
        self.global_couplings = OptimizationParameter.declare(
            denotation="global_couplings",
            variables=[CasadiInput(name=coup) for coup in couplings_global],
            ref_list=couplings_global,
        )

        multipliers = [coup.multiplier for coup in var_ref.couplings]
        self.multipliers = OptimizationParameter.declare(
            denotation="multipliers",
            variables=[CasadiInput(name=coup) for coup in multipliers],
            ref_list=multipliers,
        )

        self.local_exchange = OptimizationVariable.declare(
            denotation="local_exchange",
            variables=[model.get(name) for name in exchange_names],
            ref_list=exchange_names,
        )
        couplings_mean_diff = [coup.mean_diff for coup in var_ref.exchange]
        self.exchange_diff = OptimizationParameter.declare(
            denotation="average_diff",
            variables=[CasadiInput(name=coup) for coup in couplings_mean_diff],
            ref_list=couplings_mean_diff,
        )

        multipliers = [coup.multiplier for coup in var_ref.exchange]
        self.exchange_multipliers = OptimizationParameter.declare(
            denotation="exchange_multipliers",
            variables=[CasadiInput(name=coup) for coup in multipliers],
            ref_list=multipliers,
        )

        self.penalty_factor = OptimizationParameter.declare(
            denotation="rho",
            variables=[CasadiParameter(name="penalty_factor")],
            ref_list=["penalty_factor"],
        )

        # add admm terms to objective function
        admm_objective = 0
        rho = self.penalty_factor.full_symbolic[0]
        for i in range(len(var_ref.couplings)):
            admm_in = self.global_couplings.full_symbolic[i]
            admm_out = self.local_couplings.full_symbolic[i]
            admm_lam = self.multipliers.full_symbolic[i]
            admm_objective += admm_lam * admm_out + rho / 2 * (admm_in - admm_out) ** 2

        for i in range(len(var_ref.exchange)):
            admm_in = self.exchange_diff.full_symbolic[i]
            admm_out = self.local_exchange.full_symbolic[i]
            admm_lam = self.exchange_multipliers.full_symbolic[i]
            admm_objective += admm_lam * admm_out + rho / 2 * (admm_in - admm_out) ** 2

        self.cost_function += admm_objective


class ADMMCollocation(DirectCollocation):
    def _discretize(self, sys: CasadiADMMSystem):
        """
        Perform a direct collocation discretization.
        # pylint: disable=invalid-name
        """

        # setup the polynomial base
        collocation_matrices = self._collocation_polynomial()

        # shorthands
        n = self.options.prediction_horizon
        ts = self.options.time_step

        # Initial State
        x0 = self.add_opt_par(sys.initial_state)
        xk = self.add_opt_var(sys.states, lb=x0, ub=x0, guess=x0)
        uk = self.add_opt_par(sys.last_control)

        # Parameters that are constant over the horizon
        const_par = self.add_opt_par(sys.model_parameters)
        du_weights = self.add_opt_par(sys.r_del_u)
        rho = self.add_opt_par(sys.penalty_factor)

        # Formulate the NLP
        # loop over prediction horizon
        while self.k < n:
            # New NLP variable for the control
            u_prev = uk
            uk = self.add_opt_var(sys.controls)
            # penalty for control change between time steps
            self.objective_function += ts * ca.dot(du_weights, (u_prev - uk) ** 2)

            # perform inner collocation loop
            # perform inner collocation loop
            opt_vars_inside_inner = [
                sys.algebraics,
                sys.outputs,
                sys.local_couplings,
                sys.local_exchange,
            ]
            opt_pars_inside_inner = [
                sys.global_couplings,
                sys.multipliers,
                sys.exchange_multipliers,
                sys.exchange_diff,
                sys.non_controlled_inputs,
            ]
            constant_over_inner = {
                sys.controls: uk,
                sys.model_parameters: const_par,
                sys.penalty_factor: rho,
            }
            xk_end, constraints = self._collocation_inner_loop(
                collocation=collocation_matrices,
                state_at_beginning=xk,
                states=sys.states,
                opt_vars=opt_vars_inside_inner,
                opt_pars=opt_pars_inside_inner,
                const=constant_over_inner,
            )

            # increment loop counter and time
            self.k += 1
            self.pred_time = ts * self.k

            # New NLP variables at end of interval
            xk = self.add_opt_var(sys.states)

            # Add continuity constraint
            self.add_constraint(xk - xk_end, gap_closing=True)

            # add collocation constraints later for fatrop
            for constraint in constraints:
                self.add_constraint(*constraint)


class ADMMMultipleShooting(MultipleShooting):
    def _discretize(self, sys: CasadiADMMSystem):
        """Performs a multiple shooting discretization for ADMM-based optimization.

        This method implements the multiple shooting discretization scheme for both consensus
        and exchange ADMM variants. It handles:
        1. State continuity across shooting intervals
        2. Local coupling variables and their consensus terms
        3. Exchange variables between subsystems
        4. Integration of system dynamics
        5. Objective function construction including ADMM penalty terms

        Args:
            sys (CasadiADMMSystem): The system to be discretized, containing states,
                controls, and ADMM-specific variables
        """
        # Extract key parameters
        prediction_horizon = self.options.prediction_horizon
        timestep = self.options.time_step
        integration_options = {"t0": 0, "tf": timestep}

        # Initialize state trajectory
        initial_state = self.add_opt_par(sys.initial_state)
        current_state = self.add_opt_var(
            sys.states, lb=initial_state, ub=initial_state, guess=initial_state
        )

        # Initialize control input
        previous_control = self.add_opt_par(sys.last_control)

        # Add time-invariant parameters
        control_rate_weights = self.add_opt_par(sys.r_del_u)
        model_parameters = self.add_opt_par(sys.model_parameters)
        admm_penalty = self.add_opt_par(sys.penalty_factor)

        # Create system integrator
        dynamics_integrator = self._create_ode(
            sys, integration_options, self.options.integrator
        )

        # Perform multiple shooting discretization
        for k in range(prediction_horizon):
            # 1. Handle control inputs and their rate penalties
            current_control = self.add_opt_var(sys.controls)
            control_rate_penalty = timestep * ca.dot(
                control_rate_weights, (previous_control - current_control) ** 2
            )
            self.objective_function += control_rate_penalty
            previous_control = current_control

            # 2. Add optimization variables for current shooting interval
            disturbance = self.add_opt_par(sys.non_controlled_inputs)
            algebraic_vars = self.add_opt_var(sys.algebraics)
            output_vars = self.add_opt_var(sys.outputs)

            # 3. Add ADMM consensus variables
            local_coupling = self.add_opt_var(sys.local_couplings)
            global_coupling = self.add_opt_par(sys.global_couplings)
            coupling_multipliers = self.add_opt_par(sys.multipliers)

            # 4. Add ADMM exchange variables
            exchange_difference = self.add_opt_par(sys.exchange_diff)
            exchange_multipliers = self.add_opt_par(sys.exchange_multipliers)
            local_exchange = self.add_opt_var(sys.local_exchange)

            # 5. Construct stage-wise optimization problem
            stage_variables = {
                sys.states.name: current_state,
                sys.algebraics.name: algebraic_vars,
                sys.local_couplings.name: local_coupling,
                sys.outputs.name: output_vars,
                sys.local_exchange.name: local_exchange,
                sys.global_couplings.name: global_coupling,
                sys.multipliers.name: coupling_multipliers,
                sys.controls.name: current_control,
                sys.non_controlled_inputs.name: disturbance,
                sys.model_parameters.name: model_parameters,
                sys.penalty_factor.name: admm_penalty,
                sys.exchange_diff.name: exchange_difference,
                sys.exchange_multipliers.name: exchange_multipliers,
            }

            stage_result = self._stage_function(**stage_variables)

            # 6. Integrate system dynamics
            integration_result = dynamics_integrator(
                x0=current_state,
                p=ca.vertcat(
                    current_control,
                    local_coupling,
                    disturbance,
                    model_parameters,
                    algebraic_vars,
                    output_vars,
                ),
            )

            # 7. Add continuity constraints
            self.k = k + 1
            self.pred_time = timestep * self.k
            next_state = self.add_opt_var(sys.states)
            self.add_constraint(next_state - integration_result["xf"], gap_closing=True)

            # 8. Add model constraints and objective contributions
            self.add_constraint(
                stage_result["model_constraints"],
                lb=stage_result["lb_model_constraints"],
                ub=stage_result["ub_model_constraints"],
            )
            self.objective_function += stage_result["cost_function"] * timestep

            # Update for next interval
            current_state = next_state

    def _create_ode(
        self, sys: CasadiADMMSystem, opts: dict, integrator: Integrators
    ) -> ca.Function:
        # dummy function for empty ode, since ca.integrator would throw an error
        if sys.states.full_symbolic.shape[0] == 0:
            return lambda *args, **kwargs: {"xf": ca.MX.sym("xk_end", 0)}

        ode = sys.ode
        # create inputs
        x = sys.states.full_symbolic
        p = ca.vertcat(
            sys.controls.full_symbolic,
            sys.local_couplings.full_symbolic,
            sys.non_controlled_inputs.full_symbolic,
            sys.model_parameters.full_symbolic,
            sys.algebraics.full_symbolic,
            sys.outputs.full_symbolic,
        )
        integrator_ode = {"x": x, "p": p, "ode": ode}
        if integrator == Integrators.euler:
            xk_end = x + ode * opts["tf"]
            opt_integrator = ca.Function(
                "system", [x, p], [xk_end], ["x0", "p"], ["xf"]
            )
        else:  # rk, cvodes
            opt_integrator = ca.integrator("system", integrator, integrator_ode, opts)
        return opt_integrator


class CasADiADMMBackend(CasADiBaseBackend, ADMMBackend):
    """
    Class doing optimization of ADMM subproblems with CasADi.
    """

    system_type = CasadiADMMSystem
    discretization_types = {
        DiscretizationMethod.collocation: ADMMCollocation,
        DiscretizationMethod.multiple_shooting: ADMMMultipleShooting,
    }
    system: CasadiADMMSystem

    def __init__(self, config: dict):
        super().__init__(config)
        self.results: list[pd.DataFrame] = []
        self.result_stats: list[str] = []
        self.it: int = 0
        self.now: float = 0

    @property
    def coupling_grid(self):
        return self.discretization.grid(self.system.multipliers)

    def save_result_df(self, results: Results, now: float = 0):
        """Save ADMM results with iteration tracking."""
        writer = self._get_result_writer()
        if writer is None:
            return

        # Check if we're starting a new time step
        if now != self.now:
            self.it = 0
            self.now = now
        else:
            self.it += 1

        # For HDF5, we can write immediately
        if isinstance(writer, HDF5ResultWriter):
            if not self.results_file_exists():
                writer.initialize(results.df)
            writer.write_results(results.df, now, self.it)
            writer.write_stats(results.stats_line(f"({now}, {self.it})"))
        else:
            # For CSV, maintain existing buffering behavior
            self._buffer_csv_results(results, now)

    def _buffer_csv_results(self, results: Results, now: float):
        """Maintain existing CSV buffering behavior for ADMM."""
        # Add to buffers
        df = results.df.copy()
        df.index = [f"({now}, {self.it}, {idx})" for idx in df.index]
        self.results.append(df)

        # Append solve stats
        index = f"({now}, {self.it})"
        self.result_stats.append(results.stats_line(index))

        # Save results at the start of new sampling time, or if 1000 iterations
        # are exceeded (to prevent memory issues)
        if self.it == 0 or self.it % 1000 == 0:
            writer = self._get_result_writer()

            # Write all buffered results
            for iteration_result in self.results:
                writer.write_results(iteration_result, now, self.it)

            # Write all buffered stats
            for stat_line in self.result_stats:
                writer.write_stats(stat_line)

            # Clear buffers
            self.results = []
            self.result_stats = []
