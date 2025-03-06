from typing import Dict, Tuple, Optional, List

import casadi as ca
import numpy as np

from agentlib_mpc.data_structures.admm_datatypes import VariableReference
from agentlib_mpc.data_structures.casadi_utils import (
    DiscretizationMethod,
    GUESS_PREFIX,
    MPCInputs,
    CasadiDiscretizationOptions,
    SolverFactory,
    OptParMXContainer,
)
from agentlib_mpc.data_structures.mpc_datamodels import MPCVariable
from agentlib_mpc.models.casadi_model import CasadiParameter, CasadiInput, CasadiModel
from agentlib_mpc.modules.dmpc.aladin import aladin_datatypes
from agentlib_mpc.optimization_backends.casadi_.admm import (
    CasADiADMMBackend,
    ADMMCollocation,
    ADMMMultipleShooting,
    CasadiADMMSystem,
)
from agentlib_mpc.optimization_backends.casadi_.basic import DirectCollocation
from agentlib_mpc.optimization_backends.casadi_.core.VariableGroup import (
    OptimizationVariable,
    OptimizationParameter,
)
from agentlib_mpc.optimization_backends.casadi_.core.discretization import (
    Discretization,
    Results,
)
from agentlib_mpc.optimization_backends.casadi_.full import FullSystem


class CasadiALADINSystem(FullSystem):
    local_couplings: OptimizationVariable
    multipliers: OptimizationParameter
    penalty_factor: OptimizationParameter

    def initialize(
        self, model: CasadiModel, var_ref: aladin_datatypes.VariableReference
    ):

        super().initialize(model=model, var_ref=var_ref)

        coup_names = [c.name for c in var_ref.couplings]
        pure_outs = [m for m in model.outputs if m.name not in coup_names]
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
        multipliers = [coup.multiplier for coup in var_ref.couplings]
        self.multipliers = OptimizationParameter.declare(
            denotation="multipliers",
            variables=[CasadiInput(name=coup) for coup in multipliers],
            ref_list=multipliers,
        )

        self.penalty_factor = OptimizationParameter.declare(
            denotation="rho",
            variables=[CasadiParameter(name=aladin_datatypes.LOCAL_PENALTY_FACTOR)],
            ref_list=[aladin_datatypes.LOCAL_PENALTY_FACTOR],
        )

        # add aladin terms to objective function
        objective = 0
        for i in range(len(var_ref.couplings)):
            local_couplings = self.local_couplings.full_symbolic[i]
            multiplier = self.multipliers.full_symbolic[i]
            objective += multiplier * local_couplings

        self.cost_function += objective


class ALADINDiscretization(Discretization):
    def __init__(self, options: CasadiDiscretizationOptions):
        self.sensitivities_result = None
        self.global_variable = None
        self._sensitivities_func: Optional[ca.Function] = None
        super().__init__(options)

    def get_aladin_registration(
        self, mpc_inputs: Dict[str, ca.DM], var_ref: VariableReference
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        # answer = ald.RegistrationA2C(
        #     coup_vars=...,
        #     local_solution=...,
        # )
        guesses = self._determine_initial_guess(mpc_inputs)
        mpc_inputs.update(guesses)
        nlp_inputs: dict[str, ca.DM] = self._mpc_inputs_to_nlp_inputs(**mpc_inputs)

        local_solution = nlp_inputs["x0"].toarray()
        opt_var_length = len(local_solution)
        mpc_output = self._nlp_outputs_to_mpc_outputs(
            vars_at_optimum=list(range(opt_var_length))
        )
        result = self._process_solution(inputs=mpc_inputs, outputs=mpc_output)
        coup_vars = {c.name: result[c.name] for c in var_ref.couplings}

        return local_solution, coup_vars

    def shift_opt_var(self) -> Optional[List[float]]:
        current_optimum = {
            f"guess_{den}": var.opt for den, var in self.mpc_opt_vars.items()
        }

        # if this is the first step, just return none
        if any(value is None for value in current_optimum.values()):
            return None

        for den, var in self.mpc_opt_vars.items():
            den = f"guess_{den}"
            if var.opt.shape[0] == 0:

                # skip if there are no variables of that type
                continue
            shift_by = int(len(var.grid) / self.options.prediction_horizon)
            # have to use the key guess_... here, as the _mpc_inputs_to_nlp_inputs
            # original implementation uses that, and we want to use that function
            current_optimum[den] = ca.horzcat(  # have to use guess here as
                current_optimum[den][shift_by:], current_optimum[den][-shift_by:]
            )
        return (
            self._mpc_inputs_to_nlp_inputs(**current_optimum)["x0"].toarray().tolist()
        )

    def solve(self, mpc_inputs: MPCInputs) -> Results:
        """
        Solves the discretized trajectory optimization problem.

        Args:
            mpc_inputs: Casadi Matrices specifying the input of all different types
                of optimization parameters. Matrices consist of different variable rows
                and have a column for each time step in the discretization.
                There are separate matrices for each input type (as defined in the
                System), and also for the upper and lower boundaries of variables
                respectively.


        Returns:
            Results: The complete evolution of the states, inputs and boundaries of each
                variable and parameter over the prediction horizon, as well as solve
                statistics.

        """
        # todo get these from coordinator
        regularization_parameter = 0.0015111114529828148
        activation_margin = 0.0007901426386278718

        # collect and format inputs
        guesses = self._determine_initial_guess(mpc_inputs)
        mpc_inputs[aladin_datatypes.PRESCRIBED] = self.global_variable
        mpc_inputs.update(guesses)
        nlp_inputs: dict[str, ca.DM] = self._mpc_inputs_to_nlp_inputs(**mpc_inputs)

        # perform optimization
        nlp_output = self._optimizer(**nlp_inputs)
        debug_stats = self._optimizer.stats()
        debug_prescribed_input = {
            k: np.array(v)
            for k, v in self._nlp_outputs_to_mpc_outputs(
                vars_at_optimum=mpc_inputs["prescribed"]
            ).items()
        }

        # todo we are ignoring box constraints here. maybe that is the problem
        constraints = self._get_constraints(nlp_output["x"], nlp_inputs["p"])
        active_constraints = self._get_active_constraints(
            constraints=constraints,
            lb=nlp_inputs["lbg"],
            ub=nlp_inputs["ubg"],
            act_margin=activation_margin,  # todo get from coordinator
        )

        # get sensitivities
        nlp_output["lam_g"][~active_constraints] = 0
        self.sensitivities_result = self._sensitivities_func(
            opt_vars=nlp_output["x"],
            opt_pars=nlp_inputs["p"],
            local_constraint_multipliers=nlp_output["lam_g"],
        )
        debug_jacobian = self.sensitivities_result["J"].toarray()
        self.sensitivities_result["J"] = self.sensitivities_result["J"].toarray()[
            active_constraints
        ]
        # todo look at hessian and also with regard to lam_g, maybe this is funky. Alternatively, look at gradient, whether there should be more to that, including constraints
        self.sensitivities_result["H"] = regularize_h(
            self.sensitivities_result["H"], regularization_parameter
        )
        debug_sensitivities = {
            k: np.array(v) for k, v in self.sensitivities_result.items()
        }
        debug_sensitivities["jacobian_all"] = debug_jacobian

        # format and return solution
        mpc_output = self._nlp_outputs_to_mpc_outputs(vars_at_optimum=nlp_output["x"])
        self._remember_solution(mpc_output)
        result = self._process_solution(inputs=mpc_inputs, outputs=mpc_output)

        # todo the hessian of cooler is too big and I am missing box constraints I think
        return result

    def _get_active_constraints(
        self, constraints: np.ndarray, lb: np.ndarray, ub: np.ndarray, act_margin: float
    ) -> np.ndarray:
        """Returns a boolean array of constraints that are considered active."""
        # Check if lb and ub have the same shape as constraints
        assert (
            constraints.shape == lb.shape == ub.shape
        ), "Constraints, lb, and ub must have the same shape"

        # Create a boolean array to store active constraints
        active_constraints = np.zeros_like(constraints, dtype=bool)

        # Check equality constraints
        equality_mask = np.array(lb == ub, dtype=bool)
        active_constraints[equality_mask] = True

        # Check inequality constraints
        inequality_mask = np.array(lb < ub, dtype=bool)
        active_upper = (
            np.array(ub - constraints < act_margin, dtype=bool) & inequality_mask
        )
        active_lower = (
            np.array(constraints - lb < act_margin, dtype=bool) & inequality_mask
        )
        active_constraints[active_upper | active_lower] = True

        return active_constraints.ravel()

    def initialize(self, system: CasadiALADINSystem, solver_factory: SolverFactory):
        """Initializes the trajectory optimization problem, creating all symbolic
        variables of the OCP, the mapping function and the numerical solver."""
        self._discretize(system)
        self._finished_discretization = True
        # self._aladin_modifications(system)
        self.create_nlp_in_out_mapping(system)
        self._create_solver(solver_factory)

    def _aladin_modifications(self, system: CasadiALADINSystem):
        # inject aladin coupling term
        rho = self.add_opt_par(system.penalty_factor)
        optvars = ca.vertcat(*self.opt_vars)
        global_coupling = ca.MX.sym("global_coupling", optvars.shape[0], 1)
        par_list = self.mpc_opt_pars.setdefault(
            aladin_datatypes.PRESCRIBED, OptParMXContainer()
        )
        par_list.var.append(global_coupling)
        coupling_cost = rho / 2 * ca.norm_2(self.opt_vars - global_coupling) ** 2
        self.objective_function += coupling_cost

        # create functions for sensitivities
        objective_gradient = ca.Function("obj_grad", self.objective_function, optvars)

        constraint_jacobian = ca.Function("cons_jac", [])

    def create_nlp_in_out_mapping(self, system: CasadiALADINSystem):
        """
        Function creating mapping functions between the MPC variables ordered
        by type (as defined in `declare_quantities` and the raw input/output
        vector of the CasADi NLP.
        """
        # add penalty parameter
        rho = self.add_opt_par(system.penalty_factor)

        # Concatenate nlp variables to CasADi MX vectors
        self.opt_vars = ca.vertcat(*self.opt_vars)
        self.constraints = ca.vertcat(*self.constraints)
        self.opt_pars = ca.vertcat(*self.opt_pars)
        initial_guess = ca.vertcat(*self.initial_guess)
        opt_vars_lb = ca.vertcat(*self.opt_vars_lb)
        opt_vars_ub = ca.vertcat(*self.opt_vars_ub)
        constraints_lb = ca.vertcat(*self.constraints_lb)
        constraints_ub = ca.vertcat(*self.constraints_ub)

        # create empty lists to store all nlp inputs and outputs
        mpc_inputs = []
        aladin_inputs_dict = {}
        mpc_input_denotations = []
        mpc_outputs = []
        mpc_output_denotations = []

        # Concatenate mpc outputs and their bounds to CasADi MX matrices
        for denotation, opt_var in self.mpc_opt_vars.items():
            # mpc opt vars
            var = opt_var.var
            var = ca.horzcat(*var)
            mpc_outputs.append(var)
            mpc_output_denotations.append(denotation)

            # their bounds and guess
            lb = ca.horzcat(*opt_var.lb)
            ub = ca.horzcat(*opt_var.ub)
            guess = ca.horzcat(*opt_var.guess)
            mpc_inputs.extend([lb, ub, guess])
            mpc_input_denotations.extend(
                [f"lb_{denotation}", f"ub_{denotation}", GUESS_PREFIX + denotation]
            )

        # Concatenate mpc inputs to CasADi MX matrices
        for denotation, opt_par in self.mpc_opt_pars.items():
            var = opt_par.var
            var = ca.horzcat(*var)
            mpc_inputs.append(var)
            aladin_inputs_dict[denotation] = var
            mpc_input_denotations.append(denotation)

        # inject aladin coupling term
        global_coupling = ca.MX.sym("global_coupling", self.opt_vars.shape[0], 1)
        coupling_cost = (rho / 2) * ca.sumsqr(self.opt_vars - global_coupling)
        original_objective = self.objective_function
        self.objective_function += coupling_cost
        mpc_inputs.append(global_coupling)
        self.opt_pars = ca.vertcat(self.opt_pars, global_coupling)
        mpc_input_denotations.append(aladin_datatypes.PRESCRIBED)

        # nlp inputs
        nlp_inputs = [
            self.opt_pars,
            initial_guess,
            opt_vars_lb,
            opt_vars_ub,
            constraints_lb,
            constraints_ub,
        ]
        nlp_input_denotations = ["p", "x0", "lbx", "ubx", "lbg", "ubg"]

        # Mapping function that rearranges the variables for input into the NLP
        self._mpc_inputs_to_nlp_inputs = ca.Function(
            "mpc_inputs_to_nlp_inputs",
            mpc_inputs,
            nlp_inputs,
            mpc_input_denotations,
            nlp_input_denotations,
        )

        # Mapping function that rearranges the output of the nlp and sorts
        # by denotation
        self._nlp_outputs_to_mpc_outputs = ca.Function(
            "nlp_outputs_to_mpc_outputs",
            [self.opt_vars],
            mpc_outputs,
            ["vars_at_optimum"],
            mpc_output_denotations,
        )

        # create function to extract constraints
        self._get_constraints = ca.Function(
            "constraints",
            [self.opt_vars, self.opt_pars],
            [self.constraints],
            ["opt_vars", "opt_pars"],
            ["constraints"],
        )

        # create functions for sensitivities
        # todo in some papers, gradient here includes dot product of jacobian error times lam_g
        objective_gradient = ca.gradient(original_objective, self.opt_vars)
        aladin_inputs_dict["multipliers"] = ca.MX.zeros(
            *aladin_inputs_dict["multipliers"].shape
        )
        substituted_opt_pars = self._mpc_inputs_to_nlp_inputs(**aladin_inputs_dict)["p"]
        objective_gradient_function = ca.Function(
            "obj_grad", [self.opt_vars, self.opt_pars], [objective_gradient]
        )
        objective_gradient_lambda_zero = objective_gradient_function(
            self.opt_vars, substituted_opt_pars
        )
        constraint_jacobian = ca.jacobian(self.constraints, self.opt_vars)
        # todo somehow find out how to check for inactive inequality constraints
        constraint_multipliers = ca.MX.sym("cons_mult", self.constraints.shape[0], 1)
        hessian = ca.hessian(
            original_objective + constraint_multipliers.T @ self.constraints,
            self.opt_vars,
        )[0]
        # todo find out, if the box constraints should be included, or if we use 1s there, and why it is ones. Box constraints are not included in Engelmann implementation, but why
        hessian_function = ca.Function(
            "hessian_func",
            [self.opt_vars, self.opt_pars, constraint_multipliers],
            [hessian],
        )
        hessian_lambda_zero = hessian_function(
            self.opt_vars, substituted_opt_pars, constraint_multipliers
        )
        self._sensitivities_func = ca.Function(
            "obj_grad",
            [self.opt_vars, self.opt_pars, constraint_multipliers],
            [
                self.opt_vars,
                objective_gradient_lambda_zero,
                constraint_jacobian,
                hessian_lambda_zero,
            ],
            ["opt_vars", "opt_pars", "local_constraint_multipliers"],
            ["x", "g", "J", "H"],
        )

        matrix, col_index, full_grid, var_grids = self._create_result_format(system)
        self._result_map = ca.Function(
            "result_map", mpc_inputs, [matrix], mpc_input_denotations, ["result"]
        )

        def make_results_view(result_matrix: ca.DM, stats: dict) -> Results:
            return Results(
                matrix=result_matrix,
                columns=col_index,
                grid=full_grid,
                variable_grid_indices=var_grids,
                stats=stats,
            )

        self._create_results = make_results_view


class ALADINCollocation(ALADINDiscretization, DirectCollocation):
    def _discretize(self, sys: CasadiALADINSystem):
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

            # New parameter for inputs
            dk = self.add_opt_par(sys.non_controlled_inputs)

            # perform inner collocation loop
            # perform inner collocation loop
            opt_vars_inside_inner = [
                sys.algebraics,
                sys.outputs,
                sys.local_couplings,
            ]
            opt_pars_inside_inner = [
                sys.multipliers,
            ]
            constant_over_inner = {
                sys.controls: uk,
                sys.non_controlled_inputs: dk,
                sys.model_parameters: const_par,
                sys.penalty_factor: rho,
            }
            xk_end = self._collocation_inner_loop(
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
            self.add_constraint(xk_end - xk)

    def initialize(self, system: CasadiALADINSystem, solver_factory: SolverFactory):
        """Initializes the trajectory optimization problem, creating all symbolic
        variables of the OCP, the mapping function and the numerical solver."""
        self._construct_stage_function(system)
        super().initialize(system=system, solver_factory=solver_factory)


class ALADINMultipleShooting(ALADINDiscretization, ADMMMultipleShooting):
    def _discretize(self, sys: CasadiALADINSystem):

        # shorthands
        n = self.options.prediction_horizon
        ts = self.options.time_step
        opt_integrator = self._create_ode(
            sys, {"t0": 0, "tf": ts}, self.options.integrator
        )

        # Initial State
        x0 = self.add_opt_par(sys.initial_state)
        xk = self.add_opt_var(sys.states, lb=x0, ub=x0, guess=x0)
        uk = self.add_opt_par(sys.last_control)

        # Parameters that are constant over the horizon
        const_par = self.add_opt_par(sys.model_parameters)
        du_weights = self.add_opt_par(sys.r_del_u)
        rho = self.add_opt_par(sys.penalty_factor)

        while self.k < n:
            u_prev = uk
            uk = self.add_opt_var(sys.controls)
            # penalty for control change between time steps
            self.objective_function += ts * ca.dot(du_weights, (u_prev - uk) ** 2)
            dk = self.add_opt_par(sys.non_controlled_inputs)
            zk = self.add_opt_var(sys.algebraics)
            yk = self.add_opt_var(sys.outputs)
            v_localk = self.add_opt_var(sys.local_couplings)
            lamk = self.add_opt_par(sys.multipliers)

            # get stage_function
            stage_arguments = {
                # variables
                sys.states.name: xk,
                sys.algebraics.name: zk,
                sys.local_couplings.name: v_localk,
                sys.outputs.name: yk,
                # parameters
                sys.multipliers.name: lamk,
                sys.controls.name: uk,
                sys.non_controlled_inputs.name: dk,
                sys.model_parameters.name: const_par,
                sys.penalty_factor.name: rho,
            }
            stage = self._stage_function(**stage_arguments)

            self.add_constraint(
                stage["model_constraints"],
                lb=stage["lb_model_constraints"],
                ub=stage["ub_model_constraints"],
            )
            # TODO: add casadi 3.6 support
            fk = opt_integrator(
                x0=xk,
                p=ca.vertcat(uk, v_localk, dk, const_par),
            )
            xk_end = fk["xf"]
            # calculate model constraint
            self.k += 1
            self.pred_time = ts * self.k
            xk = self.add_opt_var(sys.states)
            self.add_constraint(xk_end - xk)
            # TODO use stage cost from the integrator, i.e. fk["qf"] and integrate
            #  everything into the stage function
            self.objective_function += stage["cost_function"] * ts

    def initialize(self, system: CasadiALADINSystem, solver_factory: SolverFactory):
        """Initializes the trajectory optimization problem, creating all symbolic
        variables of the OCP, the mapping function and the numerical solver."""
        self._construct_stage_function(system)
        super().initialize(system=system, solver_factory=solver_factory)


def regularize_h(hessian, reg_param: float):
    # Eigenvalue decomposition of the Hessian
    e, V = np.linalg.eig(hessian)

    # did_reg = False
    # if np.min(e) < 0:
    #     did_reg = True

    # Take absolute value of eigenvalues
    e = np.abs(e)

    # Modify zero and too small eigenvalues (regularization)
    e[e <= reg_param] = reg_param

    # Regularization for small stepsize
    H_reg = np.real(V @ np.diag(e) @ V.T)

    return H_reg


class CasADiALADINBackend(CasADiADMMBackend):
    discretization: ALADINDiscretization

    system_type = CasadiALADINSystem
    discretization_types = {
        DiscretizationMethod.collocation: ALADINCollocation,
        DiscretizationMethod.multiple_shooting: ALADINMultipleShooting,
    }

    def get_aladin_registration(
        self, current_vars: Dict[str, MPCVariable], now: float
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        mpc_inputs = self._get_current_mpc_inputs(agent_variables=current_vars, now=now)
        return self.discretization.get_aladin_registration(
            mpc_inputs, var_ref=self.var_ref
        )

    def shift_opt_var(self) -> Optional[List[float]]:
        return self.discretization.shift_opt_var()

    def set_global_variable(self, var: List[float]):
        var_ = ca.vertcat(var)
        self.discretization.global_variable = var_

    def get_sensitivities(self) -> aladin_datatypes.AgentToCoordinator:
        return aladin_datatypes.AgentToCoordinator(
            x=self.discretization.sensitivities_result["x"].toarray(),
            g=self.discretization.sensitivities_result["x"].toarray(),
            J=self.discretization.sensitivities_result["J"],
            H=self.discretization.sensitivities_result["H"],
        )
