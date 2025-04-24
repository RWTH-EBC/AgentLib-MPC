import logging
from typing import Dict, Tuple, Optional, List

import casadi as ca
import numpy as np
import pydantic

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
from agentlib_mpc.optimization_backends.casadi_.core.casadi_backend import (
    CasadiBackendConfig,
)
from agentlib_mpc.optimization_backends.casadi_.core.discretization import (
    Discretization,
    Results,
)
from agentlib_mpc.optimization_backends.casadi_.full import FullSystem
from agentlib_mpc.utils.plotting.mpc_debugging.aladin_debugging import (
    launch_visualization_app,
    _prepare_visualization_data,
)


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


class ALADINCasadiDiscretizationOptions(CasadiDiscretizationOptions):
    regularization_parameter: float = 0.001
    activation_margin: float = 0.001


class ALADINDiscretization(Discretization):
    options: ALADINCasadiDiscretizationOptions

    def __init__(self, options: ALADINCasadiDiscretizationOptions):
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
        result = self._process_solution(
            inputs=mpc_inputs,
            outputs=mpc_output,
            objective_value=0,
        )
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
                current_optimum[den][:, shift_by:], current_optimum[den][:, -shift_by:]
            )
        return (
            self._mpc_inputs_to_nlp_inputs(**current_optimum)["x0"].toarray().tolist()
        )

    def solve(self, mpc_inputs: MPCInputs) -> Results:
        """
        Solves the discretized trajectory optimization problem.
        ... (rest of docstring) ...
        """
        # todo get these from coordinator
        regularization_parameter = self.options.regularization_parameter
        activation_margin = self.options.activation_margin

        # collect and format inputs
        guesses = self._determine_initial_guess(mpc_inputs)
        if self.global_variable is None:
            # Handle case where global_variable might not be set yet (e.g., first iteration)
            # You might need to initialize it to zeros or based on guess
            logging.warning(
                "ALADINDiscretization.global_variable is None. Using zeros."
            )
            # Determine expected shape - requires knowing the flattened opt_vars size
            # This is tricky before create_nlp_in_out_mapping fully defines self.opt_vars
            # For now, let's assume it might fail gracefully later or needs initialization logic
            # A placeholder:
            if hasattr(self, "opt_vars") and self.opt_vars is not None:
                expected_shape = self.opt_vars.shape
                self.global_variable = ca.DM.zeros(expected_shape)
            else:
                # Cannot determine shape yet, visualization might fail if global_variable is needed later
                logging.error(
                    "Cannot determine shape for global_variable initialization."
                )
                # Or raise an error if it's essential here
                # raise ValueError("global_variable must be set before calling solve")
                pass  # Allow to proceed, hoping it's set elsewhere or not strictly needed yet

        # Ensure global_variable is set if required by the mapping
        if aladin_datatypes.PRESCRIBED in self._mpc_inputs_to_nlp_inputs.name_in():
            if self.global_variable is None:
                raise ValueError(
                    f"{aladin_datatypes.PRESCRIBED} is required by NLP mapping but global_variable is None."
                )
            mpc_inputs[aladin_datatypes.PRESCRIBED] = self.global_variable

        mpc_inputs.update(guesses)

        try:
            nlp_inputs: dict[str, ca.DM] = self._mpc_inputs_to_nlp_inputs(**mpc_inputs)
        except Exception as e:
            logging.error(f"Error during _mpc_inputs_to_nlp_inputs mapping: {e}")
            logging.error(f"Available mpc_inputs keys: {list(mpc_inputs.keys())}")
            logging.error(
                f"Expected nlp_inputs keys: {self._mpc_inputs_to_nlp_inputs.name_in()}"
            )
            # You might want to inspect the shapes of mpc_inputs here
            raise

        # perform optimization
        try:
            nlp_output = self._optimizer(**nlp_inputs)
            debug_stats = self._optimizer.stats()
        except Exception as e:
            logging.error(f"Error during NLP optimization: {e}")
            # Optionally log nlp_inputs for debugging
            # logging.error(f"NLP Inputs: {nlp_inputs}")
            raise

        # --- Post-processing and Sensitivity ---
        try:
            # Use optimal solution x and parameters p to get constraint values
            constraints = self._get_constraints(
                opt_vars=nlp_output["x"], opt_pars=nlp_inputs["p"]
            )["constraints"]
            constraints = constraints.toarray().flatten()  # Ensure numpy vector

            active_constraints = self._get_active_constraints(
                constraints=constraints,
                lb=nlp_inputs["lbg"].toarray().flatten(),
                ub=nlp_inputs["ubg"].toarray().flatten(),
                act_margin=activation_margin,
            )  # Should return flat bool numpy array

            # --- Get Sensitivities ---
            sensitivities_result = None
            debug_sensitivities = {}  # Store intermediate debug values here

            if self._sensitivities_func:
                # Ensure lam_g is correctly shaped and zeroed for inactive constraints
                lam_g_full = nlp_output["lam_g"].toarray().flatten()
                if len(lam_g_full) != len(active_constraints):
                    logging.warning(
                        f"Mismatch length lam_g ({len(lam_g_full)}) vs active_constraints ({len(active_constraints)}). Check NLP solver output."
                    )
                    # Attempt to resize/pad if possible, or handle error
                    # For now, proceed cautiously
                    min_len_sens = min(len(lam_g_full), len(active_constraints))
                    lam_g_active = lam_g_full[:min_len_sens]
                    active_mask_sens = active_constraints[:min_len_sens]
                    lam_g_active[~active_mask_sens] = 0
                else:
                    lam_g_active = lam_g_full.copy()
                    lam_g_active[~active_constraints] = 0

                try:
                    # Call sensitivity function
                    sens_result_raw = self._sensitivities_func(
                        opt_vars=nlp_output["x"],
                        opt_pars=nlp_inputs["p"],
                        local_constraint_multipliers=ca.DM(lam_g_active),
                        # Pass potentially modified lam_g
                    )

                    # Store results, converting to numpy where appropriate
                    sensitivities_result = {
                        k: v.toarray() if isinstance(v, (ca.DM, ca.MX)) else v
                        for k, v in sens_result_raw.items()
                    }

                    # --- Regularization and Debug Info ---
                    if "H" in sensitivities_result and "J" in sensitivities_result:
                        # Store original H for comparison
                        debug_sensitivities["H_original"] = sensitivities_result[
                            "H"
                        ].copy()

                        # Filter Jacobian J to only include active constraints IF NEEDED
                        # The current implementation seems to calculate J based on *all* constraints
                        # and relies on lam_g being zero for inactive ones.
                        # If J *should* only contain active rows, filter here:
                        # sensitivities_result["J"] = sensitivities_result["J"][active_constraints, :] # Filter rows

                        # Regularize Hessian
                        sensitivities_result["H"] = regularize_h(
                            sensitivities_result["H"], regularization_parameter
                        )

                        # Store full Jacobian if calculated separately for debugging
                        if hasattr(
                            self, "_full_constraint_jacobian_func"
                        ):  # Example if you had such a func
                            full_jac = self._full_constraint_jacobian_func(
                                opt_vars=nlp_output["x"], opt_pars=nlp_inputs["p"]
                            )
                            debug_sensitivities["jacobian_all"] = full_jac.toarray()
                        elif (
                            "jacobian_all" in debug_sensitivities
                        ):  # If already stored earlier
                            pass  # Already have it
                        else:
                            # Fallback: Use the J from sensitivities if it represents all constraints
                            # This depends heavily on how _sensitivities_func is defined!
                            # Assuming J from sens_func *is* the full Jacobian before filtering:
                            debug_sensitivities["jacobian_all"] = sensitivities_result[
                                "J"
                            ].copy()  # Or load from debug_jacobian if stored earlier
                            # **IMPORTANT**: Adjust this logic based on your actual implementation
                            # If J in sensitivities_result is *already* filtered, you need another way
                            # to get the full Jacobian for the 'jacobian_all' visualization.

                        # Store other debug values if calculated
                        # debug_sensitivities['gradient_lambda_zero'] = ...

                    else:
                        logging.warning(
                            "Hessian (H) or Jacobian (J) missing from sensitivity results."
                        )

                except Exception as e:
                    raise
                    logging.error(f"Error during sensitivity calculation: {e}")
                    # Optionally log inputs to sensitivity function
                    # logging.error(f"Sensitivity Inputs: opt_vars={nlp_output['x']}, opt_pars={nlp_inputs['p']}, lam_g={lam_g_active}")
                    sensitivities_result = (
                        None  # Ensure it's None if calculation failed
                    )

            else:
                logging.warning(
                    "_sensitivities_func is not defined. Cannot calculate sensitivities."
                )

        except Exception as e:
            raise

            logging.error(
                f"Error during post-processing or sensitivity calculation: {e}"
            )
            # Ensure results are handled gracefully even if errors occurred
            sensitivities_result = None
            debug_sensitivities = {}
            # Depending on severity, you might want to re-raise or return partial results
            # For debugging, let's try to continue to visualization if possible

        # --- Prepare Data for Visualization ---
        # Ensure all required components exist before preparing data
        vis_data = None
        try:
            # Pass necessary components to the preparation function
            vis_data = _prepare_visualization_data(
                discretization_obj=self,
                nlp_inputs=nlp_inputs,
                nlp_output=nlp_output,
                constraints=constraints,  # Should be flat numpy array now
                active_constraints=active_constraints,
                # Should be flat bool numpy array
                sensitivities_result=sensitivities_result,
                # Dict with numpy arrays or None
                debug_sensitivities=debug_sensitivities,  # Dict with numpy arrays
            )
        except Exception as e:
            raise
            logging.error(f"Error preparing visualization data: {e}")
            # Log the state of inputs to _prepare_visualization_data if helpful
            # logging.error(f"Inputs to prep func: nlp_inputs keys={list(nlp_inputs.keys())}, nlp_output keys={list(nlp_output.keys())}, constraints shape={constraints.shape if constraints is not None else 'None'}, ...")

        # --- Launch Visualization App ---
        if vis_data:  # Only launch if data preparation succeeded
            try:
                logging.info("Launching visualization app...")
                launch_visualization_app(vis_data)  # Launch in a thread
                # Optional: Add a small delay or wait mechanism if needed
                # time.sleep(1)
            except Exception as e:
                raise
                logging.error(f"Failed to launch visualization app: {e}")
        else:
            logging.warning("Skipping visualization launch due to missing data.")

        # --- Format and Return Solution ---
        try:
            mpc_output = self._nlp_outputs_to_mpc_outputs(
                vars_at_optimum=nlp_output["x"]
            )
            self._remember_solution(mpc_output)
            result = self._process_solution(
                inputs=mpc_inputs,  # Pass original mpc_inputs
                outputs=mpc_output,
                objective_value=float(nlp_output["f"]),
            )
            return result
        except Exception as e:
            raise
            logging.error(f"Error formatting final results: {e}")
            # Depending on requirements, return None, raise error, or return partial result
            raise  # Re-raise for now

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
        self._system = system
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
    """Direct collocation discretization for ALADIN-based optimization.

    This class implements the direct collocation discretization scheme for ALADIN algorithm
    optimization problems. It handles discretization of continuous dynamics using
    collocation polynomials.
    """

    def _discretize(self, sys: CasadiALADINSystem):
        """Perform a direct collocation discretization for ALADIN-based optimization.

        Args:
            sys: The system to be discretized
        """
        # Setup the polynomial base
        collocation_matrices = self._collocation_polynomial()

        # Shorthands
        prediction_horizon = self.options.prediction_horizon
        timestep = self.options.time_step

        # Initial State
        initial_state = self.add_opt_par(sys.initial_state)
        current_state = self.add_opt_var(
            sys.states, lb=initial_state, ub=initial_state, guess=initial_state
        )
        previous_control = self.add_opt_par(sys.last_control)

        # Parameters that are constant over the horizon
        model_parameters = self.add_opt_par(sys.model_parameters)
        control_rate_weights = self.add_opt_par(sys.r_del_u)
        aladin_penalty = self.add_opt_par(sys.penalty_factor)

        # Formulate the NLP - loop over prediction horizon
        while self.k < prediction_horizon:
            # New NLP variable for the control
            current_control = self.add_opt_var(sys.controls)
            # Penalty for control change between time steps
            self.objective_function += timestep * ca.dot(
                control_rate_weights, (previous_control - current_control) ** 2
            )
            previous_control = current_control

            # New parameter for inputs
            disturbance = self.add_opt_par(sys.non_controlled_inputs)

            # Perform inner collocation loop
            opt_vars_inside_inner = [
                sys.algebraics,
                sys.outputs,
                sys.local_couplings,
            ]
            opt_pars_inside_inner = [
                sys.multipliers,
            ]
            constant_over_inner = {
                sys.controls: current_control,
                sys.non_controlled_inputs: disturbance,
                sys.model_parameters: model_parameters,
                sys.penalty_factor: aladin_penalty,
            }
            state_end, constraints = self._collocation_inner_loop(
                collocation=collocation_matrices,
                state_at_beginning=current_state,
                states=sys.states,
                opt_vars=opt_vars_inside_inner,
                opt_pars=opt_pars_inside_inner,
                const=constant_over_inner,
            )

            # Increment loop counter and time
            self.k += 1
            self.pred_time = timestep * self.k

            # New NLP variables at end of interval
            next_state = self.add_opt_var(sys.states)

            # Add continuity constraint
            self.add_constraint(next_state - state_end, gap_closing=True)

            # Add collocation constraints
            for constraint in constraints:
                self.add_constraint(*constraint)

            # Update current state for next interval
            current_state = next_state

    def initialize(self, system: CasadiALADINSystem, solver_factory: SolverFactory):
        """Initializes the trajectory optimization problem, creating all symbolic
        variables of the OCP, the mapping function and the numerical solver."""
        self._construct_stage_function(system)
        super().initialize(system=system, solver_factory=solver_factory)


class ALADINMultipleShooting(ALADINDiscretization, ADMMMultipleShooting):
    """Multiple shooting discretization for ALADIN-based optimization.

    This class implements the multiple shooting discretization scheme for ALADIN algorithm
    optimization problems. It handles discretization of continuous dynamics, addition of
    continuity constraints, and ALADIN-specific objective augmentation.
    """

    def _discretize(self, sys: CasadiALADINSystem):
        """Performs a multiple shooting discretization for ALADIN-based optimization.

        This method implements the multiple shooting discretization scheme for ALADIN.
        It handles:
        1. State continuity across shooting intervals
        2. Local coupling variables with their multipliers
        3. Integration of system dynamics
        4. Objective function construction including ALADIN terms

        Args:
            sys (CasadiALADINSystem): The system to be discretized, containing states,
                controls, and ALADIN-specific variables
        """
        # Extract key parameters
        prediction_horizon = self.options.prediction_horizon
        timestep = self.options.time_step
        integration_options = {"t0": 0, "tf": timestep}

        # Initialize state trajectory
        initial_state = self.add_opt_par(sys.initial_state)
        next_state = self.add_opt_var(
            sys.states, lb=initial_state, ub=initial_state, guess=initial_state
        )

        # Initialize control input
        previous_control = self.add_opt_par(sys.last_control)

        # Add time-invariant parameters
        control_rate_weights = self.add_opt_par(sys.r_del_u)
        model_parameters = self.add_opt_par(sys.model_parameters)
        aladin_penalty = self.add_opt_par(sys.penalty_factor)

        # Create system integrator
        dynamics_integrator = self._create_ode(
            sys, integration_options, self.options.integrator
        )

        # Perform multiple shooting discretization
        for k in range(prediction_horizon):
            # 1. Handle control inputs and their rate penalties
            current_state = next_state
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

            # 3. Add ALADIN coupling variables and multipliers
            local_coupling = self.add_opt_var(sys.local_couplings)
            multipliers = self.add_opt_par(sys.multipliers)

            # 4. Construct stage-wise optimization problem
            stage_arguments = {
                # variables
                sys.states.name: current_state,
                sys.algebraics.name: algebraic_vars,
                sys.local_couplings.name: local_coupling,
                sys.outputs.name: output_vars,
                # parameters
                sys.multipliers.name: multipliers,
                sys.controls.name: current_control,
                sys.non_controlled_inputs.name: disturbance,
                sys.model_parameters.name: model_parameters,
                sys.penalty_factor.name: aladin_penalty,
            }
            stage = self._stage_function(**stage_arguments)

            # 5. Integrate system dynamics
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

            # 6. Add continuity constraints
            self.k = k + 1
            self.pred_time = timestep * self.k
            next_state = self.add_opt_var(sys.states)
            self.add_constraint(next_state - integration_result["xf"], gap_closing=True)

            # 7. Add objective contribution from stage
            self.objective_function += stage["cost_function"] * timestep

            # 8. Add model constraints
            self.add_constraint(
                stage["model_constraints"],
                lb=stage["lb_model_constraints"],
                ub=stage["ub_model_constraints"],
            )

    def initialize(self, system: CasadiALADINSystem, solver_factory: SolverFactory):
        """Initializes the trajectory optimization problem, creating all symbolic
        variables of the OCP, the mapping function and the numerical solver.

        Args:
            system: The system to be discretized
            solver_factory: Factory to create the numerical solver
        """
        self._construct_stage_function(system)
        super().initialize(system=system, solver_factory=solver_factory)


def regularize_h(hessian, reg_param: float):
    """Regularize a Hessian matrix to ensure it is positive definite and symmetric.

    Args:
        hessian: The Hessian matrix to regularize
        reg_param: Regularization parameter (minimum eigenvalue)

    Returns:
        H_reg: Regularized, symmetric Hessian matrix
    """
    # Ensure matrix is symmetric before eigendecomposition
    hessian = (hessian + hessian.T) / 2

    # Eigenvalue decomposition of the Hessian
    e, V = np.linalg.eig(hessian)

    # Take absolute value of eigenvalues
    e = np.abs(e)

    # Modify zero and too small eigenvalues (regularization)
    e[e <= reg_param] = reg_param

    # Regularization for small stepsize
    H_reg = np.real(V @ np.diag(e) @ V.T)

    # Final symmetry enforcement
    H_reg = (H_reg + H_reg.T) / 2

    return H_reg


class ALADINCasadiBackendConfig(CasadiBackendConfig):
    discretization_options: ALADINCasadiDiscretizationOptions = pydantic.Field(
        default_factory=ALADINCasadiDiscretizationOptions
    )


class CasADiALADINBackend(CasADiADMMBackend):
    discretization: ALADINDiscretization

    system_type = CasadiALADINSystem
    discretization_types = {
        DiscretizationMethod.collocation: ALADINCollocation,
        DiscretizationMethod.multiple_shooting: ALADINMultipleShooting,
    }
    config_type = ALADINCasadiBackendConfig

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
