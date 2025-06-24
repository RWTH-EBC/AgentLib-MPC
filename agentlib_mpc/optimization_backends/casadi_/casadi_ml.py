import casadi as ca
from typing import Dict
import collections

from agentlib_mpc.models.casadi_model import CasadiParameter

from agentlib_mpc.data_structures.casadi_utils import (
    LB_PREFIX,
    UB_PREFIX,
    DiscretizationMethod,
    SolverFactory,
    Constraint,
)
from agentlib_mpc.data_structures.ml_model_datatypes import name_with_lag
from agentlib_mpc.data_structures.mpc_datamodels import (
    VariableReference,
)
from agentlib_mpc.models.casadi_ml_model import CasadiMLModel
from agentlib_mpc.optimization_backends.casadi_.core.VariableGroup import (
    OptimizationQuantity,
    OptimizationVariable,
    OptimizationParameter,
)
from agentlib_mpc.optimization_backends.casadi_.basic import (
    MultipleShooting,
    CasADiBaseBackend,
)
from agentlib_mpc.optimization_backends.casadi_.full import FullSystem


class CasadiMLSystem(FullSystem):
    # multiple possibilities of using the MLModel
    # stage function for neural networks
    model: CasadiMLModel
    lags_dict: dict[str, int]
    sim_step: ca.Function

    def initialize(self, model: CasadiMLModel, var_ref: VariableReference):
        # define variables
        self.states = OptimizationVariable.declare(
            denotation="state",
            variables=model.get_states(var_ref.states),
            ref_list=var_ref.states,
            assert_complete=True,
        )
        self.controls = OptimizationVariable.declare(
            denotation="control",
            variables=model.get_inputs(var_ref.controls),
            ref_list=var_ref.controls,
            assert_complete=True,
        )
        self.algebraics = OptimizationVariable.declare(
            denotation="z",
            variables=model.auxiliaries,
            ref_list=[],
        )
        self.outputs = OptimizationVariable.declare(
            denotation="y",
            variables=model.outputs,
            ref_list=var_ref.outputs,
        )

        # define parameters
        self.non_controlled_inputs = OptimizationParameter.declare(
            denotation="d",
            variables=model.get_inputs(var_ref.inputs),
            ref_list=var_ref.inputs,
            assert_complete=True,
        )
        self.model_parameters = OptimizationParameter.declare(
            denotation="parameter",
            variables=model.parameters,
            ref_list=var_ref.parameters,
        )
        self.initial_state = OptimizationParameter.declare(
            denotation="initial_state",
            variables=model.get_states(var_ref.states),
            ref_list=var_ref.states,
            use_in_stage_function=False,
            assert_complete=True,
        )
        self.last_control = OptimizationParameter.declare(
            denotation="initial_control",
            variables=model.get_inputs(var_ref.controls),
            ref_list=var_ref.controls,
            use_in_stage_function=False,
            assert_complete=True,
        )
        self.cost_function = model.cost_func
        self.model_constraints = Constraint(
            function=ca.vertcat(*[c.function for c in model.get_constraints()]),
            lb=ca.vertcat(*[c.lb for c in model.get_constraints()]),
            ub=ca.vertcat(*[c.ub for c in model.get_constraints()]),
        )
        self.sim_step = model.make_predict_function_for_mpc()
        self._model = model
        self.lags_dict: dict[str, int] = model.lags_dict
        self.time = model.time

    @property
    def max_lag(self) -> int:
        if self.lags_dict:
            return max(self.lags_dict.values())
        else:
            # if there is no bb variable, we have a lag of 1
            return 1

    def all_system_quantities(self) -> dict[str, OptimizationQuantity]:
        return {var.name: var for var in self.quantities}

class MultipleShooting_ML(MultipleShooting):
    max_lag: int

    def _discretize(self, sys: CasadiMLSystem):
        n = self.options.prediction_horizon
        ts = self.options.time_step
        const_par = self.add_opt_par(sys.model_parameters)

        try:
            delta_u_objectives = sys.model.objective.get_delta_u_objectives()
        except (AttributeError, Exception) as e:
            self.logger.warning(f"Failed to get delta_u_objectives: {str(e)}")

        control_map = {}
        for i, control_name in enumerate(sys.controls.ref_names):
            control_map[control_name] = i

        pre_grid_states = [ts * i for i in range(-sys.max_lag + 1, 1)]
        inputs_lag = min(-2, -sys.max_lag)  # at least -2, to consider last control
        pre_grid_inputs = [ts * i for i in range(inputs_lag + 1, 0)]
        prediction_grid = [ts * i for i in range(0, n)]

        # sort for debugging purposes
        full_grid = sorted(
            list(set(prediction_grid + pre_grid_inputs + pre_grid_states))
        )

        # dict[time, dict[denotation, ca.MX]]
        mx_dict: dict[float, dict[str, ca.MX]] = {time: {} for time in full_grid}

        # add past state variables
        for time in pre_grid_states:
            self.pred_time = time
            x_past = self.add_opt_par(sys.initial_state)
            # add past states as optimization variables with fixed values so they can
            # be accessed by the first few steps, when there are lags
            mx_dict[time][sys.states.name] = self.add_opt_var(
                sys.states, lb=x_past, ub=x_past, guess=x_past
            )
            mx_dict[time][sys.initial_state.name] = x_past

        # add past inputs
        for time in pre_grid_inputs:
            self.pred_time = time
            d = sys.non_controlled_inputs
            mx_dict[time][d.name] = self.add_opt_par(d)
            u_past = self.add_opt_par(sys.last_control)
            mx_dict[time][sys.controls.name] = self.add_opt_var(
                sys.controls, lb=u_past, ub=u_past, guess=u_past
            )
            mx_dict[time][sys.last_control.name] = u_past

        # add all variables over future grid
        for time in prediction_grid:
            self.pred_time = time
            mx_dict[time][sys.controls.name] = self.add_opt_var(sys.controls)
            mx_dict[time][sys.non_controlled_inputs.name] = self.add_opt_par(
                sys.non_controlled_inputs
            )
            mx_dict[time][sys.algebraics.name] = self.add_opt_var(sys.algebraics)
            mx_dict[time][sys.outputs.name] = self.add_opt_var(sys.outputs)

        # create the state grid
        # x0 will always be the state at time 0 since the loop it is defined in starts
        # in the past and finishes at 0
        self.pred_time = 0
        for time in prediction_grid[1:]:
            self.pred_time = time
            mx_dict[time][sys.states.name] = self.add_opt_var(sys.states)
        self.pred_time += ts
        mx_dict[self.pred_time] = {sys.states.name: self.add_opt_var(sys.states)}

        all_quantities = sys.all_system_quantities()
        # add constraints and create the objective function for all stages
        for time in prediction_grid:
            stage_mx = mx_dict[time]

            # add penalty on control change between intervals
            u_prev = mx_dict[time - ts][sys.controls.name]
            uk = stage_mx[sys.controls.name]
            for delta_obj in delta_u_objectives:
                control_name = delta_obj.get_control_name()
                if control_name in control_map:
                    idx = control_map[control_name]
                    control_prev = u_prev[idx]
                    control_curr = uk[idx]
                    delta = control_curr - control_prev

                    if hasattr(delta_obj.weight, 'sym'):
                        param_found = False
                        for i, param_name in enumerate(sys.model_parameters.ref_names):
                            if param_name == delta_obj.weight.name:
                                weight_value = const_par[i]
                                param_found = True
                                break

                        if not param_found:
                            raise ValueError(f"Parameter {delta_obj.weight.name} not found in model parameters")
                    else:
                        weight_value = delta_obj.weight

                    self.objective_function += weight_value ** 2 * delta ** 2

            # get stage arguments from current time step
            stage_arguments = {
                # variables
                sys.states.name: stage_mx[sys.states.name],
                sys.algebraics.name: stage_mx[sys.algebraics.name],
                sys.outputs.name: stage_mx[sys.outputs.name],
                # parameters
                sys.controls.name: stage_mx[sys.controls.name],
                sys.non_controlled_inputs.name: stage_mx[
                    sys.non_controlled_inputs.name
                ],
                sys.model_parameters.name: const_par,
                "__time": time,
            }

            # collect stage arguments for lagged variables
            for lag, denotation_dict in self._lagged_input_names.items():
                for denotation, var_names in denotation_dict.items():
                    l_name = name_with_lag(denotation, lag)
                    mx_list = []
                    for v_name in var_names:
                        # add only the singular variable which has a lag on this level
                        # to the stage arguments
                        index = all_quantities[denotation].full_names.index(v_name)
                        mx_list.append(mx_dict[time - lag * ts][denotation][index])
                    stage_arguments[l_name] = ca.vertcat(*mx_list)

            # evaluate a stage, add path constraints, multiple shooting constraints
            # and add to the objective function
            stage_result = self._stage_function(**stage_arguments)
            self.add_constraint(
                stage_result["model_constraints"],
                lb=stage_result["lb_model_constraints"],
                ub=stage_result["ub_model_constraints"],
            )
            self.add_constraint(
                stage_result["next_states"] - mx_dict[time + ts][sys.states.name]
            )
            self.objective_function += stage_result["cost_function"] * ts

    def initialize(self, system: CasadiMLSystem, solver_factory: SolverFactory):
        """Initializes the trajectory optimization problem, creating all symbolic
        variables of the OCP, the mapping function and the numerical solver."""
        self._construct_stage_function(system)
        super().initialize(system=system, solver_factory=solver_factory)

    def _construct_stage_function(self, system: CasadiMLSystem):
        """
        Combine information from the model and the var_ref to create CasADi
        functions which describe the system dynamics and constraints at each
        stage of the optimization problem. Sets the stage function. It has
        all mpc variables as inputs, sorted by denotation (declared in
        self.declare_quantities) and outputs ode, cost function and 3 outputs
        per constraint (constraint, lb_constraint, ub_constraint).

        In the basic case, it has the form:
        CasadiFunction: ['x', 'z', 'u', 'y', 'd', 'p'] ->
            ['ode', 'cost_function', 'model_constraints',
            'ub_model_constraints', 'lb_model_constraints']

        Args:
            system
        """
        all_system_quantities = system.all_system_quantities()
        constraints = {"model_constraints": system.model_constraints}

        inputs = [
            q.full_symbolic
            for q in all_system_quantities.values()
            if q.use_in_stage_function
        ]
        inputs.append(system.time)
        input_denotations = [
            q.name
            for denotation, q in all_system_quantities.items()
            if q.use_in_stage_function
        ]
        input_denotations.append("__time")

        # aggregate constraints
        constraints_func = [c.function for c in constraints.values()]
        constraints_lb = [c.lb for c in constraints.values()]
        constraints_ub = [c.ub for c in constraints.values()]
        constraint_denotations = list(constraints.keys())
        constraint_lb_denotations = [LB_PREFIX + k for k in constraints]
        constraint_ub_denotations = [UB_PREFIX + k for k in constraints]

        # create a dictionary which holds all the inputs for the sim step of the model
        all_input_variables = {}
        lagged_inputs: dict[int, dict[str, ca.MX]] = {}
        # dict[lag, dict[denotation, list[var_name]]]
        lagged_input_names: dict[int, dict[str, list[str]]] = {}
        for q_name, q_obj in all_system_quantities.items():
            if not q_obj.use_in_stage_function:
                continue
            for v_id, v_name in enumerate(q_obj.full_names):
                all_input_variables[v_name] = q_obj.full_symbolic[v_id]
                lag = system.lags_dict.get(v_name, 1)

                # if lag exists, we have to create and organize new variables
                for j in range(1, lag):
                    # create an MX variable for this lag
                    l_name = name_with_lag(v_name, j)
                    new_lag_var = system.model.lags_mx_store[l_name]
                    all_input_variables[l_name] = new_lag_var

                    # add the mx variable to its lag time and denotation
                    lagged_inputs_j = lagged_inputs.setdefault(j, {})
                    lv_mx = lagged_inputs_j.setdefault(q_name, ca.DM([]))
                    lagged_inputs[j][q_name] = ca.vertcat(lv_mx, new_lag_var)

                    # keep track of the variable names that were added
                    lagged_input_names_j = lagged_input_names.setdefault(j, {})
                    lv_names = lagged_input_names_j.setdefault(q_name, [])
                    lv_names.append(v_name)

        self._lagged_input_names = lagged_input_names
        flat_lagged_inputs = {
            f"{den}_{i}": mx
            for i, subdict in lagged_inputs.items()
            for den, mx in subdict.items()
        }

        all_outputs = system.sim_step(**all_input_variables)
        state_output_it = (all_outputs[s_name] for s_name in system.states.full_names)
        state_output = ca.vertcat(*state_output_it)

        # aggregate outputs
        outputs = [
            state_output,
            system.cost_function,
            *constraints_func,
            *constraints_lb,
            *constraints_ub,
        ]
        output_denotations = [
            "next_states",
            "cost_function",
            *constraint_denotations,
            *constraint_lb_denotations,
            *constraint_ub_denotations,
        ]

        # function describing system dynamics and cost function
        self._stage_function = ca.Function(
            "f",
            inputs + list(flat_lagged_inputs.values()),
            outputs,
            # input handles to make kwarg use possible and to debug
            input_denotations + list(flat_lagged_inputs),
            # output handles to make kwarg use possible and to debug
            output_denotations,
        )

    def _create_lag_structure_for_denotations(self, system: CasadiMLSystem):
        all_system_quantities = self.all_system_quantities(system)
        all_input_variables = {}
        lagged_inputs: dict[int, dict[str, ca.MX]] = {}
        # dict[lag, dict[denotation, list[var_name]]]
        lagged_input_names: dict[int, dict[str, list[str]]] = {}
        for q_name, q_obj in all_system_quantities.items():
            if not q_obj.use_in_stage_function:
                continue
            for v_id, v_name in enumerate(q_obj.full_names):
                all_input_variables[v_name] = q_obj.full_symbolic[v_id]
                lag = system.lags_dict.get(v_name, 1)

                # if lag exists, we have to create and organize new variables
                for j in range(1, lag):
                    # create an MX variable for this lag
                    l_name = name_with_lag(v_name, j)
                    new_lag_var = ca.MX.sym(l_name)
                    all_input_variables[l_name] = new_lag_var

                    # add the mx variable to its lag time and denotation
                    lagged_inputs_j = lagged_inputs.setdefault(j, {})
                    lv_mx = lagged_inputs_j.setdefault(q_name, ca.DM([]))
                    lagged_inputs[j][q_name] = ca.vertcat(lv_mx, new_lag_var)

                    # keep track of the variable names that were added
                    lagged_input_names_j = lagged_input_names.setdefault(j, {})
                    lv_names = lagged_input_names_j.setdefault(q_name, [])
                    lv_names.append(v_name)

        return
    

class CasADiBBBackend(CasADiBaseBackend):
    """
    Class doing optimization with a MLModel.
    """

    system_type = CasadiMLSystem
    discretization_types = {DiscretizationMethod.multiple_shooting: MultipleShooting_ML}
    system: CasadiMLSystem
    # a dictionary of collections of the variable lags
    lag_collection: Dict[str, collections.deque] = {}
    max_lag: int

    def get_lags_per_variable(self) -> dict[str, float]:
        """Returns the name of variables which include lags and their lag. The MPC
        module can use this information to save relevant past data of lagged
        variables"""
        ts = self.config.discretization_options.time_step
        return {
            name: (lag - 1) * ts
            for name, lag in self.system.lags_dict.items()
            if name in self.var_ref
        }
