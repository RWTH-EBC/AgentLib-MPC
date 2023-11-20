import casadi as ca
from agentlib.core.errors import ConfigurationError

from agentlib_mpc.data_structures.ann_datatypes import name_with_lag
from agentlib_mpc.data_structures.casadi_utils import Constraint
from agentlib_mpc.data_structures.mpc_datamodels import FullVariableReference
from agentlib_mpc.models.casadi_model import CasadiParameter
from agentlib_mpc.models.casadi_model_ann import CasadiANNModel
from agentlib_mpc.optimization_backends.casadi_.basic import DirectCollocation
from agentlib_mpc.optimization_backends.casadi_.casadi_nn import (
    CasadiNNSystem,
    MultipleShooting_NN,
)
from agentlib_mpc.optimization_backends.casadi_.core.VariableGroup import (
    OptimizationVariable,
    OptimizationParameter,
)


# todo this backend is in development


class CasadiHybridSystem(CasadiNNSystem):
    states_ode: OptimizationVariable
    states_ml: OptimizationVariable

    def initialize(self, model: CasadiANNModel, var_ref: FullVariableReference):
        differential_names = [var.name for var in model.differentials]
        ml_names = [var.name for var in model.bb_states]
        if set(differential_names).union(ml_names) != set(var_ref.states):
            raise ConfigurationError("States are wrong")

        # define variables
        self.states_ode = OptimizationVariable.declare(
            denotation="state_ode",
            variables=model.differentials,
            ref_list=differential_names,
            assert_complete=True,
        )
        self.states_ml = OptimizationVariable.declare(
            denotation="state_ml",
            variables=model.bb_states,
            ref_list=ml_names,
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
            variables=model.algebraics + model.auxiliaries,
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
            denotation="initial_state",  # append the 0 as a convention to get initial guess
            variables=model.get_states(var_ref.states),
            ref_list=var_ref.states,
            use_in_stage_function=False,
            assert_complete=True,
        )
        self.last_control = OptimizationParameter.declare(
            denotation="initial_control",  # append the 0 as a convention to get initial guess
            variables=model.get_inputs(var_ref.controls),
            ref_list=var_ref.controls,
            use_in_stage_function=False,
            assert_complete=True,
        )
        self.r_del_u = OptimizationParameter.declare(
            denotation="r_del_u",
            variables=[CasadiParameter(name=r_del_u) for r_del_u in var_ref.r_del_u],
            ref_list=var_ref.r_del_u,
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
        self.model = model
        self.lags_dict: dict[str, int] = model.lags_dict


class HybridDiscretization(MultipleShooting_NN, DirectCollocation):
    def _discretize(self, sys: CasadiHybridSystem):
        n = self.options.prediction_horizon
        ts = self.options.time_step
        collocation_matrices = self._collocation_polynomial()

        const_par = self.add_opt_par(sys.model_parameters)
        du_weights = self.add_opt_par(sys.r_del_u)

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
            x_past = self.add_opt_par(sys.initial_state_ode)
            # add past states as optimization variables with fixed values so they can
            # be accessed by the first few steps, when there are lags
            mx_dict[time][sys.states_ode.name] = self.add_opt_var(
                sys.states_ode, lb=x_past, ub=x_past, guess=x_past
            )
            mx_dict[time][sys.initial_state_ode.name] = x_past
            x_past = self.add_opt_par(sys.initial_state_ml)
            # add past states as optimization variables with fixed values so they can
            # be accessed by the first few steps, when there are lags
            mx_dict[time][sys.states_ml.name] = self.add_opt_var(
                sys.states_ml, lb=x_past, ub=x_past, guess=x_past
            )
            mx_dict[time][sys.initial_state_ml.name] = x_past

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
        x0 = mx_dict[0][sys.initial_state.name]
        mx_dict[0][sys.states.name] = self.add_opt_var(
            sys.states, lb=x0, ub=x0, guess=x0
        )
        for time in prediction_grid[1:]:
            self.pred_time = time
            mx_dict[time][sys.states.name] = self.add_opt_var(sys.states)
        self.pred_time += ts
        mx_dict[self.pred_time] = {sys.states.name: self.add_opt_var(sys.states)}

        # control of last time step
        mx_dict[0 - ts][sys.controls.name] = self.add_opt_par(sys.last_control)

        all_quantities = sys.all_system_quantities()
        # add constraints and create the objective function for all stages
        for time in prediction_grid:
            stage_mx = mx_dict[time]

            # add penalty on control change between intervals
            u_prev = mx_dict[time - ts][sys.controls.name]
            uk = stage_mx[sys.controls.name]
            self.objective_function += ts * ca.dot(du_weights, (u_prev - uk) ** 2)

            # get stage arguments from current time step
            stage_arguments = {
                # variables
                sys.states_ode.name: stage_mx[sys.states_ode.name],
                sys.states_ml.name: stage_mx[sys.states_ml.name],
                sys.algebraics.name: stage_mx[sys.algebraics.name],
                sys.outputs.name: stage_mx[sys.outputs.name],
                # parameters
                sys.controls.name: uk,
                sys.non_controlled_inputs.name: stage_mx[
                    sys.non_controlled_inputs.name
                ],
                sys.model_parameters.name: const_par,
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
            stage_result_ms = self._stage_function_ms(**stage_arguments)["next_states"]
            ms_states_on_collocation_points = []
            for interpolation_factor in collocation_matrices.root[1:]:
                inter = (
                    stage_mx[sys.states_ml.name] * (1 - interpolation_factor)
                    + stage_result_ms * interpolation_factor
                )
                ms_states_on_collocation_points.append(inter)

            self.add_constraint(
                stage_result_ms - mx_dict[time + ts][sys.states_ml.name]
            )

            # perform inner collocation loop
            opt_vars_inside_inner = [sys.algebraics, sys.outputs]
            opt_pars_inside_inner = []

            constant_over_inner = {
                sys.controls: uk,
                sys.non_controlled_inputs: stage_mx[sys.non_controlled_inputs.name],
                sys.model_parameters: const_par,
            }
            pre_defined_over_inner = {sys.states_ml: ms_states_on_collocation_points}

            stage_result_collocation = self._collocation_inner_loop(
                collocation=collocation_matrices,
                state_at_beginning=mx_dict[time][sys.states_ode.name],
                states=sys.states,
                opt_vars=opt_vars_inside_inner,
                opt_pars=opt_pars_inside_inner,
                const=constant_over_inner,
                const_trajectory=pre_defined_over_inner,
            )
