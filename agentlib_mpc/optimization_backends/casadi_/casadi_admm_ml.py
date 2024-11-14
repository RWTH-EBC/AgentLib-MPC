import logging
from typing import Union

import casadi as ca

from agentlib_mpc.models.casadi_model import CasadiInput, CasadiParameter
from agentlib_mpc.data_structures.casadi_utils import (
    LB_PREFIX,
    UB_PREFIX,
    DiscretizationMethod,
    Constraint,
)
from agentlib_mpc.data_structures.ml_model_datatypes import name_with_lag
from agentlib_mpc.models.casadi_ml_model import CasadiMLModel
from agentlib_mpc.optimization_backends.casadi_.casadi_ml import (
    CasadiMLSystem,
    CasADiBBBackend,
    MultipleShooting_ML,
)
from agentlib_mpc.optimization_backends.casadi_.core.VariableGroup import (
    OptimizationVariable,
    OptimizationParameter,
)
from agentlib_mpc.data_structures import admm_datatypes
from agentlib_mpc.optimization_backends.casadi_.admm import (
    ADMMMultipleShooting,
    CasadiADMMSystem,
    CasADiADMMBackend,
)

logger = logging.getLogger(__name__)


class CasadiADMMNNSystem(CasadiADMMSystem, CasadiMLSystem):
    """
    In this class, the lags are determined by the trainer alone and the lags are
    saved in the serialized MLModel so that it doesn't have to be defined in the
    model again
    """

    past_couplings: OptimizationParameter
    past_exchange: OptimizationParameter

    def initialize(
        self, model: CasadiMLModel, var_ref: admm_datatypes.VariableReference
    ):
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
        past_coup_names = [coup.lagged for coup in var_ref.couplings]
        self.past_couplings = OptimizationParameter.declare(
            denotation="past_couplings",
            variables=[CasadiInput(name=name) for name in past_coup_names],
            ref_list=past_coup_names,
            use_in_stage_function=False,
        )
        past_exchange_names = [exchange.lagged for exchange in var_ref.exchange]
        self.past_exchange = OptimizationParameter.declare(
            denotation="past_exchange",
            variables=[CasadiInput(name=name) for name in exchange_names],
            ref_list=past_exchange_names,
            use_in_stage_function=False,
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

    @property
    def variables(self) -> list[OptimizationVariable]:
        return [
            var
            for var in self.__dict__.values()
            if isinstance(var, OptimizationVariable)
        ]

    @property
    def parameters(self) -> list[OptimizationParameter]:
        return [
            var
            for var in self.__dict__.values()
            if isinstance(var, OptimizationParameter)
        ]

    @property
    def quantities(self) -> list[Union[OptimizationParameter, OptimizationVariable]]:
        return self.variables + self.parameters

    @property
    def sim_step_quantities(
        self,
    ) -> dict[str, Union[OptimizationParameter, OptimizationVariable]]:
        omit_in_blackbox_function = {
            "global_couplings",
            "multipliers",
            "average_diff",
            "exchange_multipliers",
            "rho",
        }
        return {
            var.name: var
            for var in self.quantities
            if not var.name in omit_in_blackbox_function
        }


class MultipleShootingADMMNN(ADMMMultipleShooting, MultipleShooting_ML):
    max_lag: int

    def _discretize(self, sys: CasadiADMMNNSystem):
        n = self.options.prediction_horizon
        ts = self.options.time_step

        # Parameters that are constant over the horizon
        const_par = self.add_opt_par(sys.model_parameters)
        rho = self.add_opt_par(sys.penalty_factor)
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

            # admm quantities
            past_coup = self.add_opt_par(sys.past_couplings)
            past_exch = self.add_opt_par(sys.past_exchange)
            mx_dict[time][sys.local_couplings.name] = past_coup
            mx_dict[time][sys.local_exchange.name] = past_exch
            mx_dict[time][sys.local_exchange.name] = self.add_opt_var(
                sys.local_exchange, lb=past_exch, ub=past_exch, guess=past_exch
            )
            mx_dict[time][sys.local_couplings.name] = self.add_opt_var(
                sys.local_couplings, lb=past_coup, ub=past_coup, guess=past_coup
            )

        # add all variables over future grid
        for time in prediction_grid:
            self.pred_time = time
            mx_dict[time][sys.controls.name] = self.add_opt_var(sys.controls)
            mx_dict[time][sys.non_controlled_inputs.name] = self.add_opt_par(
                sys.non_controlled_inputs
            )
            mx_dict[time][sys.algebraics.name] = self.add_opt_var(sys.algebraics)
            mx_dict[time][sys.outputs.name] = self.add_opt_var(sys.outputs)

            # admm related quantities
            mx_dict[time][sys.multipliers.name] = self.add_opt_par(sys.multipliers)
            mx_dict[time][sys.exchange_multipliers.name] = self.add_opt_par(
                sys.exchange_multipliers
            )
            mx_dict[time][sys.exchange_diff.name] = self.add_opt_par(sys.exchange_diff)
            mx_dict[time][sys.global_couplings.name] = self.add_opt_par(
                sys.global_couplings
            )
            mx_dict[time][sys.local_exchange.name] = self.add_opt_var(
                sys.local_exchange
            )
            mx_dict[time][sys.local_couplings.name] = self.add_opt_var(
                sys.local_couplings
            )

        # create the state grid
        # x0 will always be the state at time 0 since the loop it is defined in starts
        # in the past and finishes at 0
        self.pred_time = 0
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
                sys.states.name: stage_mx[sys.states.name],
                sys.algebraics.name: stage_mx[sys.algebraics.name],
                sys.outputs.name: stage_mx[sys.outputs.name],
                # parameters
                sys.controls.name: stage_mx[sys.controls.name],
                sys.non_controlled_inputs.name: stage_mx[
                    sys.non_controlled_inputs.name
                ],
                sys.model_parameters.name: const_par,
                sys.penalty_factor.name: rho,
                # admm related quantities
                sys.multipliers.name: stage_mx[sys.multipliers.name],
                sys.exchange_multipliers.name: stage_mx[sys.exchange_multipliers.name],
                sys.exchange_diff.name: stage_mx[sys.exchange_diff.name],
                sys.global_couplings.name: stage_mx[sys.global_couplings.name],
                sys.local_exchange.name: stage_mx[sys.local_exchange.name],
                sys.local_couplings.name: stage_mx[sys.local_couplings.name],
            }

            # collect stage arguments for lagged variables
            for lag, denotation_dict in self._lagged_input_names.items():
                for denotation, var_names in denotation_dict.items():
                    l_name = name_with_lag(denotation, lag)
                    mx_list = []
                    for v_name in var_names:
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

    def _construct_stage_function(self, system: CasadiADMMNNSystem):
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
        input_denotations = [
            q.name
            for denotation, q in all_system_quantities.items()
            if q.use_in_stage_function
        ]

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

        for q_name, quantity in system.sim_step_quantities.items():
            if not quantity.use_in_stage_function:
                continue

            for v_id, v_name in enumerate(quantity.full_names):
                all_input_variables[v_name] = quantity.full_symbolic[v_id]
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


class CasADiADMMBackend_NN(CasADiADMMBackend, CasADiBBBackend):
    """
    Class doing optimization with an MLModel.
    """

    system_type = CasadiADMMNNSystem
    discretization_types = {
        DiscretizationMethod.multiple_shooting: MultipleShootingADMMNN
    }
    system: CasadiADMMNNSystem
    # a dictionary of collections of the variable lags
