import casadi as ca
from typing import Dict, Optional
import collections

from agentlib.core.errors import ConfigurationError

from agentlib_mpc.data_structures.casadi_utils import (
    LB_PREFIX,
    UB_PREFIX,
    DiscretizationMethod,
    SolverFactory,
    Constraint,
)
from agentlib_mpc.data_structures.ml_model_datatypes import name_with_lag
from agentlib_mpc.data_structures.mpc_datamodels import (
    MPCVariable,
    VariableReference,
)
from agentlib_mpc.models.casadi_ml_model import (
    CasadiMLModel,
    RNN_NEXT_STATE_SUFFIX,
    warm_up_rnn_states,
)
from agentlib_mpc.utils import sampling
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
from agentlib_mpc.optimization_backends.casadi_.core import delta_u


class CasadiMLSystem(FullSystem):
    # multiple possibilities of using the MLModel
    # stage function for neural networks
    model: CasadiMLModel
    lags_dict: dict[str, int]
    sim_step: ca.Function

    # hidden states of recurrent (multi step) ML-models. Stays None for systems which
    # do not support recurrent models.
    rnn_states: Optional[OptimizationVariable] = None
    initial_rnn_states: Optional[OptimizationParameter] = None
    rnn_warmup_steps: int = 0
    rnn_warmup_input_names: list[str] = ()
    rnn_warmup_step: Optional[ca.Function] = None

    @property
    def has_rnn_states(self) -> bool:
        """Whether any of the ML-models of this system is a recurrent model."""
        return self.rnn_states is not None and self.rnn_states.dim > 0

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
        self.model_constraints = Constraint(
            function=ca.vertcat(*[c.function for c in model.get_constraints()]),
            lb=ca.vertcat(*[c.lb for c in model.get_constraints()]),
            ub=ca.vertcat(*[c.ub for c in model.get_constraints()]),
        )
        if model.rnn_state_variables:
            # the hidden states of recurrent models are internal to the ML-model, so
            # they are not part of the results. The quantity is only declared if there
            # are recurrent models, since an empty group would be treated as a regular
            # (but never discretized) system variable.
            self.rnn_states = OptimizationVariable.declare(
                denotation="rnn_state",
                variables=model.rnn_state_variables,
                ref_list=[],
                include_in_results=False,
            )
            # the hidden states at the start of the horizon follow from measured
            # data alone, so they enter the optimization problem as a parameter
            self.initial_rnn_states = OptimizationParameter.declare(
                denotation="initial_rnn_state",
                variables=model.rnn_state_variables,
                ref_list=[],
                use_in_stage_function=False,
            )
            self.rnn_warmup_steps = model.rnn_warmup_steps
            self.rnn_warmup_input_names = model.rnn_warmup_input_names
            self.rnn_warmup_step = model.rnn_warmup_step_function

        self.sim_step = model.make_predict_function_for_mpc()
        self.lags_dict: dict[str, int] = model.lags_dict
        self.lags_mx_store = model.lags_mx_store
        self.objective = model.objective
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

        delta_u_objectives = delta_u.get_delta_u_objectives(sys)

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

        # hidden states of recurrent models over the whole grid, including the
        # warmup in the past
        if sys.has_rnn_states:
            self._discretize_rnn_states(sys, mx_dict)

        all_quantities = sys.all_system_quantities()
        # add constraints and create the objective function for all stages
        for time in prediction_grid:
            stage_mx = mx_dict[time]

            if delta_u_objectives:
                u_prev = mx_dict[time - ts][sys.controls.name]
                uk = stage_mx[sys.controls.name]
                for delta_obj in delta_u_objectives:
                    self.objective_function += delta_u.get_objective(
                        sys, delta_obj, u_prev, uk, const_par
                    )

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
            if sys.has_rnn_states:
                stage_arguments[sys.rnn_states.name] = stage_mx[sys.rnn_states.name]

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
            if sys.has_rnn_states:
                self.add_constraint(
                    stage_result["next_rnn_states"]
                    - mx_dict[time + ts][sys.rnn_states.name]
                )
            self.objective_function += stage_result["cost_function"] * ts

    def _discretize_rnn_states(
        self, sys: CasadiMLSystem, mx_dict: dict[float, dict[str, ca.MX]]
    ):
        """Adds the hidden states of all recurrent ML-models to the optimization
        problem.

        The hidden states are treated like any other state: they are optimization variables at every
        point of the grid and are linked by multiple shooting constraints. 

        The hidden states at the start of the horizon are warmed up numerically by the backend and enter as a parameter, which
        keeps the warmup out of the optimization problem entirely.
        """
        n = self.options.prediction_horizon
        ts = self.options.time_step

        self.pred_time = 0
        initial_states = self.add_opt_par(sys.initial_rnn_states)
        mx_dict[0][sys.rnn_states.name] = self.add_opt_var(
            sys.rnn_states,
            lb=initial_states,
            ub=initial_states,
            guess=initial_states,
        )
        for step in range(1, n + 1):
            self.pred_time = ts * step
            mx_dict.setdefault(self.pred_time, {})[
                sys.rnn_states.name
            ] = self.add_opt_var(sys.rnn_states)

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
                    new_lag_var = system.lags_mx_store[l_name]
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
            system.objective.get_casadi_expression(),
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

        if system.has_rnn_states:
            rnn_state_output_it = (
                all_outputs[name + RNN_NEXT_STATE_SUFFIX]
                for name in system.rnn_states.full_names
            )
            outputs.append(ca.vertcat(*rnn_state_output_it))
            output_denotations.append("next_rnn_states")

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
    model: CasadiMLModel
    # a dictionary of collections of the variable lags
    lag_collection: Dict[str, collections.deque] = {}
    max_lag: int

    def get_lags_per_variable(self) -> dict[str, float]:
        """Returns the name of variables whose past values are needed and how far the
        past has to reach. The MPC module uses this information to save the relevant
        past data. Next to the lags of the ML-models, this covers the past values that
        recurrent models need to warm up their hidden states."""
        ts = self.config.discretization_options.time_step
        return {
            name: (lag - 1) * ts
            for name, lag in self.model.history_lags_dict.items()
            if name in self.var_ref
        }

    def _get_current_mpc_inputs(
        self, agent_variables: dict[str, MPCVariable], now: float
    ) -> dict[str, ca.DM]:
        mpc_inputs = super()._get_current_mpc_inputs(
            agent_variables=agent_variables, now=now
        )
        if self.system.has_rnn_states:
            mpc_inputs[self.system.initial_rnn_states.name] = self._warm_up_rnn_states(
                agent_variables=agent_variables, now=now
            )
        return mpc_inputs

    def _warm_up_rnn_states(
        self, agent_variables: dict[str, MPCVariable], now: float
    ) -> ca.DM:
        """Determines the hidden states of the recurrent ML-models at the start of the
        prediction horizon from the measured past."""
        ts = self.config.discretization_options.time_step
        grid = [-ts * lag for lag in range(self.system.rnn_warmup_steps, 0, -1)]

        trajectories = {}
        for name in self.system.rnn_warmup_input_names if grid else ():
            variable = agent_variables[name]
            if variable.value is None:
                raise ValueError(
                    f"Input for variable {name} is empty. It is needed to warm up the "
                    f"hidden states of a recurrent ML-model."
                )
            trajectories[name] = sampling.sample(
                trajectory=variable.value,
                grid=grid,
                current=now,
                method=getattr(variable, "interpolation_method", "linear"),
            )

        return warm_up_rnn_states(
            self.system.rnn_warmup_step,
            trajectories=trajectories,
            input_names=self.system.rnn_warmup_input_names,
            steps=self.system.rnn_warmup_steps,
            dimension=self.system.rnn_states.dim,
        )
