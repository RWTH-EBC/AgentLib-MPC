import casadi as ca

from agentlib_mpc.optimization_backends.casadi_ import basic
from agentlib_mpc.data_structures.casadi_utils import (
    DiscretizationMethod,
)
from agentlib_mpc.data_structures.mpc_datamodels import (
    VariableReference,
)
from agentlib_mpc.models.casadi_model import CasadiModel, CasadiParameter
from agentlib_mpc.optimization_backends.casadi_.core.casadi_backend import CasADiBackend
from agentlib_mpc.optimization_backends.casadi_.core.VariableGroup import (
    OptimizationParameter
)


class FullSystem(basic.BaseSystem):
    last_control: OptimizationParameter

    def __init__(self):
        super().__init__()
        self._model = None

    def initialize(self, model: CasadiModel, var_ref: VariableReference):
        super().initialize(model=model, var_ref=var_ref)

        self._model = model

        self.last_control = OptimizationParameter.declare(
            denotation="u_prev",
            variables=model.get_inputs(var_ref.controls),
            ref_list=var_ref.controls,
            use_in_stage_function=False,
            assert_complete=True,
        )

        self.time = model.time

    @property
    def model(self):
        if not hasattr(self, '_model') or self._model is None:
            raise AttributeError("Model reference not initialized yet")
        return self._model

    @model.setter
    def model(self, value):
        self._model = value


class DirectCollocation(basic.DirectCollocation):
    def _discretize(self, sys: FullSystem):
        """
        Defines a direct collocation discretization.
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

        delta_u_objectives = []
        if hasattr(sys, 'model') and hasattr(sys.model, 'objective') and sys.model.objective is not None:
            try:
                delta_u_objectives = sys.model.objective.get_delta_u_objectives()
            except (AttributeError, Exception) as e:
                self.logger.warning(f"Failed to get delta_u_objectives: {str(e)}")

        control_map = {}
        for i, control_name in enumerate(sys.controls.ref_names):
            control_map[control_name] = i

        # Formulate the NLP
        # loop over prediction horizon
        while self.k < n:
            # New NLP variable for the control
            u_prev = uk
            uk = self.add_opt_var(sys.controls)

            # penalty for control change between time steps (only for new objective system)
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

            # perform inner collocation loop
            opt_vars_inside_inner = [sys.algebraics, sys.outputs]
            opt_pars_inside_inner = [sys.non_controlled_inputs]

            constant_over_inner = {
                sys.controls: uk,
                sys.model_parameters: const_par
            }
            xk_end, constraints = self._collocation_inner_loop(
                collocation=collocation_matrices,
                state_at_beginning=xk,
                states=sys.states,
                opt_vars=opt_vars_inside_inner,
                opt_pars=opt_pars_inside_inner,
                const=constant_over_inner
            )

            # increment loop counter and time
            self.k += 1
            self.pred_time = ts * self.k

            # New NLP variable for differential state at end of interval
            xk = self.add_opt_var(sys.states)

            # Add continuity constraint
            self.add_constraint(xk - xk_end, gap_closing=True)

            # add collocation constraints later for fatrop
            for constraint in constraints:
                self.add_constraint(*constraint)


class MultipleShooting(basic.MultipleShooting):
    def _discretize(self, sys: FullSystem):
        """
        Defines a multiple shooting discretization
        """
        vars_dict = {sys.states.name: {}}
        n = self.options.prediction_horizon
        ts = self.options.time_step
        opts = {"t0": 0, "tf": ts}
        # Initial State
        x0 = self.add_opt_par(sys.initial_state)
        xk = self.add_opt_var(sys.states, lb=x0, ub=x0, guess=x0)
        vars_dict[sys.states.name][0] = xk
        uk = self.add_opt_par(sys.last_control)

        # Parameters that are constant over the horizon
        const_par = self.add_opt_par(sys.model_parameters)

        delta_u_objectives = []
        if hasattr(sys, 'model') and hasattr(sys.model, 'objective') and sys.model.objective is not None:
            try:
                delta_u_objectives = sys.model.objective.get_delta_u_objectives()
            except (AttributeError, Exception) as e:
                self.logger.warning(f"Failed to get delta_u_objectives: {str(e)}")

        control_map = {}
        for i, control_name in enumerate(sys.controls.ref_names):
            control_map[control_name] = i

        # ODE is used here because the algebraics can be calculated with the stage function
        opt_integrator = self._create_ode(sys, opts, self.options.integrator)
        # initiate states
        while self.k < n:
            u_prev = uk
            uk = self.add_opt_var(sys.controls)

            # penalty for control change between time steps (only for new objective system)
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

            dk = self.add_opt_par(sys.non_controlled_inputs)
            zk = self.add_opt_var(sys.algebraics)
            yk = self.add_opt_var(sys.outputs)

            # get path constraints and objective values (stage)
            stage_arguments = {
                # variables
                sys.states.name: xk,
                sys.algebraics.name: zk,
                sys.outputs.name: yk,
                # parameters
                sys.controls.name: uk,
                sys.non_controlled_inputs.name: dk,
                sys.model_parameters.name: const_par,
                "__time": self.pred_time,
            }
            stage = self._stage_function(**stage_arguments)

            # integral and multiple shooting constraint
            fk = opt_integrator(
                x0=xk,
                p=ca.vertcat(uk, dk, const_par, zk, yk),
            )
            xk_end = fk["xf"]
            self.k += 1
            self.pred_time = ts * self.k
            xk = self.add_opt_var(sys.states)
            vars_dict[sys.states.name][self.k] = xk
            self.add_constraint(xk - xk_end, gap_closing=True)

            # add model constraints last due to fatrop
            self.add_constraint(
                stage["model_constraints"],
                lb=stage["lb_model_constraints"],
                ub=stage["ub_model_constraints"],
            )
            self.objective_function += stage["cost_function"] * ts


class CasADiFullBackend(CasADiBackend):
    """
    Class doing optimization of ADMM subproblems with CasADi.
    """

    system_type = FullSystem
    discretization_types = {
        DiscretizationMethod.collocation: DirectCollocation,
        DiscretizationMethod.multiple_shooting: MultipleShooting,
    }
    system: FullSystem
