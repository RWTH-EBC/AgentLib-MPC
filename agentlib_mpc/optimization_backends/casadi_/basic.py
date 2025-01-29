import dataclasses

import casadi as ca
import numpy as np

from agentlib_mpc.data_structures.casadi_utils import (
    Constraint,
    LB_PREFIX,
    UB_PREFIX,
    DiscretizationMethod,
    SolverFactory,
    Integrators,
)
from agentlib_mpc.data_structures.mpc_datamodels import VariableReference
from agentlib_mpc.models.casadi_model import CasadiModel
from agentlib_mpc.optimization_backends.casadi_.core.casadi_backend import CasADiBackend
from agentlib_mpc.optimization_backends.casadi_.core.VariableGroup import (
    OptimizationQuantity,
    OptimizationVariable,
    OptimizationParameter,
)
from agentlib_mpc.optimization_backends.casadi_.core.discretization import (
    Discretization,
)
from agentlib_mpc.optimization_backends.casadi_.core.system import System


class BaseSystem(System):
    # variables
    states: OptimizationVariable
    controls: OptimizationVariable
    algebraics: OptimizationVariable
    outputs: OptimizationVariable

    # parameters
    non_controlled_inputs: OptimizationParameter
    model_parameters: OptimizationParameter
    initial_state: OptimizationParameter

    # dynamics
    model_constraints: Constraint
    cost_function: ca.MX
    ode: ca.MX

    def initialize(self, model: CasadiModel, var_ref: VariableReference):
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

        # dynamics
        ode = ca.vertcat(*[sta.ode for sta in model.get_states(var_ref.states)])
        self.ode = ca.reshape(ode, -1, 1)
        self.cost_function = model.cost_func
        self.model_constraints = Constraint(
            function=ca.vertcat(*[c.function for c in model.get_constraints()]),
            lb=ca.vertcat(*[c.lb for c in model.get_constraints()]),
            ub=ca.vertcat(*[c.ub for c in model.get_constraints()]),
        )
        self.time = model.time.sym


@dataclasses.dataclass
class CollocationMatrices:
    order: int
    root: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray


class DirectCollocation(Discretization):
    def _discretize(self, sys: BaseSystem):
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

        # Parameters that are constant over the horizon
        const_par = self.add_opt_par(sys.model_parameters)

        # Formulate the NLP
        # loop over prediction horizon
        k = 0
        while k < n:
            # New NLP variable for the control
            uk = self.add_opt_var(sys.controls)
            # New parameter for inputs
            dk = self.add_opt_par(sys.non_controlled_inputs)

            # perform inner collocation loop
            opt_vars_inside_inner = [sys.algebraics, sys.outputs]
            opt_pars_inside_inner = []

            constant_over_inner = {
                sys.controls: uk,
                sys.non_controlled_inputs: dk,
                sys.model_parameters: const_par,
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
            k += 1
            self.pred_time = ts * k

            # New NLP variable for differential state at end of interval
            xk = self.add_opt_var(sys.states)

            # Add continuity constraint
            self.add_constraint(xk - xk_end, gap_closing=True)

            # add collocation constraints later for fatrop
            for constraint in constraints:
                self.add_constraint(*constraint)

    def _construct_stage_function(self, system: BaseSystem):
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
        all_system_quantities: dict[str, OptimizationQuantity] = {
            var.name: var for var in system.quantities
        }
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
        input_denotations.append("time")

        # aggregate constraints
        constraints_func = [c.function for c in constraints.values()]
        constraints_lb = [c.lb for c in constraints.values()]
        constraints_ub = [c.ub for c in constraints.values()]
        constraint_denotations = list(constraints.keys())
        constraint_lb_denotations = [LB_PREFIX + k for k in constraints]
        constraint_ub_denotations = [UB_PREFIX + k for k in constraints]

        # aggregate outputs
        outputs = [
            system.ode,
            system.cost_function,
            *constraints_func,
            *constraints_lb,
            *constraints_ub,
        ]
        output_denotations = [
            "ode",
            "cost_function",
            *constraint_denotations,
            *constraint_lb_denotations,
            *constraint_ub_denotations,
        ]

        # function describing system dynamics and cost function
        self._stage_function = ca.Function(
            "f",
            inputs,
            outputs,
            # input handles to make kwarg use possible and to debug
            input_denotations,
            # output handles to make kwarg use possible and to debug
            output_denotations,
        )

    def initialize(self, system: BaseSystem, solver_factory: SolverFactory):
        """Initializes the trajectory optimization problem, creating all symbolic
        variables of the OCP, the mapping function and the numerical solver."""
        self._construct_stage_function(system)
        super().initialize(system=system, solver_factory=solver_factory)

    def _collocation_inner_loop(
        self,
        state_at_beginning: ca.MX,
        collocation: CollocationMatrices,
        states: OptimizationVariable,
        opt_vars: list[OptimizationVariable],
        opt_pars: list[OptimizationParameter],
        const: dict[OptimizationQuantity, ca.MX]
    ) -> tuple[ca.MX, tuple]:
        """
        Constructs the inner loop of a collocation discretization. Takes the

        Args
            collocation: The collocation matrices
            state_at_beginning: The casadi MX instance representing the state at the
                beginning of the collocation interval
            states: The OptimizationVariable representing the states
            opt_vars: The OptimizationVariables which should be defined at each
                collocation point
            par_vars: The OptimizationParameters which should be defined at each
                collocation point
            const: Variables or parameters to feed into the system function that are
                constant over the inner loop. Value is the current MX to be used.

        Returns:
            state_k_end[MX]: state at the end of collocation interval
        """
        constraints = []
        constants = {var.name: mx for var, mx in const.items()}

        # remember time at start of collocation loop
        start_time = self.pred_time

        # shorthands
        ts = self.options.time_step

        # State variables at collocation points
        state_collocation = []
        opt_vars_collocation = []
        opt_pars_collocation = []

        # add variables at collocation points
        for j in range(collocation.order):  # d is collocation order
            # set time
            self.pred_time = start_time + collocation.root[j + 1] * ts

            # differential state
            state_kj = self.add_opt_var(states, post_den=f"_{j}")
            state_collocation.append(state_kj)

            opt_vars_collocation.append({})
            for opt_var in opt_vars:
                var_kj = self.add_opt_var(opt_var, post_den=f"_{j}")
                opt_vars_collocation[-1].update({opt_var.name: var_kj})

            opt_pars_collocation.append({})
            for opt_par in opt_pars:
                par_kj = self.add_opt_par(opt_par, post_den=f"_{j}")
                opt_pars_collocation[-1].update({opt_par.name: par_kj})
            opt_pars_collocation[-1].update({"time": self.pred_time})

        # Loop over collocation points
        state_k_end = collocation.D[0] * state_at_beginning
        for j in range(1, collocation.order + 1):
            # Expression for the state derivative at the collocation point
            xp = collocation.C[0, j] * state_at_beginning
            for r in range(collocation.order):
                xp = xp + collocation.C[r + 1, j] * state_collocation[r]

            stage = self._stage_function(
                **{states.name: state_collocation[j - 1]},
                **opt_pars_collocation[j - 1],
                **opt_vars_collocation[j - 1],
                **constants
            )

            constraints.append((ts * stage["ode"] - xp,))
            constraints.append(
                (
                    stage["model_constraints"],
                    stage["lb_model_constraints"],
                    stage["ub_model_constraints"],
                )
            )

            # Add contribution to the end state
            state_k_end = state_k_end + collocation.D[j] * state_collocation[j - 1]

            # Add contribution to quadrature function
            self.objective_function += collocation.B[j] * stage["cost_function"] * ts

        return state_k_end, constraints

    def _collocation_polynomial(self) -> CollocationMatrices:
        """Returns the matrices needed for direct collocation discretization."""
        # Degree of interpolating polynomial
        d = self.options.collocation_order
        polynomial = self.options.collocation_method

        # Get collocation points
        tau_root = np.append(0, ca.collocation_points(d, polynomial))

        # Coefficients of the collocation equation
        C = np.zeros((d + 1, d + 1))

        # Coefficients of the continuity equation
        D = np.zeros(d + 1)

        # Coefficients of the quadrature function
        B = np.zeros(d + 1)

        # Construct polynomial basis
        for j in range(d + 1):
            # Construct Lagrange polynomials to get the polynomial basis at
            # the collocation point
            p = np.poly1d([1])
            for r in range(d + 1):
                if r != j:
                    p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j] - tau_root[r])

            # Evaluate the polynomial at the final time to get the
            # coefficients of the continuity equation
            D[j] = p(1.0)

            # Evaluate the time derivative of the polynomial at all collocation
            # points to get the coefficients of the continuity equation
            pder = np.polyder(p)
            for r in range(d + 1):
                C[j, r] = pder(tau_root[r])

            # Evaluate the integral of the polynomial to get the coefficients
            # of the quadrature function
            pint = np.polyint(p)
            B[j] = pint(1.0)

        return CollocationMatrices(
            order=d,
            root=tau_root,
            B=B,
            C=C,
            D=D,
        )


class MultipleShooting(Discretization):
    def _discretize(self, sys: BaseSystem):
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
        const_par = self.add_opt_par(sys.model_parameters)
        # ODE is used here because the algebraics can be calculated with the stage function
        opt_integrator = self._create_ode(sys, opts, integrator=self.options.integrator)
        # initiate states
        while self.k < n:
            uk = self.add_opt_var(sys.controls)
            dk = self.add_opt_par(sys.non_controlled_inputs)
            zk = self.add_opt_var(sys.algebraics)
            yk = self.add_opt_var(sys.outputs)
            # get stage
            stage_arguments = {
                # variables
                sys.states.name: xk,
                sys.algebraics.name: zk,
                sys.outputs.name: yk,
                # parameters
                sys.controls.name: uk,
                sys.non_controlled_inputs.name: dk,
                sys.model_parameters.name: const_par,
            }
            # get stage
            stage = self._stage_function(**stage_arguments)

            self.add_constraint(
                stage["model_constraints"],
                lb=stage["lb_model_constraints"],
                ub=stage["ub_model_constraints"],
            )
            fk = opt_integrator(
                x0=xk,
                p=ca.vertcat(uk, dk, const_par, zk, yk),
            )
            xk_end = fk["xf"]
            # calculate model constraint
            self.k += 1
            self.pred_time = ts * self.k
            xk = self.add_opt_var(sys.states)
            vars_dict[sys.states.name][self.k] = xk
            self.add_constraint(xk_end - xk, gap_closing=True)
            self.objective_function += stage["cost_function"] * ts

    def _create_ode(self, sys: BaseSystem, opts: dict, integrator: Integrators):
        # dummy function for empty ode, since ca.integrator would throw an error
        if sys.states.full_symbolic.shape[0] == 0:
            return lambda *args, **kwargs: {"xf": ca.MX.sym("xk_end", 0)}

        ode = sys.ode
        # create inputs
        x = sys.states.full_symbolic
        # the order of elements here is important when calling the integrator!
        p = ca.vertcat(
            sys.controls.full_symbolic,
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

    def _construct_stage_function(self, system: BaseSystem):
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
        all_system_quantities: dict[str, OptimizationQuantity] = {
            var.name: var for var in system.quantities
        }
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
        input_denotations.append("time")

        # aggregate constraints
        constraints_func = [c.function for c in constraints.values()]
        constraints_lb = [c.lb for c in constraints.values()]
        constraints_ub = [c.ub for c in constraints.values()]
        constraint_denotations = list(constraints.keys())
        constraint_lb_denotations = [LB_PREFIX + k for k in constraints]
        constraint_ub_denotations = [UB_PREFIX + k for k in constraints]

        # aggregate outputs
        outputs = [
            system.ode,
            system.cost_function,
            *constraints_func,
            *constraints_lb,
            *constraints_ub,
        ]
        output_denotations = [
            "ode",
            "cost_function",
            *constraint_denotations,
            *constraint_lb_denotations,
            *constraint_ub_denotations,
        ]

        # function describing system dynamics and cost function
        self._stage_function = ca.Function(
            "f",
            inputs,
            outputs,
            # input handles to make kwarg use possible and to debug
            input_denotations,
            # output handles to make kwarg use possible and to debug
            output_denotations,
        )

    def initialize(self, system: BaseSystem, solver_factory: SolverFactory):
        """Initializes the trajectory optimization problem, creating all symbolic
        variables of the OCP, the mapping function and the numerical solver."""
        self._construct_stage_function(system)
        super().initialize(system=system, solver_factory=solver_factory)


class CasADiBaseBackend(CasADiBackend):
    """
    Class doing optimization of ADMM subproblems with CasADi.
    """

    system_type = BaseSystem
    discretization_types = {
        DiscretizationMethod.collocation: DirectCollocation,
        DiscretizationMethod.multiple_shooting: MultipleShooting,
    }
    system: BaseSystem
