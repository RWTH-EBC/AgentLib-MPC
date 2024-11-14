import casadi as ca
import pandas as pd

from agentlib_mpc.data_structures.casadi_utils import DiscretizationMethod, Integrators
from agentlib_mpc.data_structures.mpc_datamodels import stats_path
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

            # New parameter for inputs
            dk = self.add_opt_par(sys.non_controlled_inputs)

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
            ]
            constant_over_inner = {
                sys.controls: uk,
                sys.non_controlled_inputs: dk,
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
        """
        Performs a multiple shooting discretization for ADMM
        """
        vars_dict = {
            sys.states.name: {},
            sys.controls.name: {},
            sys.non_controlled_inputs.name: {},
            sys.local_couplings.name: {},
            sys.global_couplings.name: {},
            sys.local_exchange.name: {},
            sys.exchange_diff.name: {},
            sys.exchange_multipliers.name: {},
        }
        n = self.options.prediction_horizon
        ts = self.options.time_step
        opts = {"t0": 0, "tf": ts}
        # Initial State
        x0 = self.add_opt_par(sys.initial_state)
        xk = self.add_opt_var(sys.states, lb=x0, ub=x0, guess=x0)
        vars_dict[sys.states.name][0] = xk
        uk = self.add_opt_par(sys.last_control)

        # Parameters that are constant over the horizon
        du_weights = self.add_opt_par(sys.r_del_u)
        const_par = self.add_opt_par(sys.model_parameters)
        rho = self.add_opt_par(sys.penalty_factor)
        e_diff0 = self.add_opt_par(sys.exchange_diff)
        e_multi0 = self.add_opt_par(sys.exchange_multipliers)

        # create integrator
        opt_integrator = self._create_ode(sys, opts, self.options.integrator)
        # initiate states
        while self.k < n:
            u_prev = uk
            uk = self.add_opt_var(sys.controls)
            # penalty for control change between time steps
            self.objective_function += ts * ca.dot(du_weights, (u_prev - uk) ** 2)
            dk = self.add_opt_par(sys.non_controlled_inputs)
            zk = self.add_opt_var(sys.algebraics)
            yk = self.add_opt_var(sys.outputs)
            v_localk = self.add_opt_var(sys.local_couplings)
            v_meank = self.add_opt_par(sys.global_couplings)
            lamk = self.add_opt_par(sys.multipliers)
            vars_dict[sys.global_couplings.name][self.k] = v_meank

            e_localk = self.add_opt_var(sys.local_exchange)
            vars_dict[sys.local_exchange.name][self.k] = e_localk

            # get stage_function
            stage_arguments = {
                # variables
                sys.states.name: xk,
                sys.algebraics.name: zk,
                sys.local_couplings.name: v_localk,
                sys.outputs.name: yk,
                sys.local_exchange.name: e_localk,
                # parameters
                sys.global_couplings.name: v_meank,
                sys.multipliers.name: lamk,
                sys.controls.name: uk,
                sys.non_controlled_inputs.name: dk,
                sys.model_parameters.name: const_par,
                sys.penalty_factor.name: rho,
                sys.exchange_diff.name: e_diff0,
                sys.exchange_multipliers.name: e_multi0,
            }
            stage = self._stage_function(**stage_arguments)

            # integral and multiple shooting constraint
            fk = opt_integrator(
                x0=xk,
                p=ca.vertcat(uk, v_localk, dk, const_par, zk, yk),
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

    def save_result_df(
        self,
        results: Results,
        now: float = 0,
    ):
        """
        Save the results of `solve` into a dataframe at each time step.

        Example results dataframe:

        value_type               variable              ...     lower
        variable                      T_0   T_0_slack  ... T_0_slack mDot_0
        time_step                                      ...
        2         0.000000     298.160000         NaN  ...       NaN    NaN
                  101.431499   297.540944 -149.465942  ...      -inf    0.0
                  450.000000   295.779780 -147.704779  ...      -inf    0.0
                  798.568501   294.720770 -146.645769  ...      -inf    0.0
        Args:
            results:
            now:

        Returns:

        """
        if not self.config.save_results:
            return

        res_file = self.config.results_file

        if self.results_file_exists():
            self.it += 1
            if now != self.now:  # means we advanced to next step
                self.it = 0
                self.now = now
        else:
            self.it = 0
            self.now = now
            results.write_columns(res_file)
            results.write_stats_columns(stats_path(res_file))

        df = results.df
        df.index = list(map(lambda x: str((now, self.it, x)), df.index))
        self.results.append(df)

        # append solve stats
        index = str((now, self.it))
        self.result_stats.append(results.stats_line(index))

        # save last results at the start of new sampling time, or if 1000 iterations
        # are exceeded
        if not (self.it == 0 or self.it % 1000 == 0):
            return

        with open(res_file, "a", newline="") as f:
            for iteration_result in self.results:
                iteration_result.to_csv(f, mode="a", header=False)

        with open(stats_path(res_file), "a") as f:
            f.writelines(self.result_stats)
        self.results = []
        self.result_stats = []
