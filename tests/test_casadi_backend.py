import logging

import casadi as ca
import pytest

from agentlib_mpc.data_structures.casadi_utils import (
    CasadiDiscretizationOptions,
    SolverFactory,
)
from agentlib_mpc.data_structures.mpc_datamodels import VariableReference
from agentlib_mpc.models.casadi_model import CasadiState
from agentlib_mpc.optimization_backends.casadi_.basic import (
    BaseSystem,
    DirectCollocation,
    CasADiBaseBackend,
)
from agentlib_mpc.optimization_backends.casadi_.core.VariableGroup import (
    OptimizationVariable,
)
from agentlib_mpc.optimization_backends.casadi_.core.discretization import (
    Discretization,
)


@pytest.fixture
def var_ref() -> VariableReference:
    return VariableReference(
        states=["state"],
        controls=["myctrl"],
        inputs=["disturbance"],
        parameters=["par", "par2"],
        outputs=["myout"],
    )


@pytest.fixture
def example_casadi_system(example_casadi_model, var_ref) -> BaseSystem:
    sys = BaseSystem()
    sys.initialize(model=example_casadi_model, var_ref=var_ref)
    return sys


@pytest.fixture
def collocation_discretization(example_casadi_system) -> Discretization:
    options = CasadiDiscretizationOptions()
    dis = DirectCollocation(options=options)
    dis.initialize(
        system=example_casadi_system,
        solver_factory=SolverFactory(do_jit=False, logger=logging.getLogger()),
    )
    return dis


@pytest.fixture
def example_backend(model_type, var_ref) -> CasADiBaseBackend:
    be = CasADiBaseBackend(config={"model": {"type": model_type}})
    be.setup_optimization(var_ref)
    return be


def test_optimization_variable():
    variables = [
        CasadiState(name="s1", value=10, ub=10, lb=0),
        CasadiState(name="s2", lb=0),
    ]

    with pytest.raises(ValueError):
        optimization_variable = OptimizationVariable.declare(
            denotation="state",
            variables=variables,
            ref_list=["s1"],
            assert_complete=True,
        )

    optimization_variable_1 = OptimizationVariable.declare(
        denotation="state",
        variables=variables,
        ref_list=["s1"],
    )

    optimization_variable_2 = OptimizationVariable.declare(
        denotation="state",
        variables=variables,
        ref_list=["s1", "s2"],
        use_in_stage_function=False,
    )
    assert optimization_variable_2.ref_names == optimization_variable_2.full_names

    with pytest.raises(ValueError):
        optimization_variable_3 = OptimizationVariable.declare(
            denotation="state",
            variables=variables,
            ref_list=["s3"],
        )


def test_system(example_casadi_system):
    sys = example_casadi_system
    assert len(sys.quantities) == 7
    assert len(sys.parameters) == 3
    assert len(sys.variables) == 4

    # in the initial guess function a heuristic is used which connects a variable to
    # another one if it has 'initial_' + the denotation
    assert "initial_" + sys.states.name == sys.initial_state.name
    assert sys.initial_state.use_in_stage_function is False
    assert (
        len(sys.model_parameters.full_names)
        == sys.model_parameters.full_symbolic.shape[0]
    )


def test_discretization(
    collocation_discretization: DirectCollocation, example_casadi_system: BaseSystem
):
    sys = example_casadi_system
    dis = collocation_discretization
    all(
        i in dis.mpc_opt_vars
        for i in [
            sys.states.name,
            sys.controls.name,
            sys.algebraics.name,
            sys.outputs.name,
        ]
    )
    state_grid = dis.grid(sys.states)
    assert state_grid[0] == 0
    assert (
        len(state_grid)
        == (dis.options.collocation_order + 1) * dis.options.prediction_horizon + 1
    )


def test_add_opt(example_casadi_system: BaseSystem):
    options = CasadiDiscretizationOptions()
    dis = DirectCollocation(options=options)
    sys = example_casadi_system

    assert any([dis.opt_vars, dis.opt_pars, dis.initial_guess]) is False
    dis.add_opt_var(sys.states)
    assert all([dis.opt_vars, dis.mpc_opt_vars, dis.initial_guess])
    assert any([dis.opt_pars, dis.mpc_opt_pars]) is False

    dis.add_opt_par(sys.model_parameters)
    assert all([dis.opt_pars, dis.mpc_opt_pars])

    dis.add_opt_var(sys.controls)
    dis.pred_time += 10
    dis.add_opt_var(sys.controls)
    assert dis.mpc_opt_vars[sys.controls.name].grid == [0, 10]


def test_create_backend(example_backend):
    be = example_backend
    dis = be.discretization
    assert ca.vertcat(*dis.constraints_lb).shape[0] == dis.constraints.shape[0]
    assert ca.vertcat(*dis.constraints_ub).shape[0] == dis.constraints.shape[0]
    nlp_in = dis._mpc_inputs_to_nlp_inputs()
    assert ca.vertcat(*dis.opt_vars_lb).shape[0] == nlp_in["lbx"].shape[0]
    assert ca.vertcat(*dis.opt_vars_ub).shape[0] == nlp_in["ubx"].shape[0]
