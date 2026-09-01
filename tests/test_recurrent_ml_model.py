"""Tests for recurrent (multi step) ML-models in the CasADi backend.
"""

import logging
from pathlib import Path

import casadi as ca
import keras
import numpy as np
import pandas as pd
import pytest

from agentlib.core.errors import ConfigurationError

from agentlib_mpc.data_structures.ml_model_datatypes import Feature, OutputFeature
from agentlib_mpc.data_structures.mpc_datamodels import MPCVariable, VariableReference
from agentlib_mpc.models.casadi_predictor import CasadiRNN
from agentlib_mpc.models.serialized_ml_model import SerializedKerasRNN
from agentlib_mpc.optimization_backends.casadi_.casadi_ml import CasADiBBBackend

RNN_TYPES = ["RNN", "GRU", "LSTM"]

UNITS = 6
WARMUP_STEPS = 5
LABEL_WIDTH = 8
FEATURES = ["T_in", "load", "mDot"]
TIME_STEP = 300
HORIZON = 4
T_START = 293.15

# the inputs vary over time, so a wrong alignment between the warmup of the hidden
# states and the prediction horizon shows up as a deviation
GRID = [TIME_STEP * i for i in range(-WARMUP_STEPS, HORIZON + 1)]


def _trajectory(base: float, amplitude: float, phase: float) -> pd.Series:
    return pd.Series(
        {t: base + amplitude * np.sin(phase + t / (3 * TIME_STEP)) for t in GRID}
    )


TRAJECTORIES = {
    "T_in": _trajectory(290.15, 4.0, 0.0),
    "load": _trajectory(150.0, 60.0, 1.0),
    "mDot": _trajectory(0.02, 0.01, 2.0),
}


def _recurrent_layer(rnn_type: str, **kwargs) -> keras.layers.Layer:
    layer_types = {
        "RNN": keras.layers.SimpleRNN,
        "GRU": keras.layers.GRU,
        "LSTM": keras.layers.LSTM,
    }
    return layer_types[rnn_type](UNITS, **kwargs)


def build_keras_model(rnn_type: str) -> keras.Model:
    """Builds a model with the structure physXAI creates for multi step models.
    """
    keras.utils.set_random_seed(1)
    rng = np.random.default_rng(1)
    sample_inputs = np.stack(
        [TRAJECTORIES[name].to_numpy() for name in FEATURES], axis=-1
    ).astype("float32")

    # --- step model: [sequence, *states] -> [prediction, *states]
    sequence = keras.Input(shape=(None, len(FEATURES)))
    normalization = keras.layers.Normalization()
    normalization.adapt(sample_inputs)
    normalized = normalization(sequence)
    if rnn_type == "LSTM":
        state_inputs = [keras.Input(shape=(UNITS,)) for _ in range(2)]
    else:
        state_inputs = keras.Input(shape=(UNITS,))
    recurrent = _recurrent_layer(rnn_type, return_state=True, return_sequences=True)
    prediction, *states = recurrent(normalized, initial_state=state_inputs)
    prediction = keras.layers.Dense(1)(prediction)
    prediction = keras.layers.Rescaling(scale=20.0, offset=T_START)(prediction)
    step_model = keras.Model(
        [sequence, state_inputs], [prediction, *states], name="out_model"
    )

    # --- initialization model
    warmup_input = keras.Input(shape=(WARMUP_STEPS, 1))
    warmup_normalization = keras.layers.Normalization()
    warmup_normalization.adapt(rng.normal(size=(200, 1)).astype("float32"))
    warmed = warmup_normalization(warmup_input)
    _, *init_states = _recurrent_layer(rnn_type, return_state=True)(warmed)
    init_model = keras.Model(warmup_input, init_states, name="init_model")

    main_input = keras.Input(shape=(LABEL_WIDTH, len(FEATURES)))
    outer_warmup = keras.Input(shape=(WARMUP_STEPS, 1))
    predictions, *_ = step_model([main_input, init_model(outer_warmup)])
    outputs = keras.layers.Reshape((LABEL_WIDTH, 1))(predictions)
    return keras.Model([main_input, outer_warmup], outputs)


def keras_rollout(model: keras.Model, sequence: np.ndarray) -> np.ndarray:
    """Predicts a sequence with the step model, starting with zero hidden states."""
    step_model = model.get_layer("out_model")
    number_of_states = len(step_model.inputs) - 1
    zero_states = [
        np.zeros((1, UNITS), dtype="float32") for _ in range(number_of_states)
    ]
    if number_of_states == 1:
        # models with a single state tensor do not expect it wrapped in a list
        zero_states = zero_states[0]
    result = step_model.predict(
        [sequence.reshape(1, *sequence.shape), zero_states], verbose=0
    )
    return np.asarray(result[0]).reshape(-1)


def serialized_model(
    model_path: Path, warmup_steps: int = None, step_model: str = None
) -> SerializedKerasRNN:
    return SerializedKerasRNN(
        dt=TIME_STEP,
        model_path=model_path,
        warmup_steps=WARMUP_STEPS if warmup_steps is None else warmup_steps,
        step_model=step_model,
        input={name: Feature(name=name) for name in FEATURES},
        output={"T": OutputFeature(name="T", output_type="absolute", recursive=True)},
    )


@pytest.fixture(params=RNN_TYPES)
def keras_model_path(request, tmp_path) -> Path:
    """Saves a keras model of the given recurrent type and returns its path."""
    model = build_keras_model(request.param)
    path = Path(tmp_path, f"{request.param}.keras")
    model.save(path)
    return path


def test_warmup_is_not_a_lag(keras_model_path):
    """A recurrent model only sees the current time step, the past values it needs to
    warm up its hidden states are not lags of the model."""
    serialized = serialized_model(keras_model_path)
    assert serialized.rnn_inputs == FEATURES
    for feature in serialized.input.values():
        assert feature.lag == 1

    with pytest.raises(ValueError, match="lag of 1"):
        SerializedKerasRNN(
            dt=TIME_STEP,
            model_path=keras_model_path,
            warmup_steps=WARMUP_STEPS,
            input={name: Feature(name=name, lag=3) for name in FEATURES},
            output={
                "T": OutputFeature(name="T", output_type="absolute", recursive=True)
            },
        )


def test_predictor_matches_keras(keras_model_path):
    """The single step CasADi function reproduces a keras sequence prediction."""
    predictor = CasadiRNN(serialized_model(keras_model_path))
    assert predictor.input_shape == (1, len(FEATURES))
    assert predictor.output_shape == (1, 1)

    sequence = np.array(
        [[TRAJECTORIES[name][t] for name in FEATURES] for t in GRID], dtype="float32"
    )
    expected = keras_rollout(keras.saving.load_model(keras_model_path), sequence)

    states = ca.DM.zeros(predictor.state_dimension, 1)
    predictions = []
    for step in range(sequence.shape[0]):
        prediction, states = predictor.predict(ca.DM(sequence[step, :]), states)
        predictions.append(float(prediction))

    # keras predicts in single precision, so the tolerance is rather loose
    assert np.allclose(predictions, expected, atol=1e-4)


def build_backend(serialized: SerializedKerasRNN) -> CasADiBBBackend:
    """Sets up an MPC with the given recurrent model."""
    backend = CasADiBBBackend(
        config={
            "model": {
                "type": {
                    "file": Path(
                        Path(__file__).parent, "fixtures", "recurrent_ml_model.py"
                    ),
                    "class_name": "RecurrentRoomModel",
                },
                "ml_model_sources": [serialized],
            },
            "discretization_options": {
                "method": "multiple_shooting",
                "time_step": TIME_STEP,
                "prediction_horizon": HORIZON,
            },
            "solver": {"name": "ipopt", "options": {"ipopt.print_level": 0}},
        }
    )
    backend.register_logger(logging.getLogger(__name__))
    backend.setup_optimization(
        VariableReference(
            states=["T"],
            controls=["mDot"],
            inputs=["load", "T_in", "T_upper"],
            parameters=["s_T", "r_mDot"],
            outputs=["T_out"],
        )
    )

    return backend


def solve_mpc(serialized: SerializedKerasRNN):
    """Solves the MPC once. The controls are pinned to a trajectory, so the solution
    of the NLP is fully determined by the ML-model."""
    backend = build_backend(serialized)

    # the MPC has to collect the history of all inputs of the recurrent model
    lags = backend.get_lags_per_variable()
    warmup = serialized.warmup_steps
    assert all(lags[name] == warmup * TIME_STEP for name in FEATURES), lags

    return backend.solve(
        now=0,
        current_vars={
            "mDot": MPCVariable(
                name="mDot",
                value=TRAJECTORIES["mDot"],
                lb=TRAJECTORIES["mDot"],
                ub=TRAJECTORIES["mDot"],
            ),
            "load": MPCVariable(name="load", value=TRAJECTORIES["load"]),
            "T_in": MPCVariable(name="T_in", value=TRAJECTORIES["T_in"]),
            "T_upper": MPCVariable(name="T_upper", value=400.0),
            "T": MPCVariable(name="T", value=T_START, lb=100, ub=500),
            "T_out": MPCVariable(name="T_out", value=T_START, lb=100, ub=500),
            "s_T": MPCVariable(name="s_T", value=1),
            "r_mDot": MPCVariable(name="r_mDot", value=1),
        },
    )


@pytest.mark.parametrize("warmup_steps", [WARMUP_STEPS, 0])
def test_mpc_reproduces_keras_prediction(keras_model_path, warmup_steps):
    """The state trajectory of the MPC matches a keras rollout of the same inputs.

    The hidden states start as zeros ``warmup_steps`` before the horizon, hence the
    reference rollout covers the warmup and the horizon.
    """
    serialized = serialized_model(keras_model_path, warmup_steps=warmup_steps)
    results = solve_mpc(serialized)
    assert results.stats["return_status"] == "Solve_Succeeded"

    # the hidden states are internal, they should not blow up the result files
    variables = [name for kind, name in results.df.columns if kind == "variable"]
    assert not any("rnn_state" in name for name in variables)

    times = [TIME_STEP * i for i in range(-warmup_steps, HORIZON)]
    sequence = np.array(
        [[TRAJECTORIES[name][t] for name in FEATURES] for t in times], dtype="float32"
    )
    expected = keras_rollout(keras.saving.load_model(keras_model_path), sequence)[
        warmup_steps:
    ]
    # the trajectory has to be non trivial, otherwise the comparison is meaningless
    assert np.ptp(expected) > 0.05

    predicted = results["T"].reshape(-1)
    assert predicted[0] == pytest.approx(T_START)
    assert np.allclose(predicted[1:], expected, atol=1e-4)


def test_warmup_does_not_grow_the_problem(keras_model_path):
    """The warmup runs on measured data, so it must not enter the NLP.

    The hidden states at the start of the horizon are a parameter of the optimization
    problem, hence its size only depends on the prediction horizon.
    """
    sizes = set()
    for warmup_steps in (0, WARMUP_STEPS, 10 * WARMUP_STEPS):
        discretization = build_backend(
            serialized_model(keras_model_path, warmup_steps=warmup_steps)
        ).discretization
        sizes.add(
            (
                discretization.opt_vars.shape[0],
                discretization.constraints.shape[0],
                discretization.opt_pars.shape[0],
            )
        )
    assert len(sizes) == 1, sizes


def minimal_step_model(name: str) -> keras.Model:
    """Smallest model mapping [sequence, state] to [prediction, state]."""
    sequence = keras.Input(shape=(None, len(FEATURES)))
    state = keras.Input(shape=(UNITS,))
    prediction, next_state = keras.layers.SimpleRNN(
        UNITS, return_state=True, return_sequences=True
    )(sequence, initial_state=state)
    prediction = keras.layers.Dense(1)(prediction)
    return keras.Model([sequence, state], [prediction, next_state], name=name)


def test_step_model_is_found_by_its_signature(tmp_path):
    """Any model mapping [sequence, *states] to [prediction, *states] is accepted.

    The step model is identified by its signature, not by the way the framework that
    trained it happens to name or nest its parts.
    """
    # a model that consists of nothing but the recurrent step
    path = Path(tmp_path, "bare.keras")
    minimal_step_model("anything").save(path)
    predictor = CasadiRNN(serialized_model(path))
    assert predictor.step_model.name == "anything"
    assert predictor.state_dimensions == [UNITS]

    # a step model nested in an unrelated wrapper
    step = minimal_step_model("nested_under_any_name")
    sequence = keras.Input(shape=(HORIZON, len(FEATURES)))
    state = keras.Input(shape=(UNITS,))
    prediction, _ = step([sequence, state])
    path = Path(tmp_path, "wrapped.keras")
    keras.Model([sequence, state], prediction).save(path)
    assert (
        CasadiRNN(serialized_model(path)).step_model.name == "nested_under_any_name"
    )


def ambiguous_model() -> keras.Model:
    """A model containing two models which perform a single recurrent step."""
    first = minimal_step_model("step_a")
    second = minimal_step_model("step_b")
    sequence = keras.Input(shape=(HORIZON, len(FEATURES)))
    state = keras.Input(shape=(UNITS,))
    _, next_state = first([sequence, state])
    prediction, _ = second([sequence, next_state])
    return keras.Model([sequence, state], prediction)


def test_ambiguous_step_model_has_to_be_declared(tmp_path):
    """If several parts of a model could be the recurrent step, the user picks one."""
    path = Path(tmp_path, "ambiguous.keras")
    ambiguous_model().save(path)

    with pytest.raises(ConfigurationError, match="more than one"):
        CasadiRNN(serialized_model(path))

    predictor = CasadiRNN(serialized_model(path, step_model="step_b"))
    assert predictor.step_model.name == "step_b"

    with pytest.raises(ConfigurationError, match="does not contain a model"):
        CasadiRNN(serialized_model(path, step_model="does_not_exist"))


def test_model_without_recurrent_step_is_rejected(tmp_path):
    """A model that cannot be evaluated one step at a time is not a multi step model."""
    inputs = keras.Input(shape=(len(FEATURES),))
    path = Path(tmp_path, "feed_forward.keras")
    keras.Model(inputs, keras.layers.Dense(1)(inputs)).save(path)

    with pytest.raises(ConfigurationError, match="multi step model"):
        CasadiRNN(serialized_model(path))
