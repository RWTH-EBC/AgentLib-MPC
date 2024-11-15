"""Holds the classes for CasADi variables and the CasADi model."""

import itertools
import logging
from itertools import chain
from pathlib import Path
from typing import Dict, Union, List, Iterable, TypeVar, Any, Optional

from agentlib import AgentVariable
import pandas as pd
from agentlib.core.errors import ConfigurationError
from pydantic_core.core_schema import ValidatorFunctionWrapHandler
import casadi as ca
from pydantic import (
    field_validator,
    FieldValidationInfo,
    model_validator,
    Field,
)

from agentlib_mpc.data_structures import ml_model_datatypes
from agentlib_mpc.data_structures.ml_model_datatypes import OutputType, name_with_lag

from agentlib_mpc.models.casadi_predictor import CasadiPredictor
from agentlib_mpc.models.casadi_model import (
    CasadiModel,
    CasadiModelConfig,
    CasadiState,
    CasadiOutput,
    CasadiTypes,
)
from agentlib_mpc.models.serialized_ml_model import (
    SerializedMLModel,
)
from agentlib_mpc.utils.sampling import sample

logger = logging.getLogger(__name__)
CASADI_VERSION = float(ca.__version__[:3])


T = TypeVar("T")


def compute_dupes(collection: Iterable[T]) -> list[T]:
    """Computes the duplicate elements in a collection"""
    dupes = []
    seen = set()
    for element in collection:
        if element in seen:
            dupes.append(element)
        else:
            seen.add(element)
    return dupes


# todo
"""
2. Algebraic ML-Equations will consider the continuous evaluation of States during integration
"""


class CasadiMLModelConfig(CasadiModelConfig):
    ml_model_sources: list[Union[SerializedMLModel, Path]] = []
    dt: Union[float, int] = Field(
        default=1, title="time increment", validate_default=True
    )

    @field_validator("ml_model_sources", mode="before")
    @classmethod
    def check_or_load_models(cls, ml_model_sources, info: FieldValidationInfo):
        # load all ANNs that are paths
        for i, ml_model_src in enumerate(ml_model_sources):
            if isinstance(ml_model_src, SerializedMLModel):
                continue
            serialized = SerializedMLModel.load_serialized_model(ml_model_src)
            assert_recursive_outputs_are_states(serialized, info.data["outputs"])
            ml_model_sources[i] = serialized

        ml_model_sources: list[SerializedMLModel]
        # check that all MLModels have the same time step
        time_steps = {s_ml_model.dt for s_ml_model in ml_model_sources}
        if len(time_steps) > 1:
            raise ConfigurationError(
                f"Provided MLModel's need to have the same 'dt'. Provided dt are"
                f" {time_steps}"
            )

        # Check that model config and provided .json files match and there are no dupes
        all_outputs = list(
            chain.from_iterable(
                [ml_model.output.keys() for ml_model in ml_model_sources]
            )
        )
        all_inputs = list(
            chain.from_iterable(
                [ml_model.input.keys() for ml_model in ml_model_sources]
            )
        )
        output_names = [var.name for var in info.data["states"] + info.data["outputs"]]
        input_names = [var.name for var in info.data["inputs"] + info.data["states"]]

        output_dupes = compute_dupes(all_outputs)
        if output_dupes:
            raise ConfigurationError(
                f"The MLModel's that were provided define the same output multiple times."
                f" Duplicates are: {output_dupes}"
            )

        inputs_in_ml_model_but_not_config = set(all_inputs) - set(input_names)
        outputs_in_ml_model_but_not_config = set(all_outputs) - set(output_names)

        if inputs_in_ml_model_but_not_config:
            raise ConfigurationError(
                f"Inputs specified by MLModels do not appear in model: "
                f"{inputs_in_ml_model_but_not_config}"
            )
        if outputs_in_ml_model_but_not_config:
            raise ConfigurationError(
                f"Outputs specified by MLModels do not appear in model states / outputs: "
                f"{outputs_in_ml_model_but_not_config}"
            )

        return ml_model_sources

    @model_validator(mode="wrap")
    @classmethod
    def check_dt(cls, values, handler: ValidatorFunctionWrapHandler):
        validated: cls = handler(values)
        ml_model_dt = validated.ml_model_sources[0].dt
        model_dt = validated.dt
        if model_dt != ml_model_dt:
            logger.warning(
                f"Time step (dt) of model and supplied MLModels does not match. Setting "
                f"the model time step to {ml_model_dt}."
            )
            validated.dt = ml_model_dt
        return validated


def assert_recursive_outputs_are_states(
    serialized: SerializedMLModel, outputs: dict[str, AgentVariable]
):
    """Raises a ConfigurationError if there are recursive ML-models for outputs."""
    for out_name, out_feat in serialized.output.items():
        if out_name in outputs and out_feat.recursive:
            raise ConfigurationError(
                f"Provided ML-model defines recursive output {out_name}, however in the"
                f" model config it is listed under 'outputs'. A recursive model output"
                f" can only be associated with a 'state'."
            )


class CasadiMLModel(CasadiModel):
    """
    This class is created to handle one or multiple ML models
    used to predict the states. Compared to previous version, it is now
    only dependent on the trained models which provides information about the lags
    with the serialized_ML_Models. This way, there is no need to define the lags
    again in the model class
    """

    config_type: CasadiMLModelConfig
    config: CasadiMLModelConfig

    def __init__(self, **kwargs):
        # state variables used and predicted by the MLModel

        super().__init__(**kwargs)
        # register ARX models
        ml_model_dict, casadi_ml_model_dict = self.register_ml_models()
        self.ml_model_dict: Dict[str, SerializedMLModel] = ml_model_dict
        self.casadi_ml_model_dict: Dict[str, CasadiPredictor] = casadi_ml_model_dict

        # Register lagged variables
        lags_dict, max_lag = self._create_lags_dict()
        self.lags_dict: dict[str, int] = lags_dict
        self.max_lag: int = max_lag
        self.lags_mx_store: dict[str, ca.MX] = self._create_lags_mx_variables()
        self._fill_algebraic_equations_with_bb_output()

        self.past_values = self._create_past_values()

        # construct a stage function for optimization and simulation
        self.sim_step = self._make_unified_predict_function()

    def setup_system(self):
        return 0

    def update_ml_models(self, *ml_models: SerializedMLModel, time: float):
        """Updates the internal MLModels with the passed MLModels.

        Warning: This function does not fully check, if the result makes sense!
        Consider the following case:
        You have two ml_models with outputs out1 in ml_model1, and out2 and out3 in ml_model2.
        You call this function with an ml_model3 that defines out2.
        This function would replace ml_model2 with ml_model3, leaving the out3 undefined, causing
        an error in subequent functions. Try to make sure you specify all outputs when
        supplying ml_models, that make parts of other ml_models obsolete.
        """
        new_outputs = set(
            chain.from_iterable([ml_model.output.keys() for ml_model in ml_models])
        )
        ml_models_to_keep = []
        for ml_model in self.config.ml_model_sources:
            # if the outputs of the currently active ml_models are not part of the new ml_models
            # we just got, we keep them
            if set(ml_model.output) - new_outputs:
                ml_models_to_keep.append(ml_model)
        self.config.ml_model_sources = ml_models_to_keep + list(ml_models)

        self.lags_dict, self.max_lag = self._create_lags_dict()
        self._update_past_values(time)
        self.ml_model_dict, self.casadi_ml_model_dict = self.register_ml_models()
        self.sim_step = self._make_unified_predict_function()
        self._assert_outputs_are_defined()

    def _update_past_values(self, time: float):
        """Generates new columns and deletes old ones in the time series data, when the
        MLModels are updated."""
        new_columns = set(self.lags_dict)
        old_columns = set(self.past_values.columns)

        columns_to_remove = old_columns - new_columns
        columns_to_add = new_columns - old_columns

        self.past_values.drop(columns_to_remove, inplace=True)
        index = [time - self.dt * lag for lag in range(self.max_lag)]
        index.reverse()
        for col in columns_to_add:
            value = self.get(col).value
            for time in index:
                self.past_values.loc[(time, col)] = value

    def _create_past_values(self) -> pd.DataFrame:
        """Creates a collection which saves a history of the model's variables that
        are required in the lags. Must be executed after _create_lags_dict"""
        last_values = pd.DataFrame(columns=self.lags_dict)
        index = [-self.config.dt * lag for lag in range(self.max_lag)]
        index.reverse()
        values = [self.get(var_name).value for var_name in self.lags_dict]
        for time in index:
            last_values.loc[time] = values
        return last_values

    def _create_lags_dict(self) -> tuple[dict[str, int], int]:
        """Creates a dictionary which holds the maximum lag of each variable"""
        lags_dict = {}
        for ml_model in self.config.ml_model_sources:
            in_out = ml_model.input | ml_model.output
            for input_name, feature in in_out.items():
                current_lag = lags_dict.setdefault(input_name, 1)
                if feature.lag > current_lag:
                    lags_dict[input_name] = feature.lag
        max_lag = max(lags_dict.values())
        return lags_dict, max_lag

    def _create_lags_mx_variables(self) -> dict[str, ca.MX]:
        """Creates symbolic CasADi MX variables for all the required lags."""
        lags_mx_dict = {}
        for var_name, max_lag_of_var in self.lags_dict.items():
            for lag in range(1, max_lag_of_var):
                l_name = name_with_lag(var_name, lag)
                lags_mx_dict[l_name] = ca.MX.sym(l_name)
        return lags_mx_dict

    def set_with_timestamp(self, name: str, value: Any, timestamp: float):
        if name in self.past_values.columns:
            self.past_values.loc[(timestamp, name)] = value
        self.set(name, value)

    def _fixed_during_integration(
        self, bb_results: Optional[dict[str, ca.MX]] = None
    ) -> dict[str, ca.MX]:
        """Returns all variable names with their corresponding CasADi MX variable that
        are fixed during integration.
        Uses a heuristic to approximate blackbox defined states during integration.
        Currently, the heuristic is to use the arithmetic middle between the start and
        the end of the integration.
        If the bb_results are not available, the value at the beginning is used

        Args:
            bb_results: The results of the evaluation of the blackbox functions
        """
        all_inputs = self._all_inputs()
        exclude = [v.name for v in self.differentials + self.outputs]
        # take the mean of start/finish values of variables that have already been
        # integrated by a discrete blackbox function
        if bb_results:
            for bb_name, bb_res_mx in bb_results.items():
                all_inputs[bb_name] = (all_inputs[bb_name] + bb_res_mx) / 2
        return {name: sym for name, sym in all_inputs.items() if name not in exclude}

    def _make_integrator(self, ignore_algebraics: bool = False) -> ca.Function:
        """Creates an integrator for the white-box equations in the model. The
        integrator takes the stacked white box differential states (in order of
        self.differentials), and the stacked (parameters, inputs, mL_model_states) in that
        order as the second argument.

        Args:
            ignore_algebraics: if True, algebraic equations will not be added
                (default False)

        """
        if CASADI_VERSION < 3.6:
            args = ({"t0": 0, "tf": self.dt},)
        else:
            args = (0, self.dt, {})
        # the ml_model outputs cannot be changed during integration, so they are a
        # parameter here
        integration_params = self._fixed_during_integration()
        par = ca.vertcat(*integration_params.values())

        # if we have no differentials and no algebraics, this function should do nothing
        if (not self.differentials) and (ignore_algebraics or not self.outputs):
            return ca.Function("empty", [[], par], [[], []], ["x0", "p"], ["xf", "zf"])

        x = ca.vertcat(*[sta.sym for sta in self.differentials])
        # if we have a pure ode, we can use an ode solver which is more efficient
        if self.differentials and (ignore_algebraics or not self.outputs):
            ode = {
                "x": x,
                "p": par,
                "ode": self.system,
            }
            return ca.integrator("system", "cvodes", ode, *args)

        # if we have a dae or only algebraic equations, we use a dae solver
        dae = {
            "x": x,
            "p": par,
            "ode": self.system,
            "z": ca.vertcat(*[var.sym for var in self.outputs]),
            "alg": ca.vertcat(*self.output_equations),
        }
        # if there are no differential values, we create a dummy to make integrator
        # callable
        if not self.differentials:
            dae.update({"x": ca.MX.sym("dummy", 1), "ode": 0})

        try:
            return ca.integrator("system", "idas", dae, *args)
        except RuntimeError as e:
            free_vars = e.args[0].split("since", 1)[1]
            raise ConfigurationError(
                "Could not create model, since some equations are not defined. Please"
                " check that all states are either defined by an equation, or by a "
                f"black box model. Currently undefined are: {free_vars}"
            ) from e

    def initialize(self, **ignored):
        """
        Prepare the black- and white-box models for CasADi backend optimization and
        simulation
        """
        # load blackbox models
        pass

    def register_ml_models(
        self,
    ) -> tuple[dict[str, SerializedMLModel], dict[str, CasadiPredictor]]:
        """
        Loads a serialized MLModel and find the output states of the MLModel
        Divides the differential states of the model into states determined by white-box
        model (self._differentials) and by black-box model (self._differentials_network)
        """

        # map all outputs to their respective MLModel
        output_to_ml_model = {}
        ml_model_sources_dict = {
            tuple(ml_model.output.keys()): ml_model
            for ml_model in self.config.ml_model_sources
        }
        ml_model_dict: Dict[str, SerializedMLModel] = {}

        for output in self.config.outputs + self.config.states:
            for serialized_output_names, ml_model in ml_model_sources_dict.items():
                if output.name in serialized_output_names:
                    output_to_ml_model[
                        output.name
                    ] = CasadiPredictor.from_serialized_model(ml_model)
                    ml_model_dict[output.name] = ml_model
        casadi_ml_model_dict: Dict[str, CasadiPredictor] = output_to_ml_model
        return ml_model_dict, casadi_ml_model_dict

    def _fill_algebraic_equations_with_bb_output(self):
        """Fills empty algebraic equations with the function defined by the
        corresponding black box model."""
        for variable_name, serialized_ml_model in self.ml_model_dict.items():
            # recursive features are more like an ode, they don't represent outputs
            if serialized_ml_model.output[variable_name].recursive:
                continue
            if self.get(variable_name).alg is not None:
                raise RuntimeError("")
            inputs = ml_model_datatypes.column_order(
                inputs=serialized_ml_model.input, outputs=serialized_ml_model.output
            )
            input_mx = ca.vertcat(*(self._get_lagged_symbolic(name) for name in inputs))
            index = list(serialized_ml_model.output).index(variable_name)
            alg = self.casadi_ml_model_dict[variable_name].predict(input_mx)[index]
            self.get(variable_name).alg = alg

    def _evaluate_bb_models_symbolically(
        self, bb_inputs_mx: dict[str, ca.MX]
    ) -> dict[str, ca.MX]:
        """
        Returns the CasADi MX-Expressions that result from evaluating all black-box
        models symbolically.
        Args:
            bb_inputs_mx: Dictionary containing the variable names and symbolic MX
             Expressions of all variables, that are used as an input for the black-box
             models. The MX have the dimension of the corresponding maximum lag of said
             variable (Maximum in regard, where multiple black-box models use the same
             input but with different lag).

        Returns:
            Two dictionaries:
                - The first one contains all black-box outputs of the model with their
                 respective symbolic variable.
                - The second one contains the same outputs, with an MX-Expression that
                 defines the evaluation of the black-box model
        """

        bb_result_mx: dict[str, ca.MX] = {}
        # inputs from all MLModels of the black-box model are considered
        for output_name, serialized_ml_model in self.ml_model_dict.items():
            if not serialized_ml_model.output[output_name].recursive:
                # non-recursive outputs are handled as algebraic equations in the
                # integrator for simulation, or as constraints in MPC, so we skip them
                continue

            # for every input variable of the MLModel, create a CasAdi symbolic
            casadi_ml_model = self.casadi_ml_model_dict[output_name]
            columns_ordered = ml_model_datatypes.column_order(
                inputs=serialized_ml_model.input, outputs=serialized_ml_model.output
            )
            # todo tanja: here, we need to lookup what the user specified for the ANN as input, instead of the original mx variable
            ca_nn_input = ca.vertcat(*[bb_inputs_mx[name] for name in columns_ordered])

            # predict the result with current MLModel and add the result to the stage function
            output_index = list(serialized_ml_model.output).index(output_name)
            result = casadi_ml_model.predict(ca_nn_input)[output_index]
            if (
                serialized_ml_model.output[output_name].output_type
                == OutputType.difference
            ):
                result = result + bb_inputs_mx[output_name][0]

            bb_result_mx[output_name] = result
        return bb_result_mx

    def make_predict_function_for_mpc(self) -> ca.Function:
        """Creates a prediction step function which is suitable for MPC with multiple
        shooting."""
        return self._make_unified_predict_function(ignore_algebraics=True)

    def _get_lagged_symbolic(self, name: str):
        """Returns the symbolic ca.MX of a variable, regardless if it is lagged or not."""
        try:
            return self.get(name).sym
        except ValueError:
            return self.lags_mx_store[name]

    def _black_box_inputs(self) -> dict[str, ca.MX]:
        """Creates a dictionary with names for all inputs of the black box functions
        and their corresponding symbolic CasADi-variable."""
        bb_inputs: dict[str, ca.MX] = {}
        for name, lag in self.lags_dict.items():
            for i in range(0, lag):
                l_name = name_with_lag(name, i)
                bb_inputs[l_name] = self._get_lagged_symbolic(l_name)
        return bb_inputs

    def _all_inputs(self) -> dict[str, ca.MX]:
        """Creates a dictionary with names for all inputs of the full step function and
        their corresponding symbolic CasADi-variable."""
        all_variables = {var.name: var.sym for var in self.variables}
        all_variables.update(self._black_box_inputs())
        return all_variables

    def _make_unified_predict_function(
        self, ignore_algebraics: bool = False
    ) -> ca.Function:
        """
        This function creates a predict function which combines all available MLModels,
        their inputs and gives a unified output.
        The constructed stage-function takes the MLModel-variables with their maximum lag as
        input and gives the result of the MLModel as output

        Args:
            ignore_algebraics: When True, algebraic equations will be ignored and no
                idas solver is created. Useful for MPC, where equations can be added as
                constraints and the performance of idas is undesirable
        """
        # initiate in- and output dicts for constructing the stage function

        # create symbolic casadi variables for all inputs used in the MLModel. Each variable
        #  has the length of its maximum lag, i.e. if two MLModels use the variable var1,
        #  one with lag 3 and one with 2, we create a symbolic variable with length 3

        bb_inputs = self._black_box_inputs()
        all_variables = self._all_inputs()
        # evaluate the black box models
        bb_result_mx = self._evaluate_bb_models_symbolically(bb_inputs)
        wb_inputs = self._fixed_during_integration(bb_result_mx)

        # prepare functions that order the integrator inputs and outputs when supplied
        # with keywords names
        differentials_dict = {var.name: var.sym for var in self.differentials}
        if not ignore_algebraics:
            alg_dict = {var.name: var.sym for var in self.outputs}
        else:
            alg_dict = {}
        stacked_alg = ca.vertcat(*[mx for mx in alg_dict.values()])
        diff_states = ca.vertcat(*[mx for mx in differentials_dict.values()])
        names_to_stacked_x = ca.Function(
            "names_to_stacked_x",
            list(differentials_dict.values()),
            [diff_states],
            list(differentials_dict),
            ["x0"],
        )
        stacked_x_to_names = ca.Function(
            "stacked_x_to_names",
            [diff_states],
            list(differentials_dict.values()),
            ["x0"],
            list(differentials_dict),
        )
        stacked_z_to_names = ca.Function(
            "stacked_z_to_names",
            [stacked_alg],
            list(alg_dict.values()),
            ["algs"],
            list(alg_dict),
        )

        # perform symbolic evaluation of the white box equations
        if differentials_dict:
            int_x0_in = names_to_stacked_x(*differentials_dict.values())
        else:
            # have to handle case where differentials are empty separately because
            # CasADi will return a dict instead of an MX if the input is empty.
            int_x0_in = ca.DM([])

        int_p_in = ca.vertcat(*wb_inputs.values())
        integrator = self._make_integrator(ignore_algebraics=ignore_algebraics)
        int_result = integrator(x0=int_x0_in, p=int_p_in)
        x_names = stacked_x_to_names(x0=int_result["xf"])
        z_names = stacked_z_to_names(algs=int_result["zf"])

        opts = {"allow_duplicate_io_names": True} if CASADI_VERSION >= 3.6 else {}
        return ca.Function(
            "full_step",
            list(all_variables.values()),
            list(x_names.values())
            + list(z_names.values())
            + list(bb_result_mx.values()),
            list(all_variables),
            list(x_names) + list(z_names) + list(bb_result_mx),
            opts,
        )

    def do_step(self, *, t_start, t_sample=None):
        """
        Simulates a time step of the simulation model. In CasADi MLModel model, both black-
        and white-box models can be used in the simulation to be combined into a grey-box
        """

        if t_sample:
            ...
            assert t_sample == self.dt

        ml_model_input = self.get_ml_model_values(t_start)
        full_input = {
            var.name: var.value for var in self.variables if var.value is not None
        }
        full_input.update(ml_model_input)

        result = self.sim_step(**full_input)
        end_time = t_start + self.dt
        for var_name, value in result.items():
            self.set_with_timestamp(var_name, value, end_time)

    def get_ml_model_values(self, time: float):
        """
        gets the inputs values with the correct lags or all MLModels
        """
        ml_model_inputs: dict[str, list[float]] = {}
        for inp, lag in self.lags_dict.items():
            if lag == 1:
                continue
            target_grid = [-self.dt * t for t in range(1, lag)]
            target_grid.reverse()
            history = self.past_values[inp].dropna()
            res = sample(history, target_grid, current=time)
            for i, val in enumerate(res):
                ml_model_inputs[name_with_lag(inp, i + 1)] = val

        return ml_model_inputs

    @property
    def bb_states(self) -> List[CasadiState]:
        """List of all CasadiStates with an associated black box equation."""
        return [var for var in self.states if var.name in self.ml_model_dict]

    @property
    def bb_outputs(self) -> List[CasadiOutput]:
        """List of all CasadiStates with an associated black box equation."""
        return [var for var in self.outputs if var.name in self.ml_model_dict]

    @property
    def auxiliaries(self) -> List[CasadiState]:
        """List of all CasadiStates without an associated equation. Common
        uses for this are slack variables that appear in cost functions and
        constraints of optimization models."""
        return [var for var in self.states if self._is_auxiliary(var)]

    def _is_auxiliary(self, var: CasadiState):
        """Checks whether a state does not have any function associated with it and
        belongs to auxiliary variables"""
        if var.ode is not None:
            return False
        if var.name in self.ml_model_dict:
            return False
        return True

    def _assert_outputs_are_defined(self):
        """Raises an Error, if the output variables are not defined with an equation"""
        all_bb_outputs = [
            list(ml_model.output) for ml_model in self.config.ml_model_sources
        ]
        all_bb_outputs_flat = set(itertools.chain.from_iterable(all_bb_outputs))

        for out in self.outputs:
            if out.alg is None and out.name not in all_bb_outputs_flat:
                raise ValueError(
                    f"Output '{out.name}' was not initialized with an "
                    f"equation, nor is it specified by the provied blackbox models. "
                    f"Please sure you specify '{out.name}.alg' in 'setup_system()' or "
                    f"include a model in 'ml_model_sources'."
                )
            if out.alg is not None and out.name in all_bb_outputs:
                raise ValueError(
                    f"Output '{out.name}' is overspecified, as it has an algebraic "
                    f"equation defined in setup_system(), but also in a provided "
                    f"blackbox model. "
                )
