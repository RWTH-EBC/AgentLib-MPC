from typing import List

import numpy as np
import pandas as pd
from agentlib.core import Agent, Environment

from agentlib_mpc.models.casadi_model import (
    CasadiModelConfig,
    CasadiInput,
    CasadiState,
    CasadiParameter,
    CasadiOutput,
    CasadiModel,
)
from agentlib_mpc.modules.ml_model_training.ml_model_trainer import ANNTrainer


def read_weather(file_path):
    """Read weather data from a historic file (path), use the data as disturbance"""
    with open(file_path) as f:
        contents = f.readlines()
    contents = [x.strip() for x in contents]
    # data starts from 32th line
    contents = contents[32:]
    res_dict = {}
    title = contents[0].split()
    contents = contents[2:]
    for i in range(len(title)):
        current_list = []
        for j in range(1, len(contents)):
            current_content = contents[j].split()
            current_list.append(float(current_content[i]))
        res_dict.update({title[i]: current_list})
    weather = pd.DataFrame(res_dict)
    weather = weather.iloc[23:]
    return weather


weather_input = read_weather("TRY2015_Aachen_Jahr.dat")


class TrainConfig_nn(CasadiModelConfig):
    """To enable training, define the configuration of the model that needs to be trained"""

    inputs: List[CasadiInput] = [
        # coupling variables
        CasadiInput(name="T_v", value=295, unit="K", description="Vorlauf temperatur"),
        CasadiInput(name="T_ahu", value=295, unit="K", description="Temperatur AHU"),
        # disturbances
        CasadiInput(
            name="mDot", value=0.1, unit="K", description="Air mass flow into BKA"
        ),
        CasadiInput(
            name="mDot_ahu",
            value=0.025,
            unit="kg",
            description="air water mass flow into AHU",
        ),
        CasadiInput(name="d", value=400, unit="W", description="Heat load into zone"),
        CasadiInput(
            name="T_amb", value=290, unit="K", description="Aussenlufttemperatur"
        ),
        CasadiInput(name="Q_rad", value=300, unit="W/m²", description="Radiation"),
        # settings
        CasadiInput(
            name="T_set",
            value=298.55,
            unit="K",
            description="Set point for T in objective function",
        ),
        CasadiInput(
            name="T_upper", value=302.15, unit="K", description="Upper boundary for T."
        ),
        CasadiInput(
            name="T_lower", value=288.15, unit="K", description="Upper boundary for T."
        ),
    ]
    states: List[CasadiState] = [
        # differential variables
        CasadiState(
            name="T_wall",
            value=290.15,
            unit="K",
            description="Temperature of the wall",
        ),
        CasadiState(
            name="T_air", value=290.15, unit="K", description="Temperature of zone"
        ),
        CasadiState(
            name="T_CCA_0", value=290.15, unit="K", description="Temperatur der BKA"
        ),
        # slack variables
        CasadiState(
            name="T_slack", value=0, unit="K", description="Temperatur der BKA"
        ),
    ]
    parameters: List[CasadiParameter] = [
        CasadiParameter(
            name="cp",
            value=4200,
            unit="J/kg*K",
            description="thermal capacity of water",
        ),
        CasadiParameter(
            name="c_BKA",
            value=500000,
            unit="J/kg*K",
            description="thermal capacity of zone 0",
        ),
        CasadiParameter(
            name="cw", value=518000, unit="J/kg*K", description="Wärmekapazität Wand"
        ),
        CasadiParameter(
            name="cl", value=1000, unit="J/kg*K", description="Wärmekapazität Luft"
        ),
        CasadiParameter(
            name="hw", value=0.17, unit="J/kg*K", description="Leitfähigkeit Wand"
        ),
        CasadiParameter(
            name="hBKA", value=2, unit="J/kg*K", description="Leitfähigkeit BKA"
        ),
        CasadiParameter(
            name="hFenster",
            value=1.23,
            unit="J/kg*K",
            description="Leitfähigkeit Fenster",
        ),
        CasadiParameter(name="Aw", value=13.85, unit="m2", description="Fläche Wand"),
        CasadiParameter(name="ABKA", value=39.5, unit="m2", description="Fläche BKA"),
        CasadiParameter(
            name="AFenster", value=6.6, unit="m2", description="Fläche Fenster"
        ),
        CasadiParameter(name="mRoom", value=60, unit="kg", description="Fläche BKA"),
        CasadiParameter(
            name="q_T",
            value=1,
            unit="-",
            description="Weight for T in objective function",
        ),
        CasadiParameter(
            name="s_T",
            value=1,
            unit="-",
            description="Weight for T in objective function",
        ),
    ]
    outputs: List[CasadiOutput] = []
    dt: float = 1800


class Train_NN(CasadiModel):
    """Define the dynamics of the model with DAE, using the variables and parameters defined in the configuration"""

    config: TrainConfig_nn

    def setup_system(self):
        # Define ODE
        self.T_CCA_0.ode = self.cp * self.mDot * (self.T_v - self.T_CCA_0) / (
            self.c_BKA * self.ABKA
        ) + self.hBKA / self.c_BKA * (self.T_air - self.T_CCA_0)
        self.T_wall.ode = (
            self.hw / self.cw * (self.T_air - self.T_wall)
            + self.hw / self.cw * (self.T_amb - self.T_wall)
            + self.Q_rad / self.cw
        )
        self.T_air.ode = (
            self.hw * self.Aw / (self.cl * self.mRoom) * (self.T_wall - self.T_air)
            + self.hBKA
            * self.ABKA
            / (self.cl * self.mRoom)
            * (self.T_CCA_0 - self.T_air)
            + self.d / (self.cl * self.mRoom)
            + (self.T_ahu - self.T_air) * self.mDot_ahu / self.mRoom
            + self.hFenster
            * self.AFenster
            / (self.cl * self.mRoom)
            * (self.T_amb - self.T_air)
            + self.Q_rad * self.AFenster / (self.cl * self.mRoom)
        )

        # Define AE
        self.constraints = []

        # Objective function
        objective = sum([])

        return objective


def random_create(steps: int, iter: int, model: CasadiModel):
    """
    Creates inputs for the ANN; the inputs include:
    random state variable and historic input variables
    :param steps: inputs mit der Länge steps werden erzeugt, dies beschreibt die Länge der Simulation
    :param iter: die iter-te Simulation, die simuliert wird. Wird benutzt, um historische Daten auszuwählen
    :param model: the model to perform data creation with
    :return: The states at the start of the simulation and the inputs throughout the simulation
    """
    true_iter = iter % (len(weather_input) // steps - 1)
    amb_temp = np.array([weather_input["t"] + 273.15]).T
    solar_radiation = np.array([weather_input["B"]]).T
    x0 = np.zeros(1)
    # for all states, generate a random start value close to the model definition
    for state in model.differentials:
        x = 0.9 * state.value + 0.2 * state.value * np.random.rand(1)
        x0 = np.append(x0, x)

    # create good input values for all variables
    u0 = 1 * np.random.rand(steps, 1)
    for i, input_ in enumerate(model.inputs):
        if input_.name == "d":
            u = input_.value * np.random.rand(steps, 1)
            u0 = np.concatenate((u0, u), axis=1)
        elif input_.name == "Q_rad":
            u = solar_radiation[true_iter * steps : true_iter * steps + steps]
            u0 = np.concatenate((u0, u), axis=1)
        elif input_.name == "T_amb":
            u = amb_temp[true_iter * steps : true_iter * steps + steps]
            u0 = np.concatenate((u0, u), axis=1)
        elif input_.name == "T_v" or input_.name == "T_ahu":
            u = 275 + 40 * np.random.rand(steps, 1)
            u0 = np.concatenate((u0, u), axis=1)
        else:
            u = input_.value * np.ones((steps, 1))
            u0 = np.concatenate((u0, u), axis=1)
    return x0, u0[:, 1 : len(model.inputs) + 1]


def simulate(self, x0, u) -> np.array:
    x0 = x0[1 : len(self.output_value) + 1]
    x = np.expand_dims(x0, axis=1).T
    p = np.ones((u.shape[0], 1))
    for parval in self.par_value:
        parvector = parval * np.ones((u.shape[0], 1))
        p = np.concatenate((p, parvector), axis=1)
    p = p[:, 1 : len(p)]
    if len(self.output_value) > 0:
        for j in range(0, u.shape[0] - 1):
            x_next = np.array(
                self.do_step(x=x[j], u=np.concatenate((u[j], p[j]), axis=0)).T
            )
            x = np.concatenate((x, x_next), axis=0)
    else:
        for j in range(0, u.shape[0] - 1):
            x_next = np.array(
                self.do_step(x=x0, u=np.concatenate((u[j], p[j]), axis=0)).T
            )
            x = np.append(x, x_next, axis=0)
    data = np.concatenate([x, u], axis=1)
    return data


def generate_data(self, n_simulations: int, simulation_steps: int, ann_input):
    simulations = []
    print("-------------- Generating Data --------------")
    for j in range(n_simulations):
        print(f"{j} of {n_simulations}")
        # generate data for simulation
        x0, u = self.random_create(steps=simulation_steps, iter=j)
        # save the results of the simulation
        x = self.simulate(x0, u)
        simulations.append(x)
    x, y = None, None
    for simulation in simulations:
        data_to_train = simulation
        data_label = self.output_denotation + self.input_denotation
        data_df = pd.DataFrame(
            data_to_train, columns=data_label
        )  # ignore unmessurable variables
        trained_data_den = list(ann_input["out"].keys()) + list(
            ann_input["in"].keys()
        )  # order matters
        trained_data = data_df[trained_data_den]
        x_train, d_y = self.process_data(trained_data, ann_input)
        if x is None:
            x = x_train
            # Y = y
            y = d_y
        else:
            x = pd.concat([x, x_train], axis=0)
            y = pd.concat([y, d_y], axis=0)
    return {"input": x, "output": y}


class Datagenerator:
    """Generates data with the CasADi model"""

    def __init__(self):
        self.model = Train_NN()
        self.opt_integrator = self.model.integrator
        # Reading the CasADi model, filter the relevant input, output variables and parameters
        self.output_denotation = []
        self.output_value = []
        self.input_denotation = []
        self.input_value = []
        self.par_denotation = []
        self.par_value = []
        self.init_inout()

    def init_inout(self):
        self.get_output()
        self.get_input()
        self.get_parameter()

    def get_output(self):
        for states in self.model.differentials:
            self.output_denotation.append(states.name)
            self.output_value.append(states.value)

    def get_input(self):
        for inputs in self.model.inputs:
            self.input_denotation.append(inputs.name)
            self.input_value.append(inputs.value)
        return self.input_denotation, self.input_value

    def get_parameter(self):
        for par in self.model.parameters:
            self.par_denotation.append(par.name)
            self.par_value.append(par.value)

    def find_lags(self, ann_input):
        state_lags = []
        input_lags = []
        for state in ann_input["out"]:
            state_lags.append(ann_input["out"][state])
        for input_var in ann_input["in"]:
            input_lags.append(ann_input["in"][input_var])
        return state_lags, input_lags

    def do_step(self, x, u):
        """
        Calculate the state variables in the next timestep
        :param x: state variable now
        :param u: control variable now
        :return: value of the next timestep
        """
        fk = self.opt_integrator(x0=x, p=u)
        x_next = fk["xf"]
        # needs to be improved: also calculate the algebraics which describe the system
        # z_next = Fk['zf']
        # if len(self.output_value) > 0:
        #     return ca.vertcat(x_next,z_next)
        # else:
        return x_next

    def random_create(self, steps: int, iter: int):
        """
        Creates inputs for the ANN; the inputs include:
        random state variable and historic input variables
        :param steps: inputs mit der Länge steps werden erzeugt, dies beschreibt die Länge der Simulation
        :param iter: die iter-te Simulation, die simuliert wird. Wird benutzt, um historische Daten auszuwählen
        :return: The states at the start of the simulation and the inputs throughout the simulation
        """
        true_iter = iter % (len(weather_input) // steps - 1)
        amb_temp = np.array([weather_input["t"] + 273.15]).T
        solar_radiation = np.array([weather_input["B"]]).T
        x0 = np.zeros(1)
        for val in self.output_value:  # for all states, generate a value per state
            x = 0.9 * val + 0.2 * val * np.random.rand(1)
            x0 = np.append(x0, x)
        u0 = 1 * np.random.rand(steps, 1)
        # value range specified for each variable
        for index, inputval in enumerate(self.input_value):
            if self.input_denotation[index] == "d":
                u = inputval * np.random.rand(steps, 1)
                u0 = np.concatenate((u0, u), axis=1)
            elif self.input_denotation[index] == "Q_rad":
                u = solar_radiation[true_iter * steps : true_iter * steps + steps]
                u0 = np.concatenate((u0, u), axis=1)
            elif self.input_denotation[index] == "T_amb":
                u = amb_temp[true_iter * steps : true_iter * steps + steps]
                u0 = np.concatenate((u0, u), axis=1)
            elif (
                self.input_denotation[index] == "T_v"
                or self.input_denotation[index] == "T_ahu"
            ):
                u = 275 + 40 * np.random.rand(steps, 1)
                u0 = np.concatenate((u0, u), axis=1)
            else:
                u = inputval * np.ones((steps, 1))
                u0 = np.concatenate((u0, u), axis=1)
        return x0, u0[:, 1 : len(self.input_value) + 1]

    def simulate(self, x0, u) -> np.array:
        x0 = x0[1 : len(self.output_value) + 1]
        x = np.expand_dims(x0, axis=1).T
        p = np.ones((u.shape[0], 1))
        for parval in self.par_value:
            parvector = parval * np.ones((u.shape[0], 1))
            p = np.concatenate((p, parvector), axis=1)
        p = p[:, 1 : len(p)]
        if len(self.output_value) > 0:
            for j in range(0, u.shape[0] - 1):
                x_next = np.array(
                    self.do_step(x=x[j], u=np.concatenate((u[j], p[j], [0]), axis=0)).T
                )
                x = np.concatenate((x, x_next), axis=0)
        else:
            for j in range(0, u.shape[0] - 1):
                x_next = np.array(
                    self.do_step(x=x0, u=np.concatenate((u[j], p[j]), axis=0)).T
                )
                x = np.append(x, x_next, axis=0)
        data = np.concatenate([x, u], axis=1)
        return data

    def generate_data(self, n_simulations: int, simulation_steps: int):
        simulations = []
        print("-------------- Generating Data --------------")
        for j in range(n_simulations):
            print(f"{j} of {n_simulations}")
            # generate data for simulation
            x0, u = self.random_create(steps=simulation_steps, iter=j)
            # save the results of the simulation
            x = self.simulate(x0, u)
            columns = [var.name for var in self.model.differentials + self.model.inputs]
            simulation = pd.DataFrame(x, columns=columns)
            simulations.append(simulation)

        full_data = pd.concat(simulations)
        dt = self.model.dt
        full_data.index = np.arange(0, full_data.shape[0] * dt, dt)

        return full_data
        # x, y = None, None
        # for simulation in simulations:
        #     data_to_train = simulation
        #     data_label = self.output_denotation + self.input_denotation
        #     data_df = pd.DataFrame(
        #         data_to_train, columns=data_label
        #     )  # ignore unmessurable variables
        #     trained_data_den = list(ann_input["out"].keys()) + list(
        #         ann_input["in"].keys()
        #     )  # order matters
        #     trained_data = data_df[trained_data_den]
        #     x_train, d_y = self.process_data(trained_data, ann_input)
        #     if x is None:
        #         x = x_train
        #         # Y = y
        #         y = d_y
        #     else:
        #         x = pd.concat([x, x_train], axis=0)
        #         y = pd.concat([y, d_y], axis=0)
        # return {"input": x, "output": y}

    def process_data(self, data, ann_input):
        """
        processes the pandas dataframe so that the output is also a pd dataframe with denotations
        The data is converted to an array once at the network trainer
        """
        # find max lag
        state_lags, input_lags = self.find_lags(ann_input)
        max_lag = max(state_lags + input_lags)

        # starting the creation of dataframe
        training_df = pd.DataFrame(np.array(data.iloc[max_lag - 1 : -1, 0]))
        # concatenate the lags
        for index, key in enumerate(ann_input["out"]):  # states as input
            processed_variable = np.array(data[key])[:-1]
            lag = ann_input["out"][key]
            for j in range(lag):
                var_train = processed_variable[
                    max_lag - lag + j : len(processed_variable) - lag + 1 + j
                ]
                var_train_df = pd.DataFrame(
                    var_train, columns=[key]
                )  # no index in label for analysis purposes
                training_df = pd.concat([training_df, var_train_df], axis=1)
        for index, key in enumerate(ann_input["in"]):  # disturbances as input
            processed_variable = np.array(data[key])[:-1]
            lag = ann_input["in"][key]
            for j in range(lag):
                var_train = processed_variable[
                    max_lag - lag + j : len(processed_variable) - lag + 1 + j
                ]
                var_train_df = pd.DataFrame(var_train, columns=[key])
                training_df = pd.concat([training_df, var_train_df], axis=1)
        # delete the first column
        training_df = training_df.drop(columns=training_df.columns[0], axis=1)

        # create dy as output
        output_df = pd.DataFrame(np.array(data.iloc[max_lag:, 0]))
        # create dataframe for the delta output
        for index, key in enumerate(ann_input["out"]):  # states as input
            processed_variable = np.array(data[key])[max_lag - 1 :]
            d_out = np.zeros(shape=(len(processed_variable) - 1))
            for j in range(1, len(processed_variable)):
                d_out[j - 1] = processed_variable[j] - processed_variable[j - 1]
            var_output_df = pd.DataFrame(d_out, columns=[key])
            output_df = pd.concat([output_df, var_output_df], axis=1)
        # delete the first column
        output_df = output_df.drop(columns=output_df.columns[0], axis=1)

        return training_df, output_df

    def array_split(self, sequence: dict, training: int, validation: int, test: int):
        """
        splits the input and output data for training.
        The proportaions can be defined
        """
        # in case the sum is not 100
        dataset_sum = training + validation + test
        training = training / dataset_sum
        validation = validation / dataset_sum

        input_data = np.array(sequence["input"])
        output_data = np.array(sequence["output"])
        x = []
        y = []

        x.append(input_data[: round(input_data.shape[0] * training)])
        x.append(
            input_data[
                round(input_data.shape[0] * training)
                - 1 : round(input_data.shape[0] * (training + validation))
            ]
        )
        x.append(input_data[round(input_data.shape[0] * (training + validation)) - 1 :])

        y.append(output_data[: round(output_data.shape[0] * training)])
        y.append(
            output_data[
                round(output_data.shape[0] * training)
                - 1 : round(output_data.shape[0] * (training + validation))
            ]
        )
        y.append(
            output_data[round(output_data.shape[0] * (training + validation)) - 1 :]
        )

        return {
            "x_train": x[0],
            "x_validate": x[1],
            "x_test": x[2],
            "y_train": y[0],
            "y_validate": y[1],
            "y_test": y[2],
        }


def generate_test_data() -> pd.DataFrame:
    inputs = list(np.random.random((10000, 1)) * 100 - 50)
    func = lambda x: 2 * x
    outputs = [func(x) for x in inputs]
    func2 = lambda x: x + 10
    outputs2 = [func2(x) for x in inputs]
    df = pd.DataFrame({"inputs": inputs, "outputs": outputs, "outputs2": outputs2})
    return df

    # df.to_csv("test_data.csv")


def generate_trainers() -> tuple[ANNTrainer, ANNTrainer]:
    ag_config = {
        "modules": [],
        "id": "ann",
    }
    trainer_config = {
        "step_size": 1800,
        "type": "agentlib_mpc.ann_trainer",
        "epochs": 400,
        "batch_size": 64,
        "layers": [{32, "sigmoid"}],
        "train_share": 0.6,
        "validation_share": 0.2,
        "test_share": 0.2,
        "retrain_delay": 2,
        "save_directory": "anns",
        "use_values_for_incomplete_data": True,
        "save_data": True,
        "save_ml_model": True,
        "save_plots": True,
        "early_stopping": {"activate": "True", "patience": 500},
    }
    t_air_trainer_config = {
        **trainer_config,
        "module_id": "t_air",
        "inputs": [
            {"name": "T_CCA_0"},
            {"name": "T_ahu"},
            {"name": "mDot_ahu"},
            {"name": "d"},
            {"name": "T_amb"},
            {"name": "Q_rad"},
        ],
        "outputs": [{"name": "T_air"}],
        "output_types": {"T_air": "difference"},
        "lags": {
            "T_CCA_0": 1,
            "T_ahu": 1,
            "mDot_ahu": 1,
            "d": 2,
            "T_amb": 1,
            "Q_rad": 2,
            "T_air": 1,
        },
    }
    t_cca_trainer_config = {
        **trainer_config,
        "module_id": "t_cca",
        "inputs": [
            {"name": "T_air"},
            {"name": "T_v"},
            {"name": "d"},
            {"name": "mDot"},
        ],
        "lags": {
            "T_air": 1,
            "T_v": 3,
            "d": 1,
            "mDot": 2,
            "T_CCA_0": 1,
        },
        "outputs": [{"name": "T_CCA_0"}],
        "output_types": {"T_CCA_0": "difference"},
    }
    t_air_trainer = ANNTrainer(
        agent=Agent(env=Environment(), config=ag_config), config=t_air_trainer_config
    )
    cca_trainer = ANNTrainer(
        agent=Agent(env=Environment(), config=ag_config), config=t_cca_trainer_config
    )
    return t_air_trainer, cca_trainer


def main():
    t_air_trainer, cca_trainer = generate_trainers()
    # data = random_create()

    generator = Datagenerator()
    generated_data = generator.generate_data(
        n_simulations=4,
        simulation_steps=1000,
    )
    for trainer in t_air_trainer, cca_trainer:
        trainer.time_series_data = generated_data[trainer.time_series_data.columns]
        sampled = trainer.resample()
        inputs, outputs = trainer.create_inputs_and_outputs(sampled)
        training_data = trainer.divide_in_tvt(inputs, outputs)
        trainer.fit_ml_model(training_data)
        serialized_ann = trainer.serialize_ml_model()
        trainer.save_all(serialized_ann, training_data)

    # trainer.history_dict["outputs"] = list(test_data.index), list(test_data["outputs"])
    # trainer.history_dict["outputs2"] = list(test_data.index), list(
    #     test_data["outputs2"]
    # )
    # trainer.history_dict["inputs"] = list(test_data.index), list(test_data["inputs"])
    # trainer._update_time_series_data()


if __name__ == "__main__":
    main()
