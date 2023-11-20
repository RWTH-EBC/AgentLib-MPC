import pandas as pd
import pydantic
from agentlib.core import AgentVariables, AgentVariable
from agentlib.modules.utils.try_sensor import TRYSensorConfig, TRYSensor


class TRYPredictorConfig(TRYSensorConfig):
    prediction_length: int = pydantic.Field(
        default=24, description="prediction length in hours"
    )

    predictions: AgentVariables = [
        AgentVariable(
            name="T_oda_prediction",
            unit="K",
            description="Air temperature 2m over ground [K]",
        ),
        AgentVariable(
            name="pressure_prediction",
            unit="hPa",
            description="Air pressure in standard height [hPa]",
        ),
        AgentVariable(
            name="wind_direction_prediction",
            unit="°",
            description="Wind direction 10 m above gorund " "[Grad] {0..360;999}",
        ),
        AgentVariable(
            name="wind_speed_prediction",
            unit="m/s",
            description="Wind speed 10 m above ground [m/s]",
        ),
        AgentVariable(
            name="coverage_prediction", unit="eighth", description="[eighth]  {0..8;9}"
        ),
        AgentVariable(
            name="absolute_humidity_prediction", unit="g/kg", description="[g/kg]"
        ),
        AgentVariable(
            name="relative_humidity_prediction",
            unit="%",
            description="Relative humidity 2 m above ground " "[%] {1..100}",
        ),
        AgentVariable(
            name="beam_direct_prediction",
            unit="W/m^2",
            description="Direct beam of sun (hor. plane) "
            "[W/m^2] downwards: positive",
        ),
        AgentVariable(
            name="beam_diffuse_prediction",
            unit="/m^2",
            description="Diffuse beam of sun (hor. plane) "
            "[W/m^2] downwards: positive",
        ),
        AgentVariable(
            name="beam_atm_prediction",
            unit="/m^2",
            description="Beam of atmospheric heat (hor. plane) "
            "[W/m^2] downwards: positive",
        ),
        AgentVariable(
            name="beam_terr_prediction",
            unit="/m^2",
            description="Beam of terrestrial heat " "[W/m^2] upwards: negative",
        ),
    ]


class TRYPredictor(TRYSensor):
    config: TRYPredictorConfig
    _data: pd.DataFrame

    def process(self):
        while True:
            self.send_prediction()
            self.send_measurement()
            yield self.env.timeout(self.t_sample)

    def send_prediction(self):
        start_loc = self._data.index.get_loc(self.env.now, method="pad")
        start_time = self._data.index[start_loc]
        end_time = start_time + self.config.prediction_length * 3600
        for measurement_name, measurement_data in self._data.iteritems():
            self.set(measurement_name, measurement_data.loc[start_time:end_time])

    def send_measurement(self):
        data = self.get_data_now()
        for key, val in data.items():
            self.set(name=key, value=val)
