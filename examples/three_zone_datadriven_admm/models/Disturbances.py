import agentlib as al
import numpy as np
import pandas as pd

from examples.three_zone_datadriven_admm.three_zone_util import (
    heat_load_func,
    get_t_aussen,
    weather_data,
    irradiation_data,
)


class LoadSensorConfig(al.ModelConfig):
    """Modification through .json configuration"""

    outputs: al.ModelOutputs = [
        al.ModelOutput(name="load_measurement"),
        al.ModelOutput(name="T_amb_measure"),
        al.ModelOutput(name="Q_solar"),
        al.ModelOutput(name="set_point_low"),
        al.ModelOutput(name="set_point_high"),
    ]

    parameters: al.ModelParameters = [
        al.ModelParameter(name="uncertainty", value=0),
        al.ModelParameter(name="daystart", value=4),
        al.ModelParameter(name="dayend", value=16),
        al.ModelParameter(name="max_disturbance", value=0),
        al.ModelParameter(name="T_amb_const", value=299),
        al.ModelParameter(name="comfort_high_upper", value=301.15),
        al.ModelParameter(name="comfort_low_upper", value=296.15),
        al.ModelParameter(name="comfort_high_lower", value=294.15),
        al.ModelParameter(name="comfort_low_lower", value=290.15),
        al.ModelParameter(name="start_month", value=3),
        al.ModelParameter(name="start_day", value=15),
    ]


class LoadSensor(al.Model):
    """Sensor model for output"""

    config: LoadSensorConfig

    def do_step(self, *, t_start, t_sample=None):
        """Returns the current load measurement, with random noise added."""
        random_modifier = 1 + self.uncertainty * (np.random.random() - 0.5)
        d = random_modifier * heat_load_func(
            t_start,
            self.daystart.value,
            self.dayend.value,
            self.max_disturbance.value,
        )
        self._set_output_value("load_measurement", d)
        start_month = self.get("start_month").value
        start_day = self.get("start_day").value
        t_aussen = get_t_aussen(
            current=t_start, weather_data=weather_data, month=start_month, day=start_day
        )
        q_solar_col: pd.Series = irradiation_data[self.config.name]
        index = q_solar_col.loc[:t_start].last_valid_index()
        q_solar = int(q_solar_col[index])
        t_upper, t_lower = self.get_Temp_bounds(t_start)
        self._set_output_value("set_point_low", t_lower)
        self._set_output_value("set_point_high", t_upper)
        self._set_output_value("T_amb_measure", t_aussen)
        self._set_output_value("Q_solar", q_solar)

    def get_Temp_bounds(self, current):
        tag = current // (24 * 3600) + 1
        tageszeit = current % (24 * 3600)
        if tag < 6 and self.daystart.value <= tageszeit // 3600 < self.dayend.value:
            return self.comfort_low_upper.value, self.comfort_high_lower.value
        else:
            return self.comfort_high_upper.value, self.comfort_low_lower.value

    def initialize(self, **kwargs):
        pass
