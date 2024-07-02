import agentlib as al
import numpy as np
import pandas as pd

from examples.three_zone_datadriven_admm.three_zone_util import (
    load_weather,
    heat_load_func,
    get_t_aussen,
    irradiation_data,
)


class PredictorModuleConfig(al.BaseModuleConfig):
    """Module that outputs a prediction of the heat load at a specified
    interval."""

    outputs: al.AgentVariables = [
        al.AgentVariable(
            name="load_prediction", type="pd.Series", description="test_description"
        ),
        al.AgentVariable(
            name="set_point_schedule_high",
            type="pd.Series",
            description="test_description",
        ),
        al.AgentVariable(
            name="set_point_schedule_low",
            type="pd.Series",
            description="test_description",
        ),
        al.AgentVariable(name="set_point", type="pd.Series", description="T_set"),
        al.AgentVariable(
            name="T_amb_pred", type="pd.Series", description="Außentemperatur"
        ),
        al.AgentVariable(
            name="Q_solar", type="pd.Series", description="Solar radiation"
        ),
    ]

    parameters: al.AgentVariables = [
        # prediction times
        al.AgentVariable(
            name="sampling_time", value=10, description="Sampling time for prediction."
        ),
        al.AgentVariable(
            name="prediction_length",
            value=10,
            description="Number of sampling points for prediction.",
        ),
        al.AgentVariable(
            name="update_interval",
            value=1800,
            description="Time between prediction updates.",
        ),
        al.AgentVariable(
            name="comfort_interval",
            value=14400,
            description="Time between comfort updates.",
        ),
        # temperature boundaries
        al.AgentVariable(
            name="comfort_high_upper",
            value=301.15,
            description="High value in the comfort set point.",
        ),
        al.AgentVariable(
            name="comfort_low_upper",
            value=296.15,
            description="Low value in the comfort set point.",
        ),
        al.AgentVariable(
            name="comfort_high_lower",
            value=294.15,
            description="High value in the comfort set point.",
        ),
        al.AgentVariable(
            name="comfort_low_lower",
            value=290.15,
            description="Low value in the comfort set point.",
        ),
        # working hours
        al.AgentVariable(
            name="daystart",
            value=4,
            description="when does the working day start, 6 Uhr z.B.",
        ),
        al.AgentVariable(
            name="dayend",
            value=16,
            description="when does the working day end, 6 Uhr z.B.",
        ),
        al.AgentVariable(
            name="max_disturbance", value=250, description="maximale interne Gewinne"
        ),
        al.AgentVariable(
            name="T_amb_const", value=299, description="ambient temperature constant"
        ),
        al.AgentVariable(
            name="start_month",
            value=3,
            description="month of simulation start",
        ),
        al.AgentVariable(
            name="start_day",
            value=15,
            description="day of simulation start",
        ),
    ]

    shared_variable_fields: list[str] = ["outputs"]


class PredictorModule(al.BaseModule):
    """Module that outputs a prediction of the heat load at a specified
    interval."""

    config: PredictorModuleConfig

    def __init__(self, *, config: dict, agent: al.Agent):
        super().__init__(config=config, agent=agent)
        self.weather_data = load_weather("TRY2015_Aachen_Jahr.dat")
        room_no = int(self.id[-1])
        self.radiation_data = irradiation_data[f"Room {room_no}"]
        self.target_temp = [295, 295]

    def register_callbacks(self):
        pass

    def process(self):
        """Sets a new prediction at each time step."""
        self.env.process(self.send_comfort_trajectories())
        while True:
            ts = self.get("sampling_time").value
            n = self.get("prediction_length").value
            start_month = self.get("start_month").value
            start_day = self.get("start_day").value
            now = self.env.now
            update_interval = self.get("update_interval").value
            # temperature prediction
            grid = np.arange(now, now + n * ts, ts)
            values = heat_load_func(
                current=grid,
                begin=self.get("daystart").value,
                end=self.get("dayend").value,
                max_disturbance=self.get("max_disturbance").value,
            )
            traj = pd.Series(values, index=list(grid))
            self.set("load_prediction", traj)
            # read T-aussen trajectory
            taussen = get_t_aussen(
                current=grid,
                weather_data=self.weather_data,
                month=start_month,
                day=start_day,
            )
            taussen_traj = pd.Series(taussen, index=list(grid))
            self.set("T_amb_pred", taussen_traj)
            # read Solar radiation trajectory
            self.set("Q_solar", self.radiation_data)
            yield self.env.timeout(update_interval)

    def send_comfort_trajectories(self):
        """Sends the series for the comfort condition."""
        while True:
            now = self.env.now
            comfort_interval = self.get("comfort_interval").value
            if now < 1:
                self.boundary_low = [self.get("comfort_low_lower").value] * 2
                self.boundary_high = [self.get("comfort_high_upper").value] * 2
            tag = now // (24 * 3600) + 1
            tageszeit = now % (24 * 3600)
            daystart = self.get("daystart").value
            dayend = self.get("dayend").value
            # temperature prediction
            grid = np.arange(now, now + 7 * comfort_interval, 0.5 * comfort_interval)
            # weekends
            if tag > 5:
                values_high_pre = [self.get("comfort_high_upper")] * 12
                values_low_pre = [self.get("comfort_low_lower")] * 12
                # values_set_pre = list(values_low_pre[0]+np.random.randint(2, size=2))
                values_set_pre = [296] * 12
            # monday to friday
            else:
                values_high_pre = []
                values_low_pre = []
                values_set_pre = []
                for i in np.arange(
                    tageszeit + 1 * comfort_interval,
                    tageszeit + 7 * comfort_interval,
                    0.5 * comfort_interval,
                ):
                    tageszeit_2 = i % (24 * 3600)
                    tag2 = i // (24 * 3600) + 1
                    if daystart <= tageszeit_2 // 3600 < dayend:
                        if tag2 > 5:
                            values_high_pre.append(self.get("comfort_high_upper").value)
                            values_low_pre.append(self.get("comfort_low_lower").value)
                            values_set_pre.append(295)
                        else:
                            values_high_pre.append(self.get("comfort_low_upper").value)
                            values_low_pre.append(self.get("comfort_high_lower").value)
                            values_set_pre.append(295)
                    else:
                        values_high_pre.append(self.get("comfort_high_upper").value)
                        values_low_pre.append(self.get("comfort_low_lower").value)
                        values_set_pre.append(295)
            values = self.boundary_high + values_high_pre
            values_low = self.boundary_low + values_low_pre
            values_set = self.target_temp + values_set_pre
            self.target_temp = values_set_pre[0:2]
            self.boundary_high = values_high_pre[0:2]
            self.boundary_low = values_low_pre[0:2]
            traj_high = pd.Series(values, index=list(grid))
            traj_low = pd.Series(values_low, index=list(grid))
            traj_set = pd.Series(values_set, index=list(grid))
            self.set("set_point_schedule_high", traj_high)
            self.set("set_point_schedule_low", traj_low)
            self.set("set_point", traj_set)
            yield self.env.timeout(comfort_interval)
