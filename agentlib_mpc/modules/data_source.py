from pathlib import Path

import pandas as pd
import numpy as np
import logging
from pydantic import Field, field_validator, FilePath
from typing import List, Optional, Union
from datetime import datetime, timedelta

from agentlib.core import BaseModule, BaseModuleConfig, AgentVariable
from agentlib import Environment, Agent
from agentlib_mpc.data_structures.interpolation import InterpolationMethods


class DataSourceConfig(BaseModuleConfig):
    data: Union[pd.DataFrame, FilePath] = Field(
        title="data",
        default=pd.DataFrame(),
        description="Data that should be communicated during execution."
        "Index should be either numeric or Datetime, numeric values are interpreted as seconds.",
        validate_default=True,
    )
    columns: Optional[List[str]] = Field(
        title="columns",
        default=None,
        description="Optional list of columns of data frame that should be sent."
        "If ommited, all datapoint in frame are sent.",
    )
    t_sample: Union[float, int] = Field(
        title="t_sample",
        default=1,
        description="Sample time of data source. Default is 1 s.",
    )
    data_offset: Optional[Union[pd.Timedelta, float]] = Field(
        title="data_offset",
        default=0,
        description="Offset will be subtracted from index.",
    )
    interpolation_method: Optional[InterpolationMethods] = Field(
        title="interpolation_method",
        default=InterpolationMethods.previous,
        description="Interpolation method used for resampling of data."
        "Only 'linear' and 'previous' are allowed.",
    )

    @field_validator("data")
    @classmethod
    def check_data(cls, data):
        """Makes sure data is a data frame, and loads it if required."""
        if isinstance(data, (str, Path)) and Path(data).is_file():
            data = pd.read_csv(data, engine="python", index_col=0)
        if not isinstance(data, pd.DataFrame):
            raise ValueError(
                f"Data {data} is not a valid DataFrame or the path is not found."
            )
        if data.empty:
            raise ValueError("Dataframe 'data' is empty.")
        return data

    @field_validator("interpolation_method")
    @classmethod
    def check_interpolation_method(cls, interpolation_method):
        if interpolation_method not in {
            InterpolationMethods.linear,
            InterpolationMethods.previous,
        }:
            raise ValueError(
                "Only 'linear' and 'previous' are allowed interpolation methods."
            )
        return interpolation_method


class DataSource(BaseModule):
    config: DataSourceConfig

    def __init__(self, config: dict, agent: Agent):
        super().__init__(config=config, agent=agent)
        data = self.config.data
        data = self.transform_index(data)

        # Filter columns if specified
        if self.config.columns:
            columns_to_keep = [
                col for col in self.config.columns if col in data.columns
            ]
            if not columns_to_keep:
                raise ValueError("None of the specified columns exist in the dataframe")
            data = data[columns_to_keep]

        if data.empty:
            raise ValueError("Resulting dataframe is empty after processing")

    def transform_index(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handles the index and ensures it is numeric, with correct offset"""
        offset = self.config.data_offset
        # Convert offset to seconds if it's a Timedelta
        if isinstance(offset, pd.Timedelta):
            offset = offset.total_seconds()
        # Handle different index types
        if isinstance(data.index, pd.DatetimeIndex):
            data.index = (data.index - data.index[0]).total_seconds()
        else:
            # Try to convert to numeric if it's a string
            try:
                data.index = pd.to_numeric(data.index)
                data.index = data.index - data.index[0]
            except ValueError:
                # If conversion to numeric fails, try to convert to datetune
                try:
                    data.index = pd.to_datetime(data.index)
                    data.index = (data.index - data.index[0]).total_seconds()
                except ValueError:
                    raise ValueError("Unable to convert index to numeric format")

        data.index = data.index.astype(float) - offset
        return data

    def _get_data_at_time(
        self,
        timestamp: float,
        interpolation_method: InterpolationMethods = InterpolationMethods.previous,
    ) -> pd.Series:
        df = self.config.data
        after = df[df.index >= timestamp].first_valid_index()
        before = df[df.index <= timestamp].last_valid_index()
        if after is None:
            self.logger.warning(
                f"The timestamp {timestamp} is after the range of the data."
            )
            return df.iloc[-1]
        if before is None:
            self.logger.warning(
                f"The timestamp {timestamp} is before the range of the data."
            )
            return df.iloc[0]
        if before == after:
            return df.loc[before]
        # Extract the two points
        df_surrounding = df.loc[[before, after]]
        if interpolation_method == InterpolationMethods.linear:
            return (
                df_surrounding.reindex(df_surrounding.index.union([timestamp]))
                .interpolate(method="index")
                .loc[timestamp]
            )
        elif interpolation_method == InterpolationMethods.previous:
            return df_surrounding.iloc[0]
        else:
            self.logger.warning(
                f"Interpolation method {interpolation_method} not supported."
            )
            return df_surrounding.iloc[0]

    def process(self):
        """Write the current data values into data_broker every t_sample"""
        while True:
            current_data = self._get_data_at_time(
                self.env.now, self.config.interpolation_method
            )
            for index, value in current_data.items():
                self.logger.debug(
                    f"At {self.env.now}: Sending variable {index} with value {value} to data broker."
                )
                variable = AgentVariable(name=index, value=value, shared=True)
                self.agent.data_broker.send_variable(variable, copy=False)
            yield self.env.timeout(self.config.t_sample)

    def register_callbacks(self):
        """Don't do anything as this module is not event-triggered"""


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    date_today = datetime.now()
    time = range(5)
    time = pd.date_range(date_today, date_today + timedelta(minutes=5), freq="min")
    data1 = np.random.randint(1, high=100, size=len(time)) / 10
    data2 = np.random.randint(1, high=100, size=len(time)) / 10
    df = pd.DataFrame({"index": time, "col1": data1, "col2": data2})
    df.set_index("index", inplace=True)
    print("Dataframe:")
    df.to_csv("example_df.csv")
    print(df)
    agent_config = {
        "id": "my_agent_id",
        "modules": [
            {
                "module_id": "My_Data_Source",
                "type": "agentlib_mpc.data_source",
                "data": "example_df.csv",
                # "data_offset": pd.Timedelta("1min"),
                # "data_offset": 60,
                "interpolation_method": InterpolationMethods.linear,
                "columns": ["col1", "col2"],
            }
        ],
    }

    logging.basicConfig(level=logging.INFO)
    environment_config = {"rt": False, "factor": 1}
    env = Environment(config=environment_config)
    agent_ = Agent(config=agent_config, env=env)
    env.run(65)
