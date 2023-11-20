import pandas as pd
import numpy as np


def load_weather(path):
    """
    given a starting date, filter the data from pandas dataframe
    """
    with open(path) as f:
        contents = f.readlines()
    contents = [x.strip() for x in contents]
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
    weather = weather.set_index(["MM", "DD"])
    return weather


def heat_load_func(current, begin, end, max_disturbance):
    """
    Returns the heat load in the room in Watt, given a time in seconds.
    Since the heat load inside the room is cause by humans and devices,
    it is not dependent on
    """

    def calc_heat(time, load):
        tag = time // (24 * 3600) + 1
        tageszeit = time % (24 * 3600)
        if tag < 6 and begin <= tageszeit // 3600 < end:
            return load
        else:
            return 0

    if not isinstance(current, np.ndarray):
        return calc_heat(current, max_disturbance)
    else:
        # current = list(current)
        heat_load = []
        for t in current:
            heat = calc_heat(t, max_disturbance)
            heat_load.append(heat)
        heat_load = np.array(heat_load)
        return heat_load


def get_q_load(current, weather_data, month, day):
    """
    Takes data from weather file and the needed date.
    Returns the solar radiation which is absorbed by the thermal zone,
    since a large part of the radiation is reflected.
    """
    start_index = weather_data.index.get_loc((month, day)).start
    q_load = np.array(weather_data["B"])[start_index:] * 0.1
    if isinstance(current, np.ndarray):
        current = list(current)
        load = []
        for t in current:
            hour = t // 3600
            load.append(q_load[hour])
        load = np.array(load)
        return load
    else:
        hour = int(current) // 3600
        return q_load[hour]


def get_t_aussen(current, weather_data, month, day):
    """
    Takes data from weather file and the needed date;
    returns the ambient temperature at that day and time
    """
    start_index = weather_data.index.get_loc((month, day)).start
    t_aussen = np.array(weather_data["t"])[start_index:] + 273.15
    if isinstance(current, np.ndarray):
        current = list(current)
        taussen_list = []
        for i in current:
            hour = i // 3600
            taussen_list.append(t_aussen[hour])
        taussen_list = np.array(taussen_list)
        return taussen_list
    else:
        hour = int(current) // 3600
        return t_aussen[hour]


weather_data = load_weather("TRY2015_Aachen_Jahr.dat")
irradiation_data = pd.read_csv("radiation.csv") * 0.25
irradiation_data.index = irradiation_data.index * 3600
