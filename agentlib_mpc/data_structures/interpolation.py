from aenum import Enum

a = 1


class InterpolationMethods(str, Enum):
    linear = "linear"
    previous = "previous"
    no_interpolation = "no_interpolation"
    spline3 = "spline3"

    # this interpolation method is a custom implementation, intended for the case where
    # the source data is sampled finer that the target data. It takes the average of
    # all points between two adjacent time steps on the target grid.
    # Example:
    # source_grid: [0, 10, 20, 30, 40, 50, 60]
    # source_data: [a, b, c, d, e, f, g]
    # target_grid: [15, 35, 55]
    # Will yield: [(c+d)/2, (e+f)/2, (e+f)/2]
    # The last value is always duplicated, to get a lenght consistent with other
    # interpolation methods
    # This is intended for the case, where the target data is input for an intgration
    # / prediction between two points.
    mean_over_interval = "mean_over_interval"


c = 2
