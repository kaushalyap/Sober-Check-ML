import enum


class FeatureSet(enum.Enum):
    x_mean = 0
    x_median = 1
    x_stdev = 2
    x_raw_min = 3
    x_raw_max = 4
    x_abs_min = 5
    x_abs_max = 6
    y_mean = 7
    y_median = 8
    y_stdev = 9
    y_raw_min = 10
    y_raw_max = 11
    y_abs_min = 12
    y_abs_max = 13
    z_mean = 14
    z_median = 15
    z_stdev = 16
    z_raw_min = 17
    z_raw_max = 18
    z_abs_min = 19
    z_abs_max = 20
