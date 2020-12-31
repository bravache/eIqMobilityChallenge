from enum import Enum, auto

import numpy as np
import pandas as pd
from geopy.distance import distance


def calculate_distance_and_speed(df: pd.DataFrame) -> pd.DataFrame:
    def _calculate_geodesic_distance_from_row(row: pd.Series) -> float:
        # TODO : Find a less costly method for calculating distance
        return distance(
            (row.pickup_latitude, row.pickup_longitude),
            (row.dropoff_latitude, row.dropoff_longitude),
        ).miles

    dist = df.apply(_calculate_geodesic_distance_from_row, axis=1)  # in miles
    speed = dist / df.trip_duration * 3600  # in mph

    return pd.DataFrame({"distance": dist, "speed": speed}, index=df.index)


def _hour_of_day(dt: pd.Series):
    return dt.dt.hour + dt.dt.minute / 60 + dt.dt.second / 3600


def _hour_of_week(dt: pd.Series):
    return 24 * dt.dt.dayofweek + _hour_of_day(dt)


class DatetimePeriod(Enum):
    DAY = auto()
    WEEK = auto()


def bin_datetime(
    datetime: pd.Series,
    hour_sample: float = 1,
    period: DatetimePeriod = DatetimePeriod.DAY,
) -> pd.Series:
    """
    Bin time after resampling to the given period
    Example:
        bin_datetime([2020-12-30 14:45:00], 1, DatetimePeriod.DAY) == [14.0]
        bin_datetime([2020-12-30 14:45:00], 0.5, DatetimePeriod.DAY) == [14.5]
        bin_datetime([2020-12-30 14:45:00], 1, DatetimePeriod.WEEK) == [86.0]
    """
    if period == DatetimePeriod.DAY:
        sampled = _hour_of_day(datetime)
    elif period == DatetimePeriod.WEEK:
        sampled = _hour_of_week(datetime)
    else:
        raise ValueError(f"Unknown period: {period}")

    return pd.cut(
        sampled,
        np.arange(0, max(sampled) + hour_sample, hour_sample),
        labels=np.arange(0, max(sampled), hour_sample),
    )


def cyclical_transform_of_datetime(dt: pd.Series) -> pd.DataFrame:
    sine_t = np.sin(2 * np.pi * _hour_of_day(dt) / 24)
    cosine_t = np.cos(2 * np.pi * _hour_of_day(dt) / 24)
    weekend_mask = dt.dt.dayofweek >= 5
    return pd.DataFrame(
        {
            "sine_t": sine_t,
            "cosine_t": cosine_t,
            "weekday_sine_t": np.where(weekend_mask, 0, sine_t),
            "weekday_cosine_t": np.where(weekend_mask, 0, cosine_t),
            "weekend_sine_t": np.where(weekend_mask, sine_t, 0),
            "weekend_cosine_t": np.where(weekend_mask, cosine_t, 0),
        },
        index=dt.index,
    )


FEATURE_THRESHOLDS = {
    "speed": (0, 100),
    "distance": (0, 500),
    "trip_duration": (0, 3600 * 3),
}


def filter_value_with_threshold(df: pd.DataFrame) -> pd.DataFrame:
    mask = np.logical_and.reduce(
        [
            np.logical_and(df[feature] >= bound[0], df[feature] <= bound[1])
            for feature, bound in FEATURE_THRESHOLDS.items()
        ]
    )
    return df.loc[mask]
