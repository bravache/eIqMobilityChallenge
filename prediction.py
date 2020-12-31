import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from feature_processing import (
    calculate_distance_and_speed,
    cyclical_transform_of_datetime,
    filter_value_with_threshold,
)

COMBINED_DAYS_FEATURES = ["distance", "sine_t", "cosine_t"]
SEPARATE_WEEKEND_FEATURES = [
    "distance",
    "weekday_sine_t",
    "weekday_cosine_t",
    "weekend_sine_t",
    "weekend_cosine_t",
]


def preprocess_trip_data(df: pd.DataFrame) -> pd.DataFrame:
    distance_and_speed = calculate_distance_and_speed(df)
    cyclical_time = cyclical_transform_of_datetime(df.pickup_datetime)
    output = pd.merge(df, distance_and_speed, left_index=True, right_index=True).merge(
        cyclical_time,
        left_index=True,
        right_index=True,
    )
    return filter_value_with_threshold(output)


def predict_ride_hailing(
    train: pd.DataFrame,
    test: pd.DataFrame,
    separate_weekend: bool = False,
    target: str = "speed",
) -> np.ndarray:
    features = SEPARATE_WEEKEND_FEATURES if separate_weekend else COMBINED_DAYS_FEATURES
    pipe = make_pipeline(StandardScaler(), SGDRegressor())
    pipe.fit(train[features], train[target])
    print(f"Prediction R^2: {round(pipe.score(test[features], test[target]), 3)}")
    return pipe.predict(test[features])
