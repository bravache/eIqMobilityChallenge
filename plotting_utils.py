import pandas as pd
from plotly import express as px
from plotly.graph_objs import Figure

# This public token is required for plotting mapbox. This should be used whenever this file is sourced
px.set_mapbox_access_token(open(".mapbox_token").read())

# Human readable labels for dataframe columns
PLOT_LABELS = {
    "speed": "Trip Average Speed [mph]",
    "distance": "Trip Distance [miles]",
    "binned_distance": "Trip Distance [miles]",
    "hour_of_day": "Hour of Day",
    "dayofweek": "Day Of Week (0: Mo)",
    "relative_error": "Relative Error [%]",
    "absolute_error": "Absolute Error [s]",
}
# I just prefer the plotly_white theme to the default one.
COMMON_PLOTLY_ARGS = {"template": "plotly_white", "labels": PLOT_LABELS}


def plot_pickup_dropoff_locations(df: pd.DataFrame) -> Figure:
    """
    Join pickup and dropoff coordinates under the same columns then plot the resulting dataframe in a mapbox
    """
    pickup = pd.DataFrame(
        dict(
            latitude=df.pickup_latitude,
            longitude=df.pickup_longitude,
            type="pickup",
        )
    )
    dropoff = pd.DataFrame(
        dict(
            latitude=df.dropoff_latitude,
            longitude=df.dropoff_longitude,
            type="dropoff",
        )
    )

    return px.scatter_mapbox(
        pd.concat([pickup, dropoff]),
        lat="latitude",
        lon="longitude",
        color="type",
    )
