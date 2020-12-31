import pandas as pd


def load_trip_data(path: str) -> pd.DataFrame:
    """
    Small util to load the dataset, assigning the id column to the index and dropping the int id column.
    Typically, I would remove the int id column from the csv all together,
    but I want to make sure that script can be reused without external cleaning
    """
    df = pd.read_csv(path, index_col="id", parse_dates=["pickup_datetime"])
    # The id column will be named "Unnamed: 0".
    # Alternative for the following line could be to drop all unnamed columns
    df = df.drop(["Unnamed: 0"], axis=1)
    # TODO : Validate column names and types
    return df
