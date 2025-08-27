import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """Load the dataset from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    return pd.read_csv(path)
