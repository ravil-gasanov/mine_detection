import pandas as pd

FEATURES = ["voltage", "height", "soil"]
TARGET = ["mine"]


def load_X_y(path):
    data = pd.read_csv(path)

    X = data[FEATURES]
    y = data[TARGET]

    return X, y
