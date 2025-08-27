import os

RANDOM_STATE = 42

FEATURES = ["voltage", "height", "soil"]
TARGET = ["mine"]


MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = "mine-detection-experiment"
