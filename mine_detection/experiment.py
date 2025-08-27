from loguru import logger
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from mine_detection.config import (
    FEATURES,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    RANDOM_STATE,
    TARGET,
)
from mine_detection.data import load_data


def build_pipeline(model_name, model):
    steps = [
        (model_name, model),
    ]

    return Pipeline(steps=steps)


def get_model_params(model_name):
    params = {
        "logreg": [
            {
                "logreg__penalty": ["l1", "l2", "elasticnet"],
                "logreg__max_iter": [100, 500, 1000],
                "logreg__solver": ["liblinear"],
            },
        ],
    }
    return params[model_name]


def run_experiments(
    train_path="data/interim/train.csv",
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
):
    # prepare data
    train = load_data(path=train_path)
    X = train[FEATURES]
    y = train[TARGET]

    # prepare candidate models
    models = [
        ("logreg", LogisticRegression()),
    ]

    mlflow.sklearn.autolog()

    for model_name, model in models:
        with mlflow.start_run(run_name=model_name):
            pipeline = build_pipeline(model_name=model_name, model=model)
            param_grid = get_model_params(model_name=model_name)

            gridcv = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                cv=cv,
                scoring="f1",
            )

            gridcv.fit(X=X, y=y)

            logger.info(f"Best estimator: {gridcv.best_estimator_}")
            logger.info(f"Mean test f1-score: {gridcv.best_score_}")


if __name__ == "__main__":
    logger.add("logs/experiment.log", rotation="10 MB")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    run_experiments()
