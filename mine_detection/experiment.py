from loguru import logger
import mlflow
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from mine_detection.config import (
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    RANDOM_STATE,
)
from mine_detection.data import load_X_y


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
        "rf": [
            {
                "rf__n_estimators": [200],
                "rf__max_depth": [10],
            },
        ],
        "gbc": [
            {
                "gbc__n_estimators": [100, 200],
                "gbc__learning_rate": [0.01, 0.1],
                "gbc__max_depth": [3, 5],
            },
        ],
    }
    return params[model_name]


def run_experiments(
    train_path,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
):
    # prepare data
    X, y = load_X_y(path=train_path)

    # prepare candidate models
    models = [
        ("logreg", LogisticRegression()),
        ("rf", RandomForestClassifier(random_state=RANDOM_STATE)),
        ("gbc", GradientBoostingClassifier(random_state=RANDOM_STATE)),
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

    run_experiments(train_path="data/interim/train.csv")
