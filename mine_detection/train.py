from loguru import logger
import mlflow
from mlflow.tracking import MlflowClient

from mine_detection.config import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI
from mine_detection.data import load_X_y

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def load_best_model_from_experiment():
    """
    Load the best model from an MLflow experiment based on highest F1 score.

    Args:
        experiment_name: Name of the MLflow experiment

    Returns:
        Loaded sklearn model object
    """
    # Initialize client
    client = MlflowClient()

    # Get experiment
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if not experiment:
        raise ValueError(f"Experiment '{MLFLOW_EXPERIMENT_NAME}' not found")

    # Search for runs ordered by F1 score (highest first)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.best_cv_score DESC"],
        max_results=1,
    )

    if not runs:
        raise ValueError("No runs found in the experiment")

    best_run = runs[0]

    # Print info about the best run
    logger.info(f"Best run ID: {best_run.info.run_id}")
    logger.info(f"Best F1 score: {best_run.data.metrics.get('best_cv_score', 'N/A')}")

    model_uri = f"runs:/{best_run.info.run_id}/model"

    model = mlflow.sklearn.load_model(model_uri)

    return model.best_estimator_


def register_model(model, model_name: str):
    """
    Register an sklearn model to MLflow Model Registry.

    Args:
        model: Trained sklearn model object
        model_name: Name to register the model under in the registry

    Returns:
        ModelVersion object from MLflow
    """

    try:
        with mlflow.start_run():
            # Log the sklearn model
            mlflow.sklearn.log_model(
                sk_model=model,
                name=model_name,
                registered_model_name=model_name,
            )
    except Exception as e:
        logger.error(f"Failed to register model: {e}")
        raise e


def train_model(model, X, y):
    return model.fit(X, y)


def evaluate(best_model, test_path):
    test_X, test_y = load_X_y(path=test_path)

    test_score = best_model.score(test_X, test_y, metric="f1")
    logger.info(f"Test f1-score of the best model: {test_score}")


def train(
    train_path: str,
    test_path: str,
    model_name: str,
):
    # prepare data
    X, y = load_X_y(path=train_path)

    # load the best model from the experiment runs
    best_model = load_best_model_from_experiment()

    # fit the model on the entire training data
    best_model = train_model(model=best_model, X=X, y=y)

    # evaluate the model on test data
    evaluate(best_model=best_model, test_path=test_path)

    # register the model
    register_model(model=best_model, model_name=model_name)


if __name__ == "__main__":
    train(
        train_path="data/interim/train.csv",
        test_path="data/interim/test.csv",
        model_name="mine_detection_model",
    )
