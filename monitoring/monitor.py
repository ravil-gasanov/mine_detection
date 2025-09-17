import os
import sqlite3

from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset
import pandas as pd


def load_reference_data():
    with sqlite3.connect("local.db") as conn:
        return pd.read_sql("SELECT voltage, height, soil, mine FROM train", conn)


def load_production_data():
    with sqlite3.connect("local.db") as conn:
        return pd.read_sql("SELECT voltage, height, soil, mine FROM sensor", conn)


# evidently
def get_schema():
    schema = DataDefinition(
        numerical_columns=[
            "voltage",
            "height",
            "soil",
        ],
    )

    return schema


def calculate_metrics(reference_data, production_data):
    # evidently-related
    # 1 define schema and specify column types (categorical, numerical)
    schema = get_schema()

    # create dataset objects
    # for reference data
    reference = Dataset.from_pandas(
        pd.DataFrame(reference_data),
        data_definition=schema,
    )

    # and for the incoming data
    production = Dataset.from_pandas(
        pd.DataFrame(production_data),
        data_definition=schema,
    )

    # create a report object and specify what you want to calculate
    # DataDriftPreset contains the most common metrics for data drift detection
    report = Report([DataDriftPreset()], include_tests=True)

    metrics = report.run(
        current_data=production,
        reference_data=reference,
    )

    if os.path.exists("reports/"):
        metrics.save_html("reports/drift.html")

    return metrics


def alert(metrics):
    # example alert function
    # you can send an email, a message to Slack, etc.
    metrics = metrics.dict()

    data_drift_metrics = metrics["metrics"][0]["value"]

    print(data_drift_metrics["share"])
    print(data_drift_metrics["count"])

    if data_drift_metrics["count"] > 0:
        # since this is a critical application involving human safety
        # we alert if there is even one column that drifted
        print("ALERT! At least one column has drifted.")


def monitor():
    reference_data = load_reference_data()
    production_data = load_production_data()

    if len(production_data) < 10:
        print("Not enough production data to monitor.")
        return

    metrics = calculate_metrics(reference_data, production_data)

    alert(metrics=metrics)


if __name__ == "__main__":
    monitor()
