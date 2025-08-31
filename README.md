# Landmine Detection using Magnetic Field Distortions

This project leverages machine learning to detect buried land mines by analyzing magnetic field distortions. Land mines pose significant risks in conflict zones and post-conflict areas, and active detection methods can be slow and expensive.

By collecting magnetic field measurements and training predictive models, this system can identify potential mine locations with high accuracy, helping improve safety and speed up demining operations.

Key features:

- Processes magnetic field sensor data to detect anomalies caused by buried metal objects and predicts whether the object is a land mine.

- We deploy the system as a web API endpoint, but it could also be deployed on edge devices, such as field-deployed sensors or drones.

This project demonstrates a practical application of ML in real-world safety-critical scenarios, combining sensor data analysis, predictive modeling, and deployable inference pipelines.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         mine_detection and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── mine_detection   <- Source code for use in this project.
│    │
│   ├── __init__.py             <- Makes mine_detection a Python module
│   │
│   ├── config.py               <- Store useful variables and configuration
│   │
│   ├── data.py              <- Data-related methods and variables
│   │
│   ├── experiment.py        <- model selection experiments
│   │
│   ├── train.py             <- train the best model on full training data and eval on test data
│   │
│   └── api.py                <- Prediction endpoint implemented using FastAPI
│
├── scripts                 <- scripts to download data, start MLFlow server
│
└── tests                   <- unit tests, integration tests

```

--------

