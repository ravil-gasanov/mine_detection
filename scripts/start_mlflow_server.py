import subprocess


def start_mlflow():
    """
    Start MLflow server with SQLite backend and local artifact storage.
    Equivalent to running:
    uv run mlflow server --backend-store-uri sqlite:///local.db
    --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
    """
    cmd = [
        "uv", "run", "mlflow", "server",
        "--backend-store-uri", "sqlite:///local.db",
        "--default-artifact-root", "./mlruns",
        "--host", "0.0.0.0",
        "--port", "5000"
    ]
    
    # Stream logs to console
    process = subprocess.Popen(cmd)
    process.wait()

if __name__ == "__main__":
    start_mlflow()
