from contextlib import contextmanager
import subprocess
from typing import Mapping
from pathlib import Path
import mlflow
import pandas as pd
from urllib.parse import urlparse


def _setup_sql_dir(uri: str) -> None:
    """Create the parent directory for a local SQLite tracking URI so the DB can be created on a fresh clone."""
    if not uri.startswith('sqlite:'):
        return
    db_path = uri.removeprefix('sqlite:///')
    Path(db_path).resolve().parent.mkdir(parents=True, exist_ok=True)

def _resolve_artifact_location(artifact_location: str | None) -> str | None:
    if not artifact_location:
        return None
    if '://' in artifact_location and not artifact_location.startswith('file:'): # s3://, gs://, ... — MLflow manages it
        return artifact_location
    raw = urlparse(artifact_location).path if artifact_location.startswith('file://') else artifact_location
    path = Path(raw).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.as_uri()    # H:/Lets%20see/Projects/Malnutrition%20Risk/artifacts/mlruns ->
                            # file:///H:/Lets%20see/Projects/Malnutrition%20Risk/artifacts/mlruns

def setup_experiment(*, tracking_uri, experiment_name, artifact_location=None):
    _setup_sql_dir(tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    if mlflow.get_experiment_by_name(experiment_name) is None:
        loc = _resolve_artifact_location(artifact_location)
        mlflow.create_experiment(name=experiment_name, artifact_location=loc)
    mlflow.set_experiment(experiment_name)

@contextmanager
def start_run(*, tracking_uri, experiment_name, run_name, artifact_location=None, tags=None):
    setup_experiment(tracking_uri=tracking_uri, experiment_name=experiment_name, artifact_location=artifact_location)
    with mlflow.start_run(run_name=run_name, tags=tags) as run:
        _log_code_version()
        yield run

def log_input(
        dataset: mlflow.data.pandas_dataset.PandasDataset,
        *,
        context: str,
        model:  mlflow.entities.LoggedModelInput | None = None
):
    mlflow.log_input(dataset=dataset, context=context, model=model)          # context: "training" | "evaluation"

def log_params(params: Mapping):
    mlflow.log_params({key: str(value) for key, value in params.items()})

def log_metrics(metrics: Mapping):
    mlflow.log_metrics(dict(metrics))

def log_config(run_dir: Path):
    hydra_dir = Path(run_dir) / ".hydra"
    if hydra_dir.exists():
        mlflow.log_artifacts(local_dir=str(hydra_dir) , artifact_path='config')

def log_artifact(path: Path, artifact_path: str = None):
    mlflow.log_artifact(local_path=str(path), artifact_path=artifact_path)

def to_mlflow_dataset(df: pd.DataFrame, *, source: str, target: str, name: str):
    return mlflow.data.from_pandas(df, source=str(source), targets=target, name=name)

def _log_code_version():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
        mlflow.set_tags({"git_sha": sha, "git_dirty": str(dirty)})
    except Exception:
        pass   # not a git repo / git unavailable — MLflow's own auto-capture still applies

