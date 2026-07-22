from typing import Protocol
from dataclasses import dataclass
from pathlib import Path
from sklearn.pipeline import Pipeline
import skops.io as sio
from typing import Mapping
import json
from importlib.metadata import version

DEFAULT_MODEL_FILENAME = "model.skops"

_TRUSTED_TYPES = [
    'collections.OrderedDict', 'functools.partial', 'lightgbm.basic.Booster',
    'lightgbm.sklearn.LGBMClassifier', "sklearn.compose._column_transformer.make_column_selector",
    'malnutrition_risk.features.CPIAdjustmentTransformer',
    'malnutrition_risk.features.PostalCodeTransformer', 'malnutrition_risk.features.ToCategory',
    'malnutrition_risk.features.VulnerabilityIndexTransformer', 'malnutrition_risk.features.standardize_nan',
    'malnutrition_risk.features.ColumnPruner', 'malnutrition_risk.features.DtypeContract'
    ]

IDENTITY_KEYS = ("feature_engineering", "preprocessor", "model")

def compose_run_name(choices: Mapping[str, str], prefix: str = None):
    body = "__".join(choices[k] for k in IDENTITY_KEYS)
    return f"{prefix}__{body}" if prefix else body


# package source bundled with the model -> loads without malnutrition_risk pip-installed
_PACKAGE_ROOT = Path(__file__).resolve().parents[1]      # .../src/malnutrition_risk
_BASE_DEPS = ("scikit-learn", "skops", "numpy", "scipy", "pandas")
_FRAMEWORKS = {"lightgbm": "lightgbm", "xgboost": "xgboost", "catboost": "catboost"}

def _model_runtime_deps(model) -> list[str]:
    estimator = model.named_steps["classifier"] if hasattr(model, "named_steps") else model[-1]
    top = type(estimator).__module__.split(".")[0]        # 'lightgbm' | 'xgboost' | 'sklearn' | ...
    deps = list(_BASE_DEPS) + ([_FRAMEWORKS[top]] if top in _FRAMEWORKS else [])
    return [f"{pkg}=={version(pkg)}" for pkg in deps]

class ModelResolver(Protocol):
    def resolve(self) -> Pipeline: ...

@dataclass(frozen=True)
class RunDirModelResolver(ModelResolver):
    run_dir: Path
    model_file_name: str = DEFAULT_MODEL_FILENAME

    def resolve(self) -> Pipeline:
        path = Path(self.run_dir) / self.model_file_name
        if not path.exists():
            raise FileNotFoundError(
                f"no {self.model_file_name} in run_dir={self.run_dir}. "
                f"pass a training run directory and try again. e.g.\n"
                f"eval.run_dir=outputs/experiments/<dataset>/<fe>/<preproc>/<model>/<timestamp>"
            )
        return sio.load(path, trusted=_TRUSTED_TYPES)

@dataclass(frozen=True)
class MlflowModelResolver(ModelResolver):
    model_uri: str  # "models:/<name>@champion" | "runs:/<id>/model" | "models:/<id>"

    def resolve(self) -> Pipeline:
        import mlflow.sklearn
        return mlflow.sklearn.load_model(self.model_uri)

def save_model(model: Pipeline, run_dir: Path, file_name: str = DEFAULT_MODEL_FILENAME) -> Path:
    path = Path(run_dir) / file_name
    sio.dump(model, path)
    return path

def log_model(model, X_sample, *, name: str = 'model', registered_model_name: str = None):
    """Swap to a pyfunc wrapper here when serving."""
    import mlflow
    from mlflow.models import infer_signature

    # discover every type skops would refuse to load, then vouch for them
    trusted = sio.get_untrusted_types(data=sio.dumps(model))
    signature = infer_signature(model_input=X_sample, model_output=model.predict(X_sample))
    return mlflow.sklearn.log_model(
        sk_model=model, name=name, signature=signature,
        input_example=X_sample, registered_model_name=registered_model_name,
        skops_trusted_types=trusted,
        pip_requirements=_model_runtime_deps(model),
        code_paths=[str(_PACKAGE_ROOT)]     # serving: bundle custom transformer code
    )

def register_model(model_uri: str, *, name: str, tracking_uri: str = None, alias: str = None) -> int:
    """Register an already-logged model into the registry and (optionally) set an alias."""
    import mlflow
    from mlflow import MlflowClient

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    result = mlflow.register_model(model_uri=model_uri, name=name)
    version_ = result.version

    if alias:
        MlflowClient().set_registered_model_alias(name=name, alias=alias, version=version_)

    return int(version_)


def log_model_params(params: Mapping, *, model_id: str):
    import mlflow
    mlflow.log_model_params({k: str(v) for k, v in params.items()}, model_id=model_id)

def save_run_pointer(run_dir: Path, *, run_id: str, model_uri: str,
                     study_name: str, choices: Mapping[str, str] = None):
    path = Path(run_dir) / "mlflow_run.json"
    path.write_text(json.dumps(
        {"run_id": run_id, "model_uri": model_uri,
        "study_name": study_name, "choices": dict(choices) if choices else None},
        indent=2
    ))
    return path

def read_run_pointer(run_dir: Path):
    path = Path(run_dir) / "mlflow_run.json"
    return json.loads(path.read_text()) if path.exists() else {}