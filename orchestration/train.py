import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from hydra.core.hydra_config import HydraConfig
from pathlib import Path
from malnutrition_risk.data.loading import load_labeled_xy
from malnutrition_risk.pipeline import build_pipeline
from malnutrition_risk.core.artifacts import ArtifactWriter
from malnutrition_risk.training.runner import TrainingConfig, run_training
from malnutrition_risk.training.search_space import build_suggest_fn, get_warm_start_params
from malnutrition_risk.core import tracking, model_io

from malnutrition_risk.training.study import study_fingerprint, build_study_name

import warnings

# MLflow's dataset-source registry lists LocalArtifactDatasetSource
warnings.filterwarnings(
    "ignore",
    message=r"The specified dataset source can be interpreted in multiple ways",
    category=UserWarning,
)
# Integer-column hint from the *dataset* schema (to_mlflow_dataset).
warnings.filterwarnings(
    "ignore",
    message=r"Hint: Inferred schema contains integer column",
    category=UserWarning,
)

logger = logging.getLogger(__name__)


def _serving_input_example(X, geo_cols):
    """Provisional input-example dtype contract for the MLflow signature."""
    x = X.head(5).copy()
    geo = list(geo_cols)
    numeric = x.columns.difference(geo)
    x[numeric] = x[numeric].astype("float64")
    x[geo] = x[geo].astype("string")
    return x


@hydra.main(version_base='1.3', config_path='../conf', config_name='config')
def main(cfg: DictConfig):
    logger.info("====== Starting Training ======")
    pipeline = build_pipeline(cfg)
    X, y = load_labeled_xy(cfg.dataset.paths.train, cfg.dataset.target)
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    writer = ArtifactWriter(run_dir)

    train_df = X.copy(); train_df[cfg.dataset.target] = y
    mlflow_dataset = tracking.to_mlflow_dataset(train_df, source=cfg.dataset.paths.train, name='train', target=cfg.dataset.target)

    search_space = OmegaConf.to_container(cfg.search_space, resolve=True)

    train_cfg = TrainingConfig(
        target_col=cfg.dataset.target,
        group_col=cfg.dataset.group,
        cv_folds=cfg.training.cv_folds,
        n_trials=cfg.training.optimization.n_trials,
        n_jobs=cfg.training.n_jobs,
        seed=cfg.seed,
        objective=cfg.training.optimization.objective,
    )

    # build a unique study name for optuna
    fp = study_fingerprint(search_space=search_space, cv_folds=train_cfg.cv_folds,
                           seed=train_cfg.seed, objective=train_cfg.objective, dataset_digest=mlflow_dataset.digest)

    choices = HydraConfig.get().runtime.choices
    identity = {k: choices[k] for k in model_io.IDENTITY_KEYS}
    run_name = model_io.compose_run_name(identity, prefix="train")
    study_name = build_study_name(run_name, fp, cfg.optuna.study_version)

    tags = {**identity, "phase": "training", "optuna_study": study_name}

    with tracking.start_run(tracking_uri=cfg.mlflow.tracking_uri,
                            experiment_name=cfg.mlflow.experiment_name,
                            artifact_location=cfg.mlflow.artifact_location,
                            run_name=run_name, tags=tags) as run:

        tracking.log_input(dataset=mlflow_dataset, context='training')
        result = run_training(
            pipeline, X, y, writer, train_cfg,
            suggest_fn=build_suggest_fn(cfg.search_space),
            warm_start=get_warm_start_params(cfg.search_space),
            study_name=study_name,
            storage=cfg.optuna.storage
        )

        tracking.log_params({'seed': train_cfg.seed, 'cv_folds': train_cfg.cv_folds,
                             'n_trials': train_cfg.n_trials, 'objective': train_cfg.objective,
                             **result.best_params})
        tracking.log_metrics(result.scalar_metrics())
        tracking.log_config(run_dir)

        for path in writer.written:
            tracking.log_artifact(path)

        X_sample = _serving_input_example(X, OmegaConf.to_container(cfg.dataset.schema.geo_cols, resolve=True))
        model_info = model_io.log_model(result.model, X_sample,
                           registered_model_name=cfg.mlflow.registered_model_name)

        model_io.save_run_pointer(run_dir, run_id=run.info.run_id,
                                  model_uri=model_info.model_uri, study_name=study_name,
                                  choices=identity)

        logger.info(result.summary())
        logger.info(f"training: run_id={run.info.run_id}, model_uri={model_info.model_uri}, study={study_name}")


if __name__ == "__main__":
    main()

