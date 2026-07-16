from dataclasses import dataclass
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from malnutrition_risk.core.artifacts import ArtifactWriter
from malnutrition_risk.core.model_io import save_model
import pandas as pd
from sklearn.base import clone
from typing import Callable
from .schemas import TrainingResult, CVMetrics
from .optimizer import PipelineOptimizer
import tempfile

from logging import getLogger
logger = getLogger(__name__)

_SCORING = {
    "ap": "average_precision",
    "auc": "roc_auc",
    "recall": "recall",
    "precision": "precision",
    "brier": "neg_brier_score",
}

@dataclass(frozen=True)
class TrainingConfig:
    target_col: str
    group_col: str
    cv_folds: int = 5
    n_trials: int = 20
    n_jobs: int = -1
    seed: int = 42
    objective: str = "average_precision"



def _cross_validate(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, cfg: TrainingConfig) -> CVMetrics:
    cv = StratifiedGroupKFold(n_splits= cfg.cv_folds, shuffle=True, random_state=cfg.seed)
    r  = cross_validate(pipeline, X, y, groups= X[cfg.group_col], scoring=_SCORING, cv=cv, n_jobs=cfg.n_jobs)

    return CVMetrics(
        ap_mean=float(r["test_ap"].mean()), ap_std=float(r["test_ap"].std()),
        auc_mean=float(r["test_auc"].mean()), auc_std=float(r["test_auc"].std()),
        recall_mean=float(r["test_recall"].mean()),
        precision_mean=float(r["test_precision"].mean()),
        brier_mean=float(-r["test_brier"].mean()),
    )

def _get_processed_data(pipeline: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    preprocessor = pipeline[:-1]
    X_t = preprocessor.transform(X)
    return X_t

def run_training(
        pipeline: Pipeline,
        X: pd.DataFrame,
        y: pd.Series,
        writer: ArtifactWriter,
        cfg: TrainingConfig,
        *,
        suggest_fn: Callable,
        study_name: str,
        warm_start: dict = None,
        storage: str = None,
) -> TrainingResult:
    with tempfile.TemporaryDirectory() as cache_dir:
        pipeline.memory = cache_dir
        optimizer = PipelineOptimizer(
            pipeline, X, y, suggest_fn=suggest_fn,
            study_name=study_name, objective=cfg.objective,
            storage=storage, warm_start=warm_start,
            cv_folds=cfg.cv_folds, group=cfg.group_col,
            random_state=cfg.seed,n_jobs=cfg.n_jobs,
        )

        best_params = optimizer.optimize(n_trials=cfg.n_trials)
        trials_df = optimizer.study.trials_dataframe()

        final_pipeline = clone(pipeline)
        final_pipeline = final_pipeline.set_params(**best_params)
        cv_metrics = _cross_validate(final_pipeline, X, y, cfg)
        final_pipeline.fit(X, y)

        save_model(final_pipeline, writer.out_dir)

        # store effective parameters (the final parameter set that was effective during training)
        estimator = final_pipeline.named_steps['classifier']
        writer.write_json(estimator.get_params(), "effective_params.json")

        writer.write_metrics(cv_metrics.scalar_metrics())
        writer.write_json(best_params, 'best_params.json')
        writer.write_table(trials_df, 'optimization_history.csv')

        return TrainingResult(model=final_pipeline, cv_metrics=cv_metrics, best_params=best_params)





















