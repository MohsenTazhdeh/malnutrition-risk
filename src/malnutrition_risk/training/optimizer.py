import pandas as pd
from sklearn.pipeline import Pipeline
from typing import Callable, Any
import optuna
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.base import clone
from logging import getLogger

logger = getLogger(__name__)



class PipelineOptimizer:
    def __init__(
        self,
        base_pipeline: Pipeline,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        suggest_fn: Callable[[optuna.Trial], dict],
        cv_folds: int,
        group: str,
        random_state: int,
        study_name: str,
        objective: str,
        storage: str,
        warm_start: dict | None = None,
        n_jobs: int = -1,
        sample_weight: pd.Series = None,
        custom_cv=None
    ):
        self.base_pipeline = base_pipeline
        self.X = X
        self.y = y
        self.suggest_fn = suggest_fn
        self.cv_folds = cv_folds
        self.group = group
        self.random_state = random_state
        self.study_name = study_name
        self.objective = objective
        self.storage = storage
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.sample_weight = sample_weight
        self.custom_cv = custom_cv or StratifiedGroupKFold(
            n_splits=cv_folds, shuffle=True, random_state=random_state
        )

    def _objective(self, trial: optuna.Trial) -> float:
        trial_pipeline = clone(self.base_pipeline)
        trial_pipeline.set_params(**self.suggest_fn(trial))
        fit_params = dict()
        if self.sample_weight is not None:
            fit_params = {"classifier__sample_weight": self.sample_weight}
        scores = cross_val_score(
            trial_pipeline, self.X, self.y, cv=self.custom_cv,
            groups=self.X[self.group], scoring=self.objective,
            params=fit_params, n_jobs=self.n_jobs,
        )
        return scores.mean()

    def optimize(self, n_trials:int) -> dict[str, Any]:
        self.study = optuna.create_study(
            study_name=self.study_name, direction='maximize',
            storage=self.storage, load_if_exists=bool(self.storage) # resume same fingerprint
        )

        n_existing_trials = len(self.study.trials)
        if self.warm_start and n_existing_trials == 0:   # only seed a brand-new study
            self.study.enqueue_trial(self.warm_start)
        logger.info(f"{self.study_name}: {n_existing_trials} existing trials; adding {n_trials} new trials.")

        self.study.optimize(self._objective, n_trials=n_trials)
        logger.info(f"{self.study_name}: Best {self.objective}: {self.study.best_value:.4f} "
                    f"over {len(self.study.trials)} total trials.")
        return self.study.best_params



