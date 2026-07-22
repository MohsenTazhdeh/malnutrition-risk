import hydra
from omegaconf import DictConfig
import logging
from pathlib import Path
from hydra.core.hydra_config import HydraConfig
from malnutrition_risk.core.model_io import (RunDirModelResolver, MlflowModelResolver,
                                             read_run_pointer, compose_run_name, IDENTITY_KEYS)
from malnutrition_risk.core import tracking
from malnutrition_risk.core.artifacts import ArtifactWriter
from malnutrition_risk.data.loading import load_labeled_xy
from malnutrition_risk.evaluation.runner import EvaluationConfig, run_evaluation


logger = logging.getLogger(__name__)

@hydra.main(version_base='1.3', config_path='../conf', config_name='eval')
def main(cfg: DictConfig):
    logger.info("====== Starting Evaluation ======")
    run_dir = Path(cfg.eval.run_dir)
    model = MlflowModelResolver(cfg.model_uri).resolve() if cfg.eval.model_uri else RunDirModelResolver(run_dir).resolve()
    X, y = load_labeled_xy(cfg.dataset.paths.val, target_col=cfg.dataset.target)

    out_dir = Path(HydraConfig.get().runtime.output_dir)
    writer = ArtifactWriter(out_dir)

    val_df = X.copy(); val_df[cfg.dataset.target] = y
    mlflow_dataset = tracking.to_mlflow_dataset(val_df, source=cfg.dataset.paths.val, name="validation", target=cfg.dataset.target)

    eval_cfg = EvaluationConfig(
        target_col=cfg.dataset.target,
        beta=cfg.eval.beta,
        explain=cfg.eval.explain,
        shap_n_features=cfg.eval.shap_n_features,
        strict=cfg.eval.strict,
    )

    pointer = read_run_pointer(run_dir)
    identity = pointer.get("choices") or ""
    run_name = compose_run_name(identity, prefix='eval')
    tags = {**identity, 'phase': 'evaluation'}
    if pointer.get('run_id'):
        tags["mlflow.parentRunId"] = pointer['run_id']

    with tracking.start_run(tracking_uri=cfg.mlflow.tracking_uri,
                            experiment_name=cfg.mlflow.experiment_name,
                            artifact_location=cfg.mlflow.artifact_location,
                            run_name=run_name, tags=tags
                            ) as run:

        tracking.log_input(mlflow_dataset, context="evaluation")
        result = run_evaluation(model, X, y, writer, eval_cfg)
        tracking.log_metrics(result.scalar_metrics())
        tracking.log_config(out_dir)
        for path in writer.written:
            tracking.log_artifact(path)

        logger.info(result.summary())
        logger.info(f"evaluation: run_id={pointer['run_id']}, artifacts in {out_dir}")


if __name__ == "__main__":
    main()