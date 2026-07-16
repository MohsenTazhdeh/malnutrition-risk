import hydra
from omegaconf import DictConfig
from malnutrition_risk.core import model_io
import logging

logger = logging.getLogger(__name__)

@hydra.main(version_base="1.3", config_path='../conf', config_name="promote")
def main(cfg: DictConfig):
    pointer = model_io.read_run_pointer(cfg.promote.run_dir)
    if not pointer:
        raise FileNotFoundError(f"No mlflow_run.json file in {cfg.promote.run_dir}.\n"
                                f"pass a training run directory and try again. e.g.\n"
                                f"promote.run_dir=outputs/experiments/<dataset>/<fe>/<preproc>/<model>/<timestamp>"
                                )

    version = model_io.register_model(
        model_uri=pointer["model_uri"],
        name=cfg.promote.name,
        tracking_uri=cfg.mlflow.tracking_uri,
        alias=cfg.promote.alias
    )

    logger.info(f"registered {cfg.promote.name} v{version} as @{cfg.promote.alias} "
                f"from {pointer['model_uri']}")

if __name__ == "__main__":
    main()