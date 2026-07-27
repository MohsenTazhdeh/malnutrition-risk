import pandas as pd
import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from malnutrition_risk.data.splitter import SplitConfig, run_split
import logging
from malnutrition_risk.core import tracking
from pathlib import Path
from malnutrition_risk.utils import write_parquet

logger = logging.getLogger(__name__)

@hydra.main(version_base='1.3', config_path='../conf', config_name="split")
def main(cfg: DictConfig):
    """Split curated data into train/validation/test sets at household level."""
    logger.info("====== Starting Split ======")
    splitting_cfg = cfg.data_prep.splitting
    
    df = pd.read_parquet(cfg.dataset.paths.curated)
    logger.info(f"Loaded curated data from {cfg.dataset.paths.curated}")

    curated_dataset = tracking.to_mlflow_dataset(
        df, source=cfg.dataset.paths.curated, target=cfg.dataset.target, name='curated')

    splitter_factory = instantiate(splitting_cfg.splitter)
    split_cfg = SplitConfig(
        target=splitting_cfg.splitter.target,
        group=splitting_cfg.splitter.group,
        label_col=splitting_cfg.splitter.label_indicator_col,
        test_size=splitting_cfg.test_size,
        val_size=splitting_cfg.val_size,
        random_state=splitting_cfg.random_state,
    )

    tags = {"phase": "data_prep", "stage": "split",
            "dataset": HydraConfig.get().runtime.choices.dataset}


    with tracking.start_run(tracking_uri=cfg.mlflow.tracking_uri,
                            experiment_name=cfg.mlflow.experiment_name,
                            artifact_location=cfg.mlflow.artifact_location,
                            run_name="split", tags=tags):
        
        tracking.log_input(curated_dataset, context="split")
        tracking.log_params({
            "train_size": 1 - (split_cfg.val_size + split_cfg.test_size),
            "val_size": split_cfg.val_size,
            "test_size": split_cfg.test_size,
            "random_state": split_cfg.random_state,
            "target": split_cfg.target,
            "group": split_cfg.group,
        })

        splits, stats = run_split(df, splitter_factory, split_cfg)

        for name, path_key in [('train', 'train'), ('validation', 'val'), ('test', 'test')]:
            path = write_parquet(splits[name], cfg.dataset.paths[path_key])
            logger.info(f"saved {name} split to {path}")
            tracking.log_artifact(path, artifact_path="data")

        tracking.log_metrics(stats)

        run_dir = Path(HydraConfig.get().runtime.output_dir)
        tracking.log_config(run_dir)


if __name__ == "__main__":
    main()
