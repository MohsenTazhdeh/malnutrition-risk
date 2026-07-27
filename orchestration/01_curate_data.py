import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path, instantiate
from pathlib import Path
import pandas as pd
import logging
from malnutrition_risk.core import tracking
from malnutrition_risk.utils import write_parquet

logger = logging.getLogger(__name__)

@hydra.main(version_base='1.3', config_path='../conf', config_name='curate')
def main(cfg: DictConfig) -> None:
    logger.info("====== Starting Curation ======")

    df = pd.read_parquet(cfg.dataset.paths.raw)
    logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")

    raw_dataset = tracking.to_mlflow_dataset(df, source=cfg.dataset.paths.raw, target=cfg.dataset.target, name='raw')

    tags = {"phase": "data_prep", "stage": "curation",
            "dataset": HydraConfig.get().runtime.choices.dataset}

    with tracking.start_run(tracking_uri=cfg.mlflow.tracking_uri,
                            experiment_name=cfg.mlflow.experiment_name,
                            artifact_location=cfg.mlflow.artifact_location,
                            tags=tags, run_name='curate'):
        
        tracking.log_input(raw_dataset, context='curation')

        curator = instantiate(cfg.data_prep.curation)
        df_curated = curator.curate(df.copy())

        output_path = write_parquet(df_curated, cfg.dataset.paths.curated)
        logger.info(f"Saved curated data to {output_path}")

        tracking.log_params({
            "target": cfg.data_prep.curation.target,
            "group": cfg.data_prep.curation.group,
            "n_no_nan_cols": len(cfg.data_prep.curation.no_nan_cols),
            "raw_rows": len(df),
            "curated_rows": len(df_curated),
        })
        tracking.log_metrics({
            "raw_malnutrition_cases": float(df[cfg.data_prep.curation.target].sum()),
            "raw_malnutrition_rate": float(df[cfg.data_prep.curation.target].mean()),
            "curated_malnutrition_cases": float(df_curated[cfg.data_prep.curation.target].sum()),
            "curated_malnutrition_rate": float(df_curated[cfg.data_prep.curation.target].mean()),
        })

        run_dir = Path(HydraConfig.get().runtime.output_dir)
        tracking.log_config(run_dir)

if __name__ == '__main__':
    main()
