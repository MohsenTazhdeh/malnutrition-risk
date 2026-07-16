import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
from pathlib import Path
import pandas as pd
import mlflow
import logging
from malnutrition_risk.data.curator import DataCurator

logger = logging.getLogger(__name__)

@hydra.main(version_base='1.3', config_path='../conf', config_name='config')
def main(cfg: DictConfig) -> None:
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run(run_name="curate_data"):
        # convert Hydra config to a flat dictionary and log as MLflow parameter
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))

        # load data
        df = pd.read_parquet(to_absolute_path(cfg.dataset.paths.raw))
        logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")

        # log raw data statistics
        mlflow.log_metrics({
            "raw data total records": len(df),
            "raw_data_malnutrition_cases": df[cfg.dataset.target].sum(),
            "raw_data_malnutrition_rate": df[cfg.dataset.target].mean()
        })

        curation_cfg = cfg.data_prep.curation

        # curate
        curator = DataCurator(
            curation_cfg.target,
            curation_cfg.group,
            curation_cfg.cate_cols,
        )
        logger.info(f"initialized {curator}")
        df_curated = curator.curate(df)

        # save curated data
        output_path = Path(cfg.dataset.paths.curated)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_curated.to_parquet(output_path, index=False)
        logger.info(f"saved curated data to {output_path}")

        # log curated data as MLflow artifact
        mlflow.log_artifact(str(output_path), artifact_path="data")

        # log curated data statistics
        mlflow.log_metrics({
            "curated data malnutrition cases": df_curated[curation_cfg.target].sum(),
            "curated data malnutrition rate": df_curated[curation_cfg.target].mean()
        })

if __name__ == '__main__':
    main()
