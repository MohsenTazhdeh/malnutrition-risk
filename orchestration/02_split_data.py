import pandas as pd
import hydra
from omegaconf import DictConfig
from malnutrition_risk.data.splitter import HouseholdAwareSplitter
import mlflow
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

@hydra.main(version_base='1.3', config_path='../conf', config_name="config")
def main(cfg: DictConfig):
    """Split curated data into train/test sets at household level."""

    # setup mlflow
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run(run_name="split_data"):
        split_cfg = cfg.data_prep.splitting

        mlflow.log_params({
            "train_size": 1 - (split_cfg.val_size + split_cfg.test_size),
            "val_size": split_cfg.val_size,
            "test_size": split_cfg.test_size,
            "random_state": split_cfg.random_state,
            "target": split_cfg.target,
            "group": split_cfg.group
        })

        # load curated data
        logger.info(f"Loading curated data from {cfg.dataset.paths.curated}")
        df = pd.read_parquet(cfg.dataset.paths.curated)

        # first split: isolate test set
        train_val_df, test_df = HouseholdAwareSplitter(
            target=split_cfg.target,
            group=split_cfg.group,
            split_size=split_cfg.test_size,
            random_state=split_cfg.random_state,
        ).split(df)

        # val_size is relative to original data, so adjust for train_val size
        val_fraction = split_cfg.val_size / (1 - split_cfg.test_size)

        # second split: separate train and validation set
        # use a different seed for second split. this ensures the two splits use different random seeds,
        # to get independent randomization and avoid split patterns that might correlate in unexpected ways.
        train_df, val_df = HouseholdAwareSplitter(
            target=split_cfg.target,
            group=split_cfg.group,
            split_size=val_fraction,
            random_state=split_cfg.random_state + 1,
        ).split(train_val_df)

        # save splits
        for name, data, path_key in [
            ('train', train_df, 'train'),
            ('validation', val_df, 'val'),
            ('test', test_df, 'test'),
        ]:
            path = Path(cfg.dataset.paths[path_key])
            path.parent.mkdir(parents=True, exist_ok=True)
            data.to_parquet(path, index=False)
            logger.info(f"saved {name} split to {path}")
            mlflow.log_artifact(str(path), artifact_path="data")

        mlflow.log_metrics({
            "total individuals": len(df),
            "train individuals": len(train_df),
            "val individuals": len(val_df),
            "test individuals": len(test_df),
            "labeled individuals in total": len(df.loc[df['has_label'] == 1]),
            "labeled individuals in training set": (train_df['has_label'] == 1).sum(),
            "labeled individuals in validation set": (val_df['has_label'] == 1).sum(),
            "labeled individuals in test set": (test_df['has_label'] == 1).sum(),
            "malnutrition individuals in total": (df[split_cfg.target] == 1).sum(),
            "malnutrition individuals in training set": (train_df[split_cfg.target] == 1).sum(),
            "malnutrition individuals in validation set": (val_df[split_cfg.target] == 1).sum(),
            "malnutrition individuals in test set": (test_df[split_cfg.target] == 1).sum()
        })

if __name__ == "__main__":
    main()
