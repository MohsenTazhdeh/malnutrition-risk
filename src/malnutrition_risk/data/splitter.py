import pandas as pd
import numpy  as np
import logging
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from pathlib import Path

""" two splitting strategy:
1. Balancing Individual Labels
This strategy will prioritize balancing total number of malnutrition individuals in each split
the flaw of this strategy is that it ignores household malnutrition density. we could potentially
end up with a train/test mismatch where the training set has mostly isolated cases (one sick person
per household) while the test set has clustered cases (households with a high density of malnutrition)
for example It might put 10 households with 1 sick person each into the Train set, and 
2 households with 5 sick people each into the Test set.


2. Balancing Household Labels
This strategy will prioritize balancing total number of affected households in each split.
The flaw of this strategy is that it ignores the total volume of malnourished individuals. 
for example we could end up in a situation where it might put 5 households with 1 sick person each
into the Train set (Total: 5 malnutrition cases), and 5 households with 5 sick people each 
into the Test set (Total: 25 malnutrition cases).
"""

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HouseholdAwareSplitter:
    """
    splits data at household level with stratification on malnutrition label

    strategy:
    1. Split labeled households using StratifiedGroupKFold (stratified by presence
     of at least one positive label)
    2. split unlabeled households (households where no member has label) using GroupShuffleSplit
    3. assign all household members to same split
    """

    def __init__(self, target, group, split_size, random_state):
        self.target = target
        self.group = group
        self.split_size = split_size
        self.random_state = random_state

    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        split dataframe into train and test at household level

        returns:
            train_df, test_df
        """
        logger.info(f"Starting household-aware split (split_size={self.split_size})")

        # split households into train and test
        hh_train, hh_test = self._split_households(df)

        # assign individuals to splits based on household
        train_df = df.loc[df[self.group].isin(hh_train)]
        test_df = df.loc[df[self.group].isin(hh_test)]

        self._log_split_stats(df, train_df, test_df)

        return train_df, test_df

    def _split_households(self, df: pd.DataFrame) -> tuple[set, set]:

        # separate label and unlabel individuals
        df_labeled = df.loc[df['has_label'] == 1].copy()

        # split labeled households
        hh_train_label, hh_test_label = self._split_label_hh(df_labeled)

        # split unlabeled households
        hh_train_unlabel, hh_test_unlabel = self._split_unlabel_hh(df, hh_train_label, hh_test_label)

        return hh_train_label | hh_train_unlabel, hh_test_label | hh_test_unlabel

    def _split_label_hh(self, df_labeled: pd.DataFrame) -> tuple[set, set]:
        """Stratified split of households with at least one labeled member."""

        # create household level strata:
        # 1 = household has at least one labeled positive (after propagation)
        # 0 = household has no labeled positive (may have 0s and/or NaNs)
        hh_strata = (
            df_labeled
            .groupby(self.group)[self.target]
            .apply(lambda s: int((s == 1).any()))
            .reset_index(name='hh_has_pos_label')
        )

        logger.info(f"labeled households {len(hh_strata)}"
                    f", positive cases: {hh_strata['hh_has_pos_label'].sum()}")

        # stratified group split
        n_splits = int(1 / self.split_size)
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        y_hh = hh_strata[f"hh_has_pos_label"].to_numpy()
        g_hh = hh_strata[self.group].to_numpy()

        train_idx, test_idx = next(sgkf.split(X=np.zeros(len(hh_strata)), y=y_hh, groups=g_hh))

        hh_train_label = set(hh_strata[self.group].iloc[train_idx])
        hh_test_label = set(hh_strata[self.group].iloc[test_idx])

        return hh_train_label, hh_test_label


    def _split_unlabel_hh(self, df, hh_train_label, hh_test_label) -> tuple[set, set]:
        """Split households with only unlabeled members."""

        all_hh = set(df[self.group])
        hh_with_label = hh_train_label | hh_test_label  # households with at least one labeled member (0 or 1)
        hh_with_no_label = np.array(list(all_hh - hh_with_label))  # no member has malnutrition label

        logger.info(f"unlabeled households {len(hh_with_no_label)}")

        gss = GroupShuffleSplit(n_splits=1, test_size=self.split_size, random_state=self.random_state)
        train_idx, test_idx = next(gss.split(np.zeros(len(hh_with_no_label)), groups=hh_with_no_label))

        hh_train_unlabel = set(hh_with_no_label[train_idx])
        hh_test_unlabel = set(hh_with_no_label[test_idx])

        return hh_train_unlabel, hh_test_unlabel

    def _log_split_stats(self, df, train_df, test_df) -> None:

        logger.info(f"total individuals: {len(df)}")
        logger.info(f"train individuals: {len(train_df)}")
        logger.info(f"test individuals: {len(test_df)}")

        # label distribution in train/test
        logger.info(f"labeled individuals in total: {len(df.loc[df['has_label'] == 1])}")
        logger.info(f"labeled individuals in training set: {len(train_df.loc[train_df['has_label'] == 1])}")
        logger.info(f"labeled individuals in test set: {len(test_df.loc[test_df['has_label'] == 1])}")

        # malnutrition rate in train/test
        logger.info(f" malnutrition individuals in total: {len(df.loc[df[self.target] == 1])}")
        logger.info(f" malnutrition individuals in train: {len(train_df.loc[train_df[self.target] == 1])}")
        logger.info(f" malnutrition individuals in test: {len(test_df.loc[test_df[self.target] == 1])}")


# splitter = HouseholdAwareSplitter('Has_SoeTaghzie', 'Parent_Id', test_size=0.2, random_state=42)
# df = pd.read_parquet(r"H:\Lets see\Projects\08 Malnutrition Risk\data\curated\curated_data.parquet")
# splitter.split(df)


