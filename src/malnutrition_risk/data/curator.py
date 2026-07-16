import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class DataCurator:

    _STEP_NAMES = [
        'label_propagation',
        'add_label_indicator',
        'fix_implicit_zeros',
        'fix_car_price_logic',
        'fix_car_implicit_zeros',
        'change_zipcode_name',
        'standardize_missing_values'
    ]

    def __init__(self,
                 target: str,
                 group: str,
                 no_nan_cols: list[str]):

        self.target = target
        self.group = group
        self.no_nan_cols = list(no_nan_cols)

    def label_propagation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Malnutrition is a household-level state, not an individual state.
        this method will implement this assumption in households with at least
        one malnutrition member through label propagation to all other unlabeled
        members within the same household."""
        # count_malnutrition_cases / count_labeled_cases
        original_rate = df[self.target].mean()

        # find households with at least 1 confirmed positive (malnutrition) label
        pos_hh = df.loc[df[self.target] == 1, self.group].unique()

        # mark all unlabeled members of those households as positive(has malnutrition)
        df.loc[(df[self.group].isin(pos_hh)) & (df[self.target].isna()), self.target] = 1

        new_rate = df[self.target].mean()
        logger.info(f"original malnutrition rate: {original_rate*100:.2f}%"
                    f" \nmalnutrition rate "
                    f"after label propagation: {new_rate*100:.2f}%")
        return df

    def add_label_indicator(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add binary indicator for whether individual has observed (malnutrition) label"""
        df['has_label'] = df[self.target].isin([0,1]).astype(int)
        logger.info(f"added a new column `has_label` to indicate whether"
                    f"an individual has observed (malnutrition) label")
        logger.info(f"{df['has_label'].sum()} rows have label.")
        return df

    def fix_implicit_zeros(self, df: pd.DataFrame) -> pd.DataFrame:
        """fill nan with zeros for binary columns where nan means zero"""
        df[self.no_nan_cols] = df[self.no_nan_cols].fillna(0)
        logger.info("fixed implicit zeros for binary columns where nan means zero")
        return df

    @staticmethod
    def fix_car_price_logic(df: pd.DataFrame) -> pd.DataFrame:
        """Set Car Price to nan when CarsCount>0 but CarsPrice = 0
        note: there's no inconsistency between missing values of
        CarsCount and CarsPrice. whenever CarsCount is NaN CarsPrice is also NaN"""
        mask = (df['CarsPrice'] == 0) & (df['CarsCount'] > 0)
        df.loc[mask, 'CarsPrice'] = np.nan

        logger.info(f"fixed {mask.sum()} illogical car prices")
        return df

    @staticmethod
    def fix_car_implicit_zeros(df: pd.DataFrame) -> pd.DataFrame:
        """
        there are more than 1 million rows where CarsCount & CarsPrice is missing.
        these are not real NaNs. they just don't have cars!
        """
        mask = (df['CarsCount'].isna() & df['CarsPrice'].isna())
        df[['CarsCount', 'CarsPrice']] = df.loc[mask, ['CarsCount', 'CarsPrice']].fillna(0)
        logger.info(f"fixed implicit {len(df.loc[mask])} zeros for CarPrice & CarCount")
        return df

    @staticmethod
    def change_zipcode_name(df: pd.DataFrame) -> pd.DataFrame:
        """ Change 'Dashboard_postalcode7Digits' column name to 'postalcode' """
        logger.info("Changing zipcode 'Dashboard_postalcode7Digits' column name to 'postal_code'")
        return df.rename(columns={'Dashboard_postalcode7Digits': 'postal_code'})

    @staticmethod
    def standardize_missing_values(df:pd.DataFrame) -> pd.DataFrame:
        """Convert all pd.NA in the DataFrame to np.nan for sklearn compatibility"""
        logger.info(f"standardizing missing values")
        return df.replace({pd.NA: np.nan, None: np.nan})

    def __len__(self) -> int:
        return len(self._STEP_NAMES)

    def __repr__(self) -> str:
        # len(self) invokes the __len__ method
        return f"DataCurator({len(self)} steps: {self._STEP_NAMES})"

    def __getitem__(self, index: int):
        names = self._STEP_NAMES[index]
        if isinstance(names, list):
            return [(name, getattr(self, name)) for name in names]
        return names, getattr(self, names)

    def curate(self, df: pd.DataFrame) -> pd.DataFrame:
        """apply all curation steps in sequence"""
        logger.info(f"curating {len(df)} records with {len(self)} steps")

        for name, step in self:
            df = step(df)

        logger.info(f"curation complete")

        return df


