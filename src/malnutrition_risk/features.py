import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import validate_data
from malnutrition_risk.utils import check_required_cols


"""Custom Transformers For Feature Engineering"""

def standardize_nan(X: pd.DataFrame) -> pd.DataFrame:
    return X.replace({pd.NA: np.nan, None: np.nan})

class ToCategory(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        validate_data(self, X, reset=True, dtype=None, ensure_all_finite=False)

        # convert pinned categories to plain list to make it serializable for serving/inference
        self.categories_ = {c : X[c].dropna().astype('category').cat.categories.to_list() for c in X.columns}

        return self

    def transform(self, X):
        check_is_fitted(self, ["categories_", "feature_names_in_"])
        validate_data(self, X, reset=False, dtype=None, ensure_all_finite=False)

        X = X.copy()
        for c in X.columns:
            X[c] = pd.Categorical(X[c], categories=self.categories_[c])

        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            check_is_fitted(self, "feature_names_in_")
            input_features = self.feature_names_in_
        return np.array(input_features, dtype=object)

class PostalCodeTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, digits_to_keep: int, postal_column: str):
        self.postal_column = postal_column
        self.digits_to_keep = digits_to_keep

    @property
    def required_cols(self):
        return [self.postal_column]

    def _parse_postal(self, X):
        return X[self.postal_column].dropna().astype(int).astype(str).str[:self.digits_to_keep]

    @check_required_cols
    def fit(self, X, y=None):
        # create self.n_features_in_ and self.feature_names_in_
        validate_data(self, X, reset=True, dtype=None, ensure_all_finite=False)

        parsed_postal_code = self._parse_postal(X)
        self.postal_categories_ = {self.postal_column : parsed_postal_code.astype('category').cat.categories.to_list()}

        return self

    def transform(self, X):
        check_is_fitted(self, ["postal_categories_", "n_features_in_"])

        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"{self.__class__.__name__}.transfrom() method needs a pandas DataFrame as input")

        # check that column names match the ones seen during fit()
        validate_data(self, X, reset=False, dtype=None, ensure_all_finite=False)
        X = X.copy()

        parsed_postal_code = self._parse_postal(X)
        # pd.Categorical(values, categories=...) returns an array whose length equals len(values)
        X[f"{self.postal_column}_{self.digits_to_keep}_digits"] = pd.Series(
            pd.Categorical(parsed_postal_code.values, categories=self.postal_categories_[self.postal_column]),
            index=parsed_postal_code.index,
        )

        X = X.drop(columns=[self.postal_column])

        return X

    def get_feature_names_out(self, input_features=None):
        # Use passed features, or fall back to the features seen during fit()
        if input_features is None:
            check_is_fitted(self, "feature_names_in_")
            input_features = self.feature_names_in_

        # calculate the output names
        features = list(input_features)
        if self.postal_column in features:
            features.remove(self.postal_column)
            features.append(f"{self.postal_column}_{self.digits_to_keep}_digits")

        return np.array(features, dtype=object)

class VulnerabilityIndexTransformer(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            bimar_col: str = 'ISBimarKhas',
            malool_col: str = 'IsMalool',
            age_col: str = 'Age',
            percentile_col: str = 'Percentile',
            bime_col: str = 'is_bime_darman'
    ):
        self.bimar_col = bimar_col
        self.malool_col = malool_col
        self.age_col = age_col
        self.percentile_col = percentile_col
        self.bime_col = bime_col

    @property
    def required_cols(self):
        return [
            self.bimar_col, self.malool_col, self.age_col,
            self.percentile_col, self.bime_col
        ]

    @check_required_cols
    def fit(self, X, y=None):
        # store the input column names for validation in transform
        validate_data(self, X, reset=True, dtype=None, ensure_all_finite=False)
        return self

    def transform(self, X):
        check_is_fitted(self, "n_features_in_")

        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"{self.__class__.__name__}.transfrom() method needs a pandas DataFrame as input")

        # Check that the input features match the ones seen during fit
        validate_data(self, X, reset=False, dtype=None, ensure_all_finite=False)


        X = X.copy()

        is_bimar = (X[self.bimar_col] == 1).astype(int)
        is_malool = (X[self.malool_col] == 1).astype(int)
        is_dep_age = ((X[self.age_col] < 15) | (X[self.age_col] > 65)).astype(int)
        is_low_percentile = (X[self.percentile_col] <= 30).astype(int)
        no_bime = (X[self.bime_col] == 0).astype(int)

        X['Confirmed_Risk_Count'] = (is_bimar + is_malool + is_dep_age + is_low_percentile + no_bime)

        return X

    def get_feature_names_out(self, input_features=None):
        # If the pipeline didn't provide input_features, use the columns we memorized during fit()
        if input_features is None:
            check_is_fitted(self, "feature_names_in_")
            input_features = self.feature_names_in_

        features = list(input_features)
        features.append('Confirmed_Risk_Count')

        return np.array(features, dtype=object)

def _provincial_record(long_df: pd.DataFrame, year: str, value: str) -> dict:
    """A skops-safe snapshot of one year's provincial lookup + national fallbacks.
        Only str/int/float/list — no pandas objects, no tuple keys."""
    sub = long_df[long_df['year'] == year]
    return {
        'Province': [str(p) for p in sub['Province']],
        'isurban':  [int(u) for u in sub['isurban']],
        'values':      [float(v) for v in sub[value]],          # CPI or Inflation_rate
        "national_urban": float(sub.loc[(sub['Province'] == 'کل کشور') & (sub['isurban'] == 1), value].iloc[0]),
        "national_rural": float(sub.loc[(sub['Province'] == 'کل کشور') & (sub['isurban'] == 0), value].iloc[0]),
    }

def _lookup_with_fallback(X: pd.DataFrame, records: dict, province_col: str, isurban_col: str) -> pd.Series:
    """Rebuild a transient MultiIndex Series from the stored record and map rows,
        filling unmatched (province, isurban) with the national rate."""

    # key-mapping for CPI/Inflation table
    key_mapping = pd.Series(records['values'], index=pd.MultiIndex.from_arrays([records['Province'], records['isurban']]))

    # generate a MultiIndex object from 'SabteAhval_provincename' & 'isurban' columns
    mul_idx = pd.MultiIndex.from_arrays([X[province_col], X[isurban_col]])

    # get CPI/Inflation values
    values = pd.Series(mul_idx.map(key_mapping), index=X.index, dtype=float)

    # fallback conditions for when Province or isurban column are missing
    conditions = [X[isurban_col] == 1, X[isurban_col] == 0]
    fallbacks = np.select(
        conditions,
        [records['national_urban'], records['national_rural']],
        default=(records['national_urban'] + records['national_rural']) / 2
    )

    ### fill missing CPI/Inflation values with national CPI/Inflation values and return###
    return values.fillna(pd.Series(fallbacks, index=X.index))


class CPIAdjustmentTransformer(BaseEstimator, TransformerMixin):

    def __init__(self,
                 cpi_urban_path: str,
                 cpi_rural_path: str,
                 card_cols: list[str],
                 province_col: str = 'SabteAhval_provincename',
                 isurban_col: str = 'isurban',
                 base_year: str = '1400'
                 ):
        self.cpi_urban_path = cpi_urban_path
        self.cpi_rural_path = cpi_rural_path
        self.card_cols = card_cols
        self.province_col = province_col
        self.isurban_col = isurban_col
        self.base_year = base_year

    def _load_cpi_df(self) -> pd.DataFrame:
        # load csv files
        urban_df = pd.read_csv(self.cpi_urban_path)
        rural_df = pd.read_csv(self.cpi_rural_path)

        # add isurban col
        urban_df['isurban'] = 1
        rural_df['isurban'] = 0

        # merge urban and rural dataframes
        df = pd.concat([urban_df, rural_df], axis='index', ignore_index=True)

        # convert to a long df
        long_df = df.melt(id_vars=['Province', 'isurban'], var_name='year', value_name='CPI')
        long_df['year'] = long_df['year'].str.replace('CPI_', '')
        return long_df

    @property
    def required_cols(self):
        return self.card_cols + [self.province_col] + [self.isurban_col]

    @check_required_cols
    def fit(self, X, y=None):
        cpi_df = self._load_cpi_df()
        self.base_cpi_ = _provincial_record(cpi_df, year=self.base_year, value='CPI')
        years = [col.split('_')[-1] for col in self.card_cols]
        self.cpi_by_year_ = {year: _provincial_record(cpi_df, year, 'CPI') for year in years}

        validate_data(self, X, reset=True, dtype=None, ensure_all_finite=False)
        return self

    def transform(self, X):
        """Apply CPI adjustment to CardPerMonth features."""

        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"{self.__class__.__name__}.transform() method needs a pandas DataFrame as input")

        check_is_fitted(self, ["base_cpi_", "cpi_by_year_", "feature_names_in_"])
        validate_data(self, X, reset=False, dtype=None, ensure_all_finite=False)

        X = X.copy()
        base_cpi_values = _lookup_with_fallback(X, self.base_cpi_, self.province_col, self.isurban_col)
        for col in self.card_cols:
            year = col.split("_")[-1]
            current_cpi_values = _lookup_with_fallback(X, self.cpi_by_year_[year], self.province_col, self.isurban_col)
            # apply cpi adjustment to card_cols
            X[f"cpi_adj_{col}"] = X[col] * (base_cpi_values / current_cpi_values)
            X[f"cpi_adj_{col}"] = X[f"cpi_adj_{col}"].replace(0, np.nan)

        X = X.drop(columns=self.card_cols)

        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            check_is_fitted(self, ["base_cpi_", "cpi_by_year_", "feature_names_in_"])
            input_features = self.feature_names_in_

        features = list(input_features)

        # remove card_cols names from feature names
        features = [col for col in features if col not in self.card_cols]

        # add cpi adjusted names to feature names
        features.extend([f"cpi_adj_{col}" for col in self.card_cols])

        return np.array(features, dtype=object)


class InflationAdjustmentTransformer(BaseEstimator, TransformerMixin):

    def __init__(self,
                 inflation_urban_path: str,
                 inflation_rural_path: str,
                 card_cols: list[str],
                 province_col: str = 'SabteAhval_provincename',
                 urban_col: str = 'isurban',
                 ):

        self.inflation_urban_path = inflation_urban_path
        self.inflation_rural_path = inflation_rural_path
        self.card_cols = card_cols
        self.province_col = province_col
        self.urban_col = urban_col

    def _load_inflation_df(self):
        urban_df = pd.read_csv(self.inflation_urban_path)
        rural_df = pd.read_csv(self.inflation_rural_path)
        urban_df['isurban'] = 1
        rural_df['isurban'] = 0
        df = pd.concat([urban_df, rural_df], axis='index', ignore_index=True)
        long_df = df.melt(id_vars=['Province', 'isurban'], var_name='year', value_name='Inflation_Rate')
        long_df['year'] = long_df['year'].str.replace("Inflation_rate_", "")
        return long_df

    @property
    def required_cols(self):
        return self.card_cols + [self.province_col] + [self.urban_col]

    @check_required_cols
    def fit(self, X, y=None):
        long_df = self._load_inflation_df()
        new_years = {str(int(col.split("_")[-1]) + 1) for col in self.card_cols[:-1]}
        self.inf_by_year_ = {ny: _provincial_record(long_df, ny, "Inflation_Rate") for ny in new_years}

        validate_data(self, X, reset=True, dtype=None, ensure_all_finite=False)
        return self

    def transform(self, X):

        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"{self.__class__.__name__}.transfrom() method needs a pandas DataFrame as input")

        check_is_fitted(self, ["inf_by_year_", "feature_names_in_"])
        validate_data(self, X, reset=False, dtype=None, ensure_all_finite=False)

        X = X.copy()
        for col in self.card_cols[:-1]:
            current_year = col.split("_")[-1]
            new_year = str(int(current_year) + 1)
            inf_new_vals = _lookup_with_fallback(X, self.inf_by_year_[new_year], self.province_col, self.urban_col)
            next_col = col.replace(current_year, new_year)
            nominal_growth = np.where(X[col] != 0, (X[next_col] - X[col]) / X[col], np.nan)
            X[f"real_growth_{next_col}"] = ((1 + nominal_growth) / (1 + inf_new_vals)) - 1

        X = X.drop(columns=self.card_cols)
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            check_is_fitted(self, "feature_names_in_")
            input_features = self.feature_names_in_
        features = list(input_features)
        # features = [col for col in features if col not in self.card_cols]
        features.extend([f"real_growth_{col}" for col in self.card_cols[1:]])
        return np.array(features, dtype=object)























