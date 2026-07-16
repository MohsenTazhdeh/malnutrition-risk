import pandas as pd


def load_labeled_xy(source, target_col: str, label_col: str = 'has_label'):
    df = source if isinstance(source, pd.DataFrame) else pd.read_parquet(source)
    if label_col in df.columns:
        df = df.loc[df[label_col] == 1]
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

