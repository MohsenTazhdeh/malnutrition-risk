import functools
import numpy as np


def check_required_cols(func):
    """Decorator factory that checks for required columns in X for .fit() method inside custom transformers"""
    @functools.wraps(func)
    def wrapper(self, X, *args, **kwargs):
        X_cols = X.columns if hasattr(X, "columns") else []
        required_cols = getattr(self, "required_cols", [])
        missing_cols = set(required_cols) - set(X_cols)
        if missing_cols:
            raise ValueError(f"{self.__class__.__name__} needs these column: {required_cols}.\n"
                             f"currently missing columns: {missing_cols}")
        return func(self, X, *args, **kwargs)
    return wrapper


class TrueLabelCV:
    """
    CV wrapper for self-labeling to ensure only true labels appear in validation folds.
    Pseudo-labels can be in training folds but never in validation.
    """
    def __init__(self, base_cv, true_label_mask: np.ndarray):
        self.base_cv = base_cv
        self.true_label_mask = np.asarray(true_label_mask, dtype=bool)

    def split(self, X, y=None, groups=None):
        for tr_idx, val_idx in self.base_cv.split(X, y, groups):
            yield tr_idx, val_idx[self.true_label_mask[val_idx]]

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.base_cv.get_n_splits(X, y, groups)
