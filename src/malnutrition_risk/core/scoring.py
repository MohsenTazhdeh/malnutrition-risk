import numpy as np

def positive_class_proba(model, X):
    proba = np.asarray(model.predict_proba(X))
    if proba.ndim != 2 or proba.shape[1] != 2:
        raise ValueError(
            f"positive_class_proba expects binary predict_proba (n, 2); got shape {proba.shape}."
        )

    return proba[:, 1]