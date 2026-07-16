from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import shap
import logging

logger = logging.getLogger(__name__)

def build_shap_explainer(estimator, X_background=None):
    """
    Construct an appropriate SHAP explainer for an estimator.

    Preference is given to model-specific explainers, which are
    substantially faster and more accurate than model-agnostic methods.

    Selection order:

    1. TreeExplainer for supported tree-based models.
    2. LinearExplainer for estimators exposing ``coef_``.
    3. Unified ``shap.Explainer`` fallback.
    """
    try:
        return shap.TreeExplainer(estimator)
    except Exception:
        pass
    if hasattr(estimator, 'coef_'):
        try:
            return shap.LinearExplainer(estimator, X_background)
        except Exception:
            pass
    logger.info(f"No fast explainer for {type(estimator).__name__}. "
                f"using unified shap.Explainer.")
    return shap.Explainer(estimator, X_background)

def _split_pipeline(pipeline: Pipeline):
    """Return (preprocessor, estimator)"""
    estimator = pipeline.named_steps.get('classifier', pipeline[-1])
    return pipeline[:-1], estimator

def compute_shap(pipeline: Pipeline, X: pd.DataFrame) -> shap.Explanation:
    preprocessor, estimator = _split_pipeline(pipeline)
    X_t = preprocessor.transform(X)
    explainer = build_shap_explainer(estimator, X_t)
    return explainer(X_t)

def feature_importance_from_shap(explanation: shap.Explanation) -> pd.DataFrame:
    # extract shap values
    shap_values = np.abs(explanation.values)
    if shap_values.ndim == 3:               # (samples, features, classes)
        shap_values = shap_values[:, :, 1]
    mean_abs = shap_values.mean(axis=0)

    # extract feature names, if none, fall-back to feature_i i=1,2,...
    feature_names = explanation.feature_names
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(mean_abs.shape[0])]

    return (
        pd.DataFrame({'feature': feature_names, 'mean_abs_shap': mean_abs})
        .sort_values('mean_abs_shap', ascending=False)  # sort by most important features
        .reset_index(drop=True)
    )