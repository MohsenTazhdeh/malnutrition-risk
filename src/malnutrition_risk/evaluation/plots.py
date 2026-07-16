from .schemas import CurveData, OperatingPoint
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import shap
import numpy as np


def pr_curve_figure(curve: CurveData, op: OperatingPoint) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(curve.recalls, curve.precisions, label='Precision-Recall', color='blue')
    ax.scatter(op.recall, op.precision, label=f"Optimal Threshold: {op.threshold:.2f}", color='red', s=100, zorder=10)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve for Malnutrition Risk")
    ax.legend(loc="lower left");
    ax.grid(True, linestyle="--", alpha=0.7)
    return fig

def shap_summary_figure(explanation):
    # shap draws on the current pyplot figure; capture and return it.
    plt.figure(figsize=(10, 6))
    shap.summary_plot(explanation, explanation.data, show=False)
    fig = plt.gcf(); fig.tight_layout()
    return fig

def shap_waterfall_figure(explanation, index: int = 0):
    single = explanation[index]
    if np.asarray(single.values).ndim > 1:
        single = single[:, 1]
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(single, show=False)
    fig = plt.gcf(); fig.tight_layout()
    return fig

def feature_importance_figure(importance_df, n_features=20):
    bottom = importance_df.tail(n_features)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(bottom["feature"], bottom["mean_abs_shap"], color="coral")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Bottom {n_features} Features by SHAP Importance")
    fig.tight_layout()
    return fig
