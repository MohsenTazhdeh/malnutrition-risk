from sklearn.metrics import (precision_recall_curve, average_precision_score, roc_auc_score, confusion_matrix)
from .schemas import CurveData, OperatingPoint, ClassificationMetrics, ConfusionCounts, EvaluationResult
import numpy as np


def confusion_at_threshold(y_true, y_proba, threshold: float) -> ConfusionCounts:
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return ConfusionCounts(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))

def threshold_independent_metrics(y_true, y_proba) -> ClassificationMetrics:
    return ClassificationMetrics(
        pr_auc=float(average_precision_score(y_true, y_proba)), # average-precision(ap)
        roc_auc=float(roc_auc_score(y_true, y_proba)),
    )

def f_beta_scores(precisions: np.ndarray, recalls: np.ndarray, beta: float) -> np.ndarray:
    num = (1 + beta ** 2) * (precisions * recalls)
    den = (beta ** 2 * precisions) + recalls
    return np.divide(num, den, out=np.zeros_like(num), where=den != 0)

def pr_curve(y_true, y_proba) -> CurveData:
    p, r, t = precision_recall_curve(y_true, y_proba)
    return CurveData(precisions=p, recalls=r, thresholds=t)

def select_threshold_max_fbeta(curve: CurveData, beta: float) -> OperatingPoint:
    # thresholds has length n; precisions/recalls have length n+1.
    # restrict the search to indices that map to a real threshold.
    n = len(curve.thresholds)
    scores = f_beta_scores(curve.precisions[:n], curve.recalls[:n], beta=beta)
    best = int(np.argmax(scores))
    return OperatingPoint(
        precision=float(curve.precisions[best]),
        recall=float(curve.recalls[best]),
        threshold=float(curve.thresholds[best]),
        f_beta=float(scores[best]),
        beta=beta,
    )

def compute_metrics(y_true, y_proba, beta: float = 2.0) -> EvaluationResult:
    curve = pr_curve(y_true, y_proba)
    operating_point = select_threshold_max_fbeta(curve, beta=beta)
    return EvaluationResult(
        metrics=threshold_independent_metrics(y_true, y_proba),
        operating_point=operating_point,
        confusion=confusion_at_threshold(y_true, y_proba, operating_point.threshold),
        curve=curve,
    )

