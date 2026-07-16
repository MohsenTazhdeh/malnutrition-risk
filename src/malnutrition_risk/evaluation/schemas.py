from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CurveData:
    precisions: np.ndarray
    recalls: np.ndarray
    thresholds: np.ndarray

@dataclass(frozen=True)
class OperatingPoint:
    precision: float
    recall: float
    threshold: float
    f_beta: float
    beta: float

@dataclass(frozen=True)
class ClassificationMetrics:
    pr_auc: float   # average-precision(ap)
    roc_auc: float

@dataclass(frozen=True)
class ConfusionCounts:
    tn: int
    fp: int
    fn: int
    tp: int

@dataclass(frozen=True)
class EvaluationResult:
    metrics: ClassificationMetrics
    operating_point: OperatingPoint
    confusion: ConfusionCounts
    curve: CurveData

    def scalar_metrics(self) -> dict[str, float]:
        """Flat, JSON-serializable dict, compatible with MLflow API(mlflow.log_metrics)."""
        return {
            "pr_auc": self.metrics.pr_auc,
            "roc_auc": self.metrics.roc_auc,
            "optimal_threshold": self.operating_point.threshold,
            "precision": self.operating_point.precision,
            "recall": self.operating_point.recall,
            "f_beta": self.operating_point.f_beta,
            "tp": self.confusion.tp,
            "fp": self.confusion.fp,
            "tn": self.confusion.tn,
            "fn": self.confusion.fn,
        }

    def summary(self) -> str:
        c = self.confusion
        return (
            f"PR-AUC={self.metrics.pr_auc:.4f} ROC-AUC={self.metrics.roc_auc:.4f} | "
            f"threshold={self.operating_point.threshold:.4f} | "
            f"precision={self.operating_point.precision:.4f} "
            f"recall={self.operating_point.recall:.4f} "
            f"F{self.operating_point.beta:g}={self.operating_point.f_beta:.4f} | "
            f"TP={c.tp} FP={c.fp} TN={c.tn} FN={c.fn}"
        )