from dataclasses import dataclass
from sklearn.pipeline import Pipeline

@dataclass(frozen=True)
class CVMetrics:
    ap_mean: float
    ap_std: float
    auc_mean: float
    auc_std: float
    recall_mean: float
    precision_mean: float
    brier_mean: float

    def scalar_metrics(self) -> dict[str, float]:
        return {
            "cv_ap_mean": self.ap_mean,
            "cv_ap_std": self.ap_std,
            "cv_auc_mean": self.auc_mean,
            "cv_auc_std": self.auc_std,
            "cv_recall_mean": self.recall_mean,
            "cv_precision_mean": self.precision_mean,
            "cv_brier_mean": self.brier_mean,
        }



@dataclass(frozen=True)
class TrainingResult:
    model: Pipeline
    cv_metrics: CVMetrics
    best_params: dict

    def scalar_metrics(self) -> dict[str, float]:
        return self.cv_metrics.scalar_metrics()

    def summary(self) -> str:
        m = self.cv_metrics
        return (
            f"AP={m.ap_mean:.4f}±{m.ap_std:.4f} "
            f"AUC={m.auc_mean:.4f}±{m.auc_std:.4f} "
            f"Precision={m.precision_mean:.4f} Recall={m.recall_mean:.4f} Brier-Score={m.brier_mean:.4f}"
        )
