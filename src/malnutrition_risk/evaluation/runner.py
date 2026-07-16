from dataclasses import dataclass
from malnutrition_risk.core.artifacts import ArtifactWriter
from malnutrition_risk.core.scoring import positive_class_proba
from .metrics import compute_metrics
from .explain import compute_shap, feature_importance_from_shap
from . import plots
import pandas as pd
import logging

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class EvaluationConfig:
    target_col: str
    beta: float = 2.0
    explain: bool = True
    shap_n_features: int = 20
    strict: bool = False


def run_evaluation(model, X, y, writer: ArtifactWriter, cfg: EvaluationConfig):
    # compute and log metrics
    y_proba = positive_class_proba(model, X)
    results = compute_metrics(y, y_proba, beta=cfg.beta)
    writer.write_metrics(results.scalar_metrics())

    # log scored
    scored = pd.DataFrame(
        data={
            'y_true': list(y),
            'y_proba': y_proba,
            'y_pred': (y_proba >= results.operating_point.threshold).astype(int)
        },
        index=X.index,
    )
    writer.write_parquet(scored, 'scored.parquet')

    # save plots
    writer.write_figure(
        fig=plots.pr_curve_figure(results.curve, results.operating_point),
        name="precision_recall_curve.png"
    )

    if cfg.explain:
        try:
            explanation = compute_shap(model, X)
            importance = feature_importance_from_shap(explanation)
            writer.write_table(importance, "feature_importance_shap.csv")
            writer.write_figure(plots.shap_summary_figure(explanation), "shap_summary_plot.png")
            writer.write_figure(plots.shap_waterfall_figure(explanation), "shap_waterfall_plot.png")
            writer.write_figure(plots.feature_importance_figure(importance, cfg.shap_n_features),
                                "low_importance_features.png")

        except Exception:
            if cfg.strict:
                raise
            logger.exception("SHAP step failed; continuing without explanation artifacts.")

    return results





