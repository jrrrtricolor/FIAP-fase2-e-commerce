"""Métricas reutilizáveis para avaliação de modelos."""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class ModelEvaluator:
    """Calcula métricas simples para modelos de classificação."""

    def evaluate_classification(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        y_score: pd.Series | None = None,
    ) -> dict[str, float]:
        """Retorna métricas de classificação binária."""
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(
                y_true,
                y_pred,
                zero_division=0,
            ),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }

        if y_score is not None and y_true.nunique() > 1:
            metrics["roc_auc"] = roc_auc_score(y_true, y_score)

        return {name: float(value) for name, value in metrics.items()}
