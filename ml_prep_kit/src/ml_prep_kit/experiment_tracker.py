"""Registro reutilizável de experimentos com MLflow."""

from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn


class ExperimentTracker:
    """Centraliza operações comuns de tracking com MLflow."""

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str | None = None,
    ) -> None:
        """Configura o experimento usado pelos registros."""
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        mlflow.set_experiment(experiment_name)

    def log_training_run(
        self,
        run_name: str,
        model: Any,
        parameters: dict[str, Any],
        metrics: dict[str, float],
        artifacts: list[str | Path] | None = None,
        model_name: str = "model",
    ) -> str:
        """Registra parâmetros, métricas, artefatos e modelo treinado."""
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params(self._stringify_values(parameters))
            mlflow.log_metrics(metrics)

            for artifact in artifacts or []:
                mlflow.log_artifact(str(artifact))

            mlflow.sklearn.log_model(
                sk_model=model,
                name=model_name,
            )

            return run.info.run_id

    def _stringify_values(self, values: dict[str, Any]) -> dict[str, Any]:
        """Converte valores complexos para texto aceito pelo MLflow."""
        formatted = {}

        for key, value in values.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                formatted[key] = value
            else:
                formatted[key] = str(value)

        return formatted
