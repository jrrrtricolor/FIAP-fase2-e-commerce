"""Registro reutilizável de experimentos com MLflow."""

import logging
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pytorch
import mlflow.sklearn


logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Centraliza operações comuns de tracking com MLflow.

    Use esta classe para configurar o experimento uma única vez e registrar
    parâmetros, métricas, artefatos e modelos de forma padronizada.

    Exemplo:
        tracker = ExperimentTracker(
            experiment_name="experimento-recomendacao",
            tracking_uri="sqlite:///mlflow.db",
        )

        run_id = tracker.log_training_run(
            run_name="logistic_regression",
            model=pipeline,
            parameters={"model_name": "logistic_regression"},
            metrics={"roc_auc": 0.82},
        )
    """

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str | None = None,
    ) -> None:
        """Configura o experimento usado pelos registros.

        Exemplo:
            tracker = ExperimentTracker(
                experiment_name="experimento-recomendacao",
                tracking_uri="sqlite:///mlflow.db",
            )
        """
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
        """Registra parâmetros, métricas, artefatos e modelo treinado.

        Exemplo:
            run_id = tracker.log_training_run(
                run_name="random_forest",
                model=pipeline,
                parameters={"n_estimators": 100},
                metrics={"f1": 0.74},
                artifacts=["model/report.html"],
            )
        """
        logger.info(
            "Iniciando registro do modelo Scikit-Learn no MLflow.",
            extra={
                "evento": "registro_sklearn_mlflow_iniciado",
                "nome_execucao": run_name,
                "nome_experimento": self.experiment_name,
                "quantidade_artefatos": len(artifacts or []),
            },
        )

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params(self._stringify_values(parameters))
            mlflow.log_metrics(metrics)

            for artifact in artifacts or []:
                mlflow.log_artifact(str(artifact))

            mlflow.sklearn.log_model(
                sk_model=model,
                name=model_name,
            )

            logger.info(
                "Modelo Scikit-Learn registrado no MLflow com sucesso.",
                extra={
                    "evento": "registro_sklearn_mlflow_concluido",
                    "nome_execucao": run_name,
                    "execucao_id": run.info.run_id,
                    "nome_experimento": self.experiment_name,
                },
            )

            return run.info.run_id

    def log_pytorch_training_run(
        self,
        run_name: str,
        model: Any,
        parameters: dict[str, Any],
        metrics: dict[str, float],
        artifacts: list[str | Path] | None = None,
        input_example: Any | None = None,
        model_name: str = "model",
    ) -> str:
        """Registra parâmetros, métricas, artefatos e modelo PyTorch.

        Exemplo:
            run_id = tracker.log_pytorch_training_run(
                run_name="tabular_binary_classifier",
                model=torch_model,
                parameters={"epochs": 10, "learning_rate": 0.001},
                metrics={"roc_auc": 0.88},
                input_example=X_valid_ready[:5],
            )
        """
        logger.info(
            "Iniciando registro do modelo PyTorch no MLflow.",
            extra={
                "evento": "registro_pytorch_mlflow_iniciado",
                "nome_execucao": run_name,
                "nome_experimento": self.experiment_name,
                "quantidade_artefatos": len(artifacts or []),
            },
        )

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params(self._stringify_values(parameters))
            mlflow.log_metrics(metrics)

            for artifact in artifacts or []:
                mlflow.log_artifact(str(artifact))

            mlflow.pytorch.log_model(
                pytorch_model=model,
                name=model_name,
                input_example=input_example,
                serialization_format="pt2",
            )

            logger.info(
                "Modelo PyTorch registrado no MLflow com sucesso.",
                extra={
                    "evento": "registro_pytorch_mlflow_concluido",
                    "nome_execucao": run_name,
                    "execucao_id": run.info.run_id,
                    "nome_experimento": self.experiment_name,
                },
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
