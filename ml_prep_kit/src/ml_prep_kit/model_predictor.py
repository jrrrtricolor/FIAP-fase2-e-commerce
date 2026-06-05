from typing import Any

import joblib
import mlflow
import mlflow.pytorch
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from ml_prep_kit.sklearn_torch_binary_classifier import (
    SklearnTorchBinaryClassifier,
)


class ModelPredictor:
    """
    Carrega um modelo salvo no MLFlow e seu pré-processador.
    Fornece métodos para preparar os dados de entrada e fazer previsões.
    """
    def __init__(self, tracking_uri: str) -> None:
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()

    def find_model(self, registered_model_name: str, model_alias: str) -> None:
        self.model_version = self.client.get_model_version_by_alias(
            name=registered_model_name,
            alias=model_alias,
        )

        self.artifact_uri = self.client.get_model_version_download_uri(
            name=registered_model_name,
            version=self.model_version.version,
        )

        self.artifacts = self.client.list_artifacts(
            run_id=self.model_version.run_id,
        )

    def load_preprocessor(
        self, 
        artifact_name: str = "preprocessor.joblib"
    ) -> None:
        artifact_path = None

        for artifact in self.artifacts:
            if artifact.path.endswith(artifact_name):
                artifact_path = artifact.path
                break

        if artifact_path is None:
            raise ValueError(f"Artefato não encontrado: {artifact_name}")

        preprocessor_path = self.client.download_artifacts(
            run_id=self.model_version.run_id,
            path=artifact_path,
        )
        self.preprocessor = joblib.load(preprocessor_path)

    def load_pytorch_model(self) -> None:
        self.model = SklearnTorchBinaryClassifier()
        self.model.model_module = mlflow.pytorch.load_model(self.artifact_uri)
        
    def get_categorical_column_values(self, column_name: str) -> list[str]:
        """
        Retorna os valores únicos encontrados durante o treinamento 
        para uma coluna categórica específica.
        """

        categorical_transformers = self.preprocessor['cat']
        categorical_store = categorical_transformers['categorical_store']

        feature_names = categorical_store.categories.get(column_name)

        if feature_names is None:
            raise ValueError(
                "Nenhum armazenamento categórico "
                f"encontrado para a coluna: {column_name}"
            )
    
        return feature_names

    def prepare_input(self, X) -> Any:
        return self.preprocessor.transform(X)
    
    def predict(self, X_prepared): 
        return self.model.predict(X_prepared)

    def predict_proba(self, X_prepared):
        return self.model.predict_proba(X_prepared)
