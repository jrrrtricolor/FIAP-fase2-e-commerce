import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ml_prep_kit import ModelPredictor


class ModelPredictorTest(unittest.TestCase):
    @patch("ml_prep_kit.model_predictor.MlflowClient")
    @patch("ml_prep_kit.model_predictor.mlflow.set_tracking_uri")
    def test_init_sets_tracking_uri_and_client(
        self,
        set_tracking_uri_mock,
        mlflow_client_mock,
    ):
        predictor = ModelPredictor(tracking_uri="http://mlflow-server:5000")

        set_tracking_uri_mock.assert_called_once_with("http://mlflow-server:5000")
        mlflow_client_mock.assert_called_once_with()
        self.assertIs(predictor.client, mlflow_client_mock.return_value)

    @patch("ml_prep_kit.model_predictor.MlflowClient")
    @patch("ml_prep_kit.model_predictor.mlflow.set_tracking_uri")
    def test_find_model_sets_model_metadata(
        self,
        _set_tracking_uri_mock,
        mlflow_client_mock,
    ):
        client = mlflow_client_mock.return_value
        model_version = SimpleNamespace(version="7", run_id="run-abc")
        artifacts = [SimpleNamespace(path="preprocessor.joblib")]
        client.get_model_version_by_alias.return_value = model_version
        client.get_model_version_download_uri.return_value = "models:/uri"
        client.list_artifacts.return_value = artifacts

        predictor = ModelPredictor(tracking_uri="unused")
        predictor.find_model(
            registered_model_name="ecommerce_recommender_pytorch_mlp",
            model_alias="production",
        )

        client.get_model_version_by_alias.assert_called_once_with(
            name="ecommerce_recommender_pytorch_mlp",
            alias="production",
        )
        client.get_model_version_download_uri.assert_called_once_with(
            name="ecommerce_recommender_pytorch_mlp",
            version="7",
        )
        client.list_artifacts.assert_called_once_with(run_id="run-abc")
        self.assertEqual(predictor.model_version, model_version)
        self.assertEqual(predictor.artifact_uri, "models:/uri")
        self.assertEqual(predictor.artifacts, artifacts)

    @patch("ml_prep_kit.model_predictor.joblib.load")
    @patch("ml_prep_kit.model_predictor.MlflowClient")
    @patch("ml_prep_kit.model_predictor.mlflow.set_tracking_uri")
    def test_load_preprocessor_downloads_and_loads_artifact(
        self,
        _set_tracking_uri_mock,
        mlflow_client_mock,
        joblib_load_mock,
    ):
        client = mlflow_client_mock.return_value
        client.download_artifacts.return_value = "/tmp/preprocessor.joblib"
        loaded_preprocessor = Mock()
        joblib_load_mock.return_value = loaded_preprocessor

        predictor = ModelPredictor(tracking_uri="unused")
        predictor.model_version = SimpleNamespace(run_id="run-xyz")
        predictor.artifacts = [
            SimpleNamespace(path="folder/another.bin"),
            SimpleNamespace(path="folder/preprocessor.joblib"),
        ]

        predictor.load_preprocessor()

        client.download_artifacts.assert_called_once_with(
            run_id="run-xyz",
            path="folder/preprocessor.joblib",
        )
        joblib_load_mock.assert_called_once_with("/tmp/preprocessor.joblib")
        self.assertIs(predictor.preprocessor, loaded_preprocessor)

    @patch("ml_prep_kit.model_predictor.MlflowClient")
    @patch("ml_prep_kit.model_predictor.mlflow.set_tracking_uri")
    def test_load_preprocessor_raises_when_artifact_not_found(
        self,
        _set_tracking_uri_mock,
        _mlflow_client_mock,
    ):
        predictor = ModelPredictor(tracking_uri="unused")
        predictor.model_version = SimpleNamespace(run_id="run-xyz")
        predictor.artifacts = [SimpleNamespace(path="folder/model.pkl")]

        with self.assertRaisesRegex(ValueError, "Artefato não encontrado"):
            predictor.load_preprocessor(artifact_name="missing.joblib")

    @patch("ml_prep_kit.model_predictor.mlflow.pytorch.load_model")
    @patch("ml_prep_kit.model_predictor.SklearnTorchBinaryClassifier")
    @patch("ml_prep_kit.model_predictor.MlflowClient")
    @patch("ml_prep_kit.model_predictor.mlflow.set_tracking_uri")
    def test_load_pytorch_model_sets_wrapped_model(
        self,
        _set_tracking_uri_mock,
        _mlflow_client_mock,
        sklearn_torch_classifier_mock,
        load_model_mock,
    ):
        wrapped_classifier = Mock()
        sklearn_torch_classifier_mock.return_value = wrapped_classifier
        load_model_mock.return_value = "torch-model"

        predictor = ModelPredictor(tracking_uri="unused")
        predictor.artifact_uri = "models:/uri"

        predictor.load_pytorch_model()

        sklearn_torch_classifier_mock.assert_called_once_with()
        load_model_mock.assert_called_once_with("models:/uri")
        self.assertIs(predictor.model, wrapped_classifier)
        self.assertEqual(predictor.model.model_module, "torch-model")

    @patch("ml_prep_kit.model_predictor.MlflowClient")
    @patch("ml_prep_kit.model_predictor.mlflow.set_tracking_uri")
    def test_get_categorical_column_values_returns_categories(
        self,
        _set_tracking_uri_mock,
        _mlflow_client_mock,
    ):
        predictor = ModelPredictor(tracking_uri="unused")
        predictor.preprocessor = {
            "cat": {
                "categorical_store": SimpleNamespace(
                    categories={
                        "aisle": ["fresh vegetables", "soy milk"],
                        "department": ["produce", "dairy eggs"],
                    }
                )
            }
        }

        result = predictor.get_categorical_column_values("aisle")

        self.assertEqual(result, ["fresh vegetables", "soy milk"])

    @patch("ml_prep_kit.model_predictor.MlflowClient")
    @patch("ml_prep_kit.model_predictor.mlflow.set_tracking_uri")
    def test_get_categorical_column_values_raises_for_unknown_column(
        self,
        _set_tracking_uri_mock,
        _mlflow_client_mock,
    ):
        predictor = ModelPredictor(tracking_uri="unused")
        predictor.preprocessor = {
            "cat": {
                "categorical_store": SimpleNamespace(
                    categories={"aisle": ["fresh vegetables"]}
                )
            }
        }

        with self.assertRaisesRegex(
            ValueError,
            "Nenhum armazenamento categórico encontrado",
        ):
            predictor.get_categorical_column_values("department")

    @patch("ml_prep_kit.model_predictor.MlflowClient")
    @patch("ml_prep_kit.model_predictor.mlflow.set_tracking_uri")
    def test_prepare_input_delegates_to_preprocessor(
        self,
        _set_tracking_uri_mock,
        _mlflow_client_mock,
    ):
        predictor = ModelPredictor(tracking_uri="unused")
        preprocessor = Mock()
        preprocessor.transform.return_value = "prepared-input"
        predictor.preprocessor = preprocessor

        result = predictor.prepare_input("raw-input")

        preprocessor.transform.assert_called_once_with("raw-input")
        self.assertEqual(result, "prepared-input")

    @patch("ml_prep_kit.model_predictor.MlflowClient")
    @patch("ml_prep_kit.model_predictor.mlflow.set_tracking_uri")
    def test_predict_delegates_to_loaded_model(
        self,
        _set_tracking_uri_mock,
        _mlflow_client_mock,
    ):
        predictor = ModelPredictor(tracking_uri="unused")
        model = Mock()
        model.predict.return_value = [1, 0]
        predictor.model = model

        result = predictor.predict("prepared-input")

        model.predict.assert_called_once_with("prepared-input")
        self.assertEqual(result, [1, 0])

    @patch("ml_prep_kit.model_predictor.MlflowClient")
    @patch("ml_prep_kit.model_predictor.mlflow.set_tracking_uri")
    def test_predict_proba_delegates_to_loaded_model(
        self,
        _set_tracking_uri_mock,
        _mlflow_client_mock,
    ):
        predictor = ModelPredictor(tracking_uri="unused")
        model = Mock()
        model.predict_proba.return_value = [[0.1, 0.9]]
        predictor.model = model

        result = predictor.predict_proba("prepared-input")

        model.predict_proba.assert_called_once_with("prepared-input")
        self.assertEqual(result, [[0.1, 0.9]])


if __name__ == "__main__":
    unittest.main()