import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ml_prep_kit import ExperimentTracker


class ExperimentTrackerTest(unittest.TestCase):
    @patch("ml_prep_kit.experiment_tracker.mlflow.set_experiment")
    @patch("ml_prep_kit.experiment_tracker.mlflow.set_tracking_uri")
    def test_init_sets_tracking_uri_and_experiment(
        self,
        set_tracking_uri_mock,
        set_experiment_mock,
    ):
        ExperimentTracker(
            experiment_name="experimento-teste",
            tracking_uri="sqlite:///mlflow-test.db",
        )

        set_tracking_uri_mock.assert_called_once_with(
            "sqlite:///mlflow-test.db"
        )
        set_experiment_mock.assert_called_once_with("experimento-teste")

    @patch("ml_prep_kit.experiment_tracker.mlflow.sklearn.log_model")
    @patch("ml_prep_kit.experiment_tracker.mlflow.log_artifact")
    @patch("ml_prep_kit.experiment_tracker.mlflow.log_metrics")
    @patch("ml_prep_kit.experiment_tracker.mlflow.log_params")
    @patch("ml_prep_kit.experiment_tracker.mlflow.start_run")
    @patch("ml_prep_kit.experiment_tracker.mlflow.set_experiment")
    def test_log_training_run_logs_run_data(
        self,
        _set_experiment_mock,
        start_run_mock,
        log_params_mock,
        log_metrics_mock,
        log_artifact_mock,
        log_model_mock,
    ):
        run = Mock()
        run.info = SimpleNamespace(run_id="run-123")
        start_run_mock.return_value.__enter__.return_value = run

        tracker = ExperimentTracker(experiment_name="experimento-teste")
        run_id = tracker.log_training_run(
            run_name="baseline",
            model=Mock(),
            parameters={"modelo": "baseline", "lista": [1, 2]},
            metrics={"roc_auc": 0.95},
            artifacts=["model/report.html"],
            registered_model_name="modelo-registrado",
        )

        self.assertEqual(run_id, "run-123")
        start_run_mock.assert_called_once_with(run_name="baseline")
        log_params_mock.assert_called_once_with(
            {"modelo": "baseline", "lista": "[1, 2]"}
        )
        log_metrics_mock.assert_called_once_with({"roc_auc": 0.95})
        log_artifact_mock.assert_called_once_with("model/report.html")
        log_model_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
