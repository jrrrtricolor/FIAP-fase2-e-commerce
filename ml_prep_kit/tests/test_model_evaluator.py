import unittest

import pandas as pd

from ml_prep_kit import ModelEvaluator


class TestModelEvaluator(unittest.TestCase):
    def test_evaluate_classification_returns_expected_metrics(self):
        evaluator = ModelEvaluator()
        y_true = pd.Series([0, 0, 1, 1])
        y_pred = pd.Series([0, 1, 1, 1])
        y_score = pd.Series([0.1, 0.6, 0.8, 0.9])

        metrics = evaluator.evaluate_classification(
            y_true=y_true,
            y_pred=y_pred,
            y_score=y_score,
        )

        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["precision"], 2 / 3)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertIn("roc_auc", metrics)


if __name__ == "__main__":
    unittest.main()
