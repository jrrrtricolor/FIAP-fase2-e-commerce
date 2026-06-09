import importlib
import sys
import unittest
from unittest.mock import Mock, patch

import pandas as pd
from fastapi.testclient import TestClient

with patch("ml_prep_kit.ModelPredictor"):
    if "ecommerce_recommender.api" in sys.modules:
        del sys.modules["ecommerce_recommender.api"]
    api_module = importlib.import_module("ecommerce_recommender.api")


def _build_request_payload() -> dict[str, object]:
    return {
        "user_id": 1,
        "product_id": 10,
        "candidate_from_history": True,
        "candidate_from_cooccurrence": False,
        "candidate_from_favorite_category": True,
        "candidate_from_similar_users": False,
        "candidate_was_previously_purchased": True,
        "candidate_is_new_product_for_user": False,
        "is_favorite_department": True,
        "is_favorite_aisle": False,
        "cooccurrence_score": 0.5,
        "category_popularity_score": 0.4,
        "similar_user_score": 0.3,
        "purchase_count": 4,
        "reorder_rate": 1,
        "avg_cart_position": 2.5,
        "first_order_number": 1.0,
        "last_order_number": 10.0,
        "orders_since_last_purchase": 2.0,
        "purchase_frequency": 3,
        "user_total_orders": 20,
        "user_total_items": 120,
        "user_unique_products": 35,
        "user_reorder_rate": 1,
        "user_avg_cart_position": 3.1,
        "user_avg_days_between_orders": 7.0,
        "user_avg_order_hour": 14.0,
        "user_avg_basket_size": 8,
        "product_total_orders": 500,
        "product_unique_users": 200,
        "product_reorder_rate": 0.6,
        "product_avg_cart_position": 4.2,
        "user_department_purchase_count": 15,
        "user_department_purchase_rate": 1,
        "user_aisle_purchase_count": 9,
        "user_aisle_purchase_rate": 1,
        "aisle": "fresh vegetables",
        "department": "produce",
    }


class TestApiEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api_module.app)
        self.original_predictor = api_module.predictor

    def tearDown(self):
        api_module.predictor = self.original_predictor

    def test_not_loaded_model_middleware_returns_503(self):
        api_module.predictor = None

        response = self.client.get("/health")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["status"], "erro")
        self.assertIn("mensagem", response.json())

    def test_health_endpoint_returns_success(self):
        api_module.predictor = Mock()

        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "sucesso"})

    def test_aisles_endpoint_returns_predictor_values(self):
        predictor = Mock()
        predictor.get_categorical_column_values.return_value = [
            "fresh vegetables",
            "soy milk",
        ]
        api_module.predictor = predictor

        response = self.client.get("/aisles")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), ["fresh vegetables", "soy milk"])
        predictor.get_categorical_column_values.assert_called_once_with("aisle")

    def test_departments_endpoint_returns_predictor_values(self):
        predictor = Mock()
        predictor.get_categorical_column_values.return_value = [
            "produce",
            "dairy eggs",
        ]
        api_module.predictor = predictor

        response = self.client.get("/departments")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), ["produce", "dairy eggs"])
        predictor.get_categorical_column_values.assert_called_once_with(
            "department",
        )

    def test_recommendations_returns_400_for_empty_request(self):
        api_module.predictor = Mock()

        response = self.client.post("/recomendacoes", json=[])

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["status"], "erro")
        self.assertIn("mensagem", response.json())

    @patch("ecommerce_recommender.api.recommend_top_n")
    def test_recommendations_returns_top_n_payload(self, recommend_top_n_mock):
        api_module.predictor = Mock()

        recommendations = pd.DataFrame(
            [
                {
                    "user_id": 1,
                    "product_id": 10,
                    "recommendation_score": 0.93,
                }
            ]
        )
        recommend_top_n_mock.return_value = (
            recommendations, pd.Series([0.93])
        )

        payload = [_build_request_payload()]

        response = self.client.post("/recomendacoes", json=payload)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "recommendations": [
                    {
                        "user_id": 1,
                        "product_id": 10,
                        "recommendation_score": 0.93,
                    }
                ]
            },
        )
        recommend_top_n_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()