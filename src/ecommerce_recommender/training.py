"""Treina modelos de recomendação e registra resultados no MLflow."""

from __future__ import annotations

from pathlib import Path
import sqlite3

import pandas as pd
from sklearn.model_selection import train_test_split

from ml_prep_kit import (
    ExperimentTracker,
    FeaturePreprocessor,
    ModelEvaluator,
    ModelFactory,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_PATH = PROJECT_ROOT / "data" / "training_data.db"
MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
MLFLOW_EXPERIMENT_NAME = "tech-challenge-fase2-recommendation"
TABLE_NAME = "training_data"
RANDOM_SEED = 42
SAMPLE_SIZE = 50_000
POSITIVE_RATE = 0.2
TEST_SIZE = 0.25

METADATA_COLUMNS = ["user_id", "product_id"]
TARGET_COLUMN = "target"

BINARY_COLUMNS = [
    "candidate_from_history",
    "candidate_from_cooccurrence",
    "candidate_from_favorite_category",
    "candidate_from_similar_users",
    "candidate_was_previously_purchased",
    "candidate_is_new_product_for_user",
    "is_favorite_department",
    "is_favorite_aisle",
]

NUMERIC_COLUMNS = [
    "cooccurrence_score",
    "category_popularity_score",
    "similar_user_score",
    "purchase_count",
    "reorder_rate",
    "avg_cart_position",
    "first_order_number",
    "last_order_number",
    "orders_since_last_purchase",
    "purchase_frequency",
    "user_total_orders",
    "user_total_items",
    "user_unique_products",
    "user_reorder_rate",
    "user_avg_cart_position",
    "user_avg_days_between_orders",
    "user_avg_order_hour",
    "user_avg_basket_size",
    "product_total_orders",
    "product_unique_users",
    "product_reorder_rate",
    "product_avg_cart_position",
    "user_department_purchase_count",
    "user_department_purchase_rate",
    "user_aisle_purchase_count",
    "user_aisle_purchase_rate",
]

CATEGORICAL_COLUMNS = ["aisle", "department"]
FEATURE_COLUMNS = BINARY_COLUMNS + NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
SELECTED_COLUMNS = METADATA_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN]

MODEL_CONFIGS = [
    {
        "model_name": "dummy",
        "problem_type": "classification",
        "parameters": {"strategy": "prior"},
    },
    {
        "model_name": "logistic_regression",
        "problem_type": "classification",
        "parameters": {
            "class_weight": "balanced",
            "max_iter": 500,
            "solver": "lbfgs",
        },
    },
    {
        "model_name": "hist_gradient_boosting",
        "problem_type": "classification",
        "parameters": {
            "learning_rate": 0.08,
            "max_iter": 80,
            "max_leaf_nodes": 31,
        },
    },
]


def main() -> None:
    """Executa o treino dos modelos configurados."""
    data = load_training_sample(
        database_path=DATABASE_PATH,
        sample_size=SAMPLE_SIZE,
        positive_rate=POSITIVE_RATE,
        random_state=RANDOM_SEED,
    )

    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    evaluator = ModelEvaluator()
    tracker = ExperimentTracker(
        experiment_name=MLFLOW_EXPERIMENT_NAME,
        tracking_uri=MLFLOW_TRACKING_URI,
    )

    factory = ModelFactory(random_state=RANDOM_SEED)
    results = []

    for config in MODEL_CONFIGS:
        result = train_and_log_model(
            config=config,
            factory=factory,
            evaluator=evaluator,
            tracker=tracker,
            X_train=X_train,
            X_valid=X_valid,
            y_train=y_train,
            y_valid=y_valid,
            sample_size=len(data),
            target_rate=float(y.mean()),
            random_seed=RANDOM_SEED,
        )
        results.append(result)

    report = pd.DataFrame(results).sort_values(
        "roc_auc",
        ascending=False,
    )

    print("\nResultados registrados no MLflow:")
    print(report.to_string(index=False))


def load_training_sample(
    database_path: Path,
    sample_size: int,
    positive_rate: float,
    random_state: int,
) -> pd.DataFrame:
    """Carrega uma amostra estratificada da tabela de treino."""
    positive_size = int(sample_size * positive_rate)
    negative_size = sample_size - positive_size
    columns = ", ".join(SELECTED_COLUMNS)

    query = f"""
    SELECT {columns}
    FROM {TABLE_NAME}
    WHERE target = 1
    ORDER BY RANDOM()
    LIMIT {positive_size}
    """

    negative_query = f"""
    SELECT {columns}
    FROM {TABLE_NAME}
    WHERE target = 0
    ORDER BY RANDOM()
    LIMIT {negative_size}
    """

    with sqlite3.connect(database_path) as conn:
        positives = pd.read_sql_query(query, conn)
        negatives = pd.read_sql_query(negative_query, conn)

    data = pd.concat([positives, negatives], ignore_index=True)
    return data.sample(frac=1, random_state=random_state).reset_index(
        drop=True,
    )


def train_and_log_model(
    config: dict,
    factory: ModelFactory,
    evaluator: ModelEvaluator,
    tracker: ExperimentTracker,
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    y_train: pd.Series,
    y_valid: pd.Series,
    sample_size: int,
    target_rate: float,
    random_seed: int,
) -> dict[str, float | str]:
    """Treina, avalia e registra um modelo no MLflow."""
    model_name = config["model_name"]
    print(f"\nTreinando modelo: {model_name}")
    preprocessor = FeaturePreprocessor(
        numeric_columns=NUMERIC_COLUMNS,
        categorical_columns=CATEGORICAL_COLUMNS,
        binary_columns=BINARY_COLUMNS,
    ).create_pipeline()

    pipeline = factory.create_pipeline(
        preprocessor=preprocessor,
        problem_type=config["problem_type"],
        model_name=model_name,
        parameters=config["parameters"],
    )

    pipeline.fit(X_train, y_train)
    y_pred = pd.Series(pipeline.predict(X_valid), index=y_valid.index)
    y_score = get_positive_score(pipeline, X_valid, y_pred)

    metrics = evaluator.evaluate_classification(y_valid, y_pred, y_score)

    parameters = {
        "model_name": model_name,
        "problem_type": config["problem_type"],
        "sample_size": sample_size,
        "target_rate": target_rate,
        "random_seed": random_seed,
        "feature_count": len(FEATURE_COLUMNS),
        **config["parameters"],
    }

    run_id = tracker.log_training_run(
        run_name=model_name,
        model=pipeline,
        parameters=parameters,
        metrics=metrics,
    )

    return {
        "model_name": model_name,
        "run_id": run_id,
        **metrics,
    }


def get_positive_score(
    pipeline,
    X_valid: pd.DataFrame,
    y_pred: pd.Series,
) -> pd.Series:
    """Obtém score da classe positiva quando o modelo oferece essa saída."""
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(X_valid)
        return pd.Series(probabilities[:, 1], index=y_pred.index)

    if hasattr(pipeline, "decision_function"):
        scores = pipeline.decision_function(X_valid)
        return pd.Series(scores, index=y_pred.index)

    return y_pred.astype(float)


if __name__ == "__main__":
    main()
