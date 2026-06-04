"""Prepara a base de treino do recomendador de forma simples."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ecommerce_recommender.training import (
    DATABASE_PATH,
    RANDOM_SEED,
    SELECTED_COLUMNS,
    TABLE_NAME,
)
from ml_prep_kit import SQLiteDataFrameStore, StructuredLoggingConfigurator

LOGGER = logging.getLogger("ecommerce_recommender.prepare_data")

# Definir o caminho raiz do projeto.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Definir o diretório dos dados brutos.
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "instacart"

# Definir o caminho do arquivo de preços estimados.
MARKET_PRICES_PATH = PROJECT_ROOT / "data" / "reference" / (
    "estimated_market_prices.csv"
)

# Definir a quantidade máxima de usuários da amostra.
MAX_USERS = 5_000

# Definir a quantidade de produtos negativos por produto positivo.
NEGATIVE_RATIO = 4


def run_preparation() -> None:
    """Gera a base SQLite usada pelos modelos.

    Exemplo:
        run_preparation()
    """
    StructuredLoggingConfigurator.configure()
    LOGGER.info("Iniciando preparação dos dados.")

    # Carregar as tabelas principais do conjunto Instacart.
    products = pd.read_csv(RAW_DATA_DIR / "products.csv")
    aisles = pd.read_csv(RAW_DATA_DIR / "aisles.csv")
    departments = pd.read_csv(RAW_DATA_DIR / "departments.csv")
    orders = pd.read_csv(RAW_DATA_DIR / "orders.csv")
    order_products = pd.read_csv(RAW_DATA_DIR / "order_products__train.csv")

    # Juntar produtos com aisle e department.
    catalog = products.merge(aisles, on="aisle_id").merge(
        departments,
        on="department_id",
    )

    # Filtrar apenas pedidos marcados como treino.
    train_orders = orders.loc[orders["eval_set"].eq("train")].copy()

    # Selecionar uma amostra reproduzível de usuários.
    sampled_users = train_orders["user_id"].drop_duplicates().sample(
        n=min(MAX_USERS, train_orders["user_id"].nunique()),
        random_state=RANDOM_SEED,
    )
    train_orders = train_orders.loc[
        train_orders["user_id"].isin(sampled_users)
    ]

    # Criar exemplos positivos com produtos comprados pelo usuário.
    positive_pairs = order_products.merge(
        train_orders[["order_id", "user_id"]],
        on="order_id",
        how="inner",
    )[["user_id", "product_id"]]
    positive_pairs = positive_pairs.drop_duplicates().assign(target=1)

    # Criar exemplos negativos com produtos não comprados pelo usuário.
    negative_pairs = create_negative_pairs(positive_pairs, catalog)
    training_data = pd.concat(
        [positive_pairs, negative_pairs],
        ignore_index=True,
    )

    # Adicionar as features esperadas pelos modelos.
    training_data = add_simple_features(training_data, catalog)

    # Salvar a base final no SQLite.
    SQLiteDataFrameStore(DATABASE_PATH).save_dataframe(
        training_data,
        TABLE_NAME,
    )
    save_market_prices(catalog)

    LOGGER.info(
        "Dados preparados com sucesso.",
        extra={
            "evento": "preparacao_base_treino_concluida",
            "linhas": len(training_data),
            "usuarios": training_data["user_id"].nunique(),
        },
    )


def create_negative_pairs(
    positive_pairs: pd.DataFrame,
    catalog: pd.DataFrame,
) -> pd.DataFrame:
    """Cria exemplos negativos simples para cada usuário.

    Exemplo:
        negative_pairs = create_negative_pairs(positive_pairs, catalog)
    """
    rng = np.random.default_rng(RANDOM_SEED)
    product_ids = catalog["product_id"].to_numpy()
    negative_rows = []

    # Agrupar produtos positivos por usuário.
    for user_id, user_products in positive_pairs.groupby("user_id"):
        purchased_products = set(user_products["product_id"])

        # Definir a quantidade de negativos para o usuário.
        negative_size = len(purchased_products) * NEGATIVE_RATIO

        # Sortear produtos candidatos para exemplos negativos.
        sampled_products = rng.choice(product_ids, size=negative_size * 2)
        user_negative_rows = []

        for product_id in sampled_products:
            # Adicionar apenas produtos não comprados pelo usuário.
            if product_id not in purchased_products:
                user_negative_rows.append(
                    {
                        "user_id": user_id,
                        "product_id": int(product_id),
                        "target": 0,
                    }
                )

            # Parar quando a quantidade esperada for atingida.
            if len(user_negative_rows) >= negative_size:
                break

        negative_rows.extend(user_negative_rows)

    return pd.DataFrame(negative_rows).drop_duplicates()


def add_simple_features(
    training_data: pd.DataFrame,
    catalog: pd.DataFrame,
) -> pd.DataFrame:
    """Adiciona colunas esperadas pelos modelos de treino.

    Exemplo:
        training_data = add_simple_features(training_data, catalog)
    """
    # Juntar os pares usuário-produto com informações do catálogo.
    training_data = training_data.merge(catalog, on="product_id", how="left")

    # Criar features binárias simples.
    training_data["candidate_from_history"] = training_data["target"]
    training_data["candidate_was_previously_purchased"] = training_data[
        "target"
    ]
    training_data["candidate_is_new_product_for_user"] = (
        1 - training_data["target"]
    )
    training_data["candidate_from_favorite_category"] = training_data["target"]
    training_data["is_favorite_department"] = training_data["target"]
    training_data["is_favorite_aisle"] = training_data["target"]
    training_data["candidate_from_cooccurrence"] = 0
    training_data["candidate_from_similar_users"] = 0

    # Calcular a popularidade simples por produto.
    product_popularity = training_data.groupby("product_id")[
        "target"
    ].transform("mean")

    # Calcular a atividade simples por usuário.
    user_activity = training_data.groupby("user_id")["product_id"].transform(
        "count"
    )

    # Preencher as features numéricas esperadas pelo treino.
    default_values = {
        "cooccurrence_score": 0.0,
        "category_popularity_score": product_popularity,
        "similar_user_score": 0.0,
        "purchase_count": training_data["target"],
        "reorder_rate": training_data["target"],
        "avg_cart_position": 0.0,
        "first_order_number": 0.0,
        "last_order_number": 0.0,
        "orders_since_last_purchase": 0.0,
        "purchase_frequency": training_data["target"],
        "user_total_orders": user_activity,
        "user_total_items": user_activity,
        "user_unique_products": user_activity,
        "user_reorder_rate": training_data["target"],
        "user_avg_cart_position": 0.0,
        "user_avg_days_between_orders": 0.0,
        "user_avg_order_hour": 0.0,
        "user_avg_basket_size": user_activity,
        "product_total_orders": training_data.groupby("product_id")[
            "target"
        ].transform("count"),
        "product_unique_users": training_data.groupby("product_id")[
            "user_id"
        ].transform("nunique"),
        "product_reorder_rate": product_popularity,
        "product_avg_cart_position": 0.0,
        "user_department_purchase_count": training_data["target"],
        "user_department_purchase_rate": training_data["target"],
        "user_aisle_purchase_count": training_data["target"],
        "user_aisle_purchase_rate": training_data["target"],
    }

    # Criar cada coluna numérica no DataFrame final.
    for column, value in default_values.items():
        training_data[column] = value

    # Manter somente as colunas utilizadas pelos modelos.
    return training_data[SELECTED_COLUMNS]


def save_market_prices(catalog: pd.DataFrame) -> None:
    """Salva preços estimados no SQLite, quando o arquivo existir.

    Exemplo:
        save_market_prices(catalog)
    """
    if not MARKET_PRICES_PATH.exists():
        return

    # Salvar os preços estimados junto com o catálogo.
    prices = pd.read_csv(MARKET_PRICES_PATH)
    prices = prices.merge(catalog, on="product_id", how="left")
    SQLiteDataFrameStore(DATABASE_PATH).save_dataframe(
        prices,
        "estimated_market_prices",
    )


def main() -> None:
    """Executa a preparação pelo comando direto do módulo.

    Exemplo:
        poetry run python -m ecommerce_recommender.prepare_data
    """
    run_preparation()


if __name__ == "__main__":
    main()
