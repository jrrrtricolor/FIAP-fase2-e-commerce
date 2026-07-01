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
    prior_order_products = pd.read_csv(
        RAW_DATA_DIR / "order_products__prior.csv",
    )

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
    prior_orders = orders.loc[
        orders["eval_set"].eq("prior")
        & orders["user_id"].isin(sampled_users)
    ].copy()

    # Carregar o histórico anterior dos usuários da amostra.
    user_history = prior_order_products.merge(
        prior_orders[
            [
                "order_id",
                "user_id",
                "order_number",
                "order_dow",
                "order_hour_of_day",
                "days_since_prior_order",
            ]
        ],
        on="order_id",
        how="inner",
    )
    user_history = user_history.merge(catalog, on="product_id", how="left")

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
    training_data = add_simple_features(training_data, catalog, user_history)

    # Salvar a base final no SQLite.
    SQLiteDataFrameStore(DATABASE_PATH).save_dataframe(
        training_data,
        TABLE_NAME,
    )
    save_market_prices()

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
    user_history: pd.DataFrame,
) -> pd.DataFrame:
    """Adiciona colunas esperadas pelos modelos de treino.

    Exemplo:
        training_data = add_simple_features(
            training_data,
            catalog,
            user_history,
        )
    """
    # Juntar os pares usuário-produto com informações do catálogo.
    training_data = training_data.merge(catalog, on="product_id", how="left")

    # Calcular estatísticas usando somente o histórico anterior do usuário.
    product_history = build_product_history_features(user_history)
    user_history_features = build_user_history_features(user_history)
    catalog_history_features = build_catalog_history_features(user_history)
    category_history = build_category_history_features(user_history)

    training_data = training_data.merge(
        product_history,
        on=["user_id", "product_id"],
        how="left",
    )
    training_data = training_data.merge(
        user_history_features,
        on="user_id",
        how="left",
    )
    training_data = training_data.merge(
        catalog_history_features,
        on="product_id",
        how="left",
    )
    training_data = training_data.merge(
        category_history,
        on=["user_id", "department", "aisle"],
        how="left",
    )

    # Criar features binárias sem consultar a resposta que será prevista.
    training_data["candidate_from_history"] = (
        training_data["purchase_count"].fillna(0).gt(0).astype(int)
    )
    training_data["candidate_was_previously_purchased"] = training_data[
        "candidate_from_history"
    ]
    training_data["candidate_is_new_product_for_user"] = (
        1 - training_data["candidate_from_history"]
    )
    training_data["candidate_from_favorite_category"] = (
        training_data["user_department_purchase_rate"].fillna(0).ge(0.2)
        | training_data["user_aisle_purchase_rate"].fillna(0).ge(0.2)
    ).astype(int)
    training_data["is_favorite_department"] = (
        training_data["user_department_purchase_rate"].fillna(0).ge(0.2)
    ).astype(int)
    training_data["is_favorite_aisle"] = (
        training_data["user_aisle_purchase_rate"].fillna(0).ge(0.2)
    ).astype(int)
    training_data["candidate_from_cooccurrence"] = (
        training_data["is_favorite_aisle"]
    )
    training_data["candidate_from_similar_users"] = 0

    # Preencher as features numéricas esperadas pelo treino.
    default_values = {
        "cooccurrence_score": 0.0,
        "category_popularity_score": training_data[
            "product_reorder_rate"
        ].fillna(0.0),
        "similar_user_score": 0.0,
        "purchase_count": training_data["purchase_count"].fillna(0),
        "reorder_rate": training_data["reorder_rate"].fillna(0.0),
        "avg_cart_position": 0.0,
        "first_order_number": training_data["first_order_number"].fillna(0),
        "last_order_number": training_data["last_order_number"].fillna(0),
        "orders_since_last_purchase": training_data[
            "orders_since_last_purchase"
        ].fillna(0),
        "purchase_frequency": training_data["purchase_frequency"].fillna(0),
        "user_total_orders": training_data["user_total_orders"].fillna(0),
        "user_total_items": training_data["user_total_items"].fillna(0),
        "user_unique_products": training_data["user_unique_products"].fillna(
            0,
        ),
        "user_reorder_rate": training_data["user_reorder_rate"].fillna(0.0),
        "user_avg_cart_position": training_data[
            "user_avg_cart_position"
        ].fillna(0.0),
        "user_avg_days_between_orders": training_data[
            "user_avg_days_between_orders"
        ].fillna(0.0),
        "user_avg_order_hour": training_data["user_avg_order_hour"].fillna(
            0.0,
        ),
        "user_avg_basket_size": training_data["user_avg_basket_size"].fillna(
            0,
        ),
        "product_total_orders": training_data["product_total_orders"].fillna(
            0,
        ),
        "product_unique_users": training_data["product_unique_users"].fillna(
            0,
        ),
        "product_reorder_rate": training_data["product_reorder_rate"].fillna(
            0.0,
        ),
        "product_avg_cart_position": training_data[
            "product_avg_cart_position"
        ].fillna(0.0),
        "user_department_purchase_count": training_data[
            "user_department_purchase_count"
        ].fillna(0),
        "user_department_purchase_rate": training_data[
            "user_department_purchase_rate"
        ].fillna(0.0),
        "user_aisle_purchase_count": training_data[
            "user_aisle_purchase_count"
        ].fillna(0),
        "user_aisle_purchase_rate": training_data[
            "user_aisle_purchase_rate"
        ].fillna(0.0),
    }

    # Criar cada coluna numérica no DataFrame final.
    for column, value in default_values.items():
        training_data[column] = value

    # Manter somente as colunas utilizadas pelos modelos.
    return training_data[SELECTED_COLUMNS]


def build_product_history_features(user_history: pd.DataFrame) -> pd.DataFrame:
    """Cria estatísticas do produto no histórico anterior do usuário.

    Exemplo:
        features = build_product_history_features(user_history)
    """
    # Agrupar apenas compras anteriores, sem olhar o pedido usado como alvo.
    return (
        user_history.groupby(["user_id", "product_id"])
        .agg(
            purchase_count=("product_id", "size"),
            reorder_rate=("reordered", "mean"),
            first_order_number=("order_number", "min"),
            last_order_number=("order_number", "max"),
            orders_since_last_purchase=("days_since_prior_order", "mean"),
            purchase_frequency=("product_id", "size"),
        )
        .reset_index()
    )


def build_user_history_features(user_history: pd.DataFrame) -> pd.DataFrame:
    """Cria estatísticas gerais do histórico anterior do usuário.

    Exemplo:
        features = build_user_history_features(user_history)
    """
    basket_size = (
        user_history.groupby(["user_id", "order_id"])
        .size()
        .groupby("user_id")
        .mean()
        .rename("user_avg_basket_size")
    )

    return (
        user_history.groupby("user_id")
        .agg(
            user_total_orders=("order_id", "nunique"),
            user_total_items=("product_id", "size"),
            user_unique_products=("product_id", "nunique"),
            user_reorder_rate=("reordered", "mean"),
            user_avg_cart_position=("add_to_cart_order", "mean"),
            user_avg_days_between_orders=("days_since_prior_order", "mean"),
            user_avg_order_hour=("order_hour_of_day", "mean"),
        )
        .join(basket_size, on="user_id")
        .reset_index()
    )


def build_catalog_history_features(user_history: pd.DataFrame) -> pd.DataFrame:
    """Cria estatísticas globais do produto no histórico anterior.

    Exemplo:
        features = build_catalog_history_features(user_history)
    """
    # Essas métricas olham a popularidade do produto no histórico geral.
    return (
        user_history.groupby("product_id")
        .agg(
            product_total_orders=("order_id", "nunique"),
            product_unique_users=("user_id", "nunique"),
            product_reorder_rate=("reordered", "mean"),
            product_avg_cart_position=("add_to_cart_order", "mean"),
        )
        .reset_index()
    )


def build_category_history_features(
    user_history: pd.DataFrame,
) -> pd.DataFrame:
    """Cria estatísticas de department e aisle no histórico do usuário.

    Exemplo:
        features = build_category_history_features(user_history)
    """
    user_items = user_history.groupby("user_id")["product_id"].transform(
        "size",
    )
    history = user_history.assign(user_items=user_items)

    category_features = (
        history.groupby(["user_id", "department", "aisle"])
        .agg(
            user_department_purchase_count=("department", "size"),
            user_aisle_purchase_count=("aisle", "size"),
            user_items=("user_items", "first"),
        )
        .reset_index()
    )
    category_features["user_department_purchase_rate"] = (
        category_features["user_department_purchase_count"]
        / category_features["user_items"]
    )
    category_features["user_aisle_purchase_rate"] = (
        category_features["user_aisle_purchase_count"]
        / category_features["user_items"]
    )

    return category_features.drop(columns=["user_items"])


def save_market_prices() -> None:
    """Salva preços estimados no SQLite, quando o arquivo existir.

    Exemplo:
        save_market_prices()
    """
    if not MARKET_PRICES_PATH.exists():
        return

    # O arquivo de preços é uma referência por categoria, não por produto.
    prices = pd.read_csv(MARKET_PRICES_PATH)
    required_columns = {
        "department",
        "aisle",
        "estimated_price_usd",
        "unit",
    }
    missing_columns = required_columns.difference(prices.columns)

    # Interromper quando a referência de preços estiver fora do formato.
    if missing_columns:
        raise ValueError(
            "Arquivo de preços estimados sem colunas obrigatórias: "
            f"{sorted(missing_columns)}."
        )

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
