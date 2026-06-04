"""Gera recomendações Top-N a partir de scores de modelo."""

from __future__ import annotations

import logging

import pandas as pd

from ecommerce_recommender.training import FEATURE_COLUMNS

LOGGER = logging.getLogger("ecommerce_recommender.recommendation")


def recommend_top_n(
    model,
    candidates: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """Ordena produtos candidatos e retorna o Top-N por usuário.

    Exemplo:
        recommendations = recommend_top_n(
            model=pipeline,
            candidates=candidate_data,
            top_n=10,
        )
    """
    # Validar se os candidatos possuem as colunas necessárias.
    validate_candidates(candidates)

    # Calcular a probabilidade de recomendação para cada candidato.
    scores = model.predict_proba(candidates[FEATURE_COLUMNS])[:, 1]

    # Criar a tabela final com usuário, produto e pontuação.
    recommendations = candidates[["user_id", "product_id"]].copy()
    recommendations["recommendation_score"] = scores

    # Ordenar os produtos mais relevantes para cada usuário.
    recommendations = recommendations.sort_values(
        ["user_id", "recommendation_score"],
        ascending=[True, False],
    )

    LOGGER.info(
        "Recomendações geradas com sucesso.",
        extra={
            "evento": "recomendacoes_top_n_geradas",
            "usuarios": recommendations["user_id"].nunique(),
            "candidatos": len(recommendations),
            "top_n": top_n,
        },
    )

    return recommendations.groupby("user_id").head(top_n).reset_index(
        drop=True,
    )


def validate_candidates(candidates: pd.DataFrame) -> None:
    """Valida se os candidatos possuem as colunas necessárias.

    Exemplo:
        validate_candidates(candidate_data)
    """
    # Verificar se todas as colunas obrigatórias existem.
    required_columns = {"user_id", "product_id", *FEATURE_COLUMNS}
    missing_columns = sorted(required_columns - set(candidates.columns))

    # Interromper a execução quando existirem colunas ausentes.
    if missing_columns:
        raise ValueError(
            "Dados de candidatos sem colunas obrigatórias: "
            f"{missing_columns}."
        )
