"""Treina uma rede neural PyTorch para recomendação de produtos."""

from __future__ import annotations

import logging
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ecommerce_recommender.training import (
    BINARY_COLUMNS,
    CATEGORICAL_COLUMNS,
    DATABASE_PATH,
    FEATURE_COLUMNS,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    NUMERIC_COLUMNS,
    POSITIVE_RATE,
    RANDOM_SEED,
    SAMPLE_SIZE,
    TARGET_COLUMN,
    TEST_SIZE,
    load_training_sample,
)
from ml_prep_kit import (
    ExperimentTracker,
    FeaturePreprocessor,
    ModelEvaluator,
    SklearnTorchBinaryClassifier,
    StructuredLoggingConfigurator,
)

LOGGER = logging.getLogger("ecommerce_recommender.torch_training")


def run_training() -> None:
    """Executa o treino neural e registra o resultado no MLflow.

    Exemplo:
        run_training()
    """
    nome_modelo = "pytorch_mlp"  # Nome usado na execução do MLflow.
    epocas = 10  # Quantidade de passagens completas pela base de treino.
    tamanho_lote = 512  # Quantidade de exemplos processados por lote.
    taxa_aprendizado = 0.001  # Taxa usada pelo otimizador Adam.
    tamanho_camada_oculta = 64  # Neurônios da camada oculta.
    limiar_classificacao = 0.5  # Corte para converter score em classe.

    StructuredLoggingConfigurator.configure()
    LOGGER.info(
        "Iniciando treino PyTorch do recomendador.",
        extra={
            "evento": "treino_pytorch_iniciado",
            "nome_modelo": nome_modelo,
            "tamanho_amostra": SAMPLE_SIZE,
        },
    )

    dados = load_training_sample(
        database_path=DATABASE_PATH,
        sample_size=SAMPLE_SIZE,
        positive_rate=POSITIVE_RATE,
        random_state=RANDOM_SEED,
    )
    X = dados[FEATURE_COLUMNS]
    y = dados[TARGET_COLUMN]

    LOGGER.info(
        "Separando dados entre treino e validação.",
        extra={
            "evento": "divisao_treino_validacao_iniciada",
            "linhas": len(dados),
            "features": len(FEATURE_COLUMNS),
        },
    )
    X_treino, X_validacao, y_treino, y_validacao = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    LOGGER.info(
        "Preparando features com o preprocessador do projeto.",
        extra={
            "evento": "pre_processamento_iniciado",
            "colunas_numericas": len(NUMERIC_COLUMNS),
            "colunas_categoricas": len(CATEGORICAL_COLUMNS),
            "colunas_binarias": len(BINARY_COLUMNS),
        },
    )
    preprocessador = FeaturePreprocessor(
        numeric_columns=NUMERIC_COLUMNS,
        categorical_columns=CATEGORICAL_COLUMNS,
        binary_columns=BINARY_COLUMNS,
    )
    X_treino_pronto = preprocessador.fit_prepare(X_treino)
    X_validacao_pronto = preprocessador.prepare(X_validacao)

    classificador = SklearnTorchBinaryClassifier(
        hidden_size=tamanho_camada_oculta,
        learning_rate=taxa_aprendizado,
        epochs=epocas,
        batch_size=tamanho_lote,
        random_seed=RANDOM_SEED,
        threshold=limiar_classificacao,
    )
    classificador.fit(X_treino_pronto, y_treino)

    y_score = pd.Series(
        classificador.predict_proba(X_validacao_pronto)[:, 1],
        index=y_validacao.index,
    )
    y_pred = pd.Series(
        classificador.predict(X_validacao_pronto),
        index=y_validacao.index,
    )
    metricas = ModelEvaluator().evaluate_classification(
        y_validacao,
        y_pred,
        y_score,
    )
    metricas["train_loss"] = classificador.train_loss_

    # Chaves mantidas em inglês para padronizar a visualização no MLflow.
    parametros = {
        "model_name": nome_modelo,
        "sample_size": len(dados),
        "target_rate": float(y.mean()),
        "random_seed": RANDOM_SEED,
        "feature_count": len(FEATURE_COLUMNS),
        "input_size": X_treino_pronto.shape[1],
        "epochs": epocas,
        "batch_size": tamanho_lote,
        "learning_rate": taxa_aprendizado,
        "hidden_size": tamanho_camada_oculta,
        "threshold": limiar_classificacao,
    }

    LOGGER.info(
        "Registrando modelo PyTorch no MLflow.",
        extra={
            "evento": "registro_modelo_pytorch_iniciado",
            "nome_modelo": nome_modelo,
            "metricas": metricas,
        },
    )
    rastreador = ExperimentTracker(
        experiment_name=MLFLOW_EXPERIMENT_NAME,
        tracking_uri=MLFLOW_TRACKING_URI,
    )

    execucao_id = log_model(
        rastreador=rastreador,
        classificador=classificador,
        nome_modelo=nome_modelo,
        parametros=parametros,
        metricas=metricas,
        preprocessador=preprocessador,
        exemplo_entrada=X_validacao_pronto.head(5).to_numpy(
            dtype=np.float32,
        ),
    )

    LOGGER.info(
        "Treino PyTorch concluído com sucesso.",
        extra={
            "evento": "treino_pytorch_concluido",
            "execucao_id": execucao_id,
            "nome_modelo": nome_modelo,
            "metricas": metricas,
        },
    )


def log_model(
    rastreador: ExperimentTracker,
    classificador: SklearnTorchBinaryClassifier,
    nome_modelo: str,
    parametros: dict,
    metricas: dict[str, float],
    preprocessador: FeaturePreprocessor,
    exemplo_entrada: np.ndarray,
) -> str:
    """Registra a rede neural e o preprocessador no MLflow.

    Exemplo:
        execucao_id = log_model(
            rastreador=rastreador,
            classificador=classificador,
            nome_modelo="pytorch_mlp",
            parametros=parametros,
            metricas=metricas,
            preprocessador=preprocessador,
            exemplo_entrada=exemplo_entrada,
        )
    """
    with TemporaryDirectory() as diretorio_temporario:
        caminho_preprocessador = (
            f"{diretorio_temporario}/preprocessor.joblib"
        )
        # O preprocessador é necessário para transformar dados novos antes de
        # chamar a rede neural.
        joblib.dump(preprocessador.transformer, caminho_preprocessador)

        return rastreador.log_pytorch_training_run(
            run_name=nome_modelo,
            model=classificador.model_module,
            parameters=parametros,
            metrics=metricas,
            artifacts=[caminho_preprocessador],
            input_example=exemplo_entrada,
        )


if __name__ == "__main__":
    run_training()
