"""Executa o fluxo principal de treino do Tech Challenge Fase 2."""

from ecommerce_recommender import torch_training, training
from ml_prep_kit import StructuredLoggingConfigurator


def main() -> None:
    """Executa os treinamentos em uma ordem simples de acompanhar.

    Este arquivo é a porta de entrada do projeto. Ele mantém o fluxo linear
    para facilitar a leitura durante a avaliação: primeiro treinamos os
    modelos clássicos do Scikit-Learn e depois treinamos a rede neural em
    PyTorch.

    Exemplo:
        poetry run python -m ecommerce_recommender.main
    """
    # Configura logs estruturados para todos os passos executados abaixo.
    StructuredLoggingConfigurator.configure()

    # Primeiro fluxo: modelos clássicos usados como baseline comparativo.
    training.run_training()

    # Segundo fluxo: rede neural PyTorch registrada no mesmo experimento.
    torch_training.run_training()


if __name__ == "__main__":
    main()
