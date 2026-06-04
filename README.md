# FIAP Fase 2 - E-commerce Recommender

Sistema de recomendação de produtos para e-commerce desenvolvido para o Tech
Challenge Fase 2 da FIAP.

O projeto estima a chance de um produto ser relevante para um usuário. Esse
score pode ser usado para ordenar produtos candidatos e gerar recomendações.

## Estrutura

```text
ml_prep_kit/  -> mini framework reutilizável
src/          -> scripts específicos do Tech Challenge
notebooks/    -> exploração e comunicação
data/         -> dados locais e arquivos de referência
model/        -> artefatos e modelos finais
docs/         -> documentação da entrega
```

## Componentes

- `ml_prep_kit/`: preprocessamento, avaliação, logging, MLflow e modelos
  reutilizáveis.
- `src/ecommerce_recommender/main.py`: executa o fluxo principal.
- `src/ecommerce_recommender/training.py`: treina baselines Scikit-Learn.
- `src/ecommerce_recommender/torch_training.py`: treina o modelo PyTorch.
- `data/download_dataset.py`: baixa o dataset Instacart.
- `Makefile`: centraliza os comandos do projeto.

## Dados

Dataset usado:

```text
psparks/instacart-market-basket-analysis
```

Download:

```bash
poetry run python data/download_dataset.py
```

A base de treino usada pelos modelos é local:

```text
data/training_data.db
```

Essa base não é versionada no Git por ser grande.

## Instalação

Requisitos:

- Python 3.13
- Poetry

Instale as dependências:

```bash
poetry install
```

## Comandos

Validar o projeto:

```bash
make check
```

Rodar lint:

```bash
make lint
```

Rodar testes:

```bash
make test
```

Treinar todos os modelos:

```bash
make train
```

Treinar apenas os baselines:

```bash
make train-classic
```

Treinar apenas o modelo PyTorch:

```bash
make train-torch
```

## MLflow

Os experimentos são registrados localmente em:

```text
mlflow.db
mlruns/
```

Abrir a interface:

```bash
make mlflow-ui
```

Depois acesse:

```text
http://127.0.0.1:5001
```

## DVC

O DVC está configurado em versão inicial.

Verificar status:

```bash
make dvc-status
```

Reproduzir pipeline:

```bash
make dvc-repro
```

Stages atuais:

- `download_dataset`
- `train_models`

## Docker

Construir imagem:

```bash
make docker-build
```

Executar container:

```bash
make docker-run
```

Observação: é necessário ter o Docker daemon ativo.

## Qualidade

O projeto usa Ruff com as regras:

```text
E, F, I, N, UP
```

Também possui GitHub Actions para:

- lint e testes;
- build Docker.

## Documentação

- Checklist: `docs/requirements_checklist.md`
- Model Card: `docs/model_card.md`
- Enunciado: `docs/Tech Challenge Fase 02.pdf`

## Limitações atuais

- A rotina final de Top-N recomendações ainda precisa ser consolidada.
- A preparação de `data/training_data.db` ainda deve virar um stage próprio do
  DVC.
- O MLflow Model Registry ainda não foi usado para promover o melhor modelo.
- O Dockerfile precisa ser validado em ambiente com Docker ativo.
