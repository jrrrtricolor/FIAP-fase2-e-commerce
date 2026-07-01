# FIAP Fase 2 - E-commerce Recommender

Sistema de recomendação de produtos para e-commerce desenvolvido para o Tech
Challenge Fase 2 da FIAP.

O projeto estima a chance de um produto ser relevante para um usuário. Esse
score é usado para ordenar produtos candidatos e gerar recomendações.

## Estrutura

```text
ml_prep_kit/  -> mini framework reutilizável
src/          -> código específico do Tech Challenge
notebooks/    -> exploração e comunicação
data/         -> dados locais e arquivos de referência
docs/         -> documentação da entrega
```

## Requisitos

- Python 3.13
- Poetry
- Docker, opcional para execução em container

## Instalação

```bash
poetry install
```

## Dados

Dataset usado:

```text
psparks/instacart-market-basket-analysis
```

Baixar os dados e preparar a base local:

```bash
poetry run python data/download_dataset.py
make prepare-data
```

A base preparada fica em `data/training_data.db` e não é versionada no Git.

## Execução

Treinar todos os modelos:

```bash
make train
```

Abrir o MLflow:

```bash
make mlflow-ui
```

Executar a API:

```bash
make run-server
```

Documentação da API:

```text
http://localhost:8000/docs
```

## Validação

```bash
make check
```

Esse comando executa lint, testes e validação do `pyproject.toml`.

## DVC

Reproduzir o pipeline completo:

```bash
make dvc-repro
```

Verificar o estado do pipeline:

```bash
make dvc-status
```

Stages versionados:

- `download_dataset`
- `prepare_data`
- `train_models`

## Docker

Construir a imagem:

```bash
make docker-build
```

Subir serviços com Docker Compose:

```bash
docker compose up api
docker compose up mlflow
docker compose up trainer
```

## Documentação

- Checklist da entrega: `docs/requirements_checklist.md`
- Model Card: `docs/model_card.md`
- Enunciado: `docs/Tech Challenge Fase 02.pdf`

## Limitações

- A API recebe produtos candidatos prontos.
- A geração automática de candidatos ainda não está exposta na API.
- O deploy em cloud ainda não foi implementado.
