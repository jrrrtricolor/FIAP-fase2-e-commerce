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

Deploy em cloud da API na AWS:

```text
https://e-commerce.juhouse.com.br/docs
```

Rotas disponíveis na API:

### aisles

Endpoints relacionados à obtenção de informações sobre os corredores dos
produtos.

- `GET /aisles`: obter lista de corredores.

### departments

Endpoints relacionados à obtenção de informações sobre os departamentos dos
produtos.

- `GET /departments`: obter lista de departamentos.

### recomendações

Endpoints relacionados à geração de recomendações para os usuários.

- `POST /recomendacoes`: gerar recomendações para os usuários.

### default

Endpoints de operação e observabilidade.

- `GET /metrics`: métricas de desempenho da API para Prometheus.
- `GET /health`: verificar saúde da API.

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
- Vídeo STAR: `https://youtu.be/4Guhz0sKVVw`
- Deploy em cloud da API na AWS: `https://e-commerce.juhouse.com.br/docs`
- Enunciado: `docs/Tech Challenge Fase 02.pdf`
