FROM python:3.13-slim

LABEL maintainer="jrrrtricolor@gmail.com"
LABEL version="0.1.0"
LABEL description="Imagem para executar o treino do recomendador do Tech Challenge Fase 2."

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV POETRY_VERSION=2.2.1
ENV POETRY_NO_INTERACTION=1
ENV POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install "poetry==${POETRY_VERSION}"

COPY pyproject.toml poetry.lock README.md ./
COPY ml_prep_kit ./ml_prep_kit
COPY src ./src
COPY data/download_dataset.py ./data/download_dataset.py
COPY data/reference ./data/reference

RUN poetry install --only main

RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app

USER appuser

CMD ["poetry", "run", "python", "-m", "ecommerce_recommender.main"]
