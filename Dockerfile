FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV POETRY_VERSION=2.2.1
ENV POETRY_NO_INTERACTION=1
ENV POETRY_VIRTUALENVS_CREATE=false
ENV PYTHONPATH=/app/src:/app/ml_prep_kit/src

WORKDIR /app

RUN pip install "poetry==${POETRY_VERSION}"

COPY pyproject.toml poetry.lock README.md ./
COPY ml_prep_kit ./ml_prep_kit
COPY src ./src
COPY data/download_dataset.py ./data/download_dataset.py
COPY data/reference ./data/reference

RUN poetry install --only main --no-root

RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app

USER appuser

CMD ["poetry", "run", "python", "-m", "ecommerce_recommender.main"]
