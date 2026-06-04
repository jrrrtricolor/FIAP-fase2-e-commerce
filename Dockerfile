FROM python:3.13-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV POETRY_VERSION=2.4.1
ENV POETRY_NO_INTERACTION=1
ENV POETRY_VIRTUALENVS_IN_PROJECT=true

WORKDIR /app

RUN pip install "poetry==${POETRY_VERSION}"

COPY pyproject.toml poetry.lock README.md ./
COPY ml_prep_kit ./ml_prep_kit
COPY src ./src
COPY data/download_dataset.py ./data/download_dataset.py
COPY data/reference ./data/reference

RUN poetry install --only main --no-root


FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:${PATH}"
ENV PYTHONPATH=/app/src:/app/ml_prep_kit/src

WORKDIR /app

COPY --from=builder /app/.venv ./.venv
COPY ml_prep_kit ./ml_prep_kit
COPY src ./src
COPY data/download_dataset.py ./data/download_dataset.py
COPY data/reference ./data/reference
COPY README.md ./

RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app

USER appuser

CMD ["python", "-m", "ecommerce_recommender.main"]
