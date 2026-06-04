FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONPATH=/app/src:/app/ml_prep_kit/src

WORKDIR /app

# Instalar PyTorch CPU-only para evitar dependências CUDA na imagem.
RUN pip install --upgrade pip \
    && pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.12.0,<3.0.0"

# Instalar somente as dependências necessárias para executar o projeto.
RUN pip install \
    "colorama>=0.4.6,<0.5.0" \
    "kagglehub>=1.0.1,<2.0.0" \
    "matplotlib>=3.10.9,<4.0.0" \
    "mlflow>=3.7.0,<4.0.0" \
    "numpy>=2.4.6,<3.0.0" \
    "pandas>=2.3.3,<3.0.0" \
    "protobuf>=3.20.0,<3.21.0" \
    "scikit-learn>=1.8.0,<2.0.0"

COPY ml_prep_kit ./ml_prep_kit
COPY src ./src
COPY data/download_dataset.py ./data/download_dataset.py
COPY data/reference ./data/reference
COPY README.md ./

RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app

USER appuser

CMD ["python", "-m", "ecommerce_recommender.main"]
