.PHONY: install check lint test prepare-data train train-classic train-torch mlflow-ui clean docker-build docker-run dvc-repro dvc-status

POETRY ?= poetry
DOCKER_IMAGE ?= fiap-fase2-ecommerce-recommender
DOCKER_TAG ?= $(shell awk -F'"' '/^version =/ {print $$2; exit}' pyproject.toml)
MLFLOW_PORT ?= 5001

install:
	$(POETRY) install

check: lint test
	$(POETRY) check

lint:
	$(POETRY) run ruff check .

test:
	$(POETRY) run python -m unittest discover ./ml_prep_kit/tests
	$(POETRY) run python -m unittest discover ./tests

prepare-data:
	$(POETRY) run python -m ecommerce_recommender.prepare_data

train:
	$(POETRY) run python -m ecommerce_recommender.main

train-classic:
	$(POETRY) run python -m ecommerce_recommender.training

train-torch:
	$(POETRY) run python -m ecommerce_recommender.torch_training

mlflow-ui:
	$(POETRY) run mlflow ui --backend-store-uri sqlite:///mlflow.db --port $(MLFLOW_PORT)

run-server:
	$(POETRY) run fastapi run src/ecommerce_recommender/api.py

docker-build:
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .

docker-run:
	docker run --rm -it \
		-v "$(PWD)/data:/app/data" \
		-v "$(PWD)/mlruns:/app/mlruns" \
		$(DOCKER_IMAGE):$(DOCKER_TAG)

dvc-repro:
	$(POETRY) run dvc repro

dvc-status:
	$(POETRY) run dvc status

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -prune -exec rm -rf {} +
