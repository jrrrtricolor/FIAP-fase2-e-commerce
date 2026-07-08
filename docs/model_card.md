# Model Card - E-commerce Recommender

## Objetivo

Este modelo apoia a recomendação de produtos em um e-commerce.

A solução estima a chance de um produto ser relevante para um usuário. Esse
score pode ser usado para ordenar produtos candidatos e montar uma lista de
recomendações.

## Dados

O projeto usa o dataset Instacart Market Basket Analysis:

```text
psparks/instacart-market-basket-analysis
```

Os dados brutos são baixados pelo script:

```text
data/download_dataset.py
```

A base de treino usada pelos modelos fica em:

```text
data/training_data.db
```

Cada linha representa um par:

```text
usuário + produto candidato
```

O alvo é a coluna `target`:

- `1`: produto relevante para o usuário;
- `0`: produto não relevante para o usuário.

## Modelo

O projeto treina dois grupos de modelos:

- baselines com Scikit-Learn;
- rede neural simples com PyTorch.

Os modelos clássicos são treinados em:

```text
src/ecommerce_recommender/training.py
```

O modelo PyTorch é treinado em:

```text
src/ecommerce_recommender/torch_training.py
```

Os experimentos são registrados no MLflow.

## Features

As principais features representam:

- histórico de compra do usuário;
- popularidade de produtos e categorias;
- coocorrência entre produtos;
- similaridade entre usuários;
- preferência por departamento e aisle;
- frequência de compra e recompra.

## Métricas

As métricas atuais são:

- accuracy;
- precision;
- recall;
- f1;
- roc_auc.

Essas métricas avaliam a classificação dos pares usuário-produto e foram
mantidas para comparar os modelos de forma simples e consistente.

## Resultado Atual

Na última execução local registrada pelo pipeline DVC, o critério de escolha foi
`roc_auc`.

| Modelo | Accuracy | Precision | Recall | F1 | ROC AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| hist_gradient_boosting | 0.9435 | 0.9207 | 0.7852 | 0.8476 | 0.9677 |
| logistic_regression | 0.9181 | 0.7787 | 0.8248 | 0.8011 | 0.9517 |
| pytorch_mlp | 0.9340 | 0.9554 | 0.7028 | 0.8099 | 0.9440 |
| dummy | 0.8000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |

O melhor modelo registrado nessa execução foi `hist_gradient_boosting`. As
features passaram a ser calculadas com base no histórico anterior do usuário,
reduzindo o risco de o modelo receber a resposta diretamente pela base de
treino.

## Uso pretendido

O modelo deve ser usado para apoiar a priorização de produtos candidatos para
usuários de e-commerce.

O uso esperado é:

1. gerar candidatos para cada usuário;
2. calcular o score de cada par usuário-produto;
3. ordenar os produtos pelo score;
4. recomendar os produtos mais bem posicionados.

## API e deploy em cloud

A API pública da solução está disponível em ambiente cloud na AWS:

```text
https://e-commerce.juhouse.com.br/docs
```

Ela recebe produtos candidatos, calcula o score de recomendação e retorna os
itens mais relevantes para o usuário.

Rotas principais:

- `GET /aisles`: obter lista de corredores.
- `GET /departments`: obter lista de departamentos.
- `POST /recomendacoes`: gerar recomendações para os usuários.
- `GET /metrics`: métricas de desempenho da API para Prometheus.
- `GET /health`: verificar saúde da API.


## Riscos

- Produtos populares podem receber vantagem excessiva.
- Produtos novos podem ter baixa exposição.
- O histórico de compras pode não representar preferências futuras.
- Recomendações pouco diversas podem reduzir a qualidade da experiência.

## Próximos passos

1. Automatizar a geração de candidatos para a API.
2. Revisar a separação entre treino e validação em versões futuras.
