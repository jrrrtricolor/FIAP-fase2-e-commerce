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

Essas métricas avaliam a classificação dos pares usuário-produto. Para uma
versão final de recomendação, o projeto ainda deve incluir métricas de ranking,
como Precision@K, Recall@K ou NDCG@K.

## Resultado Atual

Na última execução local registrada pelo pipeline DVC, o critério de escolha foi
`roc_auc`.

| Modelo | Accuracy | Precision | Recall | F1 | ROC AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| logistic_regression | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| hist_gradient_boosting | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| pytorch_mlp | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| dummy | 0.8000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |

O melhor modelo registrado nessa execução foi `logistic_regression`. Como as
métricas ficaram perfeitas para mais de um modelo, esse resultado deve ser
revisado antes da entrega final para descartar vazamento de informação ou uma
separação de treino e validação fácil demais.

## Uso pretendido

O modelo deve ser usado para apoiar a priorização de produtos candidatos para
usuários de e-commerce.

O uso esperado é:

1. gerar candidatos para cada usuário;
2. calcular o score de cada par usuário-produto;
3. ordenar os produtos pelo score;
4. recomendar os produtos mais bem posicionados.

## Limitações

- A API calcula recomendações para candidatos recebidos na requisição.
- A geração automática de candidatos ainda não está exposta na API.
- As métricas atuais são de classificação, não de ranking.
- O deploy em cloud ainda não foi implementado.

## Riscos

- Produtos populares podem receber vantagem excessiva.
- Produtos novos podem ter baixa exposição.
- O histórico de compras pode não representar preferências futuras.
- Recomendações pouco diversas podem reduzir a qualidade da experiência.

## Próximos passos

1. Adicionar métricas de ranking.
2. Revisar as métricas perfeitas antes da entrega final.
3. Automatizar a geração de candidatos para a API.
4. Preparar o roteiro do vídeo STAR.
