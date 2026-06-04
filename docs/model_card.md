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

## Uso pretendido

O modelo deve ser usado para apoiar a priorização de produtos candidatos para
usuários de e-commerce.

O uso esperado é:

1. gerar candidatos para cada usuário;
2. calcular o score de cada par usuário-produto;
3. ordenar os produtos pelo score;
4. recomendar os produtos mais bem posicionados.

## Limitações

- A rotina final de geração de Top-N recomendações ainda precisa ser
  consolidada.
- A preparação da base `training_data.db` ainda precisa virar um stage próprio
  do DVC.
- O MLflow Model Registry ainda não foi usado para promover o melhor modelo.
- As métricas atuais são de classificação, não de ranking.
- O Dockerfile ainda precisa ser validado em ambiente com Docker ativo.

## Riscos

- Produtos populares podem receber vantagem excessiva.
- Produtos novos podem ter baixa exposição.
- O histórico de compras pode não representar preferências futuras.
- Recomendações pouco diversas podem reduzir a qualidade da experiência.

## Próximos passos

1. Criar a rotina de Top-N recomendações.
2. Adicionar métricas de ranking.
3. Criar stage DVC para preparação da base de treino.
4. Comparar os modelos no MLflow.
5. Promover o melhor modelo no MLflow Model Registry.
