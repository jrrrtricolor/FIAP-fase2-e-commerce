# FIAP-fase2-e-commerce
Uma empresa de e-commerce precisa de um sistema de recomendação de produtos baseado no comportamento de navegação dos usuários.

## Estrutura do projeto

O projeto segue uma separação simples entre framework reutilizável, código específico do desafio, análise e artefatos:

```text
ml_prep_kit/  -> mini framework reutilizável
notebooks/    -> exploração e comunicação
data/         -> dados
model/        -> artefatos e modelos finais
src/          -> scripts específicos do Tech Challenge
```

## Organização do código

- `ml_prep_kit/`: contém utilitários reutilizáveis para leitura, validação, preprocessamento, criação de modelos, avaliação e registro de experimentos.
- `src/`: contém os scripts e fluxos específicos do Tech Challenge, como treino dos modelos, seleção de colunas, configuração do experimento e uso da base `training_data`.
- `notebooks/`: concentra exploração, protótipos e comunicação dos resultados.
- `data/`: armazena bases brutas, preparadas e arquivos auxiliares.
- `model/`: armazena modelos exportados e artefatos finais relevantes para a entrega.

Componentes reutilizáveis, como registro no MLflow e avaliação padronizada de modelos, devem ficar no `ml_prep_kit`. Fluxos específicos do desafio devem ficar em `src`.

## Explicabilidade do modelo

O projeto utilizará SHAP/SHARP para explicar o modelo de recomendação. Essa abordagem será usada para interpretar quais características do comportamento dos usuários tiveram maior influência nas recomendações geradas, apoiando a análise técnica, o Model Card e a apresentação final.
