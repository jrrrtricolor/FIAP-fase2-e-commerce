# Checklist do Tech Challenge - Fase 2

Este checklist consolida os principais requisitos do Tech Challenge Fase 2 e
acompanha o status atual do projeto `fiap-fase2-ecommerce-recommender`.

## 1. Objetivo funcional

- [x] Definir o problema como recomendação de produtos para e-commerce.
- [x] Usar o comportamento dos usuários como base para as recomendações.
- [x] Organizar o projeto em torno do domínio de recomendação.
- [x] Documentar claramente a solução final no README.
- [x] Explicar como o modelo gera ou prioriza recomendações.

## 2. Dados

- [x] Escolher um dataset com pelo menos 10 mil interações usuário-item.
- [x] Usar um dataset com histórico de usuários, produtos e categorias.
- [x] Criar uma base de treino estruturada para recomendação.
- [x] Armazenar os dados no diretório `data/`.
- [x] Usar um banco SQLite local para a base de treino.
- [x] Documentar a origem dos dados.
- [x] Versionar o pipeline de dados com DVC em versão inicial.
- [x] Documentar em detalhe o processo de preparação da base de treino.
- [x] Criar stage DVC específico para preparação da base `training_data.db`.

## 3. Estrutura do projeto

- [x] Separar o framework reutilizável em `ml_prep_kit/`.
- [x] Separar o código específico do desafio em `src/`.
- [x] Manter os notebooks em `notebooks/`.
- [x] Manter os dados em `data/`.
- [x] Manter artefatos e modelos finais em `model/`.
- [x] Ignorar o diretório `agents/` no Git.

## 4. Poetry e ambiente

- [x] Criar o arquivo `pyproject.toml`.
- [x] Criar o arquivo `poetry.lock`.
- [x] Configurar os pacotes locais `ml_prep_kit` e `ecommerce_recommender`.
- [x] Separar dependências principais e dependências de desenvolvimento.
- [x] Garantir que o projeto rode com `poetry run`.
- [x] Adicionar PyTorch como dependência principal do projeto.
- [x] Documentar a instalação com Poetry no README.
- [x] Documentar os principais comandos de execução.

## 5. Qualidade de código

- [x] Configurar o Ruff.
- [x] Rodar `poetry run ruff check .` com sucesso.
- [x] Usar nomes descritivos nos principais componentes.
- [x] Usar type hints nos módulos principais.
- [x] Criar logging estruturado reutilizável no `ml_prep_kit`.

## 6. Testes

- [x] Criar testes para `ModelFactory`.
- [x] Criar testes para `ModelEvaluator`.
- [x] Criar testes para o classificador PyTorch tabular.
- [x] Criar testes para o classificador PyTorch com interface Scikit-Learn.
- [x] Criar testes para logging estruturado.
- [x] Rodar os testes do `ml_prep_kit` com Poetry.
- [x] Documentar o comando de testes no README.

## 7. Modelagem

- [x] Criar um baseline com Scikit-Learn.
- [x] Criar um avaliador reutilizável de modelos.
- [x] Calcular métricas de classificação.
- [x] Treinar modelos iniciais com uma amostra da base.
- [x] Definir a estratégia final de recomendação.
- [x] Criar uma rede neural tabular reutilizável no `ml_prep_kit`.
- [x] Criar um classificador PyTorch reutilizável com interface semelhante ao
      Scikit-Learn.
- [x] Implementar o modelo principal com PyTorch.
- [x] Fixar seeds de forma consistente.
- [x] Documentar estratégia de score e ranking no README e no Model Card.
- [x] Salvar ou registrar o melhor modelo final.
- [x] Documentar as métricas escolhidas e suas justificativas.

## 8. MLflow

- [x] Adicionar a dependência do MLflow.
- [x] Criar o `ExperimentTracker` reutilizável no `ml_prep_kit`.
- [x] Registrar parâmetros.
- [x] Registrar métricas.
- [x] Registrar modelos Scikit-Learn.
- [x] Configurar o tracking local com SQLite.
- [x] Registrar o modelo PyTorch.
- [x] Documentar como abrir a interface do MLflow.
- [x] Usar o MLflow Model Registry.
- [x] Promover o melhor modelo para Production ou alias equivalente.

## 9. DVC

- [x] Adicionar a dependência do DVC.
- [x] Inicializar o DVC no projeto.
- [x] Criar o arquivo `dvc.yaml`.
- [x] Versionar o arquivo `dvc.lock`.
- [x] Documentar como reproduzir o pipeline com DVC.
- [x] Versionar o treino do modelo em versão inicial.
- [x] Criar pelo menos três stages no pipeline.
- [x] Versionar a preparação dos dados.
- [x] Registrar e versionar o melhor modelo no MLflow Model Registry.

## 10. Docker

- [x] Criar o arquivo `.dockerignore`.
- [x] Criar o `Dockerfile`.
- [x] Documentar os comandos Docker no README.
- [x] Usar Dockerfile multi-stage, se aplicável.
- [x] Criar o arquivo `docker-compose.yml`, se necessário.
- [x] Garantir a execução do treino dentro do container.
- [x] Validar o build em ambiente com Docker daemon ativo.

## 11. Documentação

- [x] Criar o README inicial do projeto.
- [x] Documentar a estrutura principal do projeto.
- [x] Atualizar o README com a instalação via Poetry.
- [x] Atualizar o README com os comandos de teste.
- [x] Atualizar o README com os comandos de treino.
- [x] Atualizar o README com os comandos do MLflow.
- [x] Atualizar o README com os comandos do DVC.
- [x] Atualizar o README com os comandos Docker.
- [x] Criar o Model Card inicial do modelo.
- [x] Atualizar o Model Card com resultados atuais.
- [x] Criar um resumo da arquitetura da solução.
- [x] Publicar o vídeo final no YouTube.
- [x] Documentar o link do vídeo final no README.
- [x] Documentar o link público da API no README.
- [x] Implementar deploy em cloud da API pública na AWS.

## 12. Git e entrega

- [x] Usar commits semânticos.
- [x] Evitar versionar ambiente virtual.
- [x] Evitar versionar arquivos locais do MLflow.
- [x] Evitar versionar o diretório `agents/`.
- [x] Garantir que o repositório final esteja limpo.
- [x] Garantir que arquivos grandes estejam fora do Git ou versionados via DVC.
- [x] Conferir o checklist final antes da entrega.
- [x] Entregar o bônus de deploy em cloud na AWS.
