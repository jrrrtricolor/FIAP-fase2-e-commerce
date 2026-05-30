# Checklist do Tech Challenge - Fase 2

Este checklist consolida os principais requisitos do Tech Challenge Fase 2 e
acompanha o status atual do projeto `fiap-fase2-ecommerce-recommender`.

## 1. Objetivo funcional

- [x] Definir o problema como recomendação de produtos para e-commerce.
- [x] Usar o comportamento dos usuários como base para as recomendações.
- [x] Organizar o projeto em torno do domínio de recomendação.
- [ ] Documentar claramente a solução final no README.
- [ ] Explicar como o modelo gera ou prioriza recomendações.

## 2. Dados

- [x] Escolher um dataset com pelo menos 10 mil interações usuário-item.
- [x] Usar um dataset com histórico de usuários, produtos e categorias.
- [x] Criar uma base de treino estruturada para recomendação.
- [x] Armazenar os dados no diretório `data/`.
- [x] Usar um banco SQLite local para a base de treino.
- [ ] Documentar a origem dos dados.
- [ ] Documentar o processo de preparação da base de treino.
- [ ] Versionar o pipeline de dados com DVC.

## 3. Estrutura do projeto

- [x] Separar o framework reutilizável em `ml_prep_kit/`.
- [x] Separar o código específico do desafio em `src/`.
- [x] Manter os notebooks em `notebooks/`.
- [x] Manter os dados em `data/`.
- [x] Manter artefatos e modelos finais em `model/`.
- [x] Ignorar o diretório `agents/` no Git.
- [ ] Remover ou revisar diretórios antigos que não fazem parte da estrutura
      final, como `pipelines/`, `preprocess/` e `tests/`, caso estejam sem uso.

## 4. Poetry e ambiente

- [x] Criar o arquivo `pyproject.toml`.
- [x] Criar o arquivo `poetry.lock`.
- [x] Configurar os pacotes locais `ml_prep_kit` e `ecommerce_recommender`.
- [x] Separar dependências principais e dependências de desenvolvimento.
- [x] Garantir que o projeto rode com `poetry run`.
- [ ] Documentar a instalação com Poetry no README.
- [ ] Documentar os principais comandos de execução.

## 5. Qualidade de código

- [x] Configurar o Ruff.
- [x] Rodar `poetry run ruff check .` com sucesso.
- [x] Usar nomes descritivos nos principais componentes.
- [x] Usar type hints nos módulos principais.
- [ ] Revisar docstrings públicas no padrão Google.
- [ ] Garantir que o código final não tenha duplicações relevantes.
- [ ] Garantir que funções e classes tenham responsabilidades claras.

## 6. Testes

- [x] Criar testes para `ModelFactory`.
- [x] Criar testes para `ModelEvaluator`.
- [x] Rodar os testes do `ml_prep_kit` com Poetry.
- [ ] Criar testes para `ExperimentTracker`, usando mocks ou um ambiente local
      controlado.
- [ ] Criar um teste simples para importação do pacote
      `ecommerce_recommender`.
- [ ] Documentar o comando de testes no README.

## 7. Modelagem

- [x] Criar um baseline com Scikit-Learn.
- [x] Criar um avaliador reutilizável de modelos.
- [x] Calcular métricas de classificação.
- [x] Treinar modelos iniciais com uma amostra da base.
- [ ] Definir a estratégia final de recomendação.
- [ ] Implementar o modelo principal com PyTorch.
- [ ] Fixar seeds de forma consistente.
- [ ] Salvar ou registrar o melhor modelo final.
- [ ] Documentar as métricas escolhidas e suas justificativas.

## 8. MLflow

- [x] Adicionar a dependência do MLflow.
- [x] Criar o `ExperimentTracker` reutilizável no `ml_prep_kit`.
- [x] Registrar parâmetros.
- [x] Registrar métricas.
- [x] Registrar modelos Scikit-Learn.
- [x] Configurar o tracking local com SQLite.
- [ ] Registrar o modelo PyTorch.
- [ ] Usar o MLflow Model Registry.
- [ ] Promover o melhor modelo para Production ou alias equivalente.
- [ ] Documentar como abrir a interface do MLflow.

## 9. DVC

- [ ] Adicionar a dependência do DVC.
- [ ] Inicializar o DVC no projeto.
- [ ] Criar o arquivo `dvc.yaml`.
- [ ] Criar pelo menos três stages no pipeline.
- [ ] Versionar a preparação dos dados.
- [ ] Versionar o treino do modelo.
- [ ] Versionar a avaliação ou o registro do modelo.
- [ ] Documentar como reproduzir o pipeline com DVC.

## 10. Docker

- [ ] Criar o arquivo `.dockerignore`.
- [ ] Criar o `Dockerfile`.
- [ ] Usar Dockerfile multi-stage, se aplicável.
- [ ] Criar o arquivo `docker-compose.yml`, se necessário.
- [ ] Garantir a execução do treino dentro do container.
- [ ] Documentar os comandos Docker no README.

## 11. Documentação

- [x] Criar o README inicial do projeto.
- [x] Documentar a estrutura principal do projeto.
- [ ] Atualizar o README com a instalação via Poetry.
- [ ] Atualizar o README com os comandos de teste.
- [ ] Atualizar o README com os comandos de treino.
- [ ] Atualizar o README com os comandos do MLflow.
- [ ] Atualizar o README com os comandos do DVC.
- [ ] Atualizar o README com os comandos Docker.
- [ ] Criar o Model Card do modelo final.
- [ ] Criar um resumo da arquitetura da solução.
- [ ] Criar o roteiro do vídeo final no formato STAR.

## 12. Git e entrega

- [x] Usar commits semânticos.
- [x] Evitar versionar ambiente virtual.
- [x] Evitar versionar arquivos locais do MLflow.
- [x] Evitar versionar o diretório `agents/`.
- [ ] Garantir que o repositório final esteja limpo.
- [ ] Garantir que arquivos grandes estejam fora do Git ou versionados via DVC.
- [ ] Conferir o checklist final antes da entrega.

## 13. Critérios de avaliação

| Critério | Peso | Status |
| --- | ---: | --- |
| Clean code e estrutura | 15% | Em andamento |
| Reprodutibilidade | 15% | Em andamento |
| Docker | 15% | Pendente |
| Pipeline DVC | 15% | Pendente |
| Rede neural com PyTorch | 15% | Pendente |
| MLflow e Model Registry | 10% | Em andamento |
| Vídeo STAR | 10% | Pendente |
| Bônus de deploy em cloud | 5% | Opcional |

## Próximos passos recomendados

1. Implementar o modelo principal com PyTorch.
2. Registrar o modelo PyTorch no MLflow.
3. Criar o pipeline DVC com pelo menos três stages.
4. Criar o Dockerfile e validar a execução em container.
5. Atualizar o README e criar o Model Card.
6. Preparar o roteiro do vídeo STAR.
