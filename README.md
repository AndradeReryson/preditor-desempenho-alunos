## 🚀 Projeto de Classificação de Desempenho Escolar com Streamlit

### 🎯 Objetivo
Este projeto visa aplicar os fundamentos de Machine Learning (ML) para construir e implantar um modelo capaz de prever a situação final de um aluno (Aprovado/Reprovado). O modelo é treinado em dados sintéticos que simulam uma complexa cadeia de causa e efeito no ambiente escolar.

O aplicativo final permite que o usuário insira inputs iniciais (como nota P1 e tempo de deslocamento) para ver a previsão em tempo real.
___
### 🧠 Arquitetura e Regras de Negócio

A principal característica deste projeto é a complexa Engenharia de Features (Criação de Variáveis) que simula o comportamento do aluno. A previsão depende da Média Final das Duas Melhores Notas e da Frequência.

Fluxo da Causalidade:
- **Tempo de Deslocamento** &rarr; Penaliza Faltas.
- **Faltas** &rarr; Penaliza Horas de Estudo base.
- **Nota P1 (Input)** &rarr; Motivação: Se a P1 é baixa, o aluno aumenta as horas de estudo e tem maior probabilidade de fazer a atividade extra (para resgatar pontos).
- **Nota P2/P3** &rarr; São calculadas com base no esforço total (horas_estudo + bônus de P1) e trabalhos extras.
- **Situação Final** &rarr; Aprovado se Média das 2 Melhores Notas > 6 E Faltas < 20

___
### 📊 Tecnologias Utilizadas

- **Linguagem**: Python
- **Machine Learning**: Scikit-learn (Decision Tree Classifier)
- **Manipulação de Dados**: Pandas, NumPy
- **Web App**: Streamlit
- **Serialização**: Joblib

___
### 📁 Estrutura do Repositório

| Pasta/Arquivo | Conteúdo |
| :--- | :--- |
| `app.py` | Aplicação Streamlit principal. Carrega o modelo e a lógica de cálculo para a previsão. |
| `src/` | **Código Fonte:** Lógica de geração de dados e treinamento. |
| `src/gerar_dados.py` | Script que contém a função `calcular_dados_aluno` e gera o `desempenho_alunos.csv`. |
| `src/treinar_modelo.py` | Script para carregar, pré-processar, treinar o modelo e salvar os artefatos (`.pkl`). |
| `data/` | Contém o `desempenho_alunos.csv` (Base de dados gerada). |
| `models/` | Contém os artefatos de ML salvos (`modelo_desempenho.pkl`, `model_metrics.pkl`). |
| `requirements.txt` | Lista todas as dependências do projeto. |
| `run_pipeline.py` | Script orquestrador para rodar as etapas (geração, treinamento e app) em sequência. |

___
### ⚙️ Como Reproduzir o Projeto

#### 1\. Configurar o Ambiente

Crie um ambiente virtual (recomendado) e instale todas as bibliotecas necessárias:

```bash
pip install -r requirements.txt
```

#### 2\. Executar o Pipeline Completo

O `run_pipeline.py` automatiza todo o processo, garantindo que a base de dados seja gerada e o modelo seja treinado antes de lançar o aplicativo web.

```bash
python run_pipeline.py
```

#### 3\. Execução Manual (Passo a Passo)

Se precisar rodar cada etapa individualmente:

1.  **Geração de Dados:**
    ```bash
    python src/gerar_dados.py
    ```
2.  **Treinamento do Modelo:**
    ```bash
    python src/treinar_modelo.py
    ```
3.  **Lançamento do Streamlit App:**
    ```bash
    streamlit run app.py
    ```
___
### 📈 Desempenho do Modelo

As métricas de desempenho são calculadas no conjunto de teste (20%) e salvas no `model_metrics.pkl`.

| Métrica | Valor |
| :--- | :--- |
| **Acurácia Geral** | 99.38% |
| **F1-Score (Ponderado)** | 0.9938 |
| **Precisão (Ponderada)** | 0.9938 |
| **Suporte Total** | 10000.0 |
