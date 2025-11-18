
import streamlit as st
import joblib
import pandas as pd
import os # (Já vem pronto, para verificar se o arquivo existe)
import random
import time
from pathlib import Path
from src.gerar_dados import config, get_random_float, get_random_int, get_random_bool, get_valor_ou_limite, calcular_dados_aluno

# Montando o PATH
BASE_DIR = Path(__file__).resolve()

# Mapeamento para Cor Favorita
OPCOES_COR = {
    "Não informado": 0,
    "Azul": 1,
    "Verde": 2,
    "Vermelho": 3,
    "Amarelo": 4,
    "Roxo": 5,
    "Preto": 6,
    "Branco": 7
}

# Mapeamento para Letra da Turma
OPCOES_TURMA = {
    "Turma A": 0,
    "Turma B": 1,
    "Turma C": 2,
    "Turma D": 3
}


# Caminho de arquivos necessários
NOME_ARQUIVO_MODELO = BASE_DIR.parent / 'models' / 'modelo_desempenho.pkl'
NOME_ARQUIVO_METRICAS = BASE_DIR.parent / 'models' / 'model_metrics.pkl'

# Variaveis globais do modelo e suas classes
MODELO = None
CLASSES_MODELO = None

@st.cache_resource # Cache para carregar o modelo apenas uma vez
def carregar_modelo(caminho_modelo):
  if not os.path.exists(caminho_modelo):
    return None, None
  try:
    modelo = joblib.load(caminho_modelo)
    classes_modelo = modelo.classes_
    return modelo, classes_modelo
  except Exception as e:
    st.error(f"Erro ao carregar o modelo: {e}")
    return None, None
    
@st.cache_data
def carregar_métricas(caminho_metricas):
  if not os.path.exists(caminho_metricas):
    return None
  try:
    metricas = joblib.load(caminho_metricas)
    return metricas
  except Exception as e:
    st.error(f"Erro ao carregar as métricas: {e}")
    return None

def sidebar():
  """
    Função auxiliar que constroi o sidebar da página
  """
  
  # --- Interface do Usuário (Entradas) na Sidebar ---
  st.sidebar.header('Insira os dados do Aluno:', width='stretch')

  tempo_deslocamento = st.sidebar.slider('Tempo de deslocamento até a escola (Minutos)', 15, 150, 60, 5, width='stretch')

  nota_p1 = st.sidebar.slider('Primeira nota do aluno (P1): ', 0.0, 10.0, 5.0, 0.1, width='stretch')

  # input e conversão para valor numérico da cor favorita
  cor_selecionada = st.sidebar.selectbox(
      "Selecione a Cor Favorita:",
      options=list(OPCOES_COR.keys()),
      width='stretch'
  )
  cod_cor_favorita = OPCOES_COR[cor_selecionada]

  # input e conversão para valor numérico da letra da turma
  turma_selecionada = st.sidebar.selectbox(
      "Selecione a Letra da Turma:",
      options=list(OPCOES_TURMA.keys()),
      width='stretch'
  )
  cod_letra_turma = OPCOES_TURMA[turma_selecionada]
  
  quant_irmaos = st.sidebar.slider('Quantidade de irmãos', 0, 4, 0, width='stretch')

  # BOTÃO PARA FAZER A PREVISÃO        
  if st.sidebar.button('Fazer Previsão', width='stretch'):
      with st.spinner('Processando...'):
        time.sleep(1)            
      
      # Montando dataframe com dados indenpendentes do aluno
      inputs_usuario = {
        'ID': 99999,
        'tempo_desloc_minutos': tempo_deslocamento,
        'nota_p1': nota_p1,
        'cod_cor_favorita': cod_cor_favorita,
        'quant_irmaos': quant_irmaos,
        'cod_letra_turma': cod_letra_turma,
      }

      # Calculando os dados restantes do aluno (dict) e convertendo em dataframe
      aluno_dict = calcular_dados_aluno(**inputs_usuario)
      dados_aluno = pd.DataFrame([aluno_dict])
      dados_aluno.drop(['ID', 'situacao'], axis=1, inplace=True)

      # Escrevendo os dados na tela
      st.header('Previsão de Situação do Aluno Informado', width='stretch')
      
      st.subheader('Dados do Aluno', width='stretch')
      st.dataframe(dados_aluno.T, width='stretch')
      
      # Fazendo a previsão
      prev = MODELO.predict(dados_aluno)
      probabilidades = MODELO.predict_proba(dados_aluno)
      
      if prev == 'reprovado':
          st.error("Aluno Reprovado!")
      else:
          st.success("Aluno Aprovado!")
          
      # (Bloco pronto para exibir métricas)
      st.subheader("Análise de Confiança da IA", )
      prob_aprovado = probabilidades[0][list(CLASSES_MODELO).index('aprovado')]
      prob_reprovado = probabilidades[0][list(CLASSES_MODELO).index('reprovado')]
      col1, col2 = st.columns(2)
      col1.metric("Confiança em 'Aprovado'", f"{prob_aprovado*100:.2f}%", width='stretch')
      col2.metric("Confiança em 'Reprovado'", f"{prob_reprovado*100:.2f}%", width='stretch')
      
  else:
      st.info("Informe os dados e Clique para fazer a previsão...")
   
def main():
    """
    Função principal que executa o App Streamlit.
    """
    global MODELO 
    global CLASSES_MODELO
    
    # --- Carregamento do Modelo e das métricas (variaveis globais) ---
    MODELO, CLASSES_MODELO = carregar_modelo(NOME_ARQUIVO_MODELO)
    metricas = carregar_métricas(NOME_ARQUIVO_METRICAS)

    st.title('Previsão de Desempenho de Alunos usando ML', width='stretch')
    st.subheader('Estudo de Caso da Imersão em IA (Aulas 1-3)', width='stretch')
    
    # Se o modelo não existir, exibe um aviso 
    if MODELO is None:
        st.error(f"Arquivo do modelo ('{NOME_ARQUIVO_MODELO}') não encontrado.")
        st.warning("Execute o script 'python train.py' no terminal para treinar e criar o modelo.")
        st.stop() # Para a execução do app

    # --- Exibindo as métricas do nosso modelo treinado ---
    if metricas:
      st.markdown("---")
      st.header("📈 Desempenho do Modelo na Base de Testes (20%)")

      # --- Seção 1: Métricas de Resumo Geral ---
      st.subheader("1. Resumo Geral")
      st.caption("A média ponderada ('Weighted Avg') é mais relevante, pois nossa base é desbalanceada.")

      # 1.1 Exibição de métricas de resumo
      col1, col2, col3, col4 = st.columns(4)

      with col1:
        st.metric(label="Acurácia Geral", 
                  value=f"{metricas['accuracy'] * 100:.2f}%",
                  help="Proporção de todas as previsões corretas.")

      with col2:
          st.metric(label="F1-Score (Ponderado)", 
                    value=f"{metricas['weighted avg']['f1-score']:.4f}",
                    help="Média F1 de todas as classes, ponderada pelo suporte (ideal para classes desbalanceadas).")
      
      with col3:
          st.metric(label="Precisão (Ponderada)", 
                    value=f"{metricas['weighted avg']['precision']:.4f}")
      
      with col4:
          st.metric(label="Suporte Total", 
                    value=f"{metricas['weighted avg']['support']}", 
                    help="Número total de amostras no conjunto de testes (20%).")
      
      st.markdown("---")

      # --- Seção 2: Métricas Detalhadas por Classe (DataFrame) ---
      st.subheader("2. Detalhe por Classe (Aprovado vs. Reprovado)")

      # Extrai as métricas para as classes 'aprovado' e 'reprovado'
      class_metrics = {
          'Métrica': ['Precisão', 'Recall', 'F1-Score', 'Suporte'],
          'Aprovado': [
              metricas['aprovado']['precision'],
              metricas['aprovado']['recall'],
              metricas['aprovado']['f1-score'],
              metricas['aprovado']['support']
          ],
          'Reprovado': [
              metricas['reprovado']['precision'],
              metricas['reprovado']['recall'],
              metricas['reprovado']['f1-score'],
              metricas['reprovado']['support']
          ]
      }
      
      df_metrics = pd.DataFrame(class_metrics).set_index('Métrica')

      # Aplica formatação de duas casas decimais para floats
      st.dataframe(df_metrics.style.format("{:.4f}"), use_container_width=True)

      # Destaque para o F1-Score da classe positiva (Aprovado)
      st.caption(f"**F1-Score para 'Aprovado': {metricas['aprovado']['f1-score']:.4f}**")

      st.markdown("---")

    # --- Carregando código da sidebar --- 
    sidebar()
    
    st.markdown("---")
    st.write("Este App foi construído no curso de Programação em IA Generativa.")

# (Bloco pronto)
if __name__ == "__main__":
    random.seed(42)
    main()