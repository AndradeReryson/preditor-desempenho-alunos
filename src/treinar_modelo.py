import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pathlib import Path
import traceback
import io

# Montando os caminhos relevantes
BASE_DIR = Path(__file__).resolve().parent

URL_DADOS = BASE_DIR.parent / 'data' / 'desempenho_alunos.csv'

NOME_ARQUIVO_MODELO = BASE_DIR.parent / 'models' / 'modelo_desempenho.pkl'
NOME_ARQUIVO_METRICAS = BASE_DIR.parent / 'models' / 'model_metrics.pkl'

def treinar_modelo():
    print(f"Importando dados da fonte...")
    
    # LACUNA 5: Use 'pd.read_csv()' para ler a 'URL_DADOS'
    try:
      data = pd.read_csv(URL_DADOS)
      data.dropna(inplace=True)
      
      print("\n--- Dados Carregados ---")
      print(data.head())

      print("\n--- Preparando dados para o treino ---")
      features = data.drop(columns=['ID','situacao'], axis=1)
      target = 'situacao'
      X = features
      y = data[target]

      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  
      modelo = DecisionTreeClassifier(random_state=42)
      modelo.fit(X_train, y_train)
      print(f"\n--- Modelo Treinado! Classes: {modelo.classes_} ---")
      
      joblib.dump(modelo, NOME_ARQUIVO_MODELO)
      print(f"--- Modelo salvo com sucesso ---")

      y_pred = modelo.predict(X_test)
      metrics_data = classification_report(y_test, y_pred, target_names=modelo.classes_, output_dict=True)
      joblib.dump(metrics_data, NOME_ARQUIVO_METRICAS) # Salvando as métricas para acesso no streamlit
      print("--- Métricas calculadas e salvas. ---")
    except FileNotFoundError:
      print(f'Erro FileNotFoundError: \n Arquivo CSV não encontrado... Rode o script "gerar_csv.py" para criá-lo!')
      traceback.print_exc()
    except Exception as any: 
      print(f'Erro: {any}')
      traceback.print_exc()

# (Bloco pronto)
if __name__ == "__main__":
    treinar_modelo()