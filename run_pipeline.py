import subprocess
import sys
import os

# ----------------------------------------------------
# 1. FUNÇÃO AUXILIAR PARA RODAR SCRIPTS
# ----------------------------------------------------
def run_script(script_path, name):
    """Executa um script Python e trata possíveis erros."""
    print(f"\n========================================================")
    print(f"🚀 INICIANDO {name}...")
    print(f"========================================================")
    
    # O comando usa o interpretador Python atual (sys.executable)
    # e o caminho do script.
    result = subprocess.run([sys.executable, script_path], check=True)
    
    if result.returncode == 0:
        print(f"✅ {name} CONCLUÍDO com sucesso.")
    else:
        # Isso só deve acontecer se 'check=True' falhar (retcode != 0), 
        # mas é uma boa prática
        print(f"❌ ERRO ao executar {name}. Código de retorno: {result.returncode}")
        sys.exit(1)

# ----------------------------------------------------
# 2. FLUXO PRINCIPAL DO PIPELINE
# ----------------------------------------------------
if __name__ == "__main__":
    
    # Define os caminhos dos scripts (relativos à pasta raiz, onde este script será executado)
    GERAR_DADOS_SCRIPT = os.path.join('src', 'gerar_dados.py')
    TREINAR_MODELO_SCRIPT = os.path.join('src', 'treinar_modelo.py')
    APP_STREAMLIT_SCRIPT = os.path.join('app.py')

    try:
        # 1. Geração de Dados (Gerar CSV)
        run_script(GERAR_DADOS_SCRIPT, "Geração de Dados")
        
        # 2. Treinamento do Modelo (Cria os PKLs)
        run_script(TREINAR_MODELO_SCRIPT, "Treinamento do Modelo")
        
        # 3. Inicia o Aplicativo Streamlit
        print(f"\n========================================================")
        print(f"🌐 INICIANDO STREAMLIT APP...")
        print(f"========================================================")
        
        # Para Streamlit, use o comando "streamlit run"
        subprocess.run(['streamlit', 'run', APP_STREAMLIT_SCRIPT], check=True)

    except subprocess.CalledProcessError as e:
        print(f"\n❌ PIPELINE INTERROMPIDO. Detalhes: {e}")
    except FileNotFoundError:
        print("\n❌ ERRO: Verifique se as pastas 'src' e 'data' existem e se os nomes dos arquivos estão corretos.")