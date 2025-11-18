import pandas as pd
import numpy as np
import random
import joblib
import heapq
from pathlib import Path

"""
  1. COLUNAS DE REGRAS REAIS QUE VAMOS CRIAR
    - ID                          (INT) valor de identificação do aluno
    - tempo_desloc_minutos        (INT) tempo de deslocamento em minutos do aluno até a escola
    - faltas                      (INT) total de faltas. Influenciada pelo tempo de deslocamento
    - horas_estudo                (INT) quantidade de horas de estudo. Influenciada pelas faltas
    - fez_atividade_extra         (BOOL) indica se o aluno fez a atividade extra
    - nota_p1                     (FLOAT) a primeira nota do aluno. Influenciada pelo tempo de estudo e trabalho em grupo
    - nota_p2                     (FLOAT) a segunda nota do aluno. Influenciada pelo tempo de estudo e atividade individual
    - nota_p3                     (FLOAT) a nota de recuperacao do aluno. Pode substituir uma das duas notas mas só usa como influencia as horas de estudo

  2. COLUNAS DE RUIDO QUE NÃO AFETAM O RESULTADO FINAL
    - cod_cor_favorita    (INT) numero de 0 a 7 que representa a cor favorita do aluno
    - quant_irmaos        (INT) quantidade de irmãos que o aluno tem
    - letra_turma         (STRING) letra da turma (A, B, C ou D)
"""


# __________ Montando os caminhos relevantes _______________

BASE_DIR = Path(__file__).resolve().parent

URL_SAIDA_DADOS = BASE_DIR.parent / 'data' / 'desempenho_alunos.csv'

# __________ VARIAVEIS DE ALEATORIEDADE E LIMITES __________

config = {
  "RND_MIN_FALTAS": 0,
  "RND_MAX_FALTAS": 10,
  "TOTAL_MAX_FALTAS": 20,
  "RND_MIN_HORAS_ESTUDO": 10,
  "RND_MAX_HORAS_ESTUDO": 50,
  "PROB_BAIXA_FAZER_ATV": [0.3, 0.7],
  "PROB_ALTA_FAZER_ATV": [0.5, 0.5],
  "PONTOS_ATIVIDADE": 2,
  "RND_MIN_NOTA_P2": 0,
  "RND_MAX_NOTA_P2": 5,
  "P2_DIVISOR_HORAS_ESTUDO": 8,
  "LIMITE_RECUPERACAO": 5,
  "RND_MIN_NOTA_P3": 0,
  "RND_MAX_NOTA_P3": 5,
  "P3_DIVISOR_HORAS_ESTUDO": 8,
  "MEDIA_CORTE": 6
}

# __________ FUNÇÕES _____________________________

def get_random_float(start, end):
  return round(random.uniform(start, end+0.01), 2)

def get_random_int(start, end):
  return random.randint(start, end)

def get_random_bool():
   return random.randint(0, 1)

def get_valor_ou_limite(valor, limite):
   return min(valor, limite)

def calcular_dados_aluno(ID,
                         tempo_desloc_minutos, 
                         nota_p1,
                         cod_cor_favorita, 
                         quant_irmaos,
                         cod_letra_turma):
  """Recebe alguns dados independentes sobre um aluno e calcula outros dados dependentes, retornando um dict que representa o aluno

  Args:
    ID (int): Numero de identificacao do aluno.
    tempo_desloc_minutos (int): Numero que representa a quantidade de minutos que o aluno estuda.
    nota_p1 (float): Numero que representa a primeira nota do aluno.
    cod_cor_favorita (int): Numero que representa a cor favorita do aluno.
    quant_irmaos (int): Quantidade de irmãos que o aluno possui.
    fez_atividade_extra (int): Numero que indica se o aluno fez ou não a atividade extra.
    cod_letra_turma (int): Numero que representa a letra da turma a qual o aluno faz parte.

  Returns:
    dict: Um dicionário contendo todos os dados (independentes e calculados) de um aluno.
  """

  # Com base no deslocamento, adicionamos algumas faltas a mais:
  # _ 0 a 10 faltas aleatórias
  # + um décimo arredondado para cima do tempo de deslocamento 
  faltas = (
    get_random_int(config['RND_MIN_FALTAS'], config['RND_MAX_FALTAS'])
    + round(tempo_desloc_minutos / 10)
  )

  # definindo se a nota_p1 foi alta ou baixa para influenciar a chance de fazer o trabalho
  nota_p1_alta = 1 if nota_p1 > 6 else 0

  # horas de estudo será um valor aleatório o qual sofrerá uma subtração baseado no numero de faltas do aluno
  horas_estudo = (
    get_random_int(config['RND_MIN_HORAS_ESTUDO'], config['RND_MAX_HORAS_ESTUDO']) 
    - round(faltas / 2)
  )
  horas_estudo = np.clip(horas_estudo, 0, None) # Limita o valor mínimo em 0.

  # Se a Nota_P1 não for alta, o aluno ganha + 25% de horas de estudo
  if not nota_p1_alta:
    horas_estudo += round(horas_estudo * 0.25) 
  
  # Se a Nota_P1 não for alta, o aluno tem mais chance de fazer a atividade extra
  if nota_p1_alta:
    fez_atividade_extra = random.choices([0, 1], weights=config['PROB_BAIXA_FAZER_ATV'], k=1)[0]
  else:
    fez_atividade_extra = random.choices([0, 1], weights=config['PROB_ALTA_FAZER_ATV'], k=1)[0]

  
  # Definindo notas p1 e p2 pelas notas base + horas_estudo + trabalho ou atividade
  nota_p2 = (
    get_random_float(config['RND_MIN_NOTA_P2'], config['RND_MAX_NOTA_P2']) 
    + (horas_estudo / config['P2_DIVISOR_HORAS_ESTUDO']) 
    + fez_atividade_extra * config['PONTOS_ATIVIDADE']
  )
  nota_p2 = round(nota_p2, 2)
  nota_p2 = get_valor_ou_limite(nota_p2, 10.00)

  # Nota_p3 é a nota de recuperação
  nota_p3 = -1

  # Definindo se o aluno fica ou não de recuperação
  # Se a p1 ou p2 for menor que tres, o aluno pode fazer a recuperacao e obter uma nota p3 que substituirá a menor nota
  recuperacao = None
  if nota_p1 < config['LIMITE_RECUPERACAO'] or nota_p2 < config['LIMITE_RECUPERACAO']:
      recuperacao = 1
      horas_estudo += 20
      nota_p3 = (
        get_random_float(config['RND_MIN_NOTA_P3'], config['RND_MAX_NOTA_P3']) 
        + (horas_estudo / config['P3_DIVISOR_HORAS_ESTUDO'])
      )
      nota_p3 = round(nota_p3, 2)
      nota_p3 = get_valor_ou_limite(nota_p3, 10.00)
  else:
      recuperacao = 0

  # _____________ Calculando se aluno foi aprovado ou não ________________

  list_notas = [nota_p1, nota_p2, nota_p3]
  maiores_notas = heapq.nlargest(2, list_notas)
  media = round(sum(maiores_notas) / 2, 2)
  
  situacao = None
  if faltas > config['TOTAL_MAX_FALTAS'] or media < config['MEDIA_CORTE']:
    situacao = 'reprovado'
  else:
    situacao = 'aprovado' 

  # _____________ Montando o dict do aluno _______________________________

  aluno = {
    'ID': ID,
    'tempo_desloc_minutos': tempo_desloc_minutos,
    'faltas': faltas,
    'cod_cor_favorita': cod_cor_favorita,
    'quant_irmaos': quant_irmaos,
    'horas_estudo': horas_estudo,
    'fez_atividade_extra': fez_atividade_extra,
    'cod_letra_turma': cod_letra_turma,
    'nota_p1': nota_p1,
    'nota_p2': nota_p2,
    'nota_p3': nota_p3,
    'recuperacao': recuperacao,
    'situacao': situacao
  }

  return aluno

def gerar_registros(quant_registros):
  """Gera N registros de alunos e retorna uma lista

  Args:
      quant_registros (int): The length of the rectangle.

  Returns:
      list: Lista contendo a quantidade de alunos especificada.
  """
  
  LISTA_ALUNOS = []
  
  for i in range(0, quant_registros):
    # Definindo um tempo de deslocamento aleatorio
    tempo_desloc_minutos = get_random_int(15, 150)

    # Definindo a nota_p1 do aluno
    nota_p1 = get_random_float(0, 10)
    
    # _____________ Adicionando colunas de "barulho" _______________________
    
    cod_cor_favorita = get_random_int(0, 7)

    cod_letra_turma = get_random_int(0, 3)

    quant_irmaos = get_random_int(0, 4)

    aluno = calcular_dados_aluno(i, tempo_desloc_minutos, nota_p1, cod_cor_favorita, quant_irmaos, cod_letra_turma)

    LISTA_ALUNOS.append(aluno)

  return LISTA_ALUNOS


if __name__ == "__main__":
  random.seed(42)
  lista_registros = gerar_registros(50000)

  df = pd.DataFrame(lista_registros)
  df.to_csv(URL_SAIDA_DADOS, index=False)

  print(f'\n --- RESULTADO SITUACAO --- \n'\
        f'{df.groupby('situacao')['situacao'].count()}\n')

  print(f'--- Dados gerados e salvos com sucesso em formato CSV ---\n'\
        f'--- Nome do arquivo: "desempenho_alunos.csv" ---\n'\
        f'--- Função "Calculdora_features.pkl" salva com sucesso ---\n')
  print(df.head(5))
