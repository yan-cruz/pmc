import pandas as pd
import os
import math

PATH = os.path.join(os.getcwd(), './datasets/validacao.xlsx')

# ==========================================
# FUNÇÃO DE ATIVAÇÃO
# ==========================================
def sigmoid(u):
    if u > 500: return 1.0
    if u < -500: return 0.0
    return 1.0 / (1.0 + math.exp(-u))

# ==========================================
# FUNÇÃO DE VALIDAÇÃO PMC
# ==========================================
def validar(W1, W2, treino):
    # Carregamos a planilha a cada chamada para não sobrescrever na memória
    df_validacao = pd.read_excel(PATH)
    
    # Cria a coluna nova para este treinamento específico (Y_T1, Y_T2...)
    df_validacao[f'Y_T{treino}'] = pd.NA 

    # O número de neurônios escondidos é o tamanho da matriz W1
    n_escondidos = len(W1)

    for index, row in df_validacao.iterrows():
        # Forçamos a leitura como float e já adicionamos o Bias (1.0) na posição 0
        x_atual = [1.0, float(row['x1']), float(row['x2']), float(row['x3'])]
        
        # ----------------------------------------------------
        # PASSO FORWARD (Processamento do sinal através da rede)
        # ----------------------------------------------------
        
        # 1. Passa pelas entradas da Camada Oculta
        y_oculta = [1.0] # Inicia com o Bias (1.0) para a próxima camada
        for j in range(n_escondidos):
            u_j = sum(w * x for w, x in zip(W1[j], x_atual))
            y_oculta.append(sigmoid(u_j))
            
        # 2. Passa para a Camada de Saída
        u_saida = sum(w * y for w, y in zip(W2, y_oculta))
        y_final = sigmoid(u_saida) # Resultado contínuo da estimativa
        
        # ----------------------------------------------------
        
        # Salva a estimativa de energia na coluna correta
        df_validacao.at[index, f'Y_T{treino}'] = y_final
  
    # Grava fisicamente a planilha atualizada no disco
    df_validacao.to_excel(PATH, index=False)
    print(f"Validação do Treinamento {treino} salva com sucesso!")