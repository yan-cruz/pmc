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
# FUNÇÃO DE VALIDAÇÃO (PP04)
# ==========================================
def validar(W1, W2, treino):
    df_validacao = pd.read_excel(PATH)
    
    # Limpa cabeçalhos ocultos do Excel
    df_validacao.columns = df_validacao.columns.str.strip().str.replace('_x000d_', '')
    
    colunas_x = ['x1', 'x2', 'x3', 'x4']
    
    # Converte para float
    for col in colunas_x:
        df_validacao[col] = df_validacao[col].astype(str).str.replace('_x000d_', '').astype(float)
        
    n_escondidos = len(W1)
    n_saidas = len(W2) # Agora ele sabe que W2 tem 3 linhas (3 saídas)
    
    for index, row in df_validacao.iterrows():
        x_atual = [1.0, row['x1'], row['x2'], row['x3'], row['x4']]
        
        # --- FORWARD (Camada Oculta) ---
        y_oculta = [1.0]
        for j in range(n_escondidos):
            u_j = sum(w * x for w, x in zip(W1[j], x_atual))
            y_oculta.append(sigmoid(u_j))
            
        # --- FORWARD (Camada de Saída) ---
        for k in range(n_saidas):
            # A CORREÇÃO ESTÁ AQUI: W2[k] acede à linha correta da matriz
            u_k = sum(w * y for w, y in zip(W2[k], y_oculta))
            y_calc = sigmoid(u_k)
            
            # Pós-processamento: Filtro Binário
            y_binario = 1 if y_calc >= 0.5 else 0
            
            df_validacao.at[index, f'Y{k+1}_T{treino}'] = y_binario
  
    # Grava fisicamente na folha de cálculo
    df_validacao.to_excel(PATH, index=False)