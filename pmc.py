import pandas as pd
import random
import os
import math
import matplotlib.pyplot as plt
from math import sqrt
from classificar import validar

LOCAL_PATH = os.path.join(os.getcwd(), './datasets/treinamento.xlsx')
RESULTADOS_PATH = os.path.join(os.getcwd(), './datasets/resultados.xlsx')

# ==========================================
# 1. FUNÇÕES DE ATIVAÇÃO
# ==========================================
def sigmoid(u):
    if u > 500: return 1.0
    if u < -500: return 0.0
    return 1.0 / (1.0 + math.exp(-u))

def derivada_sigmoid(y):
    return y * (1.0 - y)

# ==========================================
# 2. LEITURA DE DADOS
# ==========================================
df_treinamento = pd.read_excel(LOCAL_PATH)
df_resultados = pd.read_excel(RESULTADOS_PATH)

# Limpa cabeçalhos ocultos para evitar erros de leitura
df_treinamento.columns = df_treinamento.columns.str.strip().str.replace('_x000d_', '')

colunas_x = ['x1', 'x2', 'x3', 'x4']
colunas_d = ['d1', 'd2', 'd3']

# Converte os dados para float
for col in colunas_x + colunas_d:
    df_treinamento[col] = df_treinamento[col].astype(str).str.replace('_x000d_', '').astype(float)

X_treino = df_treinamento[colunas_x].values.tolist()
D_treino = df_treinamento[colunas_d].values.tolist()

# ==========================================
# 3. PARÂMETROS DO PROJETO (PP04)
# ==========================================
taxaDeAprendizagem = 0.1
precisao = 1e-6
n_entradas = 4
n_escondidos = 15
n_saidas = 3

col_mse = 'erro-quadratico-medio'
col_epocas = 'Numero-de-epocas'

df_resultados[col_mse] = None
df_resultados[col_epocas] = None

# ==========================================
# 4. LAÇO PRINCIPAL DOS 5 TREINAMENTOS
# ==========================================
for treinamento in range(1, 6):
    print(f"\n--- A iniciar Treinamento {treinamento} ---")
    
    epocas = 0
    rmse_por_epoca = []
    mse_anterior = float('inf')

    # Inicialização entre -0.5 e 0.5
    W1 = [[random.uniform(-0.5, 0.5) for _ in range(n_entradas + 1)] for _ in range(n_escondidos)]
    W2 = [[random.uniform(-0.5, 0.5) for _ in range(n_escondidos + 1)] for _ in range(n_saidas)]

    while epocas < 10000:
        epocas += 1
        soma_erro_quadratico = 0

        # Embaralha a ordem de leitura para evitar ciclos viciosos (Stochastic Gradient Descent)
        indices = list(range(len(X_treino)))
        random.shuffle(indices)

        for i in indices:
            x_atual = [1.0] + X_treino[i]
            d_atual = D_treino[i]
            
            # --- FORWARD ---
            y_oculta = [1.0]
            for j in range(n_escondidos):
                u_j = sum(w * x for w, x in zip(W1[j], x_atual))
                y_oculta.append(sigmoid(u_j))
                
            y_final = []
            for k in range(n_saidas):
                u_k = sum(w * y for w, y in zip(W2[k], y_oculta))
                y_final.append(sigmoid(u_k))
            
            # --- BACKWARD ---
            deltas_saida = []
            for k in range(n_saidas):
                erro_k = d_atual[k] - y_final[k]
                
                # O multiplicador 0.5 pertence à fórmula do erro
                soma_erro_quadratico += 0.5 * (erro_k ** 2) 
                deltas_saida.append(erro_k * derivada_sigmoid(y_final[k]))
            
            delta_oculta = []
            for j in range(n_escondidos):
                soma_deltas = sum(deltas_saida[k] * W2[k][j + 1] for k in range(n_saidas))
                delta_oculta.append(soma_deltas * derivada_sigmoid(y_oculta[j + 1]))
                
            # --- ATUALIZAÇÃO DOS PESOS ---
            for k in range(n_saidas):
                for j in range(len(y_oculta)):
                    W2[k][j] += taxaDeAprendizagem * deltas_saida[k] * y_oculta[j]
                
            for j in range(n_escondidos):
                for m in range(len(x_atual)):
                    W1[j][m] += taxaDeAprendizagem * delta_oculta[j] * x_atual[m]

        # --- AVALIAÇÃO DO CRITÉRIO DE PARADA ---
        mse_atual = soma_erro_quadratico / len(X_treino)
        rmse_por_epoca.append(sqrt(mse_atual))

        if abs(mse_atual - mse_anterior) <= precisao:
            break  # Interrompe o ciclo 'while' se a rede estagnar
            
        mse_anterior = mse_atual

    print(f"Treinamento concluído em {epocas} épocas. Erro Final: {mse_atual:.6f}")

    # ==========================================
    # 5. SALVAR DADOS E PÓS-PROCESSAMENTO
    # ==========================================
    for index, row in df_treinamento.iterrows():
        x_atual = [1.0, row['x1'], row['x2'], row['x3'], row['x4']]
        
        y_oculta = [1.0]
        for j in range(n_escondidos):
            u_j = sum(w * x for w, x in zip(W1[j], x_atual))
            y_oculta.append(sigmoid(u_j))
            
        for k in range(n_saidas):
            u_k = sum(w * y for w, y in zip(W2[k], y_oculta))
            y_calc = sigmoid(u_k)
            
            # Filtro Binário: se >= 0.5 regista 1, senão regista 0
            y_binario = 1 if y_calc >= 0.5 else 0
            
            df_treinamento.at[index, f'Y{k+1}_T{treinamento}'] = y_binario

    df_treinamento.to_excel(LOCAL_PATH, index=False)

    # Regista os dados na Tabela 1
    idx_linha = treinamento - 1
    df_resultados.at[idx_linha, col_mse] = mse_atual
    df_resultados.at[idx_linha, col_epocas] = epocas
    df_resultados.to_excel(RESULTADOS_PATH, index=False)

    # Cria a curva de erro
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(rmse_por_epoca)+1), rmse_por_epoca)
    plt.title(f'Curva de Aprendizado PMC - Treinamento {treinamento}')
    plt.xlabel('Épocas')
    plt.ylabel('RMSE')
    plt.grid(True)
    
    grafico_dir = './graphics/Evolucao_do_erro/'
    if not os.path.exists(grafico_dir):
        os.makedirs(grafico_dir)
    plt.savefig(f'{grafico_dir}treinamento_pmc_{treinamento}.png')
    plt.close()

    # Chama o classificar.py para prever a folha de cálculo de validação
    validar(W1, W2, treinamento)

print("\nProcesso Finalizado com Sucesso!")