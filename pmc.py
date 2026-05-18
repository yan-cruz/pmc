import pandas as pd
import random
import os
import math
import matplotlib.pyplot as plt
from math import sqrt
from classificar import validar

# NOTA: O arquivo resultados.xlsx agora é manipulado diretamente via Pandas abaixo,
# dispensando a necessidade de atualizar o módulo externo resultados.py.
# import classificar
# import metricas

LOCAL_PATH = os.path.join(os.getcwd(), './datasets/treinamento.xlsx')
RESULTADOS_PATH = os.path.join(os.getcwd(), './datasets/resultados.xlsx')

df_treinamento = pd.read_excel(LOCAL_PATH)
# DESCOMENTADO: Carrega a planilha de resultados globalmente antes do loop
df_resultados = pd.read_excel(RESULTADOS_PATH)

# ==========================================
# FUNÇÕES DE ATIVAÇÃO
# ==========================================
def sigmoid(u):
    if u > 500: return 1.0
    if u < -500: return 0.0
    return 1.0 / (1.0 + math.exp(-u))

def derivada_sigmoid(y):
    return y * (1.0 - y)

# ==========================================
# EXTRAÇÃO DE DADOS (Otimização de Velocidade)
# ==========================================
df_treinamento['x1'] = df_treinamento['x1'].astype(str).str.replace('_x000d_', '').astype(float)
df_treinamento['x2'] = df_treinamento['x2'].astype(str).str.replace('_x000d_', '').astype(float)
df_treinamento['x3'] = df_treinamento['x3'].astype(str).str.replace('_x000d_', '').astype(float)
df_treinamento['d'] = df_treinamento['d'].astype(str).str.replace('_x000d_', '').astype(float)

x1_list = df_treinamento['x1'].tolist()
x2_list = df_treinamento['x2'].tolist()
x3_list = df_treinamento['x3'].tolist()
d_list = df_treinamento['d'].tolist()

X_treino = []
for i in range(len(x1_list)):
    X_treino.append([x1_list[i], x2_list[i], x3_list[i]])

# ==========================================
# PARÂMETROS DO PMC 
# ==========================================
taxaDeAprendizagem = 0.1
precisao = 1e-6
n_escondidos = 10
n_entradas = 3

treinamento = 1

# LÓGICA DE LIMPEZA: Identifica e limpa os dados antigos antes de iniciar os treinamentos
col_mse = 'erro-quadratico-medio'
col_epocas = 'Numero-de-epocas'

df_resultados[col_mse] = None
df_resultados[col_epocas] = None

while treinamento <= 5:
    epocas = 0
    rmse_por_epoca = []
    mse_anterior = float('inf')

    # INICIALIZAÇÃO DE PESOS (0 a 1)
    W1 = [[random.uniform(0, 1) for _ in range(n_entradas + 1)] for _ in range(n_escondidos)]
    W2 = [random.uniform(0, 1) for _ in range(n_escondidos + 1)]

    while epocas < 1000:
        epocas += 1
        soma_erro_quadratico = 0

        for i in range(len(X_treino)):
            x_atual = [1.0] + X_treino[i]
            d = d_list[i]
            
            # Forward
            y_oculta = [1.0]
            for j in range(n_escondidos):
                u_j = sum(w * x for w, x in zip(W1[j], x_atual))
                y_oculta.append(sigmoid(u_j))
                
            u_saida = sum(w * y for w, y in zip(W2, y_oculta))
            y_final = sigmoid(u_saida)
            
            # Backward
            erro = d - y_final
            soma_erro_quadratico += (erro ** 2)
            
            delta_saida = erro * derivada_sigmoid(y_final)
            
            delta_oculta = []
            for j in range(n_escondidos):
                soma_deltas = delta_saida * W2[j + 1]
                delta_j = soma_deltas * derivada_sigmoid(y_oculta[j + 1])
                delta_oculta.append(delta_j)
                
            for j in range(len(W2)):
                W2[j] = W2[j] + taxaDeAprendizagem * delta_saida * y_oculta[j]
                
            for j in range(n_escondidos):
                for k in range(len(x_atual)):
                    W1[j][k] = W1[j][k] + taxaDeAprendizagem * delta_oculta[j] * x_atual[k]

        mse_atual = soma_erro_quadratico / len(X_treino)
        rmse_epoca = sqrt(mse_atual)
        rmse_por_epoca.append(rmse_epoca)

        if abs(mse_atual - mse_anterior) <= precisao:
            break
            
        mse_anterior = mse_atual

    # ==========================================
    # SALVAR RESULTADOS NA PLANILHA DE TREINAMENTO
    # ==========================================
    for index, row in df_treinamento.iterrows():
        x_atual = [1.0, row['x1'], row['x2'], row['x3']]
        
        y_oculta = [1.0]
        for j in range(n_escondidos):
            u_j = sum(w * x for w, x in zip(W1[j], x_atual))
            y_oculta.append(sigmoid(u_j))
            
        u_saida = sum(w * y for w, y in zip(W2, y_oculta))
        y_final = sigmoid(u_saida)
        
        df_treinamento.at[index, f'Y_{treinamento}'] = y_final
        
    df_treinamento.to_excel(LOCAL_PATH, index=False)
    print(f'Treinamento {treinamento} concluído em {epocas} épocas. Erro Final: {mse_atual:.6f}')

    # ==========================================
    # LÓGICA ADICIONADA: SALVAR NA PLANILHA DE RESULTADOS (TABELA 1)
    # ==========================================
    # Mapeia a linha correspondente ao índice (T1 -> linha 0, T2 -> linha 1...)
    idx_linha = treinamento - 1
    df_resultados.at[idx_linha, col_mse] = mse_atual
    df_resultados.at[idx_linha, col_epocas] = epocas
    
    # Grava fisicamente as atualizações no arquivo resultados.xlsx
    df_resultados.to_excel(RESULTADOS_PATH, index=False)
    # ==========================================

    # Gráfico de Evolução do Erro
    plt.figure(figsize=(10, 6))
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

    if epocas < 1000:
     validar(W1, W2, treinamento)    
            
    treinamento += 1

