import os
import random
import math
import pandas as pd
import matplotlib.pyplot as plt

import resultados
import classificar
import metricas

TREINAMENTO_PATH = os.path.join(os.getcwd(), 'datasets', 'treinamento.xlsx')
GRAFICOS_ERRO_PATH = os.path.join(os.getcwd(), 'graphics', 'Evolucao_do_erro')

ETA = 0.1
EPSILON = 1e-6
MAX_EPOCAS = 50000
N_ENTRADAS = 3
N_ESCONDIDOS = 10


def sigmoid(u):
    if u > 500: return 1.0
    if u < -500: return 0.0
    return 1.0 / (1.0 + math.exp(-u))


def derivada_sigmoid(y):
    return y * (1.0 - y)


def forward(W1, W2, x_bias):
    y_oculta = [1.0]
    for j in range(N_ESCONDIDOS):
        u_j = sum(w * x for w, x in zip(W1[j], x_bias))
        y_oculta.append(sigmoid(u_j))

    u_k = sum(w * y for w, y in zip(W2[0], y_oculta))
    y_saida = sigmoid(u_k)

    return y_oculta, y_saida


def calcular_eqm(X, D, W1, W2):
    total = 0.0
    for x, d in zip(X, D):
        _, y = forward(W1, W2, [1.0] + x)
        total += 0.5 * (d - y) ** 2
    return total / len(X)


def plotar_eqm(eqm_por_epoca, treinamento):
    os.makedirs(GRAFICOS_ERRO_PATH, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(eqm_por_epoca) + 1), eqm_por_epoca, linewidth=1)
    ax.set_title(f'Evolução do Erro - Treinamento {treinamento}')
    ax.set_xlabel('Épocas')
    ax.set_ylabel('Erro Quadrático Médio (EQM)')
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(GRAFICOS_ERRO_PATH, f'treinamento_pmc_{treinamento}.png'))
    plt.close(fig)


def treinar():
    df = pd.read_excel(TREINAMENTO_PATH)
    X = df[['x1', 'x2', 'x3']].values.tolist()
    D = df['d'].tolist()

    resultados.limpar()
    classificar.limpar_validacao()

    for treinamento in range(1, 6):
        print(f'\n--- Iniciando Treinamento {treinamento} ---')

        # Pesos inicializados entre 0 e 1 conforme roteiro
        W1 = [[random.uniform(0, 1) for _ in range(N_ENTRADAS + 1)] for _ in range(N_ESCONDIDOS)]
        W2 = [[random.uniform(0, 1) for _ in range(N_ESCONDIDOS + 1)]]

        eqm_anterior = float('inf')
        eqm_por_epoca = []
        epocas = 0

        for epoca in range(1, MAX_EPOCAS + 1):
            epocas = epoca

            indices = list(range(len(X)))
            random.shuffle(indices)

            for i in indices:
                x_bias = [1.0] + X[i]
                d_atual = D[i]

                y_oculta, y_saida = forward(W1, W2, x_bias)

                erro = d_atual - y_saida
                delta_saida = erro * derivada_sigmoid(y_saida)

                delta_oculta = [
                    delta_saida * W2[0][j + 1] * derivada_sigmoid(y_oculta[j + 1])
                    for j in range(N_ESCONDIDOS)
                ]

                for j_idx in range(len(y_oculta)):
                    W2[0][j_idx] += ETA * delta_saida * y_oculta[j_idx]

                for j in range(N_ESCONDIDOS):
                    for m in range(len(x_bias)):
                        W1[j][m] += ETA * delta_oculta[j] * x_bias[m]

            eqm_atual = calcular_eqm(X, D, W1, W2)
            eqm_por_epoca.append(eqm_atual)

            if abs(eqm_atual - eqm_anterior) <= EPSILON:
                break

            eqm_anterior = eqm_atual

        print(f'Treinamento {treinamento} concluído em {epocas} épocas. EQM: {eqm_atual:.8f}')

        resultados.preencher(treinamento, eqm_atual, epocas)
        plotar_eqm(eqm_por_epoca, treinamento)
        classificar.validar(W1, W2, treinamento)

    print('\nGerando gráficos de validação e métricas...')
    metricas.calcular()
    print('\nProcesso finalizado com sucesso!')


if __name__ == '__main__':
    treinar()
