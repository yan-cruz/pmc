import os
import math
import pandas as pd

PATH = os.path.join(os.getcwd(), 'datasets', 'validacao.xlsx')


def sigmoid(u):
    if u > 500: return 1.0
    if u < -500: return 0.0
    return 1.0 / (1.0 + math.exp(-u))


def limpar_validacao():
    df = pd.read_excel(PATH)
    colunas_y = [col for col in df.columns if str(col).startswith('y_T')]
    if colunas_y:
        df = df.drop(columns=colunas_y)
        df.to_excel(PATH, index=False)


def validar(W1, W2, treino):
    df = pd.read_excel(PATH)
    n_escondidos = len(W1)

    for index, row in df.iterrows():
        x_bias = [1.0, row['x1'], row['x2'], row['x3']]

        y_oculta = [1.0]
        for j in range(n_escondidos):
            u_j = sum(w * x for w, x in zip(W1[j], x_bias))
            y_oculta.append(sigmoid(u_j))

        u_k = sum(w * y for w, y in zip(W2[0], y_oculta))
        y_out = sigmoid(u_k)

        df.at[index, f'y_T{treino}'] = round(y_out, 4)

    df.to_excel(PATH, index=False)
