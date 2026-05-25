import os
import pandas as pd

RESULTADOS_PATH = os.path.join(os.getcwd(), 'datasets', 'resultados.xlsx')


def _criar_template():
    return pd.DataFrame({
        'Treinamento': [f'T{i}' for i in range(1, 6)],
        'Erro quadratico medio': [pd.NA] * 5,
        'Numero de epocas': [pd.NA] * 5,
    })


def limpar():
    os.makedirs(os.path.dirname(RESULTADOS_PATH), exist_ok=True)
    _criar_template().to_excel(RESULTADOS_PATH, index=False)


def preencher(treinamento, mse, epocas):
    df = pd.read_excel(RESULTADOS_PATH)
    df.at[treinamento - 1, 'Erro quadratico medio'] = mse
    df.at[treinamento - 1, 'Numero de epocas'] = epocas
    df.to_excel(RESULTADOS_PATH, index=False)
