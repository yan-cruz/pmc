import os
import pandas as pd

DOWNLOADS = os.path.join(os.path.expanduser('~'), 'Downloads')
DATASETS = os.path.join(os.getcwd(), 'datasets')

TREINO_XLS = os.path.join(DOWNLOADS, 'PP03_dados-treinamento.xls')
VAL_XLS = os.path.join(DOWNLOADS, 'PP03_dados-validacao.xls')


def preparar():
    os.makedirs(DATASETS, exist_ok=True)
    os.makedirs(os.path.join(os.getcwd(), 'graphics', 'Evolucao_do_erro'), exist_ok=True)
    os.makedirs(os.path.join(os.getcwd(), 'graphics', 'validacao'), exist_ok=True)

    df_treino = pd.read_excel(TREINO_XLS)
    df_val = pd.read_excel(VAL_XLS)

    df_treino.columns = df_treino.columns.str.strip().str.lower()
    df_val.columns = df_val.columns.str.strip().str.lower()

    df_treino.to_excel(os.path.join(DATASETS, 'treinamento.xlsx'), index=False)

    df_val.insert(0, 'Amostra', range(1, len(df_val) + 1))
    df_val.to_excel(os.path.join(DATASETS, 'validacao.xlsx'), index=False)

    pd.DataFrame({
        'Treinamento': [f'T{i}' for i in range(1, 6)],
        'Erro quadratico medio': [pd.NA] * 5,
        'Numero de epocas': [pd.NA] * 5,
    }).to_excel(os.path.join(DATASETS, 'resultados.xlsx'), index=False)

    pd.DataFrame({
        'Rede': [f'T{i}' for i in range(1, 6)],
        'Erro relativo medio (%)': [pd.NA] * 5,
        'Variancia (%)': [pd.NA] * 5,
    }).to_excel(os.path.join(DATASETS, 'metricas.xlsx'), index=False)

    print('Planilhas preparadas com sucesso!')
    print(f'  Treinamento: {len(df_treino)} amostras')
    print(f'  Validacao:   {len(df_val)} amostras')


if __name__ == '__main__':
    preparar()
