import pandas as pd
import os

RESULTADOS_PATH = os.path.join(os.getcwd(), './datasets/resultados.xlsx')


def limpar(df_resultados):
    for index, row in df_resultados.iterrows():
        df_resultados.at[index, 'W1-inicial'] = pd.NA
        df_resultados.at[index, 'W2-inicial'] = pd.NA
        df_resultados.at[index, 'W3-inicial'] = pd.NA

        df_resultados.at[index, 'W1-final'] = pd.NA
        df_resultados.at[index, 'W2-final'] = pd.NA
        df_resultados.at[index, 'W3-final'] = pd.NA
        df_resultados.at[index, 'Numero-de-epocas'] = pd.NA
        df_resultados.to_excel(RESULTADOS_PATH, index=False)

def preencher_w_iniciais(df_resultados, treinamento, pesos, limiarDeAtivacao):
    df_resultados.at[treinamento - 1, 'W0-inicial'] = limiarDeAtivacao
    df_resultados.at[treinamento - 1, 'W1-inicial'] = pesos[0]
    df_resultados.at[treinamento - 1, 'W2-inicial'] = pesos[1]
    df_resultados.at[treinamento - 1, 'W3-inicial'] = pesos[2]
    df_resultados.to_excel(RESULTADOS_PATH, index=False)

def preencher_w_finais(df_resultados, treinamento, pesos_fiais, epocas, limiarDeAtivacao):
    df_resultados.at[treinamento - 1, 'W0-final'] = limiarDeAtivacao
    df_resultados.at[treinamento - 1, 'W1-final'] = pesos_fiais[0]
    df_resultados.at[treinamento - 1, 'W2-final'] = pesos_fiais[1]
    df_resultados.at[treinamento - 1, 'W3-final'] = pesos_fiais[2]
    df_resultados.at[treinamento - 1, 'Numero-de-epocas'] = epocas
    df_resultados.to_excel(RESULTADOS_PATH, index=False)

