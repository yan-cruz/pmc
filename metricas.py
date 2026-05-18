import pandas as pd
import os 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

PATH = os.path.join(os.getcwd(), './datasets/validacao.xlsx')
METRICAS_PATH = os.path.join(os.getcwd(), './datasets/metricas.xlsx')

df_validacao = pd.read_excel(PATH)
df_metricas = pd.read_excel(METRICAS_PATH)

def calcular():
    d = df_validacao['d']
    redes = ['Y_T1', 'Y_T2', 'Y_T3', 'Y_T4', 'Y_T5']


    for i, rede in enumerate(redes, start=1):
      y_predito = df_validacao[rede]
      matizDeConfusao = confusion_matrix(d, y_predito, labels=[-1, 1])
      verdadeiroNegativo, falsoPositivo, falsoNegativo, verdadeiroPositivo = matizDeConfusao.ravel()

      
      acuracia = (verdadeiroPositivo + verdadeiroNegativo) / len(d)
      sensibilidade = verdadeiroPositivo/ (verdadeiroPositivo + falsoNegativo) if (verdadeiroPositivo + falsoNegativo) > 0 else 0
      especificiddade = verdadeiroNegativo / (verdadeiroNegativo + falsoPositivo) if (verdadeiroNegativo + falsoPositivo) > 0 else 0
      precisao = verdadeiroPositivo / (verdadeiroPositivo + falsoPositivo) if (verdadeiroPositivo + falsoPositivo) > 0 else 0
      numero_de_erros = falsoPositivo + falsoNegativo
      numero_de_acertos = verdadeiroPositivo + verdadeiroNegativo

      df_metricas.at[i-1, 'Acertos'] = numero_de_acertos
      df_metricas.at[i-1, 'Erros'] = numero_de_erros
      df_metricas.at[i-1, 'Acurácia'] = acuracia
      df_metricas.at[i-1, 'Sensibilidade'] = sensibilidade
      df_metricas.at[i-1, 'Especificidade'] = especificiddade
      df_metricas.at[i-1, 'Precisao'] = precisao
      df_metricas.to_excel(METRICAS_PATH, index=False)



      fig, ax = plt.subplots(figsize=(6, 5))
      display = ConfusionMatrixDisplay(confusion_matrix=matizDeConfusao,display_labels=['Erro (-1)', 'P2 (+1)'])
      display.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
      ax.set_title(f"Rede T{i} - Matriz de Confusão")
      plt.tight_layout()
      plt.savefig(f'./graphics/matrizes_de_confusao/rede_T{i}.png')


