import pandas as pd
import os 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

PATH = os.path.join(os.getcwd(), './datasets/validacao.xlsx')
METRICAS_PATH = os.path.join(os.getcwd(), './datasets/metricas.xlsx')

def traduzir_codigo_para_classe(v1, v2, v3):
    """Traduz o formato binário para o nome da classe. Se a rede gerar um código inválido, regista como Invalido"""
    if v1 == 1 and v2 == 0 and v3 == 0: return 'A'
    if v1 == 0 and v2 == 1 and v3 == 0: return 'B'
    if v1 == 0 and v2 == 0 and v3 == 1: return 'C'
    return 'Invalido' # Caso a rede tenha previsto [1, 1, 0] por exemplo

def calcular():
    df_validacao = pd.read_excel(PATH)
    
    # Limpa cabeçalhos caso venham sujos do Excel
    df_validacao.columns = df_validacao.columns.str.strip().str.replace('_x000d_', '')
    
    try:
        df_metricas = pd.read_excel(METRICAS_PATH)
    except FileNotFoundError:
        df_metricas = pd.DataFrame(columns=['Rede', 'Acertos', 'Erros', 'Acurácia', 'Sensibilidade', 'Especificidade', 'Precisao'])

    # 1. Traduz o gabarito real (d1, d2, d3)
    d_verdadeiro = []
    for _, row in df_validacao.iterrows():
        classe = traduzir_codigo_para_classe(row['d1'], row['d2'], row['d3'])
        d_verdadeiro.append(classe)

    classes_labels = ['A', 'B', 'C']

    for i in range(1, 6):
        y_predito = []
        
        # 2. Traduz as previsões da rede T(i)
        for _, row in df_validacao.iterrows():
            c1, c2, c3 = row[f'Y1_T{i}'], row[f'Y2_T{i}'], row[f'Y3_T{i}']
            y_predito.append(traduzir_codigo_para_classe(c1, c2, c3))

        # 3. Gera a Matriz de Confusão 3x3
        # As classes 'Invalido' contam como erro natural e ficam fora da diagonal principal
        todas_classes = ['A', 'B', 'C', 'Invalido']
        matriz_confusao = confusion_matrix(d_verdadeiro, y_predito, labels=todas_classes)
        
        # Recorta só a matriz 3x3 principal para os cálculos matemáticos
        mc_3x3 = matriz_confusao[0:3, 0:3]
        
        acertos = np.trace(mc_3x3) # Soma da diagonal principal
        erros = len(d_verdadeiro) - acertos
        acuracia = acertos / len(d_verdadeiro)

        # 4. Cálculo Macro (Média das 3 classes) para métricas complexas
        sensibilidade_lista = []
        especificidade_lista = []
        precisao_lista = []

        for c in range(3):
            TP = mc_3x3[c, c]
            FN = np.sum(mc_3x3[c, :]) - TP
            FP = np.sum(mc_3x3[:, c]) - TP
            TN = np.sum(mc_3x3) - (TP + FP + FN)

            sens = TP / (TP + FN) if (TP + FN) > 0 else 0
            espec = TN / (TN + FP) if (TN + FP) > 0 else 0
            prec = TP / (TP + FP) if (TP + FP) > 0 else 0

            sensibilidade_lista.append(sens)
            especificidade_lista.append(espec)
            precisao_lista.append(prec)

        sensibilidade_macro = np.mean(sensibilidade_lista)
        especificidade_macro = np.mean(especificidade_lista)
        precisao_macro = np.mean(precisao_lista)

        # 5. Regista no Excel
        idx = i - 1
        df_metricas.at[idx, 'Rede'] = f'T{i}'
        df_metricas.at[idx, 'Acertos'] = acertos
        df_metricas.at[idx, 'Erros'] = erros
        df_metricas.at[idx, 'Acurácia'] = acuracia
        df_metricas.at[idx, 'Sensibilidade'] = sensibilidade_macro
        df_metricas.at[idx, 'Especificidade'] = especificidade_macro
        df_metricas.at[idx, 'Precisao'] = precisao_macro

        # 6. Salva o Gráfico da Matriz
        fig, ax = plt.subplots(figsize=(6, 5))
        # Remove a coluna 'Invalido' do gráfico se não houver
        display_matriz = confusion_matrix(d_verdadeiro, y_predito, labels=['A', 'B', 'C'])
        display = ConfusionMatrixDisplay(confusion_matrix=display_matriz, display_labels=['Tipo A', 'Tipo B', 'Tipo C'])
        display.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
        ax.set_title(f"Rede T{i} - Matriz de Confusão")
        
        plt.tight_layout()
        
        grafico_dir = './graphics/matrizes_de_confusao/'
        if not os.path.exists(grafico_dir):
            os.makedirs(grafico_dir)
            
        plt.savefig(f'{grafico_dir}rede_T{i}.png')
        plt.close()

    df_metricas.to_excel(METRICAS_PATH, index=False)
    print("\nMétricas e Matrizes de Confusão geradas com sucesso!")

# Executar caso o arquivo seja chamado diretamente
if __name__ == '__main__':
    calcular()