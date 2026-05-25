import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

VALIDACAO_PATH = os.path.join(os.getcwd(), 'datasets', 'validacao.xlsx')
METRICAS_PATH = os.path.join(os.getcwd(), 'datasets', 'metricas.xlsx')
GRAFICOS_PATH = os.path.join(os.getcwd(), 'graphics', 'validacao')


def calcular():
    df = pd.read_excel(VALIDACAO_PATH)
    os.makedirs(GRAFICOS_PATH, exist_ok=True)

    d = df['d'].values.astype(float)
    amostras = list(range(1, len(df) + 1))

    linhas_metricas = []

    for i in range(1, 6):
        col = f'y_T{i}'
        y = df[col].values.astype(float)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(amostras, d, 'b-o', label='Saída desejada (d)', markersize=5, linewidth=1.5)
        ax.plot(amostras, y, 'r-s', label=f'Saída da rede T{i} (y)', markersize=5, linewidth=1.5)
        ax.set_title(f'Validação T{i}: Saída Desejada vs. Saída da Rede')
        ax.set_xlabel('Número da amostra')
        ax.set_ylabel('Saída')
        ax.legend()
        ax.grid(True)
        ax.set_xticks(amostras)
        fig.tight_layout()
        fig.savefig(os.path.join(GRAFICOS_PATH, f'validacao_T{i}.png'))
        plt.close(fig)

        erros_relativos = np.abs(d - y) / np.abs(d) * 100
        linhas_metricas.append({
            'Rede': f'T{i}',
            'Erro relativo medio (%)': round(float(np.mean(erros_relativos)), 4),
            'Variancia (%)': round(float(np.var(erros_relativos)), 4),
        })

    pd.DataFrame(linhas_metricas).to_excel(METRICAS_PATH, index=False)
    print('Gráficos de validação e métricas gerados com sucesso!')


if __name__ == '__main__':
    calcular()
