import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# DADOS (fonte canônica — salva em Excel no início de cada run)
# ============================================================

DADOS_TREINAMENTO = np.array([
    [0.2563, 0.9503, -1], [0.2405, 0.9018, -1], [0.1157, 0.3676,  1],
    [0.5147, 0.0167,  1], [0.4127, 0.3275,  1], [0.2809, 0.5830,  1],
    [0.8263, 0.9301, -1], [0.9359, 0.8724, -1], [0.1096, 0.9165, -1],
    [0.5158, 0.8545, -1], [0.1334, 0.1362,  1], [0.6371, 0.1439,  1],
    [0.7052, 0.6277, -1], [0.8703, 0.8666, -1], [0.2612, 0.6109,  1],
    [0.0244, 0.5279,  1], [0.9588, 0.3672, -1], [0.9332, 0.5499, -1],
    [0.9623, 0.2961, -1], [0.7297, 0.5776, -1], [0.4560, 0.1871,  1],
    [0.1715, 0.7713,  1], [0.5571, 0.5485, -1], [0.3344, 0.0259,  1],
    [0.4803, 0.7635, -1], [0.9721, 0.4850, -1], [0.8318, 0.7844, -1],
    [0.1373, 0.0292,  1], [0.3660, 0.8581, -1], [0.3626, 0.7302, -1],
    [0.6474, 0.3324,  1], [0.3461, 0.2398,  1], [0.1353, 0.8120,  1],
    [0.3463, 0.1017,  1], [0.9086, 0.1947, -1], [0.5227, 0.2321,  1],
    [0.5153, 0.2041,  1], [0.1832, 0.0661,  1], [0.5015, 0.9812, -1],
    [0.5024, 0.5274, -1],
])

DADOS_VALIDACAO = np.array([
    [0.8705, 0.9329, -1], [0.0388, 0.2703,  1], [0.8236, 0.4458, -1],
    [0.7075, 0.1502,  1], [0.9587, 0.8663, -1], [0.6115, 0.9365, -1],
    [0.3534, 0.3646,  1], [0.3268, 0.2766,  1], [0.6129, 0.4518, -1],
    [0.9948, 0.4962, -1],
])

# ============================================================
# HIPERPARÂMETROS
# ============================================================

ETA        = 0.01
EPSILON    = 1e-7
MAX_EPOCAS = 100_000
K          = 2
N_REDES    = 5

DATASETS_PATH = os.path.join(os.getcwd(), 'datasets')
EVOLUCAO_PATH = os.path.join(os.getcwd(), 'graphics', 'Evolucao_do_erro')
MATRIZES_PATH = os.path.join(os.getcwd(), 'graphics', 'matrizes_de_confusao')

# ============================================================
# I/O DE PLANILHAS
# ============================================================

def preparar_datasets():
    """Limpa e reescreve rbf_treinamento.xlsx e rbf_validacao.xlsx (dados brutos)."""
    os.makedirs(DATASETS_PATH, exist_ok=True)

    df_treino = pd.DataFrame(DADOS_TREINAMENTO, columns=['x1', 'x2', 'd'])
    df_treino['d'] = df_treino['d'].astype(int)
    df_treino.to_excel(os.path.join(DATASETS_PATH, 'rbf_treinamento.xlsx'), index=False)

    df_val = pd.DataFrame(DADOS_VALIDACAO, columns=['x1', 'x2', 'd'])
    df_val['d'] = df_val['d'].astype(int)
    df_val.to_excel(os.path.join(DATASETS_PATH, 'rbf_validacao.xlsx'), index=False)

    print("  Datasets preparados em datasets/")


def carregar_dados():
    """Lê X_treino, D_treino, X_val, D_val de datasets/."""
    df_treino = pd.read_excel(os.path.join(DATASETS_PATH, 'rbf_treinamento.xlsx'))
    df_val    = pd.read_excel(os.path.join(DATASETS_PATH, 'rbf_validacao.xlsx'))

    X_treino = df_treino[['x1', 'x2']].values
    D_treino = df_treino['d'].values.astype(float)
    X_val    = df_val[['x1', 'x2']].values
    D_val    = df_val['d'].values.astype(float)

    return X_treino, D_treino, X_val, D_val


def salvar_validacao(X_val, D_val, predicoes_cont, predicoes_bin):
    """
    Reescreve rbf_validacao.xlsx com colunas y_T1..y_T5 e ybin_T1..ybin_T5.
    predicoes_cont / predicoes_bin: dict {T: lista de valores}
    """
    df = pd.DataFrame({
        'x1': X_val[:, 0],
        'x2': X_val[:, 1],
        'd':  [int(d) for d in D_val],
    })
    for T in range(1, N_REDES + 1):
        df[f'y_T{T}']    = [round(v, 8) for v in predicoes_cont[T]]
        df[f'ybin_T{T}'] = predicoes_bin[T]

    df.to_excel(os.path.join(DATASETS_PATH, 'rbf_validacao.xlsx'), index=False)
    print("  Previsões salvas em datasets/rbf_validacao.xlsx")


def salvar_metricas(resultados):
    """
    Salva métricas de todas as redes em datasets/rbf_metricas.xlsx.
    resultados: lista de dicts (um por rede).
    """
    linhas = []
    for r in resultados:
        linhas.append({
            'Rede':               f"T{r['T']}",
            'Épocas':             r['epocas'],
            'RMSE Final':         round(r['rmse_final'], 6),
            'VP':                 r['VP'],
            'VN':                 r['VN'],
            'FP':                 r['FP'],
            'FN':                 r['FN'],
            'Acertos':            r['acertos'],
            'Erros':              r['erros'],
            'Acurácia (%)':       round(r['acuracia']       * 100, 2),
            'Sensibilidade (%)':  round(r['sensibilidade']  * 100, 2),
            'Especificidade (%)': round(r['especificidade'] * 100, 2),
            'Precisão (%)':       round(r['precisao']       * 100, 2),
        })

    pd.DataFrame(linhas).to_excel(
        os.path.join(DATASETS_PATH, 'rbf_metricas.xlsx'), index=False
    )
    print("  Métricas salvas em datasets/rbf_metricas.xlsx")

# ============================================================
# FUNÇÕES DE ATIVAÇÃO E PROPAGAÇÃO
# ============================================================

def gaussiana(x, centro, sigma2):
    """Função de base radial gaussiana: exp(-||x-c||² / 2σ²)."""
    dist2    = float(np.sum((x - centro) ** 2))
    expoente = -dist2 / (2.0 * sigma2)
    return math.exp(max(expoente, -500.0))


def calcular_phi(x, centros, variancias):
    """Vetor de ativações da camada oculta para a entrada x."""
    return np.array([gaussiana(x, centros[i], variancias[i]) for i in range(K)])


def propagacao_direta(x, centros, variancias, W, theta):
    """Forward pass: retorna (phi, y_linear)."""
    phi = calcular_phi(x, centros, variancias)
    y   = float(np.dot(W, phi)) + theta
    return phi, y


def pos_processar(y):
    """Limiarização: y >= 0 → +1, y < 0 → -1."""
    return 1 if y >= 0.0 else -1

# ============================================================
# ESTÁGIO 1 — K-MEANS (somente amostras com d = +1)
# ============================================================

def kmeans(X_pos, k=K, max_iter=1000):
    """K-Means sobre amostras positivas; retorna (centros, rotulos)."""
    n        = len(X_pos)
    idx_init = np.random.choice(n, k, replace=False)
    centros  = X_pos[idx_init].copy().astype(float)

    for _ in range(max_iter):
        dists   = np.array([[np.sum((x - c) ** 2) for c in centros] for x in X_pos])
        rotulos = np.argmin(dists, axis=1)

        novos_centros = np.zeros_like(centros)
        for i in range(k):
            membros = X_pos[rotulos == i]
            novos_centros[i] = membros.mean(axis=0) if len(membros) > 0 else X_pos[np.random.randint(n)]

        if np.allclose(centros, novos_centros, atol=1e-12):
            break
        centros = novos_centros

    return centros, rotulos


def calcular_variancias(X_pos, centros, rotulos, k=K):
    """Variância de cada cluster: distância quadrática média ao centróide."""
    variancias = []
    for i in range(k):
        membros = X_pos[rotulos == i]
        if len(membros) > 1:
            var = float(np.mean(np.sum((membros - centros[i]) ** 2, axis=1)))
        else:
            var = 1e-6
        variancias.append(max(var, 1e-12))
    return np.array(variancias)

# ============================================================
# ESTÁGIO 2 — REGRA DELTA (camada de saída linear)
# ============================================================

def treinar_saida(X_treino, D_treino, centros, variancias):
    """Treina W e θ via Regra Delta; retorna (W, theta, rmse_por_epoca)."""
    W     = np.random.uniform(-0.1, 0.1, K)
    theta = float(np.random.uniform(-0.1, 0.1))

    rmse_por_epoca = []
    rmse_anterior  = float('inf')

    for epoca in range(MAX_EPOCAS):
        erros_quad = []
        for x, d in zip(X_treino, D_treino):
            phi, y = propagacao_direta(x, centros, variancias, W, theta)
            e      = d - y
            erros_quad.append(e ** 2)
            W     += ETA * e * phi
            theta += ETA * e

        rmse = math.sqrt(float(np.mean(erros_quad)))
        rmse_por_epoca.append(rmse)

        if abs(rmse_anterior - rmse) < EPSILON:
            print(f"  Convergência atingida na época {epoca + 1}  |  RMSE = {rmse:.2e}")
            break
        rmse_anterior = rmse
    else:
        print(f"  Limite de {MAX_EPOCAS} épocas atingido  |  RMSE = {rmse_por_epoca[-1]:.2e}")

    return W, theta, rmse_por_epoca

# ============================================================
# MÉTRICAS DE VALIDAÇÃO
# ============================================================

def calcular_metricas(y_pred, y_real):
    """Matriz de confusão e métricas para classificação binária {+1, -1}."""
    VP = sum(1 for p, r in zip(y_pred, y_real) if p ==  1 and r ==  1)
    VN = sum(1 for p, r in zip(y_pred, y_real) if p == -1 and r == -1)
    FP = sum(1 for p, r in zip(y_pred, y_real) if p ==  1 and r == -1)
    FN = sum(1 for p, r in zip(y_pred, y_real) if p == -1 and r ==  1)

    total   = len(y_pred)
    acertos = VP + VN
    erros   = FP + FN

    return {
        'VP': VP, 'VN': VN, 'FP': FP, 'FN': FN,
        'acertos': acertos, 'erros': erros,
        'acuracia':       acertos / total,
        'sensibilidade':  VP / (VP + FN) if (VP + FN) > 0 else 0.0,
        'especificidade': VN / (VN + FP) if (VN + FP) > 0 else 0.0,
        'precisao':       VP / (VP + FP) if (VP + FP) > 0 else 0.0,
    }

# ============================================================
# PLOTAGEM
# ============================================================

def plotar_rmse(rmse_por_epoca, T):
    """Salva curva RMSE × época em graphics/Evolucao_do_erro/treinamento_rbf_{T}.png."""
    os.makedirs(EVOLUCAO_PATH, exist_ok=True)
    epocas = list(range(1, len(rmse_por_epoca) + 1))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epocas, rmse_por_epoca, linewidth=1.2, color='steelblue')
    ax.set_title(f'Evolução do RMSE — Rede RBF T{T} (Regra Delta)')
    ax.set_xlabel('Épocas')
    ax.set_ylabel('RMSE')
    ax.grid(True, alpha=0.4)
    fig.tight_layout()

    caminho = os.path.join(EVOLUCAO_PATH, f'treinamento_rbf_{T}.png')
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    print(f"  Gráfico RMSE salvo em: {caminho}")


def plotar_matriz_confusao(m, T):
    """Gera e salva matriz de confusão em graphics/matrizes_de_confusao/rede_rbf_T{T}.png."""
    os.makedirs(MATRIZES_PATH, exist_ok=True)

    matriz = np.array([[m['VP'], m['FN']],
                       [m['FP'], m['VN']]])

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matriz, cmap='Blues', vmin=0)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['+1\n(Previsto)', '-1\n(Previsto)'], fontsize=10)
    ax.set_yticklabels(['+1 (Real)', '-1 (Real)'], fontsize=10)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matriz[i, j]),
                    ha='center', va='center', fontsize=16, fontweight='bold',
                    color='white' if matriz[i, j] > matriz.max() / 2 else 'black')

    ax.set_title(
        f'Matriz de Confusão — RBF T{T}\n'
        f'Acurácia: {m["acuracia"]*100:.2f}%  |  '
        f'Precisão: {m["precisao"]*100:.2f}%',
        fontsize=10,
    )
    plt.colorbar(im, ax=ax)
    fig.tight_layout()

    caminho = os.path.join(MATRIZES_PATH, f'rede_rbf_T{T}.png')
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    print(f"  Matriz de confusão salva em: {caminho}")

# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================

def main():
    # ----------------------------------------------------------
    # PREPARAR E CARREGAR DADOS
    # ----------------------------------------------------------
    print("=" * 62)
    print("  PREPARANDO DATASETS")
    print("=" * 62)
    preparar_datasets()
    X_treino, D_treino, X_val, D_val = carregar_dados()
    print(f"  Treinamento: {len(X_treino)} amostras  |  Validação: {len(X_val)} amostras")

    X_pos = X_treino[D_treino == 1]

    # Acumuladores para salvar ao final
    predicoes_cont = {}
    predicoes_bin  = {}
    resultados     = []

    # ----------------------------------------------------------
    # LOOP: 5 REDES
    # ----------------------------------------------------------
    for T in range(1, N_REDES + 1):
        print("\n" + "=" * 62)
        print(f"  REDE T{T}")
        print("=" * 62)

        # ---- ESTÁGIO 1 — K-MEANS ----
        print(f"\n  [Estágio 1] K-Means  (amostras positivas: {len(X_pos)})")
        centros, rotulos = kmeans(X_pos)
        variancias       = calcular_variancias(X_pos, centros, rotulos)

        for i in range(K):
            n_membros = int(np.sum(rotulos == i))
            print(f"    Cluster {i+1}: c=({centros[i][0]:.6f}, {centros[i][1]:.6f})"
                  f"  σ²={variancias[i]:.6f}  n={n_membros}")

        # ---- ESTÁGIO 2 — REGRA DELTA ----
        print(f"\n  [Estágio 2] Regra Delta  eta={ETA}  eps={EPSILON}  max={MAX_EPOCAS}")
        W, theta, rmse_por_epoca = treinar_saida(X_treino, D_treino, centros, variancias)

        print(f"    W1={W[0]:+.8f}  W2={W[1]:+.8f}  θ={theta:+.8f}")
        print(f"    Épocas: {len(rmse_por_epoca)}  |  RMSE final: {rmse_por_epoca[-1]:.2e}")

        # ---- GRÁFICO RMSE ----
        plotar_rmse(rmse_por_epoca, T)

        # ---- VALIDAÇÃO ----
        print(f"\n  [Validação] Rede T{T}")
        cab = f"  {'#':<4} {'x1':>8} {'x2':>8} {'d':>5} {'y':>12} {'y_bin':>7}"
        print(cab)
        print("  " + "-" * (len(cab) - 2))

        y_cont = []
        y_bin  = []
        for i, (x, d) in enumerate(zip(X_val, D_val)):
            _, y  = propagacao_direta(x, centros, variancias, W, theta)
            y_pos = pos_processar(y)
            y_cont.append(y)
            y_bin.append(y_pos)
            acerto = "OK" if y_pos == int(d) else "XX"
            print(f"  {i+1:<4} {x[0]:>8.4f} {x[1]:>8.4f} {int(d):>5} {y:>12.6f} {y_pos:>7}  {acerto}")

        predicoes_cont[T] = y_cont
        predicoes_bin[T]  = y_bin

        # ---- MÉTRICAS ----
        m = calcular_metricas(y_bin, [int(d) for d in D_val])

        print(f"\n  Matriz de Confusão — T{T}")
        print("               Previsto  +1    -1")
        print(f"  Real  +1  |  VP={m['VP']:>2}  |  FN={m['FN']:>2}  |")
        print(f"        -1  |  FP={m['FP']:>2}  |  VN={m['VN']:>2}  |")
        print(f"  Acurácia: {m['acuracia']*100:.2f}%  "
              f"Sensibilidade: {m['sensibilidade']*100:.2f}%  "
              f"Especificidade: {m['especificidade']*100:.2f}%  "
              f"Precisão: {m['precisao']*100:.2f}%")

        # ---- GRÁFICO MATRIZ ----
        plotar_matriz_confusao(m, T)

        resultados.append({
            'T':             T,
            'epocas':        len(rmse_por_epoca),
            'rmse_final':    rmse_por_epoca[-1],
            **m,
        })

    # ----------------------------------------------------------
    # SALVAR RESULTADOS CONSOLIDADOS
    # ----------------------------------------------------------
    print("\n" + "=" * 62)
    print("  SALVANDO RESULTADOS CONSOLIDADOS")
    print("=" * 62)
    salvar_validacao(X_val, D_val, predicoes_cont, predicoes_bin)
    salvar_metricas(resultados)

    # ---- RESUMO FINAL ----
    print("\n" + "=" * 62)
    print("  RESUMO — 5 REDES")
    print("=" * 62)
    cab = f"  {'Rede':<6} {'Épocas':>8} {'RMSE':>10} {'Acurácia':>10} {'Precisão':>10}"
    print(cab)
    print("  " + "-" * (len(cab) - 2))
    for r in resultados:
        print(f"  T{r['T']:<5} {r['epocas']:>8} {r['rmse_final']:>10.2e}"
              f" {r['acuracia']*100:>9.2f}% {r['precisao']*100:>9.2f}%")
    print()


if __name__ == '__main__':
    main()
