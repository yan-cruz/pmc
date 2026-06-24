# RBF — Rede de Função de Base Radial

Projeto acadêmico: classificador binário RBF implementado do zero em Python. Treina 5 redes independentes e compara resultados. Sem frameworks de ML — apenas `numpy`, `pandas` e `math`. Todos os comentários, variáveis e documentação em **português**.

---

## O que o projeto faz

Treina 5 redes RBF independentes sobre o mesmo conjunto de dados, comparando convergência (épocas, RMSE) e desempenho de validação (acurácia, sensibilidade, especificidade, precisão). Persiste todos os resultados em Excel e gráficos PNG.

Cada rede usa pesos e inicialização K-Means aleatórios independentes — a variância entre runs é intencional.

---

## Arquitetura

```
Entrada (2 features: x1, x2)
    ↓  φ_i(x) = exp(-‖x − c_i‖² / 2σ²_i)  para i = 1..K
Camada Oculta (K=2 neurônios RBF) — Gaussiana
    ↓  y = W·φ + θ
Saída (1 neurônio) — Linear + limiarização {+1, -1}
```

- Função de base: `φ_i(x) = exp(-‖x − c_i‖² / 2σ²_i)`
- Saída linear: `y = W·φ + θ`
- Limiarização: `y ≥ 0 → +1`, `y < 0 → -1`
- Pesos: init uniforme `[-0.1, 0.1]` — aleatório por rede, sem semente fixa

---

## Hiperparâmetros (hardcoded em `rbf.py`)

| Parâmetro | Valor | Significado |
|-----------|-------|-------------|
| `ETA` | `0.01` | Taxa de aprendizado |
| `EPSILON` | `1e-7` | Limiar de convergência (`\|ΔRMSE\| < EPSILON`) |
| `MAX_EPOCAS` | `100000` | Máximo de épocas (Estágio 2) |
| `K` | `2` | Número de neurônios / clusters |
| `N_REDES` | `5` | Número de redes treinadas por execução |

---

## Mapa de arquivos

### Script Python

| Arquivo | Papel |
|---------|-------|
| `rbf.py` | Único arquivo: dados, K-Means, Regra Delta, I/O Excel, validação, plotagem |

### Datasets (`datasets/`)

| Arquivo | Conteúdo |
|---------|---------|
| `rbf_treinamento.xlsx` | 40 amostras: cols `x1, x2, d` |
| `rbf_validacao.xlsx` | 10 amostras: cols `x1, x2, d, y_T1..y_T5, ybin_T1..ybin_T5` |
| `rbf_metricas.xlsx` | Uma linha por rede: Rede, Épocas, RMSE Final, VP, VN, FP, FN, Acertos, Erros, Acurácia (%), Sensibilidade (%), Especificidade (%), Precisão (%) |

### Gráficos

| Caminho | Conteúdo |
|---------|---------|
| `graphics/Evolucao_do_erro/treinamento_rbf_1..5.png` | Curva RMSE × época por rede |
| `graphics/matrizes_de_confusao/rede_rbf_T1..T5.png` | Matriz de confusão por rede |

---

## Pipeline

```
rbf.py
  │
  ├── preparar_datasets()
  │     └── DADOS_TREINAMENTO → datasets/rbf_treinamento.xlsx
  │     └── DADOS_VALIDACAO   → datasets/rbf_validacao.xlsx (colunas brutas)
  │
  ├── carregar_dados()
  │     └── lê X_treino, D_treino, X_val, D_val dos .xlsx
  │
  └── for T in 1..5:
        ├── ESTÁGIO 1 — kmeans(X_pos)
        │     └── centros, rotulos = kmeans(X_pos)
        │     └── variancias = calcular_variancias(X_pos, centros, rotulos)
        │
        ├── ESTÁGIO 2 — treinar_saida(...)
        │     └── W, theta, rmse_por_epoca
        │     └── plotar_rmse(T) → graphics/Evolucao_do_erro/treinamento_rbf_{T}.png
        │
        ├── VALIDAÇÃO
        │     └── propagacao_direta() + pos_processar() por amostra
        │     └── calcular_metricas() → m
        │     └── plotar_matriz_confusao(m, T) → graphics/matrizes_de_confusao/rede_rbf_T{T}.png
        │
        └── acumula em predicoes_cont[T], predicoes_bin[T], resultados[]
  │
  └── CONSOLIDAR
        ├── salvar_validacao()   → datasets/rbf_validacao.xlsx (y_T1..T5 + ybin_T1..T5)
        └── salvar_metricas()    → datasets/rbf_metricas.xlsx  (5 linhas)
```

---

## Setup & Execução

```bash
# Criar venv
py -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

# Instalar dependências
pip install numpy pandas openpyxl matplotlib

# Executar
py rbf.py
```

---

## Dependências

```
numpy       # Operações matriciais
pandas      # Leitura e escrita de .xlsx
openpyxl    # Backend do pandas para .xlsx
matplotlib  # Geração de gráficos
```

Sem frameworks de ML (scikit-learn, TensorFlow, PyTorch). Rede implementada do zero com `math` e `numpy`.

---

## Terminologia (PT → EN)

| Português | English |
|-----------|---------|
| RMSE / Raiz do erro quadrático médio | RMSE (Root Mean Squared Error) |
| Épocas | Epochs |
| Treinamento | Training |
| Validação | Validation |
| Amostra | Sample |
| Centro | Center / centroid |
| Variância | Variance |
| Escondido / Oculto | Hidden |
| Saída | Output |
| Desejado (`d`) | Target / desired output |
| Taxa de aprendizado | Learning rate |
| Limiarização | Thresholding |
| Acurácia | Accuracy |
| Sensibilidade | Sensitivity / Recall |
| Especificidade | Specificity |
| Precisão | Precision |
| VP / VN / FP / FN | TP / TN / FP / FN |

---

## Funções principais

### `rbf.py`

**I/O de planilhas**
- `preparar_datasets()` — limpa e reescreve `rbf_treinamento.xlsx` e `rbf_validacao.xlsx`
- `carregar_dados()` — lê os .xlsx; retorna `(X_treino, D_treino, X_val, D_val)`
- `salvar_validacao(X_val, D_val, predicoes_cont, predicoes_bin)` — escreve `rbf_validacao.xlsx` com `y_T1..y_T5` e `ybin_T1..ybin_T5`
- `salvar_metricas(resultados)` — escreve `rbf_metricas.xlsx` com uma linha por rede

**Ativação e propagação**
- `gaussiana(x, centro, sigma2)` — RBF gaussiana, expoente limitado em -500
- `calcular_phi(x, centros, variancias)` — vetor `φ` da camada oculta
- `propagacao_direta(x, centros, variancias, W, theta)` — forward pass; retorna `(phi, y_linear)`
- `pos_processar(y)` — limiarização `y ≥ 0 → +1`, `y < 0 → -1`

**Estágio 1 — K-Means**
- `kmeans(X_pos, k, max_iter)` — K-Means sobre amostras positivas; retorna `(centros, rotulos)`
- `calcular_variancias(X_pos, centros, rotulos)` — distância quadrática média por cluster

**Estágio 2 — Regra Delta**
- `treinar_saida(X_treino, D_treino, centros, variancias)` — retorna `(W, theta, rmse_por_epoca)`

**Métricas**
- `calcular_metricas(y_pred, y_real)` — retorna dict com VP, VN, FP, FN, acurácia, sensibilidade, especificidade, precisão

**Plotagem**
- `plotar_rmse(rmse_por_epoca, T)` — salva `graphics/Evolucao_do_erro/treinamento_rbf_{T}.png`
- `plotar_matriz_confusao(m, T)` — salva `graphics/matrizes_de_confusao/rede_rbf_T{T}.png`

---

## Comportamentos importantes

- **`preparar_datasets()` sempre executa** — sobrescreve os .xlsx a cada run; dados canônicos vivem no script
- **5 redes sempre** — `N_REDES = 5`, hardcoded; sem argumento CLI
- **K-Means determinístico** — inicialização com as primeiras K amostras positivas (`X_pos[:K]`), conforme pseudocódigo do professor (passo `<2>`). Mesmos clusters garantidos em todos os treinamentos
- **Cluster unitário** — variância mínima `1e-6` para evitar divisão por zero na gaussiana
- **Consolidação no final** — `salvar_validacao` e `salvar_metricas` escrevem uma vez com dados de todas as redes
- **Arquivos RBF isolados** — usa `rbf_*.xlsx` para não sobrescrever os datasets do PMC

---

## Git

- Remote: `https://github.com/Italovini223/pmc.git`
- Branch: `main`
- `.gitignore`: `__pycache__`, `venv`
