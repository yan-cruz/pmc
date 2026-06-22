# RBF — Rede de Função de Base Radial

Projeto acadêmico: classificador binário RBF implementado do zero em Python. Treina 5 redes independentes e compara resultados. Sem frameworks de ML — apenas `numpy`, `pandas` e `math`.

## Requisitos

```
numpy
pandas
openpyxl
matplotlib
```

## Como rodar

### 1. Criar e ativar ambiente virtual

Windows:
```
py -m venv venv
venv\Scripts\activate
```

Mac/Linux:
```
python3 -m venv venv
source venv/bin/activate
```

### 2. Instalar dependências

```
pip install numpy pandas openpyxl matplotlib
```

### 3. Treinar e validar

```
py rbf.py
```

Os datasets são gerados automaticamente e sobrescritos a cada run.

## Arquivos gerados

| Saída | Conteúdo |
|---|---|
| `datasets/rbf_treinamento.xlsx` | 40 amostras de treinamento `(x1, x2, d)` |
| `datasets/rbf_validacao.xlsx` | 10 amostras + previsões `y_T1..y_T5` e `ybin_T1..ybin_T5` |
| `datasets/rbf_metricas.xlsx` | Acurácia, Sensibilidade, Especificidade, Precisão, VP/VN/FP/FN por rede |
| `graphics/Evolucao_do_erro/treinamento_rbf_1..5.png` | Curva RMSE × época por rede |
| `graphics/matrizes_de_confusao/rede_rbf_T1..T5.png` | Matriz de confusão por rede |
