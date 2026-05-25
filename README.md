## Requisitos

```
pandas
openpyxl
matplotlib
numpy
xlrd
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
pip install pandas openpyxl matplotlib numpy xlrd
```

### 3. Preparar planilhas (apenas na primeira vez)

Coloque os arquivos `PP03_dados-treinamento.xls` e `PP03_dados-validacao.xls` na pasta Downloads e execute:

```
py preparar_planilhas.py
```

### 4. Treinar as redes

```
py pmc.py
```

## Arquivos gerados

| Arquivo | Conteúdo |
|---|---|
| `datasets/resultados.xlsx` | Tabela 1 — EQM e épocas dos 5 treinamentos |
| `datasets/validacao.xlsx` | Tabela 2 — saídas y(T1)…y(T5) para as 20 amostras de validação |
| `datasets/metricas.xlsx` | Erro relativo médio (%) e Variância (%) por rede |
| `graphics/Evolucao_do_erro/` | 5 gráficos de EQM por época |
| `graphics/validacao/` | 5 gráficos de saída desejada vs. saída da rede |
