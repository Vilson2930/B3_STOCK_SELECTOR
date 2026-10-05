# B3 STOCK SELECTOR

Sistema quantitativo para seleção de ações da B3 baseado em fundamentos, valuation, controle de risco e arquitetura Point-in-Time (PIT).

O projeto foi estruturado para transformar o estudo científico realizado sobre ações vencedoras da B3 em um motor reproduzível, auditável e protegido contra look-ahead.

---

## Objetivo

Selecionar empresas da B3 com maior probabilidade de apresentar características fundamentalistas favoráveis, utilizando somente informações que estariam disponíveis na data da análise.

O sistema separa claramente:

- seleção operacional;
- pesquisa científica;
- controle de risco;
- auditoria.

Retorno futuro nunca é utilizado como variável de decisão operacional.

---

## Estrutura

```text
B3_STOCK_SELECTOR/
│
├── main.py
├── config.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── cvm.py
│   ├── market.py
│   └── pit.py
│
├── engines/
│   ├── universe.py
│   ├── investability.py
│   ├── fundamentals.py
│   ├── quality.py
│   ├── turnaround.py
│   ├── valuation.py
│   ├── ranking.py
│   └── risk.py
│
├── reports/
│   └── report.py
│
├── outputs/
│
└── .github/
    └── workflows/
        └── run.yml
