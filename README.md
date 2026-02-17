# Credit Score Model

Projeto simples e direto para estudar risco de crédito com o dataset **German Credit** (versão traduzida em PT-BR).
A ideia aqui é manter um fluxo prático: entender os dados, validar qualidade mínima e ter um baseline interpretável antes de partir para modelos mais sofisticados.

Dataset original:
https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)

## O que tem neste repositório

- `ML_Score.ipynb`: notebook de exploração e modelagem.
- `translated_database.csv`: base de dados usada no projeto.
- `scripts/analise_credito.py`: análise rápida + baseline baseado em regra de negócio (sem dependências externas).
- `tests/test_metrics.py`: teste unitário das métricas calculadas no script.
- `german_dataset_dictionary.txt`: dicionário de variáveis.

## Como rodar (sem instalar bibliotecas)

```bash
python scripts/analise_credito.py
```

Saída esperada:
- volume de dados,
- distribuição de `default`,
- checagens básicas de qualidade (duplicados e campos vazios),
- resumo das variáveis numéricas,
- métricas de baseline (accuracy, precision, recall, f1 e matriz de confusão).

## Rodando os testes

```bash
python -m unittest discover -s tests
```

## Próximos passos recomendados

1. Criar split estratificado para preservar melhor o balanceamento do alvo.
2. Comparar o baseline de regra com modelos supervisionados (regressão logística, árvore e gradient boosting).
3. Incluir análise de explicabilidade (ex.: importância de variáveis / SHAP).
4. Definir um critério de negócio explícito (ex.: minimizar falso negativo de inadimplência).

---

Se quiser, na próxima iteração eu já deixo uma pipeline completa com treino + validação cruzada + export de relatório em CSV.
