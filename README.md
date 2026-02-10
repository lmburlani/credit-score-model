# Credit Score Model

Este projeto tem como objetivo desenvolver e validar um modelo de classificação para pontuação de crédito, utilizando o dataset **German Credit** da UCI Machine Learning Repository, em versão traduzida.

Dataset original:
https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)

O foco do projeto é explorar um fluxo completo de modelagem:
- análise e preparação dos dados,
- treinamento de um modelo de **Logistic Regression**,
- avaliação do desempenho com métricas apropriadas para classificação,
- validação do modelo ao longo do desenvolvimento.

O dataset contém informações socioeconômicas e financeiras de indivíduos, e o modelo busca estimar a probabilidade de concessão de crédito com base nessas variáveis.

## Modelo
- Algoritmo: **Logistic Regression**
- Tipo de problema: **Classificação**
- Dataset utilizado: `translated_database.csv`
- Principais métricas avaliadas:
  - Accuracy
  - ROC AUC
  - Confusion Matrix
  - Classification Report

## Estrutura do projeto
- `ML_Score.ipynb` — notebook com todo o processo de exploração, modelagem e avaliação
- `translated_database.csv` — versão traduzida do dataset German Credit
- `german_dataset_dictionary.txt` — descrição das variáveis do dataset


---
