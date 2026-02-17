#!/usr/bin/env python3
"""Análise rápida e baseline interpretável para o dataset de crédito.

Uso:
    python scripts/analise_credito.py
"""

from __future__ import annotations

import csv
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable

DATASET_PATH = Path("translated_database.csv")
RANDOM_SEED = 42
TEST_SIZE = 0.2


@dataclass(frozen=True)
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    tp: int
    tn: int
    fp: int
    fn: int


def compute_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> Metrics:
    tp = tn = fp = fn = 0
    for real, pred in zip(y_true, y_pred):
        if real == 1 and pred == 1:
            tp += 1
        elif real == 0 and pred == 0:
            tn += 1
        elif real == 0 and pred == 1:
            fp += 1
        else:
            fn += 1

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    return Metrics(accuracy, precision, recall, f1, tp, tn, fp, fn)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def split_train_test(rows: list[dict[str, str]], test_size: float = TEST_SIZE) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    random.seed(RANDOM_SEED)
    shuffled = rows[:]
    random.shuffle(shuffled)

    cutoff = int(len(shuffled) * (1 - test_size))
    return shuffled[:cutoff], shuffled[cutoff:]


def parse_target(row: dict[str, str]) -> int:
    return int(row["default"])


def regra_baseline(row: dict[str, str]) -> int:
    """Regra simples e interpretável para priorizar recall de inadimplência.

    Prediz inadimplência quando há histórico de crédito crítico ou conta corrente negativa.
    """
    historico = row["historico_credito"].lower()
    conta = row["conta_corrente"].lower()

    return int("critical" in historico or "< 0 dm" in conta)


def describe_numeric(rows: list[dict[str, str]], cols: list[str]) -> None:
    print("\nResumo de variáveis numéricas:")
    for col in cols:
        values = [float(row[col]) for row in rows]
        print(f"- {col}: média={mean(values):.2f} | mín={min(values):.2f} | máx={max(values):.2f}")


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset não encontrado em: {DATASET_PATH}")

    rows = read_rows(DATASET_PATH)
    print(f"Registros carregados: {len(rows)}")

    target_counts = Counter(parse_target(r) for r in rows)
    print(f"Distribuição do target (default): {dict(target_counts)}")

    duplicated = len(rows) - len({tuple(sorted(row.items())) for row in rows})
    print(f"Linhas duplicadas exatas: {duplicated}")

    missing = sum(1 for row in rows for value in row.values() if value.strip() == "")
    print(f"Valores ausentes vazios: {missing}")

    describe_numeric(
        rows,
        [
            "prazo_emprestimo_meses",
            "valor_emprestimo",
            "idade",
            "taxa_comp_salario",
            "n_creditos_banco",
        ],
    )

    _, test = split_train_test(rows)
    y_true = [parse_target(row) for row in test]
    y_pred = [regra_baseline(row) for row in test]
    metrics = compute_metrics(y_true, y_pred)

    print("\nAvaliação da regra baseline (conjunto de teste):")
    print(f"- Accuracy:  {metrics.accuracy:.3f}")
    print(f"- Precision: {metrics.precision:.3f}")
    print(f"- Recall:    {metrics.recall:.3f}")
    print(f"- F1-score:  {metrics.f1:.3f}")
    print("- Matriz de confusão:")
    print(f"  TP={metrics.tp} | FP={metrics.fp}")
    print(f"  FN={metrics.fn} | TN={metrics.tn}")


if __name__ == "__main__":
    main()
