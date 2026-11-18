"""Глобальные константы и пути проекта."""

from pathlib import Path

# Пути
ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_RAW = ROOT / "data" / "raw"
FIGURES = ROOT / "figures"
REPORTS = ROOT / "reports"

# Сид для воспроизводимости
SEED = 42

# Сетка K
K_GRID = (5, 10, 20, 50, 100, 200, 230)

# Условные стоимости стадий (единиц на документ)
COST_CHEAP = 1.0
COST_EXPENSIVE = 10.0

# Разбиение по запросам
SPLIT_A = 0.6
SPLIT_B = 0.2
SPLIT_C = 0.2

# Значения eps для таргета K
EPSILONS = [0.0, 0.01, 0.02]

NDCG_K = 10