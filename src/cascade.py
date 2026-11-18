"""Двухстадийный каскад: дешёвый LightGBM + дорогой CatBoost."""

from typing import List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool

import config


# ---------- Подготовка данных ----------

def prepare_lgb_data(df: pd.DataFrame, feature_cols: List, label_col: str = "target"):
    df_sorted = df.sort_values("query_id").reset_index(drop=True)
    X = df_sorted[feature_cols].values.astype("float32")
    y = df_sorted[label_col].values.astype("float32")
    group = df_sorted.groupby("query_id").size().values
    return X, y, group, df_sorted


def prepare_catboost_data(df: pd.DataFrame, feature_cols: List, label_col: str = "target"):
    df_sorted = df.sort_values("query_id").reset_index(drop=True)
    X = df_sorted[feature_cols].values.astype("float32")
    y = df_sorted[label_col].values.astype("float32")
    group_id = df_sorted["query_id"].values
    return X, y, group_id, df_sorted


def get_all_features(df: pd.DataFrame, label_col: str = "target") -> List:
    excl = {str(label_col), "query_id"}
    return [c for c in df.columns if str(c) not in excl]


# ---------- Обучение ----------

def train_cheap_ranker(
    df, cheap_features, label_col="target",
    n_estimators=80, learning_rate=0.1, num_leaves=15, max_depth=5, seed=None,
) -> lgb.LGBMRanker:
    X, y, group, _ = prepare_lgb_data(df, cheap_features, label_col=label_col)
    model = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        max_depth=max_depth,
        random_state=seed if seed is not None else config.SEED,
        verbose=-1,
    )
    model.fit(X, y, group=group)
    return model


def train_expensive_ranker(
    df, all_features, label_col="target",
    iterations=300, learning_rate=0.05, depth=7, seed=None, verbose=50,
) -> CatBoostRanker:
    X, y, group_id, _ = prepare_catboost_data(df, all_features, label_col=label_col)
    assert len(X) == len(group_id), f"data={len(X)}, group_id={len(group_id)}"
    pool = Pool(data=X, label=y, group_id=group_id)
    model = CatBoostRanker(
        loss_function="YetiRank",
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        random_seed=seed if seed is not None else config.SEED,
        verbose=verbose,
    )
    model.fit(pool)
    return model


# ---------- Сохранение / загрузка ----------

def save_models(cheap, expensive) -> None:
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    booster = cheap.booster_ if hasattr(cheap, "booster_") else cheap
    booster.save_model(str(config.DATA_PROCESSED / "cheap_ranker.txt"))
    expensive.save_model(str(config.DATA_PROCESSED / "expensive_ranker.cbm"))
    print("Модели сохранены в", config.DATA_PROCESSED)


def load_models() -> Tuple[lgb.Booster, CatBoostRanker]:
    cheap = lgb.Booster(model_file=str(config.DATA_PROCESSED / "cheap_ranker.txt"))
    expensive = CatBoostRanker()
    expensive.load_model(str(config.DATA_PROCESSED / "expensive_ranker.cbm"))
    return cheap, expensive