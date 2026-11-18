import numpy as np
import pandas as pd

def ndcg_at_k(y_true, y_score, k=10):
    """NDCG@k. Документы со score=-inf (вне top-K каскада) не дают вклада в DCG."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    order = np.argsort(y_score)[::-1][:k]
    y_true_sorted = y_true[order].copy()

    # Документы вне top-K приходят как -inf -> их вклад в DCG обнуляем
    unscored = ~np.isfinite(y_score[order])
    y_true_sorted[unscored] = 0

    discounts = 1.0 / np.log2(np.arange(2, len(y_true_sorted) + 2))
    dcg = np.sum((2.0 ** y_true_sorted - 1.0) * discounts)

    ideal_order = np.argsort(y_true)[::-1][:k]
    ideal_sorted = y_true[ideal_order]
    idcg = np.sum((2.0 ** ideal_sorted - 1.0) * discounts[:len(ideal_sorted)])

    return float(dcg / idcg) if idcg > 0 else 0.0


def cascade_predict(cheap_model, expensive_model, df, k,
                    cheap_features, all_features):
    """Прогон каскада. Возвращает DataFrame по ВСЕМ документам запроса:
    top-K дешёвой модели получают скоры дорогой, остальные — -inf.
    """
    results = []
    for qid, group in df.groupby("query_id"):
        group = group.reset_index(drop=True)

        X_cheap = group[cheap_features].values
        cheap_scores = cheap_model.predict(X_cheap)

        top_k_idx = np.argsort(cheap_scores)[::-1][:k]
        X_expensive = group.iloc[top_k_idx][all_features].values
        expensive_scores = expensive_model.predict(X_expensive)

        # Выравниваем по всем строкам запроса: вне top-K -> -inf
        aligned_scores = np.full(len(group), -np.inf, dtype=np.float64)
        aligned_scores[top_k_idx] = expensive_scores

        results.append(pd.DataFrame({
            "query_id": qid,
            "target": group["target"].values,
            "score": aligned_scores,
        }))

    return pd.concat(results, ignore_index=True)