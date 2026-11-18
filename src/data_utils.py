import numpy as np

def split_by_query(df, frac_a=0.6, frac_b=0.2, seed=42):
    qids = df["query_id"].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(qids)
    n = len(qids)
    a = set(qids[:int(n * frac_a)])
    b = set(qids[int(n * frac_a):int(n * (frac_a + frac_b))])
    c = set(qids[int(n * (frac_a + frac_b)):])
    return a, b, c


def prepare_lgb_data(df, feature_cols, label_col="target"):
    df_sorted = df.sort_values("query_id").reset_index(drop=True)
    X = df_sorted[feature_cols].values
    y = df_sorted[label_col].values
    group = df_sorted.groupby("query_id").size().values
    return X, y, group, df_sorted