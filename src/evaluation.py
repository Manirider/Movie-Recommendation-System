
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def ndcg_at_k(
    model,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    uid_to_idx: Dict[int, int],
    mid_to_idx: Dict[int, int],
    k: int = 10,
    threshold: float = 3.5,
) -> float:
    idx_to_mid = {v: k_ for k_, v in mid_to_idx.items()}
    discounts = 1.0 / np.log2(np.arange(2, k + 2))

    test_relevant = (
        test_df[test_df["rating"] >= threshold]
        .groupby("user_id")["movie_id"]
        .apply(set)
        .to_dict()
    )
    train_items = train_df.groupby("user_id")["movie_id"].apply(set).to_dict()

    ndcgs = []
    for uid, relevant_mids in test_relevant.items():
        u = uid_to_idx.get(int(uid))
        if u is None or len(relevant_mids) == 0:
            continue

        scores = model.predict_for_user(u)
        seen = train_items.get(uid, set())
        for mid in seen:
            idx = mid_to_idx.get(int(mid))
            if idx is not None:
                scores[idx] = -np.inf

        top_k_indices = np.argsort(scores)[::-1][:k]
        top_k_mids = [idx_to_mid[j] for j in top_k_indices]

        gains = np.array(
            [1.0 if mid in relevant_mids else 0.0 for mid in top_k_mids])
        dcg = float(gains @ discounts[:len(gains)])

        n_rel = min(len(relevant_mids), k)
        idcg = float(np.ones(n_rel) @ discounts[:n_rel])
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    return float(np.mean(ndcgs)) if ndcgs else 0.0


def predict_test_ratings(
    model,
    test_df: pd.DataFrame,
    uid_to_idx: Dict[int, int],
    mid_to_idx: Dict[int, int],
) -> np.ndarray:
    fallback = getattr(model, "global_mean", 3.0)
    preds = np.full(len(test_df), fallback, dtype=np.float64)

    u_indices = test_df["user_id"].map(uid_to_idx)
    i_indices = test_df["movie_id"].map(mid_to_idx)

    cache: Dict[int, np.ndarray] = {}
    for pos in range(len(test_df)):
        u = u_indices.iat[pos]
        i = i_indices.iat[pos]
        if pd.isna(u) or pd.isna(i):
            continue
        u, i = int(u), int(i)
        if u not in cache:
            cache[u] = model.predict_for_user(u)
        preds[pos] = cache[u][i]

    return preds


def precision_recall_at_k(
    model,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    uid_to_idx: dict,
    mid_to_idx: dict,
    k: int = 10,
    threshold: float = 3.5,
) -> Tuple[float, float]:
    idx_to_mid = {v: k_ for k_, v in mid_to_idx.items()}
    n_items = len(mid_to_idx)

    test_relevant = (
        test_df[test_df["rating"] >= threshold]
        .groupby("user_id")["movie_id"]
        .apply(set)
        .to_dict()
    )

    train_items = train_df.groupby("user_id")["movie_id"].apply(set).to_dict()

    precisions, recalls = [], []

    for uid, relevant_mids in test_relevant.items():
        u = uid_to_idx.get(int(uid))
        if u is None or len(relevant_mids) == 0:
            continue

        scores = model.predict_for_user(u)
        seen = train_items.get(uid, set())

        for mid in seen:
            idx = mid_to_idx.get(int(mid))
            if idx is not None:
                scores[idx] = -np.inf

        top_k_indices = np.argsort(scores)[::-1][:k]
        top_k_mids = {idx_to_mid[j] for j in top_k_indices}

        hits = len(top_k_mids & relevant_mids)
        precisions.append(hits / k)
        recalls.append(hits / len(relevant_mids))

    avg_prec = float(np.mean(precisions)) if precisions else 0.0
    avg_rec = float(np.mean(recalls)) if recalls else 0.0
    return avg_prec, avg_rec


def evaluate_model(
    model,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    uid_to_idx: Dict[int, int],
    mid_to_idx: Dict[int, int],
    k: int = 10,
) -> Dict[str, float]:
    y_pred = predict_test_ratings(model, test_df, uid_to_idx, mid_to_idx)
    y_true = test_df["rating"].values.astype(np.float64)
    rmse_val = rmse(y_true, y_pred)
    mae_val = mae(y_true, y_pred)

    prec, rec = precision_recall_at_k(
        model, train_df, test_df, uid_to_idx, mid_to_idx, k=k,
    )
    ndcg_val = ndcg_at_k(
        model, train_df, test_df, uid_to_idx, mid_to_idx, k=k,
    )
    return {
        "RMSE": round(rmse_val, 4),
        "MAE": round(mae_val, 4),
        f"Precision@{k}": round(prec, 4),
        f"Recall@{k}": round(rec, 4),
        f"NDCG@{k}": round(ndcg_val, 4),
    }


def ranking_protocol_note(n_test_per_user: int = 2, n_movies: int = 3_706) -> str:
    return (
        f"\n  ℹ️  Ranking metrics use the FULL-CATALOGUE protocol:\n"
        f"     Each user has only {n_test_per_user} held-out items, but the model\n"
        f"     must surface them among ~{n_movies:,} candidates — a deliberately\n"
        f"     challenging needle-in-a-haystack test.\n"
        f"     Protocols that rank only the test items inflate Recall to ~1.0.\n"
        f"     The relative ordering across models is the important signal.\n"
    )


def comparison_table(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    df = pd.DataFrame(results).T
    df.index.name = "Model"
    return df
