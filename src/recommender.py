
from typing import Tuple, List

import numpy as np
import pandas as pd


def recommend_top_n(
    model,
    uid_to_idx: dict,
    mid_to_idx: dict,
    user_id: int,
    movies_df: pd.DataFrame,
    train_df: pd.DataFrame,
    n: int = 10,
) -> pd.DataFrame:
    u = uid_to_idx.get(int(user_id))
    if u is None:
        pop = popular_movies(train_df, movies_df, n=n)
        pop = pop.rename(columns={"avg_rating": "predicted_rating"})
        return pop[["movie_id", "title", "predicted_rating"]]

    scores = model.predict_for_user(u)

    idx_to_mid = {v: k for k, v in mid_to_idx.items()}
    seen = set(train_df.loc[train_df["user_id"] == user_id, "movie_id"].values)

    candidates = []
    for i, sc in enumerate(scores):
        mid = idx_to_mid[i]
        if mid not in seen:
            candidates.append((mid, float(sc)))

    candidates.sort(key=lambda x: x[1], reverse=True)
    top = candidates[:n]

    out = pd.DataFrame(top, columns=["movie_id", "predicted_rating"])
    out["predicted_rating"] = out["predicted_rating"].round(3)
    out = out.merge(movies_df[["movie_id", "title"]], on="movie_id", how="left")
    return out[["movie_id", "title", "predicted_rating"]]


def popular_movies(
    train_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    n: int = 10,
    min_ratings: int = 50,
) -> pd.DataFrame:
    stats = (
        train_df.groupby("movie_id")["rating"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "avg_rating", "count": "n_ratings"})
        .reset_index()
    )
    C = train_df["rating"].mean()
    m = min_ratings

    stats["score"] = (
        (stats["n_ratings"] * stats["avg_rating"] + m * C)
        / (stats["n_ratings"] + m)
    )
    stats = stats.sort_values("score", ascending=False).head(n)
    stats = stats.merge(movies_df[["movie_id", "title"]], on="movie_id", how="left")
    return stats[["movie_id", "title", "avg_rating", "n_ratings"]].reset_index(drop=True)


def get_item_embeddings(
    svd_model, mid_to_idx: dict,
) -> Tuple[np.ndarray, List[int]]:
    import warnings
    from src.embeddings import extract_item_embeddings

    warnings.warn(
        "get_item_embeddings() is deprecated — use "
        "src.embeddings.extract_item_embeddings() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return extract_item_embeddings(svd_model, mid_to_idx)
