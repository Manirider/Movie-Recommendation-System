
import os
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def load_ratings(data_dir: str) -> pd.DataFrame:
    fp = os.path.join(data_dir, "ratings.dat")
    df = pd.read_csv(
        fp, sep="::", engine="python", header=None,
        names=["user_id", "movie_id", "rating", "timestamp"],
        dtype={"user_id": np.int32, "movie_id": np.int32,
               "rating": np.float32, "timestamp": np.int64},
    )
    return df


def load_movies(data_dir: str) -> pd.DataFrame:
    fp = os.path.join(data_dir, "movies.dat")
    df = pd.read_csv(
        fp, sep="::", engine="python", header=None,
        names=["movie_id", "title", "genres"], encoding="latin-1",
    )
    df["genres"] = df["genres"].str.split("|")
    return df


def load_users(data_dir: str) -> pd.DataFrame:
    fp = os.path.join(data_dir, "users.dat")
    return pd.read_csv(
        fp, sep="::", engine="python", header=None,
        names=["user_id", "gender", "age", "occupation", "zip_code"],
    )


def dataset_summary(ratings: pd.DataFrame) -> Dict[str, object]:
    n_users = ratings["user_id"].nunique()
    n_movies = ratings["movie_id"].nunique()
    n_ratings = len(ratings)
    sparsity = 1.0 - n_ratings / (n_users * n_movies)
    return dict(
        n_users=n_users, n_movies=n_movies, n_ratings=n_ratings,
        sparsity=sparsity,
        rating_mean=float(ratings["rating"].mean()),
        rating_std=float(ratings["rating"].std()),
    )


def build_interaction_matrix(
    ratings: pd.DataFrame,
    user_ids: np.ndarray | None = None,
    movie_ids: np.ndarray | None = None,
) -> Tuple[csr_matrix, np.ndarray, np.ndarray, Dict[int, int], Dict[int, int]]:
    if user_ids is None:
        user_ids = np.sort(ratings["user_id"].unique())
    if movie_ids is None:
        movie_ids = np.sort(ratings["movie_id"].unique())

    uid_to_idx = {int(uid): i for i, uid in enumerate(user_ids)}
    mid_to_idx = {int(mid): i for i, mid in enumerate(movie_ids)}

    rows = ratings["user_id"].map(uid_to_idx).values
    cols = ratings["movie_id"].map(mid_to_idx).values
    vals = ratings["rating"].values.astype(np.float32)

    R = csr_matrix((vals, (rows, cols)),
                   shape=(len(user_ids), len(movie_ids)))
    return R, user_ids, movie_ids, uid_to_idx, mid_to_idx


def temporal_train_test_split(
    ratings: pd.DataFrame,
    n_test_per_user: int = 2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sorted_df = ratings.sort_values(["user_id", "timestamp"])
    sorted_df["_rank"] = (
        sorted_df.groupby("user_id")["timestamp"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    test_df = sorted_df[sorted_df["_rank"] <= n_test_per_user].drop(columns="_rank")
    train_df = sorted_df[sorted_df["_rank"] > n_test_per_user].drop(columns="_rank")
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
