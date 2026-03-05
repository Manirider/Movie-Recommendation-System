
from typing import Dict, List, Tuple
import itertools
import time

import numpy as np
import pandas as pd

from src.data_loader import (
    build_interaction_matrix,
    temporal_train_test_split,
)
from src.models import FunkSVD, KNNRecommender, train_from_df
from src.evaluation import evaluate_model


def temporal_cv_folds(
    ratings: pd.DataFrame,
    n_folds: int = 3,
    n_test_per_user: int = 2,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    sorted_df = ratings.sort_values(["user_id", "timestamp"]).copy()
    sorted_df["_global_rank"] = (
        sorted_df.groupby("user_id")["timestamp"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    folds = []
    for fold_idx in range(n_folds):
        drop_count = (n_folds - 1 - fold_idx) * n_test_per_user

        if drop_count > 0:
            window = sorted_df[sorted_df["_global_rank"] > drop_count].copy()
        else:
            window = sorted_df.copy()

        window_clean = window.drop(columns="_global_rank")
        train, val = temporal_train_test_split(window_clean, n_test_per_user)

        if len(train) == 0 or len(val) == 0:
            continue
        folds.append((train, val))

    return folds


def grid_search_svd(
    ratings: pd.DataFrame,
    param_grid: Dict[str, List],
    n_folds: int = 3,
    n_test_per_user: int = 2,
    eval_k: int = 10,
    verbose: bool = True,
) -> pd.DataFrame:
    all_user_ids = np.sort(ratings["user_id"].unique())
    all_movie_ids = np.sort(ratings["movie_id"].unique())

    folds = temporal_cv_folds(ratings, n_folds, n_test_per_user)
    if verbose:
        print(f"Generated {len(folds)} temporal CV folds")

    keys = sorted(param_grid.keys())
    combos = list(itertools.product(*(param_grid[k] for k in keys)))
    if verbose:
        print(f"Testing {len(combos)} parameter combinations × {len(folds)} folds "
              f"= {len(combos) * len(folds)} runs\n")

    records = []
    for combo_idx, vals in enumerate(combos):
        params = dict(zip(keys, vals))
        fold_results = []

        for fold_idx, (train_df, val_df) in enumerate(folds):
            t0 = time.time()
            model = FunkSVD(
                n_factors=params.get("n_factors", 100),
                n_epochs=params.get("n_epochs", 30),
                lr=params.get("lr", 0.005),
                reg=params.get("reg", 0.02),
            )
            model, uid2idx, mid2idx = train_from_df(
                model, train_df, all_user_ids, all_movie_ids,
            )
            metrics = evaluate_model(
                model, train_df, val_df, uid2idx, mid2idx, k=eval_k,
            )
            elapsed = time.time() - t0

            row = {**params, "fold": fold_idx, **metrics, "time_s": round(elapsed, 1)}
            records.append(row)
            fold_results.append(metrics)

        mean_rmse = np.mean([r["RMSE"] for r in fold_results])
        if verbose:
            param_str = ", ".join(f"{k}={params[k]}" for k in keys)
            print(f"  [{combo_idx+1}/{len(combos)}] {param_str}  →  "
                  f"mean RMSE = {mean_rmse:.4f}")

    results_df = pd.DataFrame(records)
    return results_df


def grid_search_knn(
    ratings: pd.DataFrame,
    param_grid: Dict[str, List],
    n_folds: int = 3,
    n_test_per_user: int = 2,
    eval_k: int = 10,
    verbose: bool = True,
) -> pd.DataFrame:
    all_user_ids = np.sort(ratings["user_id"].unique())
    all_movie_ids = np.sort(ratings["movie_id"].unique())

    folds = temporal_cv_folds(ratings, n_folds, n_test_per_user)
    if verbose:
        print(f"Generated {len(folds)} temporal CV folds")

    keys = sorted(param_grid.keys())
    combos = list(itertools.product(*(param_grid[k] for k in keys)))
    if verbose:
        print(f"Testing {len(combos)} parameter combinations × {len(folds)} folds "
              f"= {len(combos) * len(folds)} runs\n")

    records = []
    for combo_idx, vals in enumerate(combos):
        params = dict(zip(keys, vals))
        fold_results = []

        for fold_idx, (train_df, val_df) in enumerate(folds):
            t0 = time.time()
            model = KNNRecommender(
                k=params.get("k", 40),
                user_based=params.get("user_based", True),
            )
            model, uid2idx, mid2idx = train_from_df(
                model, train_df, all_user_ids, all_movie_ids,
            )
            metrics = evaluate_model(
                model, train_df, val_df, uid2idx, mid2idx, k=eval_k,
            )
            elapsed = time.time() - t0

            row = {**params, "fold": fold_idx, **metrics, "time_s": round(elapsed, 1)}
            records.append(row)
            fold_results.append(metrics)

        mean_rmse = np.mean([r["RMSE"] for r in fold_results])
        if verbose:
            param_str = ", ".join(f"{k}={params[k]}" for k in keys)
            print(f"  [{combo_idx+1}/{len(combos)}] {param_str}  →  "
                  f"mean RMSE = {mean_rmse:.4f}")

    return pd.DataFrame(records)


def best_params(
    results_df: pd.DataFrame,
    metric: str = "RMSE",
    lower_is_better: bool = True,
) -> Dict:
    metric_cols = {"fold", "time_s", "RMSE", "MAE"} | {
        c for c in results_df.columns
        if c.startswith("Precision") or c.startswith("Recall") or c.startswith("NDCG")
    }
    param_cols = [c for c in results_df.columns if c not in metric_cols]

    agg = results_df.groupby(param_cols)[metric].mean().reset_index()
    agg.rename(columns={metric: f"mean_{metric}"}, inplace=True)

    if lower_is_better:
        best_row = agg.loc[agg[f"mean_{metric}"].idxmin()]
    else:
        best_row = agg.loc[agg[f"mean_{metric}"].idxmax()]

    return best_row.to_dict()
