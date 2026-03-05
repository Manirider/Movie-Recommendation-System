
import numpy as np
import pandas as pd
import pytest

from src.tuning import (
    temporal_cv_folds,
    grid_search_svd,
    grid_search_knn,
    best_params,
)


class TestTemporalCVFolds:
    def test_returns_list_of_tuples(self, tiny_ratings):
        folds = temporal_cv_folds(tiny_ratings, n_folds=2, n_test_per_user=1)
        assert isinstance(folds, list)
        assert all(isinstance(f, tuple) and len(f) == 2 for f in folds)

    def test_fold_count(self, tiny_ratings):
        folds = temporal_cv_folds(tiny_ratings, n_folds=2, n_test_per_user=1)
        assert len(folds) <= 2

    def test_no_overlap_in_folds(self, tiny_ratings):
        folds = temporal_cv_folds(tiny_ratings, n_folds=2, n_test_per_user=1)
        for train, val in folds:
            train_keys = set(zip(train["user_id"], train["movie_id"]))
            val_keys = set(zip(val["user_id"], val["movie_id"]))
            assert train_keys.isdisjoint(val_keys)

    def test_temporal_order_preserved(self, tiny_ratings):
        folds = temporal_cv_folds(tiny_ratings, n_folds=2, n_test_per_user=1)
        for train, val in folds:
            for uid in train["user_id"].unique():
                user_train = train[train["user_id"] == uid]
                user_val = val[val["user_id"] == uid]
                if len(user_train) > 0 and len(user_val) > 0:
                    assert user_train["timestamp"].max() <= user_val["timestamp"].min()

    def test_later_folds_have_more_data(self, tiny_ratings):
        folds = temporal_cv_folds(tiny_ratings, n_folds=2, n_test_per_user=1)
        if len(folds) >= 2:
            assert len(folds[1][0]) >= len(folds[0][0])


class TestGridSearchSVD:
    def test_returns_dataframe(self, tiny_ratings):
        results = grid_search_svd(
            tiny_ratings,
            param_grid={"n_factors": [4], "n_epochs": [5], "lr": [0.01], "reg": [0.01]},
            n_folds=2,
            n_test_per_user=1,
            verbose=False,
        )
        assert isinstance(results, pd.DataFrame)
        assert "RMSE" in results.columns
        assert "MAE" in results.columns
        assert "fold" in results.columns

    def test_row_count_matches_combos_times_folds(self, tiny_ratings):
        results = grid_search_svd(
            tiny_ratings,
            param_grid={"n_factors": [4, 8], "n_epochs": [5], "lr": [0.01], "reg": [0.01]},
            n_folds=2,
            n_test_per_user=1,
            verbose=False,
        )
        folds = temporal_cv_folds(tiny_ratings, n_folds=2, n_test_per_user=1)
        assert len(results) == 2 * len(folds)

    def test_rmse_is_positive(self, tiny_ratings):
        results = grid_search_svd(
            tiny_ratings,
            param_grid={"n_factors": [4], "n_epochs": [5], "lr": [0.01], "reg": [0.01]},
            n_folds=2,
            n_test_per_user=1,
            verbose=False,
        )
        assert (results["RMSE"] > 0).all()


class TestGridSearchKNN:
    def test_returns_dataframe(self, tiny_ratings):
        results = grid_search_knn(
            tiny_ratings,
            param_grid={"k": [2], "user_based": [True]},
            n_folds=2,
            n_test_per_user=1,
            verbose=False,
        )
        assert isinstance(results, pd.DataFrame)
        assert "RMSE" in results.columns

    def test_both_user_and_item(self, tiny_ratings):
        results = grid_search_knn(
            tiny_ratings,
            param_grid={"k": [2], "user_based": [True, False]},
            n_folds=2,
            n_test_per_user=1,
            verbose=False,
        )
        folds = temporal_cv_folds(tiny_ratings, n_folds=2, n_test_per_user=1)
        assert len(results) == 2 * len(folds)


class TestBestParams:
    def test_extracts_best_rmse(self, tiny_ratings):
        results = grid_search_svd(
            tiny_ratings,
            param_grid={"n_factors": [4, 8], "n_epochs": [5], "lr": [0.01], "reg": [0.01]},
            n_folds=2,
            n_test_per_user=1,
            verbose=False,
        )
        bp = best_params(results, metric="RMSE", lower_is_better=True)
        assert "n_factors" in bp
        assert "mean_RMSE" in bp

    def test_higher_is_better(self, tiny_ratings):
        results = grid_search_svd(
            tiny_ratings,
            param_grid={"n_factors": [4, 8], "n_epochs": [5], "lr": [0.01], "reg": [0.01]},
            n_folds=2,
            n_test_per_user=1,
            verbose=False,
        )
        ndcg_col = [c for c in results.columns if c.startswith("NDCG")][0]
        bp = best_params(results, metric=ndcg_col, lower_is_better=False)
        assert f"mean_{ndcg_col}" in bp
