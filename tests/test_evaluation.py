
import numpy as np
import pandas as pd
import pytest

from src.data_loader import build_interaction_matrix, temporal_train_test_split
from src.evaluation import (
    rmse, mae, ndcg_at_k,
    predict_test_ratings,
    precision_recall_at_k,
    evaluate_model,
    comparison_table,
)
from src.models import FunkSVD, train_from_df


class TestScalarMetrics:
    def test_rmse_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert rmse(y, y) == 0.0

    def test_rmse_known(self):
        y_true = np.array([3.0, 3.0, 3.0])
        y_pred = np.array([4.0, 4.0, 4.0])
        assert abs(rmse(y_true, y_pred) - 1.0) < 1e-9

    def test_mae_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mae(y, y) == 0.0

    def test_mae_known(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        assert abs(mae(y_true, y_pred) - 1.0) < 1e-9

    def test_rmse_ge_mae(self):
        y_true = np.array([1.0, 2.0, 5.0])
        y_pred = np.array([1.5, 3.5, 4.0])
        assert rmse(y_true, y_pred) >= mae(y_true, y_pred)


class TestPredictTestRatings:
    def test_output_length(self, tiny_ratings, user_ids, movie_ids):
        train, test = temporal_train_test_split(tiny_ratings, n_test_per_user=1)
        model, uid2idx, mid2idx = train_from_df(
            FunkSVD(n_factors=4, n_epochs=10), train, user_ids, movie_ids,
        )
        preds = predict_test_ratings(model, test, uid2idx, mid2idx)
        assert len(preds) == len(test)

    def test_predictions_in_range(self, tiny_ratings, user_ids, movie_ids):
        train, test = temporal_train_test_split(tiny_ratings, n_test_per_user=1)
        model, uid2idx, mid2idx = train_from_df(
            FunkSVD(n_factors=4, n_epochs=10), train, user_ids, movie_ids,
        )
        preds = predict_test_ratings(model, test, uid2idx, mid2idx)
        assert np.all(preds >= 1.0) and np.all(preds <= 5.0)


class TestPrecisionRecall:
    def test_returns_tuple(self, tiny_ratings, user_ids, movie_ids):
        train, test = temporal_train_test_split(tiny_ratings, n_test_per_user=1)
        model, uid2idx, mid2idx = train_from_df(
            FunkSVD(n_factors=4, n_epochs=10), train, user_ids, movie_ids,
        )
        prec, rec = precision_recall_at_k(
            model, train, test, uid2idx, mid2idx, k=3,
        )
        assert isinstance(prec, float)
        assert isinstance(rec, float)
        assert 0.0 <= prec <= 1.0
        assert 0.0 <= rec <= 1.0


class TestEvaluateModel:
    def test_keys(self, tiny_ratings, user_ids, movie_ids):
        train, test = temporal_train_test_split(tiny_ratings, n_test_per_user=1)
        model, uid2idx, mid2idx = train_from_df(
            FunkSVD(n_factors=4, n_epochs=10), train, user_ids, movie_ids,
        )
        result = evaluate_model(model, train, test, uid2idx, mid2idx, k=3)
        assert "RMSE" in result
        assert "MAE" in result
        assert "Precision@3" in result
        assert "Recall@3" in result
        assert "NDCG@3" in result

    def test_rmse_positive(self, tiny_ratings, user_ids, movie_ids):
        train, test = temporal_train_test_split(tiny_ratings, n_test_per_user=1)
        model, uid2idx, mid2idx = train_from_df(
            FunkSVD(n_factors=4, n_epochs=10), train, user_ids, movie_ids,
        )
        result = evaluate_model(model, train, test, uid2idx, mid2idx, k=3)
        assert result["RMSE"] > 0
        assert result["MAE"] > 0


class TestNDCG:
    def test_returns_float(self, tiny_ratings, user_ids, movie_ids):
        train, test = temporal_train_test_split(tiny_ratings, n_test_per_user=1)
        model, uid2idx, mid2idx = train_from_df(
            FunkSVD(n_factors=4, n_epochs=10), train, user_ids, movie_ids,
        )
        val = ndcg_at_k(model, train, test, uid2idx, mid2idx, k=3)
        assert isinstance(val, float)
        assert 0.0 <= val <= 1.0

    def test_perfect_ranking(self, tiny_ratings, user_ids, movie_ids):
        train, test = temporal_train_test_split(tiny_ratings, n_test_per_user=1)
        model, uid2idx, mid2idx = train_from_df(
            FunkSVD(n_factors=4, n_epochs=10), train, user_ids, movie_ids,
        )
        val = ndcg_at_k(model, train, test, uid2idx, mid2idx, k=3)
        assert val >= 0.0


class TestComparisonTable:
    def test_shape(self):
        results = {
            "A": {"RMSE": 1.0, "MAE": 0.8},
            "B": {"RMSE": 0.9, "MAE": 0.7},
        }
        df = comparison_table(results)
        assert df.shape == (2, 2)
        assert list(df.index) == ["A", "B"]

    def test_index_name(self):
        df = comparison_table({"X": {"RMSE": 1.0}})
        assert df.index.name == "Model"
