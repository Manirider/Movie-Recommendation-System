
import numpy as np
import pandas as pd
import pytest

from src.data_loader import (
    dataset_summary,
    build_interaction_matrix,
    temporal_train_test_split,
)


class TestDatasetSummary:
    def test_counts(self, tiny_ratings):
        s = dataset_summary(tiny_ratings)
        assert s["n_users"] == 5
        assert s["n_movies"] == 6
        assert s["n_ratings"] == len(tiny_ratings)

    def test_sparsity(self, tiny_ratings):
        s = dataset_summary(tiny_ratings)
        expected = 1.0 - len(tiny_ratings) / (5 * 6)
        assert abs(s["sparsity"] - expected) < 1e-6

    def test_mean_std(self, tiny_ratings):
        s = dataset_summary(tiny_ratings)
        assert abs(s["rating_mean"] - tiny_ratings["rating"].mean()) < 1e-4
        assert abs(s["rating_std"] - tiny_ratings["rating"].std()) < 1e-4


class TestBuildInteractionMatrix:
    def test_shape(self, tiny_ratings, user_ids, movie_ids):
        R, uids, mids, uid2idx, mid2idx = build_interaction_matrix(
            tiny_ratings, user_ids, movie_ids,
        )
        assert R.shape == (5, 6)
        assert len(uids) == 5
        assert len(mids) == 6

    def test_nonzero_count(self, tiny_ratings, user_ids, movie_ids):
        R, *_ = build_interaction_matrix(tiny_ratings, user_ids, movie_ids)
        assert R.nnz == len(tiny_ratings)

    def test_values_correct(self, tiny_ratings, user_ids, movie_ids):
        R, uids, mids, uid2idx, mid2idx = build_interaction_matrix(
            tiny_ratings, user_ids, movie_ids,
        )
        assert R[uid2idx[1], mid2idx[1]] == 5.0
        assert R[uid2idx[4], mid2idx[6]] == 5.0
        assert R[uid2idx[2], mid2idx[6]] == 0.0

    def test_inferred_ids(self, tiny_ratings):
        R, uids, mids, uid2idx, mid2idx = build_interaction_matrix(tiny_ratings)
        assert R.shape == (5, 6)
        np.testing.assert_array_equal(uids, np.array([1, 2, 3, 4, 5]))

    def test_superset_ids(self, tiny_ratings):
        big_uids = np.array([1, 2, 3, 4, 5, 99])
        big_mids = np.array([1, 2, 3, 4, 5, 6, 100])
        R, *_ = build_interaction_matrix(tiny_ratings, big_uids, big_mids)
        assert R.shape == (6, 7)
        assert R[5].nnz == 0


class TestTemporalSplit:
    def test_sizes(self, tiny_ratings):
        train, test = temporal_train_test_split(tiny_ratings, n_test_per_user=1)
        assert len(test) == 5
        assert len(train) + len(test) == len(tiny_ratings)

    def test_no_overlap(self, tiny_ratings):
        train, test = temporal_train_test_split(tiny_ratings, n_test_per_user=1)
        train_keys = set(zip(train["user_id"], train["movie_id"]))
        test_keys = set(zip(test["user_id"], test["movie_id"]))
        assert train_keys.isdisjoint(test_keys)

    def test_temporal_order(self, tiny_ratings):
        train, test = temporal_train_test_split(tiny_ratings, n_test_per_user=1)
        for uid in tiny_ratings["user_id"].unique():
            max_train_ts = train.loc[train["user_id"] == uid, "timestamp"].max()
            min_test_ts = test.loc[test["user_id"] == uid, "timestamp"].min()
            assert min_test_ts >= max_train_ts, (
                f"User {uid}: test ts {min_test_ts} < train ts {max_train_ts}"
            )

    def test_two_per_user(self, tiny_ratings):
        train, test = temporal_train_test_split(tiny_ratings, n_test_per_user=2)
        for uid in tiny_ratings["user_id"].unique():
            user_test = test[test["user_id"] == uid]
            n_total = len(tiny_ratings[tiny_ratings["user_id"] == uid])
            expected = min(2, n_total)
            assert len(user_test) == expected
