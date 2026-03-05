
import numpy as np
import pytest

from src.data_loader import build_interaction_matrix
from src.models import KNNRecommender, FunkSVD, train_from_df


def _fit_model(model, tiny_ratings, user_ids, movie_ids):
    R, uids, mids, uid2idx, mid2idx = build_interaction_matrix(
        tiny_ratings, user_ids, movie_ids,
    )
    model.fit(R, uids, mids)
    return model, uid2idx, mid2idx


class TestKNNUserBased:
    @pytest.fixture(autouse=True)
    def setup(self, tiny_ratings, user_ids, movie_ids):
        self.model, self.uid2idx, self.mid2idx = _fit_model(
            KNNRecommender(k=3, user_based=True),
            tiny_ratings, user_ids, movie_ids,
        )

    def test_predict_range(self):
        for u in range(5):
            for i in range(6):
                p = self.model.predict(u, i)
                assert 1.0 <= p <= 5.0, f"predict({u},{i}) = {p}"

    def test_predict_for_user_shape(self):
        scores = self.model.predict_for_user(0)
        assert scores.shape == (6,)

    def test_predict_for_user_range(self):
        for u in range(5):
            scores = self.model.predict_for_user(u)
            assert np.all(scores >= 1.0) and np.all(scores <= 5.0)

    def test_name(self):
        assert "User" in self.model.name
        assert "KNN" in self.model.name


class TestKNNItemBased:
    @pytest.fixture(autouse=True)
    def setup(self, tiny_ratings, user_ids, movie_ids):
        self.model, self.uid2idx, self.mid2idx = _fit_model(
            KNNRecommender(k=3, user_based=False),
            tiny_ratings, user_ids, movie_ids,
        )

    def test_predict_range(self):
        for u in range(5):
            for i in range(6):
                p = self.model.predict(u, i)
                assert 1.0 <= p <= 5.0

    def test_predict_for_user_shape(self):
        scores = self.model.predict_for_user(0)
        assert scores.shape == (6,)

    def test_name(self):
        assert "Item" in self.model.name


class TestFunkSVD:
    @pytest.fixture(autouse=True)
    def setup(self, tiny_ratings, user_ids, movie_ids):
        self.model, self.uid2idx, self.mid2idx = _fit_model(
            FunkSVD(n_factors=8, n_epochs=50, lr=0.01, reg=0.01),
            tiny_ratings, user_ids, movie_ids,
        )

    def test_predict_range(self):
        for u in range(5):
            for i in range(6):
                p = self.model.predict(u, i)
                assert 1.0 <= p <= 5.0

    def test_predict_for_user_shape(self):
        scores = self.model.predict_for_user(0)
        assert scores.shape == (6,)

    def test_predict_for_user_range(self):
        for u in range(5):
            scores = self.model.predict_for_user(u)
            assert np.all(scores >= 1.0) and np.all(scores <= 5.0)

    def test_fitting_reduces_error(self, tiny_ratings, user_ids, movie_ids):
        from src.data_loader import build_interaction_matrix
        R, _, _, uid2idx, mid2idx = build_interaction_matrix(
            tiny_ratings, user_ids, movie_ids,
        )
        errors = []
        for _, row in tiny_ratings.iterrows():
            u = uid2idx[int(row["user_id"])]
            i = mid2idx[int(row["movie_id"])]
            errors.append(abs(self.model.predict(u, i) - row["rating"]))
        avg_err = np.mean(errors)
        assert avg_err < 2.0, f"Avg absolute error on training data = {avg_err:.2f}"

    def test_item_embeddings_shape(self):
        Q = self.model.get_item_embeddings()
        assert Q.shape == (6, 8)

    def test_item_embeddings_are_copy(self):
        Q1 = self.model.get_item_embeddings()
        Q2 = self.model.get_item_embeddings()
        Q1[:] = 0.0
        assert not np.allclose(Q2, 0.0), "get_item_embeddings should return a copy"

    def test_name(self):
        assert "SVD" in self.model.name


class TestTrainFromDf:
    def test_returns_fitted_model(self, tiny_ratings, user_ids, movie_ids):
        model, uid2idx, mid2idx = train_from_df(
            FunkSVD(n_factors=4, n_epochs=5),
            tiny_ratings, user_ids, movie_ids,
        )
        assert hasattr(model, "P")
        assert len(uid2idx) == 5
        assert len(mid2idx) == 6

    def test_knn_via_helper(self, tiny_ratings, user_ids, movie_ids):
        model, uid2idx, mid2idx = train_from_df(
            KNNRecommender(k=2, user_based=True),
            tiny_ratings, user_ids, movie_ids,
        )
        scores = model.predict_for_user(uid2idx[1])
        assert scores.shape == (6,)
