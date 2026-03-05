
import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_loader import (
    dataset_summary,
    build_interaction_matrix,
    temporal_train_test_split,
)
from src.models import KNNRecommender, FunkSVD, train_from_df
from src.evaluation import evaluate_model, comparison_table
from src.recommender import recommend_top_n, popular_movies
from src.embeddings import (
    extract_item_embeddings,
    reduce_to_2d,
    build_genre_labels,
    plot_embedding_scatter,
    visualize_embeddings,
)


class TestFullPipeline:

    @pytest.fixture(autouse=True)
    def run_pipeline(self, tiny_ratings, tiny_movies, user_ids, movie_ids):
        self.ratings = tiny_ratings
        self.movies = tiny_movies
        self.user_ids = user_ids
        self.movie_ids = movie_ids

        self.stats = dataset_summary(tiny_ratings)

        self.train_df, self.test_df = temporal_train_test_split(
            tiny_ratings, n_test_per_user=1,
        )

        self.uknn, self.uid2idx_uk, self.mid2idx_uk = train_from_df(
            KNNRecommender(k=3, user_based=True),
            self.train_df, user_ids, movie_ids,
        )
        self.iknn, self.uid2idx_ik, self.mid2idx_ik = train_from_df(
            KNNRecommender(k=3, user_based=False),
            self.train_df, user_ids, movie_ids,
        )
        self.svd, self.uid2idx_sv, self.mid2idx_sv = train_from_df(
            FunkSVD(n_factors=4, n_epochs=10, lr=0.01, reg=0.01),
            self.train_df, user_ids, movie_ids,
        )

        self.res_uknn = evaluate_model(
            self.uknn, self.train_df, self.test_df,
            self.uid2idx_uk, self.mid2idx_uk, k=3,
        )
        self.res_iknn = evaluate_model(
            self.iknn, self.train_df, self.test_df,
            self.uid2idx_ik, self.mid2idx_ik, k=3,
        )
        self.res_svd = evaluate_model(
            self.svd, self.train_df, self.test_df,
            self.uid2idx_sv, self.mid2idx_sv, k=3,
        )

        self.results = {
            "User-KNN": self.res_uknn,
            "Item-KNN": self.res_iknn,
            "SVD": self.res_svd,
        }

    def test_summary_correct(self):
        assert self.stats["n_users"] == 5
        assert self.stats["n_movies"] == 6
        assert self.stats["n_ratings"] == 18

    def test_split_sizes(self):
        assert len(self.train_df) + len(self.test_df) == 18
        assert len(self.test_df) == 5

    def test_split_no_leakage(self):
        for uid in self.ratings["user_id"].unique():
            t_train = self.train_df[self.train_df["user_id"] == uid]["timestamp"]
            t_test = self.test_df[self.test_df["user_id"] == uid]["timestamp"]
            if len(t_train) > 0 and len(t_test) > 0:
                assert t_train.max() <= t_test.min()

    def test_all_models_have_metrics(self):
        for name, res in self.results.items():
            assert "RMSE" in res, f"{name} missing RMSE"
            assert "MAE" in res, f"{name} missing MAE"
            assert "Precision@3" in res, f"{name} missing Precision@3"
            assert "Recall@3" in res, f"{name} missing Recall@3"
            assert "NDCG@3" in res, f"{name} missing NDCG@3"

    def test_rmse_positive(self):
        for name, res in self.results.items():
            assert res["RMSE"] > 0, f"{name} RMSE should be > 0"

    def test_comparison_table(self):
        comp = comparison_table(self.results)
        assert comp.shape == (3, 5)
        assert list(comp.index) == ["User-KNN", "Item-KNN", "SVD"]

    def test_recommend_known_user(self):
        recs = recommend_top_n(
            self.svd, self.uid2idx_sv, self.mid2idx_sv,
            user_id=1, movies_df=self.movies, train_df=self.train_df, n=3,
        )
        assert list(recs.columns) == ["movie_id", "title", "predicted_rating"]
        rated = set(self.train_df[self.train_df["user_id"] == 1]["movie_id"])
        assert set(recs["movie_id"]).isdisjoint(rated)

    def test_recommend_cold_start(self):
        recs = recommend_top_n(
            self.svd, self.uid2idx_sv, self.mid2idx_sv,
            user_id=9999, movies_df=self.movies, train_df=self.train_df, n=3,
        )
        assert list(recs.columns) == ["movie_id", "title", "predicted_rating"]

    def test_popular_movies(self):
        pop = popular_movies(self.train_df, self.movies, n=3, min_ratings=1)
        assert len(pop) == 3

    def test_embedding_extraction(self):
        Q, raw_ids = extract_item_embeddings(self.svd, self.mid2idx_sv)
        assert Q.shape == (6, 4)
        assert len(raw_ids) == 6

    def test_pca_reduction(self):
        Q, _ = extract_item_embeddings(self.svd, self.mid2idx_sv)
        coords, pca = reduce_to_2d(Q, method="pca")
        assert coords.shape == (6, 2)
        assert hasattr(pca, "explained_variance_ratio_")

    def test_tsne_reduction(self):
        Q, _ = extract_item_embeddings(self.svd, self.mid2idx_sv)
        coords, _ = reduce_to_2d(Q, method="tsne", perplexity=2)
        assert coords.shape == (6, 2)

    def test_genre_labels(self):
        Q, raw_ids = extract_item_embeddings(self.svd, self.mid2idx_sv)
        labels = build_genre_labels(raw_ids, self.movies)
        assert len(labels) == 6
        assert all(isinstance(g, str) for g in labels)

    def test_embedding_plot(self):
        Q, raw_ids = extract_item_embeddings(self.svd, self.mid2idx_sv)
        coords, _ = reduce_to_2d(Q, method="pca")
        genres = build_genre_labels(raw_ids, self.movies)
        fig, ax = plt.subplots()
        plot_embedding_scatter(
            coords, genres, raw_ids, self.movies, ax,
            title="Integration Test", label_movies=[1, 4],
        )
        plt.close(fig)

    def test_visualize_embeddings_end_to_end(self):
        fig_pca, fig_tsne = visualize_embeddings(
            self.svd, self.mid2idx_sv, self.movies,
            landmark_ids=[1, 4],
        )
        assert isinstance(fig_pca, plt.Figure)
        assert isinstance(fig_tsne, plt.Figure)
        plt.close(fig_pca)
        plt.close(fig_tsne)


class TestCrossModelConsistency:

    def test_predict_for_user_shape_all_models(
        self, tiny_ratings, user_ids, movie_ids,
    ):
        train, _ = temporal_train_test_split(tiny_ratings, n_test_per_user=1)
        models = [
            KNNRecommender(k=3, user_based=True),
            KNNRecommender(k=3, user_based=False),
            FunkSVD(n_factors=4, n_epochs=5),
        ]
        for model in models:
            m, uid2idx, mid2idx = train_from_df(model, train, user_ids, movie_ids)
            for u in range(len(user_ids)):
                scores = m.predict_for_user(u)
                assert scores.shape == (len(movie_ids),), f"{m.name} wrong shape"
                assert np.all(scores >= 1.0) and np.all(scores <= 5.0), (
                    f"{m.name} predictions out of [1, 5]"
                )

    def test_evaluate_model_keys_consistent(
        self, tiny_ratings, user_ids, movie_ids,
    ):
        train, test = temporal_train_test_split(tiny_ratings, n_test_per_user=1)
        models = [
            KNNRecommender(k=3, user_based=True),
            FunkSVD(n_factors=4, n_epochs=5),
        ]
        all_keys = None
        for model in models:
            m, uid2idx, mid2idx = train_from_df(model, train, user_ids, movie_ids)
            res = evaluate_model(m, train, test, uid2idx, mid2idx, k=3)
            if all_keys is None:
                all_keys = set(res.keys())
            else:
                assert set(res.keys()) == all_keys, (
                    f"{m.name} keys {set(res.keys())} != {all_keys}"
                )
