
import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models import FunkSVD, train_from_df
from src.embeddings import (
    extract_item_embeddings,
    reduce_to_2d,
    build_genre_labels,
    plot_embedding_scatter,
    visualize_embeddings,
    LANDMARK_MOVIES,
)


@pytest.fixture()
def trained_svd(tiny_ratings, user_ids, movie_ids):
    model = FunkSVD(n_factors=4, n_epochs=10, lr=0.01, reg=0.01)
    model, uid2idx, mid2idx = train_from_df(model, tiny_ratings, user_ids, movie_ids)
    return model, uid2idx, mid2idx


@pytest.fixture()
def embeddings_and_ids(trained_svd):
    model, _, mid2idx = trained_svd
    Q, raw_ids = extract_item_embeddings(model, mid2idx)
    return Q, raw_ids


class TestExtractItemEmbeddings:
    def test_shape(self, embeddings_and_ids, movie_ids):
        Q, raw_ids = embeddings_and_ids
        assert Q.shape == (len(movie_ids), 4)

    def test_raw_ids_match_movies(self, embeddings_and_ids, movie_ids):
        _, raw_ids = embeddings_and_ids
        assert sorted(raw_ids) == sorted(movie_ids.tolist())

    def test_returns_copy(self, trained_svd):
        model, _, mid2idx = trained_svd
        Q, _ = extract_item_embeddings(model, mid2idx)
        original_sum = model.Q.sum()
        Q[:] = 0
        assert model.Q.sum() == pytest.approx(original_sum)


class TestReduceTo2d:
    def test_pca_shape(self, embeddings_and_ids):
        Q, _ = embeddings_and_ids
        coords, reducer = reduce_to_2d(Q, method="pca")
        assert coords.shape == (Q.shape[0], 2)

    def test_tsne_shape(self, embeddings_and_ids):
        Q, _ = embeddings_and_ids
        coords, reducer = reduce_to_2d(Q, method="tsne", perplexity=2)
        assert coords.shape == (Q.shape[0], 2)

    def test_pca_variance_ratio(self, embeddings_and_ids):
        Q, _ = embeddings_and_ids
        _, reducer = reduce_to_2d(Q, method="pca")
        ratios = reducer.explained_variance_ratio_
        assert len(ratios) == 2
        assert all(0 <= r <= 1 for r in ratios)

    def test_invalid_method_raises(self, embeddings_and_ids):
        Q, _ = embeddings_and_ids
        with pytest.raises(ValueError, match="Unknown method"):
            reduce_to_2d(Q, method="umap")

    def test_deterministic_pca(self, embeddings_and_ids):
        Q, _ = embeddings_and_ids
        c1, _ = reduce_to_2d(Q, method="pca", random_state=0)
        c2, _ = reduce_to_2d(Q, method="pca", random_state=0)
        np.testing.assert_array_equal(c1, c2)


class TestBuildGenreLabels:
    def test_length_matches(self, tiny_movies):
        raw_ids = [1, 2, 3, 4, 5, 6]
        labels = build_genre_labels(raw_ids, tiny_movies)
        assert len(labels) == 6

    def test_primary_genre_extracted(self, tiny_movies):
        labels = build_genre_labels([2], tiny_movies)
        assert labels == ["Action"]

    def test_unknown_movie(self, tiny_movies):
        labels = build_genre_labels([9999], tiny_movies)
        assert labels == ["Unknown"]


class TestPlotEmbeddingScatter:
    def test_no_error(self, embeddings_and_ids, tiny_movies):
        Q, raw_ids = embeddings_and_ids
        coords, _ = reduce_to_2d(Q, method="pca")
        genres = build_genre_labels(raw_ids, tiny_movies)
        fig, ax = plt.subplots()
        plot_embedding_scatter(
            coords, genres, raw_ids, tiny_movies, ax,
            title="Test", label_movies=[1, 4],
        )
        plt.close(fig)

    def test_with_no_labels(self, embeddings_and_ids, tiny_movies):
        Q, raw_ids = embeddings_and_ids
        coords, _ = reduce_to_2d(Q, method="pca")
        genres = build_genre_labels(raw_ids, tiny_movies)
        fig, ax = plt.subplots()
        plot_embedding_scatter(
            coords, genres, raw_ids, tiny_movies, ax,
            title="No Labels",
        )
        plt.close(fig)


class TestVisualizeEmbeddings:
    def test_returns_two_figures(self, trained_svd, tiny_movies):
        model, _, mid2idx = trained_svd
        fig_pca, fig_tsne = visualize_embeddings(
            model, mid2idx, tiny_movies, landmark_ids=[1, 4],
        )
        assert isinstance(fig_pca, plt.Figure)
        assert isinstance(fig_tsne, plt.Figure)
        plt.close(fig_pca)
        plt.close(fig_tsne)


class TestLandmarkMovies:
    def test_is_non_empty_list(self):
        assert isinstance(LANDMARK_MOVIES, list)
        assert len(LANDMARK_MOVIES) >= 10

    def test_all_ints(self):
        assert all(isinstance(m, int) for m in LANDMARK_MOVIES)
