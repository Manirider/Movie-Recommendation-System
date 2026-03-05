
import numpy as np
import pandas as pd
import pytest

from src.data_loader import build_interaction_matrix
from src.models import FunkSVD, train_from_df
from src.recommender import recommend_top_n, popular_movies
from src.embeddings import extract_item_embeddings


class TestRecommendTopN:
    @pytest.fixture(autouse=True)
    def setup(self, tiny_ratings, tiny_movies, user_ids, movie_ids):
        self.model, self.uid2idx, self.mid2idx = train_from_df(
            FunkSVD(n_factors=4, n_epochs=20), tiny_ratings, user_ids, movie_ids,
        )
        self.tiny_ratings = tiny_ratings
        self.tiny_movies = tiny_movies

    def test_returns_dataframe(self):
        recs = recommend_top_n(
            self.model, self.uid2idx, self.mid2idx,
            user_id=1, movies_df=self.tiny_movies,
            train_df=self.tiny_ratings, n=3,
        )
        assert isinstance(recs, pd.DataFrame)
        assert len(recs) <= 3

    def test_no_already_rated(self):
        rated = set(self.tiny_ratings[self.tiny_ratings["user_id"] == 1]["movie_id"])
        recs = recommend_top_n(
            self.model, self.uid2idx, self.mid2idx,
            user_id=1, movies_df=self.tiny_movies,
            train_df=self.tiny_ratings, n=10,
        )
        recommended_ids = set(recs["movie_id"])
        assert recommended_ids.isdisjoint(rated)

    def test_columns(self):
        recs = recommend_top_n(
            self.model, self.uid2idx, self.mid2idx,
            user_id=1, movies_df=self.tiny_movies,
            train_df=self.tiny_ratings, n=3,
        )
        assert list(recs.columns) == ["movie_id", "title", "predicted_rating"]

    def test_unknown_user_fallback(self):
        recs = recommend_top_n(
            self.model, self.uid2idx, self.mid2idx,
            user_id=9999, movies_df=self.tiny_movies,
            train_df=self.tiny_ratings, n=3,
        )
        assert isinstance(recs, pd.DataFrame)
        assert len(recs) <= 3
        assert list(recs.columns) == ["movie_id", "title", "predicted_rating"]


class TestPopularMovies:
    def test_length(self, tiny_ratings, tiny_movies):
        pop = popular_movies(tiny_ratings, tiny_movies, n=3, min_ratings=1)
        assert len(pop) == 3

    def test_descending_score(self, tiny_ratings, tiny_movies):
        pop = popular_movies(tiny_ratings, tiny_movies, n=6, min_ratings=1)
        assert pop.iloc[0]["avg_rating"] >= 3.0

    def test_columns(self, tiny_ratings, tiny_movies):
        pop = popular_movies(tiny_ratings, tiny_movies, n=3, min_ratings=1)
        assert "movie_id" in pop.columns
        assert "title" in pop.columns
        assert "avg_rating" in pop.columns
        assert "n_ratings" in pop.columns

    def test_min_ratings_filter(self, tiny_ratings, tiny_movies):
        pop = popular_movies(tiny_ratings, tiny_movies, n=3, min_ratings=100)
        assert isinstance(pop, pd.DataFrame)


class TestExtractItemEmbeddingsFromRecommender:
    def test_shape_and_ids(self, tiny_ratings, user_ids, movie_ids):
        model, uid2idx, mid2idx = train_from_df(
            FunkSVD(n_factors=4, n_epochs=5), tiny_ratings, user_ids, movie_ids,
        )
        Q, raw_ids = extract_item_embeddings(model, mid2idx)
        assert Q.shape == (6, 4)
        assert len(raw_ids) == 6
        assert set(raw_ids) == {1, 2, 3, 4, 5, 6}
