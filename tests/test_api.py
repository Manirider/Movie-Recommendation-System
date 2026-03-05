
import numpy as np
import pandas as pd
import pytest

from src.models import FunkSVD, train_from_df
from src.embeddings import extract_item_embeddings

from src.api import _AppState, app

try:
    from fastapi.testclient import TestClient
except ImportError:
    pytest.skip("fastapi / httpx not installed", allow_module_level=True)


@pytest.fixture()
def client(tiny_ratings, tiny_movies, user_ids, movie_ids):
    from src.api import _state

    model, uid2idx, mid2idx = train_from_df(
        FunkSVD(n_factors=4, n_epochs=10, lr=0.01, reg=0.01),
        tiny_ratings, user_ids, movie_ids,
    )
    _state.model = model
    _state.uid2idx = uid2idx
    _state.mid2idx = mid2idx
    _state.train_df = tiny_ratings
    _state.movies_df = tiny_movies
    _state.stats = {"n_users": 5, "n_movies": 6, "n_ratings": 18}

    Q, raw_ids = extract_item_embeddings(model, mid2idx)
    _state.Q = Q
    _state.raw_movie_ids = raw_ids

    with TestClient(app, raise_server_exceptions=True) as c:
        yield c

    _state.model = None
    _state.Q = None


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["n_users"] == 5
        assert body["n_movies"] == 6


class TestRecommendEndpoint:
    def test_recommend_known_user(self, client):
        resp = client.get("/recommend/1?n=3")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) <= 3
        assert all("movie_id" in r and "title" in r and "predicted_rating" in r for r in data)

    def test_recommend_unknown_user_fallback(self, client):
        resp = client.get("/recommend/9999?n=3")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) <= 3

    def test_recommend_default_n(self, client):
        resp = client.get("/recommend/1")
        assert resp.status_code == 200


class TestPopularEndpoint:
    def test_popular_returns_list(self, client):
        resp = client.get("/popular?n=3&min_ratings=1")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 3
        assert all("avg_rating" in r for r in data)


class TestSimilarEndpoint:
    def test_similar_known_movie(self, client):
        resp = client.get("/similar/1?n=3")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) <= 3
        assert all("similarity" in r for r in data)
        assert all(r["movie_id"] != 1 for r in data)

    def test_similar_unknown_movie_404(self, client):
        resp = client.get("/similar/99999")
        assert resp.status_code == 404

    def test_similarity_scores_descending(self, client):
        resp = client.get("/similar/1?n=5")
        data = resp.json()
        sims = [r["similarity"] for r in data]
        assert sims == sorted(sims, reverse=True)
