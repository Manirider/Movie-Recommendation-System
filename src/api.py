
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import List, Optional

import numpy as np
import pandas as pd

try:
    from fastapi import FastAPI, HTTPException, Query
    from pydantic import BaseModel
except ImportError:
    raise ImportError(
        "FastAPI is required for the serving layer. "
        "Install with:  pip install fastapi uvicorn"
    )

from src.data_loader import (
    load_ratings,
    load_movies,
    dataset_summary,
    temporal_train_test_split,
)
from src.models import BaseRecommender, FunkSVD, train_from_df
from src.recommender import recommend_top_n, popular_movies
from src.embeddings import extract_item_embeddings


class Recommendation(BaseModel):
    movie_id: int
    title: str
    predicted_rating: float


class PopularMovie(BaseModel):
    movie_id: int
    title: str
    avg_rating: float
    n_ratings: int


class SimilarMovie(BaseModel):
    movie_id: int
    title: str
    similarity: float


class HealthResponse(BaseModel):
    status: str
    n_users: int
    n_movies: int
    model: str


class _AppState:
    model: Optional[BaseRecommender] = None
    uid2idx: dict = {}
    mid2idx: dict = {}
    train_df: Optional[pd.DataFrame] = None
    movies_df: Optional[pd.DataFrame] = None
    Q: Optional[np.ndarray] = None
    raw_movie_ids: Optional[List[int]] = None
    stats: dict = {}


_state = _AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    data_dir = os.environ.get("DATA_DIR", "")
    if not data_dir:
        print(
            "WARNING: DATA_DIR not set. The API will start but endpoints "
            "will return 503 until a model is loaded."
        )
        yield
        return

    print(f"[api] Loading data from {data_dir}...")
    ratings = load_ratings(data_dir)
    _state.movies_df = load_movies(data_dir)
    _state.stats = dataset_summary(ratings)

    train_df, _ = temporal_train_test_split(ratings, n_test_per_user=2)
    _state.train_df = train_df

    all_user_ids = np.sort(ratings["user_id"].unique())
    all_movie_ids = np.sort(ratings["movie_id"].unique())

    print("[api] Training SVD model...")
    svd = FunkSVD(n_factors=100, n_epochs=30, lr=0.005, reg=0.02)
    _state.model, _state.uid2idx, _state.mid2idx = train_from_df(
        svd, train_df, all_user_ids, all_movie_ids,
    )

    _state.Q, _state.raw_movie_ids = extract_item_embeddings(
        _state.model, _state.mid2idx,
    )
    print(f"[api] Ready — {_state.stats['n_users']:,} users, "
          f"{_state.stats['n_movies']:,} movies")

    yield

    print("[api] Shutting down.")


app = FastAPI(
    title="Movie Recommender API",
    description="REST API for collaborative-filtering recommendations (Funk SVD)",
    version="1.0.0",
    lifespan=lifespan,
)


def _require_model():
    if _state.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Set DATA_DIR and restart.",
        )


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if _state.model else "no_model",
        n_users=_state.stats.get("n_users", 0),
        n_movies=_state.stats.get("n_movies", 0),
        model=_state.model.name if _state.model else "none",
    )


@app.get("/recommend/{user_id}", response_model=List[Recommendation])
def recommend(user_id: int, n: int = Query(10, ge=1, le=100)):
    _require_model()
    recs_df = recommend_top_n(
        _state.model,
        _state.uid2idx,
        _state.mid2idx,
        user_id,
        _state.movies_df,
        _state.train_df,
        n=n,
    )
    return [
        Recommendation(
            movie_id=int(row["movie_id"]),
            title=str(row["title"]),
            predicted_rating=round(float(row["predicted_rating"]), 3),
        )
        for _, row in recs_df.iterrows()
    ]


@app.get("/popular", response_model=List[PopularMovie])
def popular(n: int = Query(10, ge=1, le=100), min_ratings: int = Query(50, ge=1)):
    _require_model()
    pop_df = popular_movies(_state.train_df, _state.movies_df, n=n, min_ratings=min_ratings)
    return [
        PopularMovie(
            movie_id=int(row["movie_id"]),
            title=str(row["title"]),
            avg_rating=round(float(row["avg_rating"]), 3),
            n_ratings=int(row["n_ratings"]),
        )
        for _, row in pop_df.iterrows()
    ]


@app.get("/similar/{movie_id}", response_model=List[SimilarMovie])
def similar(movie_id: int, n: int = Query(10, ge=1, le=100)):
    _require_model()
    if _state.Q is None:
        raise HTTPException(status_code=503, detail="Embeddings not computed.")

    if movie_id not in _state.mid2idx:
        raise HTTPException(status_code=404, detail=f"Movie {movie_id} not found.")

    idx = _state.mid2idx[movie_id]
    query_vec = _state.Q[idx]

    norms = np.linalg.norm(_state.Q, axis=1)
    query_norm = np.linalg.norm(query_vec)
    sims = (_state.Q @ query_vec) / (norms * query_norm + 1e-10)

    sims[idx] = -np.inf

    top_indices = np.argsort(sims)[::-1][:n]
    title_map = _state.movies_df.set_index("movie_id")["title"].to_dict()

    results = []
    for i in top_indices:
        mid = _state.raw_movie_ids[i]
        results.append(SimilarMovie(
            movie_id=mid,
            title=title_map.get(mid, f"Movie {mid}"),
            similarity=round(float(sims[i]), 4),
        ))
    return results
