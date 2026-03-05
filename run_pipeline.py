#!/usr/bin/env python

import argparse
import os
import time

import numpy as np
import pandas as pd

from src.data_loader import (
    load_ratings, load_movies, dataset_summary,
    build_interaction_matrix, temporal_train_test_split,
)
from src.models import KNNRecommender, FunkSVD, train_from_df
from src.evaluation import evaluate_model, comparison_table, ranking_protocol_note
from src.recommender import recommend_top_n, popular_movies


def main():
    parser = argparse.ArgumentParser(
        description="Movie Recommendation System — Full Pipeline",
    )
    parser.add_argument(
        "--data-dir", required=True,
        help="Path to the ml-1m directory containing ratings.dat, movies.dat, users.dat",
    )
    parser.add_argument(
        "--output-dir", default="outputs",
        help="Directory for metrics.csv (default: outputs/)",
    )
    parser.add_argument(
        "--n-factors", type=int, default=100,
        help="SVD latent factors (default: 100)",
    )
    parser.add_argument(
        "--n-epochs", type=int, default=30,
        help="SVD training epochs (default: 30)",
    )
    parser.add_argument(
        "--knn-k", type=int, default=40,
        help="Number of neighbours for KNN (default: 40)",
    )
    parser.add_argument(
        "--top-k", type=int, default=10,
        help="K for Precision@K / Recall@K (default: 10)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("  MOVIE RECOMMENDATION SYSTEM — FULL PIPELINE")
    print("=" * 60)

    print("\n[1/6] Loading data...")
    ratings = load_ratings(args.data_dir)
    movies = load_movies(args.data_dir)
    stats = dataset_summary(ratings)
    print(f"  {stats['n_users']:,} users, {stats['n_movies']:,} movies, "
          f"{stats['n_ratings']:,} ratings  (sparsity {stats['sparsity']:.2%})")

    print("\n[2/6] Temporal train/test split...")
    train_df, test_df = temporal_train_test_split(ratings, n_test_per_user=2)
    print(f"  Train: {len(train_df):,}  |  Test: {len(test_df):,}  "
          f"({len(test_df)/len(ratings):.1%} held out)")

    all_user_ids = np.sort(ratings["user_id"].unique())
    all_movie_ids = np.sort(ratings["movie_id"].unique())

    print(f"\n[3/6] Training models...")

    t0 = time.time()
    print(f"\n  User-KNN (k={args.knn_k})...")
    user_knn, uid2idx_uk, mid2idx_uk = train_from_df(
        KNNRecommender(k=args.knn_k, user_based=True),
        train_df, all_user_ids, all_movie_ids,
    )
    print(f"  Done in {time.time() - t0:.1f}s")

    t0 = time.time()
    print(f"\n  Item-KNN (k={args.knn_k})...")
    item_knn, uid2idx_ik, mid2idx_ik = train_from_df(
        KNNRecommender(k=args.knn_k, user_based=False),
        train_df, all_user_ids, all_movie_ids,
    )
    print(f"  Done in {time.time() - t0:.1f}s")

    t0 = time.time()
    print(f"\n  Funk SVD (f={args.n_factors}, epochs={args.n_epochs})...")
    svd_model, uid2idx_sv, mid2idx_sv = train_from_df(
        FunkSVD(n_factors=args.n_factors, n_epochs=args.n_epochs),
        train_df, all_user_ids, all_movie_ids,
    )
    print(f"  Done in {time.time() - t0:.1f}s")

    print(f"\n[4/6] Evaluating (full-catalogue ranking, k={args.top_k})...")

    models = [
        ("User-KNN", user_knn, uid2idx_uk, mid2idx_uk),
        ("Item-KNN", item_knn, uid2idx_ik, mid2idx_ik),
        ("SVD",      svd_model, uid2idx_sv, mid2idx_sv),
    ]
    results = {}
    for name, model, uid2idx, mid2idx in models:
        t0 = time.time()
        res = evaluate_model(model, train_df, test_df,
                             uid2idx, mid2idx, k=args.top_k)
        elapsed = time.time() - t0
        results[name] = res
        print(f"  {name:12s}  RMSE={res['RMSE']:.4f}  MAE={res['MAE']:.4f}  "
              f"P@{args.top_k}={res[f'Precision@{args.top_k}']:.4f}  "
              f"R@{args.top_k}={res[f'Recall@{args.top_k}']:.4f}  "
              f"NDCG@{args.top_k}={res[f'NDCG@{args.top_k}']:.4f}  ({elapsed:.1f}s)")

    n_movies = len(all_movie_ids)
    print(ranking_protocol_note(n_test_per_user=2, n_movies=n_movies))

    print("\n[5/6] Saving metrics...")
    comp = comparison_table(results)
    csv_path = os.path.join(args.output_dir, "metrics.csv")
    comp.to_csv(csv_path)
    print(f"  Saved to {csv_path}")

    print("\n[6/6] Sample recommendations (SVD, top-5)...")
    for uid in [1, 42, 610]:
        recs = recommend_top_n(
            svd_model, uid2idx_sv, mid2idx_sv,
            uid, movies, train_df, n=5,
        )
        n_rated = train_df[train_df["user_id"] == uid].shape[0]
        print(f"\n  User {uid} (rated {n_rated} movies):")
        for _, row in recs.iterrows():
            print(f"    {row['predicted_rating']:.3f}  {row['title']}")

    print("\n  Cold-start fallback (top-5 popular):")
    pop = popular_movies(train_df, movies, n=5, min_ratings=100)
    for _, row in pop.iterrows():
        print(
            f"    {row['avg_rating']:.2f} ({row['n_ratings']:,} ratings)  {row['title']}")

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
