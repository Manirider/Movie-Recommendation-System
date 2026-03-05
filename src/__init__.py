
from src.data_loader import (
    load_ratings,
    load_movies,
    load_users,
    dataset_summary,
    build_interaction_matrix,
    temporal_train_test_split,
)
from src.models import KNNRecommender, FunkSVD, train_from_df
from src.evaluation import evaluate_model, comparison_table, ndcg_at_k, rmse, mae, ranking_protocol_note
from src.recommender import recommend_top_n, popular_movies, get_item_embeddings
from src.embeddings import (
    extract_item_embeddings,
    reduce_to_2d,
    build_genre_labels,
    plot_embedding_scatter,
    visualize_embeddings,
)
from src.tuning import (
    temporal_cv_folds,
    grid_search_svd,
    grid_search_knn,
    best_params,
)

__all__ = [
    "load_ratings", "load_movies", "load_users",
    "dataset_summary", "build_interaction_matrix", "temporal_train_test_split",
    "KNNRecommender", "FunkSVD", "train_from_df",
    "evaluate_model", "comparison_table", "ndcg_at_k", "rmse", "mae", "ranking_protocol_note",
    "recommend_top_n", "popular_movies", "get_item_embeddings",
    "extract_item_embeddings", "reduce_to_2d", "build_genre_labels",
    "plot_embedding_scatter", "visualize_embeddings",
    "temporal_cv_folds", "grid_search_svd", "grid_search_knn", "best_params",
]
