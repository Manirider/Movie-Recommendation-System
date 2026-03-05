
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def extract_item_embeddings(
    svd_model,
    mid_to_idx: Dict[int, int],
) -> Tuple[np.ndarray, List[int]]:
    Q = svd_model.get_item_embeddings()
    idx_to_mid = {v: k for k, v in mid_to_idx.items()}
    raw_ids = [idx_to_mid[i] for i in range(Q.shape[0])]
    return Q, raw_ids


def reduce_to_2d(
    embeddings: np.ndarray,
    method: str = "pca",
    random_state: int = 42,
    **kwargs,
) -> Tuple[np.ndarray, object]:
    method = method.lower()
    if method == "pca":
        reducer = PCA(n_components=2, random_state=random_state, **kwargs)
    elif method == "tsne":
        tsne_defaults = dict(perplexity=30, max_iter=1000, init="pca", learning_rate="auto")
        tsne_defaults.update(kwargs)
        reducer = TSNE(n_components=2, random_state=random_state, **tsne_defaults)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'pca' or 'tsne'.")

    coords_2d = reducer.fit_transform(embeddings)
    return coords_2d, reducer


def build_genre_labels(
    raw_movie_ids: List[int],
    movies_df: pd.DataFrame,
) -> List[str]:
    genre_map = movies_df.set_index("movie_id")["genres"].to_dict()
    labels = []
    for mid in raw_movie_ids:
        genres = genre_map.get(mid, ["Unknown"])
        labels.append(genres[0] if isinstance(genres, list) else "Unknown")
    return labels


def plot_embedding_scatter(
    coords_2d: np.ndarray,
    genre_labels: List[str],
    raw_movie_ids: List[int],
    movies_df: pd.DataFrame,
    ax,
    title: str = "SVD Item Embeddings",
    top_n_genres: int = 8,
    label_movies: Optional[List[int]] = None,
    xlabel: str = "Dim 1",
    ylabel: str = "Dim 2",
):
    df = pd.DataFrame({
        "x": coords_2d[:, 0],
        "y": coords_2d[:, 1],
        "genre": genre_labels,
        "movie_id": raw_movie_ids,
    })

    top_genres = df["genre"].value_counts().head(top_n_genres).index.tolist()
    plot_df = df[df["genre"].isin(top_genres)]

    for genre in top_genres:
        subset = plot_df[plot_df["genre"] == genre]
        ax.scatter(subset["x"], subset["y"], label=genre, alpha=0.5, s=15)

    if label_movies:
        title_map = movies_df.set_index("movie_id")["title"].to_dict()
        for mid in label_movies:
            row = df[df["movie_id"] == mid]
            if row.empty:
                continue
            x, y = row.iloc[0]["x"], row.iloc[0]["y"]
            name = title_map.get(mid, str(mid))
            short = name[:30] + "…" if len(name) > 30 else name
            ax.annotate(
                short, (x, y),
                fontsize=7, fontweight="bold",
                xytext=(6, 6), textcoords="offset points",
                arrowprops=dict(arrowstyle="-", color="grey", lw=0.5),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="grey", alpha=0.8),
            )

    ax.legend(title="Primary Genre", loc="best", fontsize=7, markerscale=1.5)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


LANDMARK_MOVIES = [
    260,
    1196,
    2571,
    318,
    858,
    2028,
    1,
    1210,
    593,
    2762,
    480,
    110,
    527,
    1197,
    356,
]


def visualize_embeddings(
    svd_model,
    mid_to_idx: Dict[int, int],
    movies_df: pd.DataFrame,
    save_dir: Optional[str] = None,
    landmark_ids: Optional[List[int]] = None,
    figsize: Tuple[int, int] = (12, 8),
):
    import matplotlib.pyplot as plt

    if landmark_ids is None:
        landmark_ids = LANDMARK_MOVIES

    Q, raw_ids = extract_item_embeddings(svd_model, mid_to_idx)
    genres = build_genre_labels(raw_ids, movies_df)

    coords_pca, pca_obj = reduce_to_2d(Q, method="pca")
    fig_pca, ax_pca = plt.subplots(figsize=figsize)
    var1 = pca_obj.explained_variance_ratio_[0]
    var2 = pca_obj.explained_variance_ratio_[1]
    plot_embedding_scatter(
        coords_pca, genres, raw_ids, movies_df, ax_pca,
        title="PCA Projection of SVD Item Embeddings",
        xlabel=f"PC1 ({var1:.1%} variance)",
        ylabel=f"PC2 ({var2:.1%} variance)",
        label_movies=landmark_ids,
    )
    fig_pca.tight_layout()
    if save_dir:
        import os
        fig_pca.savefig(os.path.join(save_dir, "svd_embeddings_pca.png"), dpi=150, bbox_inches="tight")

    tsne_perplexity = min(30, Q.shape[0] - 1)
    coords_tsne, _ = reduce_to_2d(Q, method="tsne", perplexity=tsne_perplexity)
    fig_tsne, ax_tsne = plt.subplots(figsize=figsize)
    plot_embedding_scatter(
        coords_tsne, genres, raw_ids, movies_df, ax_tsne,
        title="t-SNE Projection of SVD Item Embeddings",
        xlabel="t-SNE 1",
        ylabel="t-SNE 2",
        label_movies=landmark_ids,
    )
    fig_tsne.tight_layout()
    if save_dir:
        import os
        fig_tsne.savefig(os.path.join(save_dir, "svd_embeddings_tsne.png"), dpi=150, bbox_inches="tight")

    return fig_pca, fig_tsne
