
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from typing import Dict, Tuple


class BaseRecommender:
    name: str = "base"
    global_mean: float = 3.0

    def fit(self, R: csr_matrix, user_ids: np.ndarray, movie_ids: np.ndarray):
        raise NotImplementedError

    def predict(self, u: int, i: int) -> float:
        raise NotImplementedError

    def predict_for_user(self, u: int) -> np.ndarray:
        raise NotImplementedError


class KNNRecommender(BaseRecommender):

    def __init__(self, k: int = 40, user_based: bool = True):
        self.k = k
        self.user_based = user_based
        self.name = f"{'User' if user_based else 'Item'}-KNN (k={k})"

    def __repr__(self) -> str:
        return f"KNNRecommender(k={self.k}, user_based={self.user_based})"


    def fit(self, R: csr_matrix, user_ids: np.ndarray, movie_ids: np.ndarray):
        self.user_ids = user_ids
        self.movie_ids = movie_ids
        self.R = R.toarray().astype(np.float64)
        self.n_users, self.n_items = self.R.shape
        self.mask = (self.R != 0).astype(np.float64)

        counts = self.mask.sum(axis=1)
        self.user_means = np.divide(
            self.R.sum(axis=1), counts,
            out=np.full(self.n_users, 3.0), where=counts != 0,
        )
        item_counts = self.mask.sum(axis=0)
        self.item_means = np.divide(
            self.R.sum(axis=0), item_counts,
            out=np.full(self.n_items, 3.0), where=item_counts != 0,
        )
        self.global_mean = float(self.R[self.R != 0].mean())

        if self.user_based:
            self.sim = cosine_similarity(self.R)
        else:
            self.sim = cosine_similarity(self.R.T)
        np.fill_diagonal(self.sim, 0.0)

        return self


    def predict(self, u: int, i: int) -> float:
        if self.user_based:
            return self._predict_ub(u, i)
        return self._predict_ib(u, i)

    def _predict_ub(self, u: int, i: int) -> float:
        sims = self.sim[u].copy()
        sims[~self.mask[:, i].astype(bool)] = 0.0

        top_idx = np.argsort(sims)[::-1][: self.k]
        top_sims = sims[top_idx]
        denom = np.abs(top_sims).sum()
        if denom == 0:
            return float(self.user_means[u])

        devs = self.R[top_idx, i] - self.user_means[top_idx]
        return float(np.clip(self.user_means[u] + top_sims @ devs / denom, 1, 5))

    def _predict_ib(self, u: int, i: int) -> float:
        sims = self.sim[i].copy()
        sims[~self.mask[u].astype(bool)] = 0.0

        top_idx = np.argsort(sims)[::-1][: self.k]
        top_sims = sims[top_idx]
        denom = np.abs(top_sims).sum()
        if denom == 0:
            return float(self.user_means[u])

        devs = self.R[u, top_idx] - self.item_means[top_idx]
        return float(np.clip(self.item_means[i] + top_sims @ devs / denom, 1, 5))


    def predict_for_user(self, u: int) -> np.ndarray:
        if self.user_based:
            return self._full_ub(u)
        return self._full_ib(u)

    def _full_ub(self, u: int) -> np.ndarray:
        sims = self.sim[u]
        top_idx = np.argsort(sims)[::-1][: self.k]
        top_sims = sims[top_idx]

        nbr_mask = self.mask[top_idx]
        nbr_devs = self.R[top_idx] - self.user_means[top_idx, None]

        numer = (top_sims[:, None] * nbr_devs * nbr_mask).sum(axis=0)
        denom = (np.abs(top_sims)[:, None] * nbr_mask).sum(axis=0)
        denom = np.where(denom == 0, 1.0, denom)

        return np.clip(self.user_means[u] + numer / denom, 1.0, 5.0)

    def _full_ib(self, u: int) -> np.ndarray:
        rated = np.where(self.mask[u].astype(bool))[0]
        if len(rated) == 0:
            return np.full(self.n_items, self.global_mean)

        sim_sub = self.sim[:, rated]
        rat_sub = self.R[u, rated]

        if sim_sub.shape[1] > self.k:
            top_k_idx = np.argpartition(sim_sub, -self.k, axis=1)[:, -self.k:]
            mask_k = np.zeros_like(sim_sub)
            np.put_along_axis(mask_k, top_k_idx, 1.0, axis=1)
            sim_sub = sim_sub * mask_k

        devs = rat_sub - self.item_means[rated]
        numer = sim_sub @ devs
        denom = np.abs(sim_sub).sum(axis=1)
        denom = np.where(denom == 0, 1.0, denom)
        return np.clip(self.item_means + numer / denom, 1.0, 5.0)


class FunkSVD(BaseRecommender):

    def __init__(self, n_factors: int = 100, n_epochs: int = 30,
                 lr: float = 0.005, reg: float = 0.02):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.name = f"SVD (f={n_factors}, e={n_epochs})"

    def __repr__(self) -> str:
        return (f"FunkSVD(n_factors={self.n_factors}, n_epochs={self.n_epochs}, "
                f"lr={self.lr}, reg={self.reg})")

    def fit(self, R: csr_matrix, user_ids: np.ndarray, movie_ids: np.ndarray):
        self.user_ids = user_ids
        self.movie_ids = movie_ids
        dense = R.toarray().astype(np.float64)
        self.n_users, self.n_items = dense.shape

        obs_u, obs_i = np.nonzero(dense)
        obs_r = dense[obs_u, obs_i]
        self.global_mean = float(obs_r.mean())

        rng = np.random.RandomState(42)
        self.P = rng.normal(0, 0.1, (self.n_users, self.n_factors))
        self.Q = rng.normal(0, 0.1, (self.n_items, self.n_factors))
        self.bu = np.zeros(self.n_users)
        self.bi = np.zeros(self.n_items)

        for epoch in range(self.n_epochs):
            order = rng.permutation(len(obs_r))
            sq_err = 0.0
            for idx in order:
                u, i, r = obs_u[idx], obs_i[idx], obs_r[idx]
                pred = (self.global_mean + self.bu[u] + self.bi[i]
                        + self.P[u] @ self.Q[i])
                err = r - pred
                sq_err += err * err

                self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                self.bi[i] += self.lr * (err - self.reg * self.bi[i])
                pu = self.P[u].copy()
                self.P[u] += self.lr * (err * self.Q[i] - self.reg * self.P[u])
                self.Q[i] += self.lr * (err * pu       - self.reg * self.Q[i])

            train_rmse = np.sqrt(sq_err / len(obs_r))
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  epoch {epoch+1:3d}/{self.n_epochs}  "
                      f"train RMSE = {train_rmse:.4f}")

        return self


    def predict(self, u: int, i: int) -> float:
        p = self.global_mean + self.bu[u] + self.bi[i] + self.P[u] @ self.Q[i]
        return float(np.clip(p, 1.0, 5.0))

    def predict_for_user(self, u: int) -> np.ndarray:
        preds = self.global_mean + self.bu[u] + self.bi + self.P[u] @ self.Q.T
        return np.clip(preds, 1.0, 5.0)

    def get_item_embeddings(self) -> np.ndarray:
        return self.Q.copy()


def train_from_df(
    model: BaseRecommender,
    train_df: pd.DataFrame,
    user_ids: np.ndarray,
    movie_ids: np.ndarray,
) -> Tuple[BaseRecommender, Dict[int, int], Dict[int, int]]:
    from src.data_loader import build_interaction_matrix

    R, _, _, uid_to_idx, mid_to_idx = build_interaction_matrix(
        train_df, user_ids, movie_ids
    )
    model.fit(R, user_ids, movie_ids)
    return model, uid_to_idx, mid_to_idx
