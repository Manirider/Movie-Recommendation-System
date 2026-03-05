
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix


@pytest.fixture()
def tiny_ratings() -> pd.DataFrame:
    data = [
        (1, 1, 5.0, 100), (1, 2, 4.0, 200), (1, 3, 4.0, 300), (1, 5, 2.0, 400),
        (2, 1, 4.0, 100), (2, 2, 5.0, 200), (2, 3, 3.0, 300),
        (3, 4, 5.0, 100), (3, 5, 4.0, 200), (3, 6, 4.0, 300),
        (4, 4, 4.0, 100), (4, 5, 5.0, 200), (4, 6, 5.0, 300), (4, 1, 2.0, 400),
        (5, 1, 3.0, 100), (5, 3, 4.0, 200), (5, 4, 3.0, 300), (5, 6, 5.0, 400),
    ]
    return pd.DataFrame(data, columns=["user_id", "movie_id", "rating", "timestamp"])


@pytest.fixture()
def tiny_movies() -> pd.DataFrame:
    return pd.DataFrame({
        "movie_id": [1, 2, 3, 4, 5, 6],
        "title": [
            "Alpha (2000)", "Beta (2001)", "Gamma (2002)",
            "Delta (2003)", "Epsilon (2004)", "Zeta (2005)",
        ],
        "genres": [
            ["Action"], ["Action", "Thriller"], ["Drama"],
            ["Comedy"], ["Comedy", "Romance"], ["Drama"],
        ],
    })


@pytest.fixture()
def user_ids(tiny_ratings) -> np.ndarray:
    return np.sort(tiny_ratings["user_id"].unique())


@pytest.fixture()
def movie_ids(tiny_ratings) -> np.ndarray:
    return np.sort(tiny_ratings["movie_id"].unique())
