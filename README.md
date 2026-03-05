# Movie Recommendation System

**Ever wondered how Netflix knows what you want to watch next?** This project dives deep into the mechanics behind movie recommendations — building a complete recommendation engine from the ground up.

I built three collaborative-filtering algorithms **entirely from scratch** using NumPy, SciPy, and scikit-learn. No black-box recommendation libraries. Every line of the core logic — from cosine similarity to stochastic gradient descent — is written by hand so you can see exactly how these systems work under the hood.

## What Makes This Project Stand Out

- **Built from scratch** — No Surprise library, no magic. Pure algorithmic implementation you can read, understand, and learn from.
- **Realistic evaluation** — Temporal train/test split ensures the model never peeks into the future. Just like in production.
- **Honest metrics** — Full-catalogue ranking protocol (Precision@K, Recall@K, NDCG@K) that doesn't cheat by inflating scores.
- **Cold-start ready** — New user with zero history? No problem. Bayesian-adjusted popularity kicks in gracefully.
- **Visual insights** — PCA and t-SNE projections of learned movie embeddings, colored by genre, with landmark films labeled.
- **Production-ready API** — FastAPI endpoints for real-time recommendations, popularity lists, and similar-movie search.
- **104 tests** — Every module covered. Runs in under 5 seconds on a synthetic dataset.

## Dataset

This project uses the well-known **[MovieLens 1M](https://grouplens.org/datasets/movielens/1m/)** dataset:

- **1,000,209** ratings on a 1–5 star scale
- **6,040** users and **3,706** movies
- Collected from April 2000 to February 2003
- Matrix sparsity: **95.5%** (most users have rated only a tiny fraction of all movies)

> Citation: F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context.* ACM TiiS 5(4), Article 19.

## System Architecture

```
┌────────────────────────────────────────────────────────────┐
│                      User / Client                         │
│              (Notebook · CLI · REST API)                    │
└──────────────────────┬─────────────────────────────────────┘
                       │
          ┌────────────▼────────────┐
          │     run_pipeline.py     │   CLI entrypoint
          │     notebooks/main.ipynb│   Interactive walkthrough
          └────────────┬────────────┘
                       │
    ┌──────────────────▼──────────────────┐
    │            src/ Package             │
    │                                     │
    │  data_loader.py ──► models.py       │
    │       │                │            │
    │       │          ┌─────┴─────┐      │
    │       │          │           │      │
    │       ▼          ▼           ▼      │
    │  evaluation.py  recommender.py      │
    │       │              │              │
    │       ▼              ▼              │
    │  embeddings.py   tuning.py          │
    │                                     │
    │  api.py ── FastAPI serving layer    │
    └─────────────────────────────────────┘
                       │
          ┌────────────▼────────────┐
          │       outputs/          │
          │   metrics.csv + plots/  │
          └─────────────────────────┘
```

## Project Structure

```
Movie-Recommendation-System/
├── notebooks/
│   └── main.ipynb              # Full walkthrough — load, train, evaluate, visualize
├── src/
│   ├── __init__.py             # Clean package exports
│   ├── data_loader.py          # Data ingestion, sparse matrix, temporal split
│   ├── models.py               # KNNRecommender + FunkSVD (from scratch)
│   ├── evaluation.py           # RMSE, MAE, Precision@K, Recall@K, NDCG@K
│   ├── recommender.py          # Top-N generation + cold-start fallback
│   ├── embeddings.py           # SVD factor extraction, PCA/t-SNE, scatter plots
│   ├── tuning.py               # Temporal cross-validation + grid search
│   └── api.py                  # FastAPI serving layer
├── tests/                      # 104 tests across 9 files
├── outputs/
│   ├── metrics.csv             # Model comparison table
│   └── plots/                  # All generated visualizations
├── data/                       # Dataset goes here (.dat files)
├── run_pipeline.py             # Run everything from the command line
├── download_data.py            # One-command dataset download
├── requirements.txt
├── pyproject.toml
└── .gitignore
```

Everything in `src/` is modular, importable, and independently testable. The notebook ties it all together with narrative analysis.

## Models Implemented

| Model | Algorithm | Key Parameters |
|-------|-----------|---------------|
| **User-Based KNN** | Cosine similarity over user vectors, mean-centered deviations, top-k neighbor selection | k = 40 |
| **Item-Based KNN** | Cosine similarity over item vectors, mean-centered deviations, vectorized prediction | k = 40 |
| **Funk SVD** | Biased SGD on latent factors with L2 regularization: r̂ = μ + bᵤ + bᵢ + pᵤ · qᵢ | f=100, epochs=30, lr=0.005, λ=0.02 |

All three models are implemented entirely from scratch using NumPy and SciPy — no Surprise library, no pre-built recommendation packages. This demonstrates a deep understanding of how these algorithms work mathematically, not just how to call an API.

## How It Works

### Step 1 — Load and Explore the Data

The pipeline starts by parsing the raw `.dat` files and computing key statistics — how sparse is the matrix, how are ratings distributed, which genres dominate, and how user activity varies over time.

### Step 2 — Split the Data (Without Cheating)

This is where many toy projects go wrong. A random split leaks future information into training. Instead, I use a **temporal split**: for every user, the last 2 ratings by timestamp become the test set. Everything earlier is training. The model never sees the future — exactly like a deployed system.

### Step 3 — Train Three Models

Each model learns user preferences from the training set only. User-KNN and Item-KNN build similarity matrices from the sparse interaction data, while Funk SVD learns low-rank latent factors through stochastic gradient descent over 30 epochs.

### Step 4 — Evaluate Honestly

**Rating accuracy**: RMSE and MAE on the held-out ratings.

**Ranking quality**: For each user, the model scores *every single unseen movie* (~3,700 items), masks out training items, picks the top-K, and checks how many of the user's actual relevant test items appear. This is the standard full-catalogue protocol used in academic RecSys research — much harder and more realistic than ranking a handful of test items.

### Step 5 — Handle Cold-Start Users

When a brand-new user shows up with no rating history, the system falls back to a **Bayesian-adjusted popularity list**. Instead of naively showing the highest-average movie (which could be a niche film with 3 perfect ratings), it shrinks scores toward the global mean, naturally favoring well-rated *and* well-known titles.

### Step 6 — Visualize What the Model Learned

The SVD model doesn't know anything about genres — it only sees user-movie-rating triples. Yet when I project the learned item embeddings to 2D using PCA and t-SNE, **genre clusters emerge on their own**. Action movies huddle together. Dramas form their own region. Landmark films like *Star Wars*, *The Matrix*, and *Shawshank Redemption* are annotated directly on the plots so you can see the structure for yourself.

## Evaluation Metrics

| Metric | Type | What It Measures |
|--------|------|-----------------|
| **RMSE** | Rating accuracy | Root mean squared error between predicted and actual ratings |
| **MAE** | Rating accuracy | Mean absolute error — less sensitive to outliers than RMSE |
| **Precision@K** | Ranking quality | Fraction of top-K recommendations that are actually relevant |
| **Recall@K** | Ranking quality | Fraction of relevant items that appear in the top-K list |
| **NDCG@K** | Ranking quality | Normalized discounted cumulative gain — rewards relevant items ranked higher |

All ranking metrics use the **full-catalogue protocol** — scoring all ~3,700 unseen items per user, not just the test set.

## Model Comparison Results

Evaluated on the temporal split with full-catalogue ranking:

| Model | RMSE ↓ | MAE ↓ | P@10 ↑ | R@10 ↑ | NDCG@10 ↑ |
|-------|--------|-------|--------|--------|-----------|
| User-KNN (k=40) | 1.0036 | 0.7790 | 0.0008 | 0.0047 | 0.0025 |
| Item-KNN (k=40) | 0.9273 | 0.7286 | 0.0003 | 0.0020 | 0.0009 |
| **SVD (f=100)** | **0.9248** | **0.7250** | **0.0048** | **0.0306** | **0.0169** |

**"Why are ranking metrics so low?"** — This is actually the *correct* behavior. Each user has only 2 held-out items, but the model has to find them among ~3,700 candidates. It's a genuine needle-in-a-haystack test. Many tutorials cheat by ranking only the test set, which inflates Recall to nearly 1.0. The important takeaway here is the **relative ordering**: SVD dominates on ranking, while Item-KNN and SVD run neck-and-neck on rating accuracy.

## SVD Embedding Visualization

The SVD item-factor matrix Q ∈ ℝⁿˣᵏ captures latent taste dimensions learned purely from user-movie interactions. By projecting this 100-dimensional space down to 2D using PCA and t-SNE, we can visually verify that the model has learned meaningful structure:

- **PCA projection** — Captures the directions of maximum variance. Genre clusters (Action, Comedy, Drama, Sci-Fi) emerge naturally without the model ever seeing genre labels.
- **t-SNE projection** — Preserves local neighborhood structure. Movies that "feel" similar cluster tightly together.
- **Landmark annotations** — 15 well-known films (Star Wars, The Matrix, Toy Story, Shawshank Redemption, etc.) are labeled directly on the scatter plots, making it easy to verify semantic coherence.

These visualizations confirm that the latent factors are not random noise — they capture genuine taste dimensions that align with human-interpretable categories.

## Cold-Start Strategy

New users with zero history receive a **Bayesian-adjusted popularity list**:

```
score(i) = (nᵢ · r̄ᵢ + m · C) / (nᵢ + m)
```

Where C is the global mean rating, m is the prior strength, and nᵢ, r̄ᵢ are the item's rating count and average. This shrinks niche films toward the global average, preventing a movie with three 5-star ratings from outranking a well-known classic with thousands of reviews.

## Sample Output

Here's what the SVD model recommends for real users:

```
User 1 (rated 51 movies):
  4.940  Godfather: Part II, The (1974)
  4.905  Godfather, The (1972)
  4.832  GoodFellas (1990)
  4.824  Patton (1970)
  4.717  Sanjuro (1962)

User 42 (rated 229 movies):
  4.746  Sanjuro (1962)
  4.602  Lamerica (1994)
  4.597  Firelight (1997)
  4.524  Army of Darkness (1993)
  4.473  For All Mankind (1989)

Cold-start fallback (new user, no history):
  4.55 (2,202 ratings)  Shawshank Redemption, The (1994)
  4.52 (2,205 ratings)  Godfather, The (1972)
  4.51 (2,286 ratings)  Schindler's List (1993)
  4.52 (1,771 ratings)  Usual Suspects, The (1995)
  4.48 (2,500 ratings)  Raiders of the Lost Ark (1981)
```

> Exact scores vary slightly with the random seed, but the relative ordering is stable across runs.

## Demo

To see the full pipeline in action:

```bash
python download_data.py
python run_pipeline.py --data-dir data
```

Expected output includes the evaluation table, sample recommendations for three users, and a cold-start popularity fallback.

To record a terminal session:

```bash
# Windows PowerShell
Start-Transcript -Path demo_output.txt
python run_pipeline.py --data-dir data
Stop-Transcript
```

## Installation

### Prerequisites

- Python 3.10 or later (tested on 3.14)
- About 25 MB of disk space for the dataset

### Setup

```bash
git clone https://github.com/Manirider/Movie-Recommendation-System.git
cd Movie-Recommendation-System

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt

python download_data.py         # downloads MovieLens 1M into data/
```

> You can also [download the dataset manually from GroupLens](https://grouplens.org/datasets/movielens/1m/) and place the `.dat` files in `data/`.

## How to Run

### Option 1 — Jupyter Notebook (Recommended)

Open `notebooks/main.ipynb` in VS Code or JupyterLab and hit **Run All**. Takes about 5–10 minutes depending on your machine.

### Option 2 — Command Line

```bash
python run_pipeline.py --data-dir data
```

Optional flags: `--n-factors`, `--n-epochs`, `--knn-k`, `--top-k`.

### Option 3 — REST API

```bash
DATA_DIR=data uvicorn src.api:app --port 8000
curl http://localhost:8000/recommend/42?n=5
```

### Run Tests

```bash
pip install fastapi uvicorn httpx
python -m pytest tests/ -v
```

All **104 tests** pass against a synthetic dataset in under 5 seconds.

## Generated Outputs

After a full run, you'll find these in `outputs/`:

| File | What It Shows |
|------|--------------|
| `metrics.csv` | Side-by-side comparison of all three models |
| `plots/rating_distribution.png` | How users rate — the classic J-curve |
| `plots/user_activity.png` | Some users rate 20 movies, others rate 2,000 |
| `plots/movie_popularity.png` | A few blockbusters get thousands of ratings; most get a handful |
| `plots/ratings_over_time.png` | Monthly trends in volume and average score |
| `plots/genre_analysis.png` | Which genres are most popular and best rated |
| `plots/model_comparison.png` | Visual comparison across all five metrics |
| `plots/svd_embeddings_pca.png` | PCA projection — genre clusters emerge from pure interaction data |
| `plots/svd_embeddings_tsne.png` | t-SNE projection — tighter clusters, local structure preserved |

## Design Decisions

| What I Did | Why |
|-----------|-----|
| Built models from scratch | Shows I understand the math, not just the API calls. Also avoids Cython issues on newer Python versions. |
| Temporal split instead of random | Random splits leak future data. Temporal splits mirror real deployment. |
| Full-catalogue ranking evaluation | The honest, industry-standard protocol. Many tutorials skip this because the numbers look less impressive — but they're more *meaningful*. |
| Bayesian popularity for cold-start | Raw averages are misleading when a movie has 3 ratings. The Bayesian prior naturally handles this. |
| Vectorized KNN prediction | `predict_for_user` scores all items in one pass. Without this, ranking evaluation would take hours. |
| Modular `src/` package | Keeps the notebook clean. Every function is independently importable and testable. |
| 104 unit + integration tests | Confidence that refactoring won't silently break things. All run on synthetic data. |

## Future Improvements

There's always more to build. Here's what I'd explore next:

- **Hybrid models** — Combine collaborative signals with content features (genres, plot text, cast metadata)
- **Neural collaborative filtering** — NeuMF, variational autoencoders, or two-tower architectures for capturing non-linear user-item interactions
- **Implicit feedback** — Extend to click/watch-time data using BPR or Weighted ALS
- **Online learning** — Incremental SVD updates (fold-in) as new ratings stream in, so the model stays fresh without full retraining
- **Caching layer** — Redis cache in front of the API for sub-millisecond inference at scale
- **Fairness and diversity** — Calibrated recommendations, intra-list diversity metrics, and exposure parity across content categories

## License

MIT — see [LICENSE](LICENSE) for details.

This project uses the MovieLens dataset under the [GroupLens research license](https://grouplens.org/datasets/movielens/).

## Author

**Manikanta Suryasai**
AI/ML Developer & Engineer

- GitHub: [@Manirider](https://github.com/Manirider)
- LinkedIn: [Manikanta Suryasai](https://linkedin.com/in/smanikanta)