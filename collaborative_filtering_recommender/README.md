# Collaborative Recommender System

A user-based collaborative filtering recommender system that identifies
users with similar taste and recommends movies based on their ratings.

---

## How It Works

Collaborative filtering finds users in the dataset whose rating patterns
are similar to the target user, then recommends movies those users loved
that the target user hasn't seen yet.

| Step | What happens |
|---|---|
| 1. Input | User rates 5 movies |
| 2. Matching | Find all dataset users who rated those same movies |
| 3. Similarity | Compute Pearson correlation between user and each match |
| 4. Weighting | Multiply each candidate movie's rating by similarity score |
| 5. Output | Rank unseen movies by weighted average score |

---

## Key Concept — Pearson Correlation

Implemented from scratch without any recommendation library.
Measures the linear similarity between two users' rating patterns:

- **+1** — identical taste, rate everything the same way
- **0** — no relationship
- **-1** — opposite taste, one rates high what the other rates low

```python
sxx = sum([i**2 for i in ratings_a]) - pow(sum(ratings_a), 2) / n
syy = sum([i**2 for i in ratings_b]) - pow(sum(ratings_b), 2) / n
sxy = sum(i*j for i, j in zip(ratings_a, ratings_b)) - sum(ratings_a)*sum(ratings_b) / n
correlation = sxy / sqrt(sxx * syy)
```

---

## Difference from Content-Based Filtering

| | Collaborative Filtering | Content-Based Filtering |
|---|---|---|
| Based on | User behaviour (ratings) | Item properties (genres) |
| Discovers new genres | Yes | No |
| Needs user history | Yes | No |
| Cold start problem | Yes | No |

---

## Dataset

[MovieLens](https://grouplens.org/datasets/movielens/) — two files needed:
- `movies.csv` — movieId, title, genres
- `ratings.csv` — userId, movieId, rating, timestamp

---

## Tech Stack

- Pandas, NumPy, Matplotlib
- Dataset: MovieLens

---

*Part of ML-notebooks — coursework exercises*
