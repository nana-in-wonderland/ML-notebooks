# Content-Based Movie Recommender System

A content-based filtering recommender system that builds a user preference
profile from rated movies and recommends new movies based on genre similarity.

---

## How It Works

Unlike collaborative filtering (which finds similar users), content-based
filtering focuses on the **properties of the items themselves** — in this
case, movie genres.

| Step | What happens |
|---|---|
| 1. Input | User rates 5 movies |
| 2. Genre matrix | Each movie is represented as a binary vector of genres |
| 3. User profile | Weighted sum of genre vectors by user ratings |
| 4. Scoring | Every movie in the dataset is scored against the user profile |
| 5. Output | Top 20 most similar movies recommended |

---

## Key Concept — User Profile Vector

The user profile is built by multiplying the genre table of rated movies
by their ratings and summing the result:

```
user_profile = genre_table.T · ratings
```

This produces a weight for each genre — genres in highly-rated movies
get higher weights. Every unrated movie is then scored by how closely
its genre vector aligns with this profile.

---

## Dataset

[MovieLens](https://grouplens.org/datasets/movielens/) — two files needed:
- `movies.csv` — movieId, title, genres
- `ratings.csv` — userId, movieId, rating, timestamp

---

## Difference from Collaborative Filtering

| | Content-Based | Collaborative Filtering |
|---|---|---|
| Based on | Item properties (genres) | User behaviour (ratings) |
| Cold start | Handles new movies well | Struggles with new items |
| Diversity | Lower (stays in known genres) | Higher (discovers new genres) |
| Data needed | Item metadata | User rating history |

---

## Tech Stack

- Pandas, NumPy, Matplotlib
- Dataset: MovieLens

---

*Part of ML-notebooks — coursework exercises*
