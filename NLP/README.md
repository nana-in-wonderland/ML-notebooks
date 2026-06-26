# Restaurant Review Sentiment Analysis

An NLP pipeline that classifies restaurant reviews as positive or negative
using a Bag of Words model and a Naive Bayes classifier.

---

## How It Works

The text is cleaned and transformed into numerical features, then fed into
a Gaussian Naive Bayes classifier trained on labelled restaurant reviews.

| Step | What happens |
|---|---|
| 1. Input | 1,000 labelled restaurant reviews (.tsv) |
| 2. Cleaning | Remove punctuation, lowercase, stem, strip stopwords |
| 3. Vectorising | Build a Bag of Words matrix (top 1,500 features) |
| 4. Training | Fit a Gaussian Naive Bayes classifier on 80% of data |
| 5. Output | Predict sentiment on the remaining 20%, evaluate accuracy |

---

## Key Concept — Bag of Words

Each review is represented as a vector of word counts across a fixed vocabulary.

- Punctuation and numbers are stripped with regex
- Words are lowercased and reduced to their root form via **Porter Stemming**
- Common stopwords are removed — except **"not"**, which carries sentiment signal
- `CountVectorizer` keeps only the top 1,500 most frequent words

```python
review = re.sub('[^a-zA-Z]', ' ', review)
review = [ps.stem(w) for w in review.split() if w not in all_stopwords]

cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
```

---

## Results

| Metric | Value |
|---|---|
| Accuracy | 73% |
| True Positives | 91 |
| True Negatives | 55 |
| False Positives | 42 |
| False Negatives | 12 |

---

## Dataset

`Restaurant_Reviews.tsv` — tab-separated file, two columns:

- `Review` — raw review text
- `Liked` — sentiment label (1 = positive, 0 = negative)

---

## Tech Stack

- Pandas, NumPy, Matplotlib
- NLTK — stopwords, Porter Stemmer
- scikit-learn — CountVectorizer, GaussianNB, confusion matrix

---

*Part of ML-notebooks — coursework exercises*
