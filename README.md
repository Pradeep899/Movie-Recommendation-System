# Movie-Recommendation-System
# 🎬 Movie Recommendation System

A content-based movie recommender that suggests similar movies based on plot, genre, keywords, cast, and director — with a head-to-head comparison between classic **TF-IDF** vectorization and modern **transformer-based sentence embeddings**.

Built on [The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) (45,000+ movies), with a Streamlit UI on top of a FastAPI backend.

---

## Why this project exists

Most tutorial-level recommenders stop at "vectorize the overview, compute cosine similarity, done." This one goes a step further: it builds **two independent recommendation pipelines** (TF-IDF and BGE sentence embeddings) on the same enriched dataset, then evaluates them against each other using genre/keyword/cast/director overlap and recommendation diversity — instead of just assuming the fancier model wins.

The result is a genuinely mixed finding, not a marketing claim, which is arguably the more interesting part of the project (see [Evaluation](#evaluation--tf-idf-vs-embeddings) below).

---

## Features

- 🔍 **Search-driven recommendations** — type a movie title, get 10 similar movies
- ⚖️ **Two recommendation engines** — TF-IDF (sparse, exact-match) vs. BGE embeddings (dense, semantic)
- 🧩 **Rich metadata fusion** — combines plot overview, genres, tagline, keywords, top-billed cast, and director into a single similarity signal (not just plot text)
- 📊 **Quantitative evaluation** — genre/keyword/cast/director overlap scores and recommendation diversity, measured across 50 randomly sampled movies
- 🖥️ **Streamlit UI** — clean, browsable interface with movie posters (TMDB API) and a details view
- ⚡ **FastAPI backend** — serves recommendations and TMDB metadata to the frontend

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data processing | `pandas`, `numpy` |
| Text cleaning | `nltk` (stopwords, WordNet lemmatizer), `re` |
| Vectorization | `scikit-learn` (`TfidfVectorizer`) |
| Embeddings | `sentence-transformers` (`BAAI/bge-small-en-v1.5`) |
| Similarity | `scikit-learn` (`cosine_similarity`) |
| Backend API | `FastAPI` |
| Frontend | `Streamlit` |
| External data | [TMDB API](https://www.themoviedb.org/documentation/api) (posters, live metadata) |
| Dataset | [The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) (Kaggle) |

---

## How it works

1. **Data merge** — `movies_metadata.csv` is merged with `credits.csv` and `keywords.csv` on movie `id`, deduplicated to avoid row-count blowups from repeated ids.
2. **Metadata extraction** — genres and keywords are parsed from JSON-like strings; only the **director** (not full crew) and **top 3 billed cast** (not the entire cast list) are extracted, to keep the similarity signal focused rather than diluted by minor names.
3. **Tag construction** — overview, genre, tagline, keywords, cast, and director are combined into a single `tags` field per movie.
4. **Text cleaning** — lowercased, punctuation stripped, stopwords removed, lemmatized.
5. **Two vector representations built from the same `tags`:**
   - **TF-IDF** (`max_features=50000`, uni+bigrams)
   - **BGE sentence embeddings** (`BAAI/bge-small-en-v1.5`, 384-dim, normalized)
6. **Recommendation** — cosine similarity against the query movie's vector, top-N returned (excluding the movie itself).
7. **Evaluation** — genre/keyword/cast/director Jaccard overlap + within-recommendation diversity, computed for both pipelines across the same 50 random movies.

---

## Evaluation — TF-IDF vs. Embeddings

Measured across 50 randomly sampled movies, top-10 recommendations each.

| Metric | TF-IDF | BGE Embeddings |
|---|---|---|
| Genre overlap | 0.361 | **0.390** |
| Keyword overlap | **0.097** | 0.051 |
| Cast overlap | **0.026** | 0.016 |
| Director overlap | **0.026** | 0.022 |
| Diversity (avg. pairwise similarity *within* a recommendation set — lower is more diverse) | **0.143** | 0.684 |

**Takeaway:** this isn't a clean "embeddings win" story, and I think that's the more honest and useful result. Embeddings edge out TF-IDF on genre-level thematic overlap — their real strength is capturing broad semantic similarity. But TF-IDF wins on keyword, cast, and director overlap, because it rewards exact token matches, which is exactly what those fields are. More strikingly, embedding-based recommendations were far less diverse from each other (0.684 vs. 0.143) — the model tends to cluster tightly around a movie's dominant theme rather than surfacing a varied set of related films.

**Practical implication:** for this dataset and short-text (overview-length) inputs, TF-IDF's exact-match behavior actually produces more varied, arguably more useful recommendation sets, while embeddings are stronger when genre/theme consistency matters more than exact entity matches. A hybrid approach (blend both scores) is a natural next step — see [Future Work](#future-work).

---

## Project Structure

```
Movie-Recommendation-System/
├── notebook.ipynb              # Full pipeline: data prep → TF-IDF → embeddings → evaluation
├── app.py                      # Streamlit frontend
├── api/                        # FastAPI backend (recommendation + TMDB endpoints)
├── tfidf_matrix.pkl            # Saved TF-IDF matrix
├── tfidf.pkl                   # Saved TF-IDF vectorizer
├── indices.pkl                 # Title → row index lookup
├── df.pkl                      # Cleaned, merged movie dataframe
├── movie_embeddings.npy        # Saved BGE embeddings (384-dim, normalized)
├── .gitattributes              # Git LFS tracking for .pkl / .npy
└── README.md
```

---

## Screenshots

`[add 1-2 screenshots of the Streamlit UI here]`
<img width="1918" height="917" alt="image" src="https://github.com/user-attachments/assets/78d3169e-0313-45c2-a60b-3dd7eb0faefa" />
<img width="1917" height="967" alt="Screenshot 2026-07-18 122044" src="https://github.com/user-attachments/assets/98a0235d-3128-4b2b-8b28-303f1cb8e786" />

---

## Future Work

- **Hybrid scoring** — weighted combination of TF-IDF and embedding similarity, rather than treating them as separate pipelines
- **Collaborative filtering** — incorporate `ratings.csv` (user-movie ratings) for a hybrid content + collaborative system
- **Re-ranking** — use a cross-encoder to re-rank the top-N candidates for higher precision
- **Weighted rating filter** — apply an IMDB-style Bayesian rating adjustment so low-vote-count outliers don't dominate recommendations

---
