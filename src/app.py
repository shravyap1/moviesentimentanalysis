"""
app.py  —  Flask backend
Fetches real TMDB reviews, runs them through your sentiment classifier,
returns results + sentiment gradient info to the Lovable frontend.

Setup:
  pip install flask flask-cors requests scikit-learn pandas
  export TMDB_API_KEY=your_key_here
  python app.py
"""

import os
import re
import pickle
import requests
from flask import Flask, jsonify, request
from flask_cors import CORS

# ── your existing modules ──────────────────────────────────────────────
from preprocess import clean_text
from features import create_features
from models import train_logistic

app = Flask(__name__)
CORS(app)  # allows Lovable frontend to call this API

TMDB_KEY = os.environ.get("TMDB_API_KEY", "YOUR_TMDB_KEY_HERE")
TMDB_BASE = "https://api.themoviedb.org/3"
IMG_BASE  = "https://image.tmdb.org/t/p/w500"

# ── load or train model ────────────────────────────────────────────────
# If you have a pre-trained model saved as model.pkl + vectorizer.pkl, load them.
# Otherwise train from your CSV on startup (slower but works out of the box).

MODEL_PATH      = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

def load_or_train_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(VECTORIZER_PATH, "rb") as f:
            vectorizer = pickle.load(f)
        print("✅ Loaded saved model.")
    else:
        print("⚙️  No saved model found — training from CSV...")
        from utils import parse_data
        texts, labels = parse_data("data/reviews.csv")   # <-- point to your CSV
        texts_clean   = [clean_text(t) for t in texts]
        X, vectorizer = create_features(texts_clean)
        model         = train_logistic(X, labels)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        with open(VECTORIZER_PATH, "wb") as f:
            pickle.dump(vectorizer, f)
        print("✅ Model trained and saved.")
    return model, vectorizer

model, vectorizer = load_or_train_model()


# ── helpers ────────────────────────────────────────────────────────────

def predict_sentiment(text: str) -> dict:
    """Run a single review through the classifier."""
    cleaned = clean_text(text)
    vec     = vectorizer.transform([cleaned])
    label   = model.predict(vec)[0]          # "positive" or "negative"

    # Probability score (only works for models with predict_proba)
    try:
        proba      = model.predict_proba(vec)[0]
        classes    = list(model.classes_)
        confidence = float(proba[classes.index(label)])
    except AttributeError:
        confidence = 1.0  # LinearSVC has no predict_proba

    return {"label": label, "confidence": round(confidence, 3)}


def sentiment_to_gradient(pos_pct: float) -> dict:
    """
    Convert positive-review percentage (0–100) into CSS gradient colours.
    Green  → mostly positive
    Amber  → mixed
    Red    → mostly negative
    """
    if pos_pct >= 65:
        return {
            "from": "#0a2e1a",
            "via":  "#041508",
            "to":   "#010904",
            "label": "Positive",
            "color": "#34d399"
        }
    elif pos_pct >= 40:
        return {
            "from": "#2e240a",
            "via":  "#151004",
            "to":   "#090701",
            "label": "Mixed",
            "color": "#fbbf24"
        }
    else:
        return {
            "from": "#2e0a0a",
            "via":  "#150404",
            "to":   "#090101",
            "label": "Negative",
            "color": "#ef4444"
        }


def tmdb_get(path: str, params: dict = {}) -> dict:
    params["api_key"] = TMDB_KEY
    r = requests.get(f"{TMDB_BASE}{path}", params=params, timeout=8)
    r.raise_for_status()
    return r.json()


# ── routes ─────────────────────────────────────────────────────────────

@app.route("/api/movies/popular")
def popular_movies():
    """Return a list of popular movies with poster URLs."""
    page = request.args.get("page", 1)
    data = tmdb_get("/movie/popular", {"page": page, "language": "en-US"})
    movies = []
    for m in data.get("results", []):
        movies.append({
            "id":          m["id"],
            "title":       m["title"],
            "year":        (m.get("release_date") or "")[:4],
            "poster":      IMG_BASE + m["poster_path"] if m.get("poster_path") else None,
            "backdrop":    "https://image.tmdb.org/t/p/w1280" + m["backdrop_path"] if m.get("backdrop_path") else None,
            "overview":    m.get("overview", ""),
            "rating":      m.get("vote_average", 0),
            "genres":      [],          # populated in /movie/<id>
        })
    return jsonify(movies)


@app.route("/api/movie/<int:movie_id>")
def movie_detail(movie_id: int):
    """
    Full movie detail + real TMDB reviews run through your sentiment model.
    """
    # 1. movie metadata
    details = tmdb_get(f"/movie/{movie_id}", {"language": "en-US"})

    # 2. all review pages from TMDB
    reviews_raw = []
    page = 1
    while True:
        rv = tmdb_get(f"/movie/{movie_id}/reviews", {"language": "en-US", "page": page})
        reviews_raw.extend(rv.get("results", []))
        if page >= rv.get("total_pages", 1):
            break
        page += 1
        if page > 5:          # safety cap
            break

    # 3. run each review through your classifier
    classified = []
    for r in reviews_raw:
        content = r.get("content", "")
        if len(content.strip()) < 20:   # skip empty/very short
            continue
        sentiment = predict_sentiment(content)
        classified.append({
            "author":     r.get("author", "Anonymous"),
            "content":    content,
            "url":        r.get("url", ""),
            "created_at": r.get("created_at", ""),
            "label":      sentiment["label"],
            "confidence": sentiment["confidence"],
        })

    # 4. compute overall sentiment stats
    total    = len(classified)
    pos      = sum(1 for r in classified if r["label"] == "positive")
    neg      = total - pos
    pos_pct  = round(pos / total * 100) if total else 50
    gradient = sentiment_to_gradient(pos_pct)

    return jsonify({
        "id":          details["id"],
        "title":       details["title"],
        "tagline":     details.get("tagline", ""),
        "overview":    details.get("overview", ""),
        "poster":      IMG_BASE + details["poster_path"] if details.get("poster_path") else None,
        "backdrop":    "https://image.tmdb.org/t/p/w1280" + details["backdrop_path"] if details.get("backdrop_path") else None,
        "rating":      details.get("vote_average", 0),
        "year":        (details.get("release_date") or "")[:4],
        "genres":      [g["name"] for g in details.get("genres", [])],
        "reviews":     classified,
        "sentiment": {
            "total":   total,
            "positive": pos,
            "negative": neg,
            "pos_pct": pos_pct,
            "gradient": gradient,
        }
    })


@app.route("/api/search")
def search():
    """Search movies by title."""
    query = request.args.get("q", "")
    if not query:
        return jsonify([])
    data = tmdb_get("/search/movie", {"query": query, "language": "en-US"})
    results = []
    for m in data.get("results", [])[:10]:
        results.append({
            "id":     m["id"],
            "title":  m["title"],
            "year":   (m.get("release_date") or "")[:4],
            "poster": IMG_BASE + m["poster_path"] if m.get("poster_path") else None,
        })
    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True, port=5000)