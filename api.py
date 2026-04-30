import os
import re
import json
import pickle
from collections import Counter
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

PORT = int(os.environ.get("PORT", 10000))
OMDB_API_KEY = os.environ.get("OMDB_API_KEY", "")

# ── Load model + vectorizer once on startup ──────────────────────────────────
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# ── Load pre-computed movies data ────────────────────────────────────────────
with open("movies.json", "r") as f:
    movies_data = json.load(f)


# ── Helpers ──────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Strip HTML tags, punctuation, lowercase."""
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return text.lower().strip()


def predict_single(review: str) -> dict:
    """
    Run one review through TF-IDF vectorizer → Logistic Regression.
    Returns:
        { "label": "positive"|"negative", "confidence": 94.2 }
    LogisticRegression supports predict_proba; LinearSVC does not,
    so we fall back gracefully.
    """
    cleaned = clean_text(review)
    features = vectorizer.transform([cleaned])
    label = model.predict(features)[0]

    try:
        proba = model.predict_proba(features)[0]
        confidence = round(float(max(proba)) * 100, 1)
    except AttributeError:
        # LinearSVC has no predict_proba
        confidence = 100.0

    return {"label": str(label), "confidence": confidence}


def analyze_review_list(reviews: list) -> dict:
    """
    Classify a list of review strings.
    Returns aggregate stats used for drift chart + mood board.
    """
    if not reviews:
        return {
            "positive": 0, "negative": 0,
            "score": 50,
            "positive_pct": 50.0, "negative_pct": 50.0,
        }

    labels = [predict_single(r)["label"] for r in reviews]
    positive = sum(1 for l in labels if l == "positive")
    negative = len(labels) - positive
    score = round(positive / len(labels) * 100)

    return {
        "positive": positive,
        "negative": negative,
        "score": score,
        "positive_pct": round(positive / len(labels) * 100, 1),
        "negative_pct": round(negative / len(labels) * 100, 1),
    }


def get_top_words(reviews: list, n: int = 30) -> list:
    """Most frequent meaningful words across a list of reviews."""
    stopwords = {
        "the", "and", "this", "that", "with", "have", "from",
        "they", "what", "been", "were", "when", "their", "there",
        "just", "more", "also", "would", "could", "should", "very",
        "film", "movie", "films", "movies", "watch", "story", "scene",
    }
    words = []
    for r in reviews:
        for w in clean_text(r).split():
            if len(w) > 3 and w not in stopwords:
                words.append(w)
    return [word for word, _ in Counter(words).most_common(n)]


def calculate_mood_colors(overall_score: float, drift: float) -> dict:
    """
    Map % positive score → gradient hex colors.
    'drift' = early_score - late_score
      positive drift → hype decay (started hot, cooled)
      negative drift → slow burn (grew on people)
    """
    if overall_score >= 80:
        return {
            "topLeft": "#FF2D6B", "topRight": "#FFB347",
            "bottomLeft": "#9B59B6", "bottomRight": "#00BFFF",
            "centerOpacity": 0.35, "label": "Beloved",
        }
    elif overall_score >= 65:
        return {
            "topLeft": "#FFB347", "topRight": "#2ECC71",
            "bottomLeft": "#00BFFF", "bottomRight": "#9B59B6",
            "centerOpacity": 0.25, "label": "Well Liked",
        }
    elif overall_score >= 50:
        return {
            "topLeft": "#FF2D6B", "topRight": "#9B59B6",
            "bottomLeft": "#FFB347", "bottomRight": "#00BFFF",
            "centerOpacity": 0.20, "label": "Divisive",
        }
    elif overall_score >= 35:
        return {
            "topLeft": "#9B59B6", "topRight": "#00BFFF",
            "bottomLeft": "#FF2D6B", "bottomRight": "#2C3E50",
            "centerOpacity": 0.15, "label": "Disappointing",
        }
    else:
        return {
            "topLeft": "#C0392B", "topRight": "#FF2D6B",
            "bottomLeft": "#2C3E50", "bottomRight": "#9B59B6",
            "centerOpacity": 0.10, "label": "Hated",
        }


def build_movie_response(title: str, data: dict) -> dict:
    """
    Given raw movies.json entry, run model on reviews and return
    the full structured response Lovable needs.
    """
    reviews = data.get("reviews", {})
    early_reviews = reviews.get("early", [])
    mid_reviews   = reviews.get("mid",   [])
    late_reviews  = reviews.get("late",  [])

    early_stats = analyze_review_list(early_reviews)
    mid_stats   = analyze_review_list(mid_reviews)
    late_stats  = analyze_review_list(late_reviews)

    overall_score = round(
        (early_stats["score"] + mid_stats["score"] + late_stats["score"]) / 3
    )
    drift = early_stats["score"] - late_stats["score"]
    twist_score = abs(drift)
    overall_label = "positive" if overall_score >= 50 else "negative"

    mood_colors = calculate_mood_colors(overall_score, drift)

    return {
        "title": title,
        "overallScore": overall_score,
        "overallLabel": overall_label,
        "twistScore": twist_score,
        "moodColors": mood_colors,
        "sentiment": {
            "early": early_stats,
            "mid":   mid_stats,
            "late":  late_stats,
        },
        "topWordsEarly": get_top_words(early_reviews),
        "topWordsLate":  get_top_words(late_reviews),
        "sampleReviews": {
            "early": early_reviews[0]  if early_reviews else "",
            "mid":   mid_reviews[0]    if mid_reviews   else "",
            "late":  late_reviews[-1]  if late_reviews  else "",
        },
    }


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/movies", methods=["GET"])
def list_movies():
    """Return all available movie titles."""
    return jsonify(list(movies_data.keys()))


@app.route("/movie/<path:title>", methods=["GET"])
def get_movie(title):
    """
    Return full sentiment analysis for a movie.
    Case-insensitive match against movies.json keys.
    """
    for key in movies_data:
        if key.lower() == title.lower():
            result = build_movie_response(key, movies_data[key])
            return jsonify(result)
    return jsonify({"error": f"Movie '{title}' not found"}), 404


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Classify an arbitrary list of review strings.
    Body: { "reviews": ["review text 1", "review text 2", ...] }
    """
    body = request.get_json(force=True)
    reviews = body.get("reviews", [])

    if not reviews:
        return jsonify({"error": "No reviews provided"}), 400

    results = [predict_single(r) for r in reviews]
    positive = sum(1 for r in results if r["label"] == "positive")
    negative = len(results) - positive
    score = round(positive / len(results) * 100)

    return jsonify({
        "results": results,
        "summary": {
            "total":    len(results),
            "positive": positive,
            "negative": negative,
            "score":    score,
            "label":    "positive" if score >= 50 else "negative",
        },
    })


@app.route("/quiz/recommend", methods=["POST"])
def quiz_recommend():
    """
    Accept quiz answers and return top 3 matching movie titles.
    Body: { "vibe": "mind-bending", "feelAfter": "thoughtful", ... }
    Matching is tag-based — each movie in movies.json can have a 'tags' array.
    Falls back to top 3 by overallScore if no tags match.
    """
    prefs = request.get_json(force=True)
    vibe       = prefs.get("vibe", "").lower()
    feel_after = prefs.get("feelAfter", "").lower()

    scored = []
    for title, data in movies_data.items():
        tags = [t.lower() for t in data.get("tags", [])]
        score = 0
        if vibe and vibe in tags:
            score += 2
        if feel_after and feel_after in tags:
            score += 2
        # bonus: surface highly positive movies
        reviews = data.get("reviews", {})
        all_reviews = (
            reviews.get("early", []) +
            reviews.get("mid",   []) +
            reviews.get("late",  [])
        )
        if all_reviews:
            stats = analyze_review_list(all_reviews)
            score += stats["score"] / 50  # normalised 0-2 bonus
        scored.append((title, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    top3 = [t for t, _ in scored[:3]]
    return jsonify({"recommendations": top3})


# ── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)