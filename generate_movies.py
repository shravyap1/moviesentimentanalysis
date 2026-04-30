import json
import re
import pickle
from collections import Counter

with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower().strip()

def analyze_reviews(reviews):
    results = []
    for r in reviews:
        cleaned = clean_text(r)
        features = vectorizer.transform([cleaned])
        label = model.predict(features)[0]
        results.append(label)
    
    positive = sum(1 for r in results if r == 'positive')
    negative = len(results) - positive
    score = round((positive / len(results)) * 100) if results else 0
    return {
        "positive": positive,
        "negative": negative,
        "score": score,
        "positive_pct": round(positive / len(results) * 100, 1),
        "negative_pct": round(negative / len(results) * 100, 1)
    }

def get_top_words(reviews, n=30):
    all_words = []
    for r in reviews:
        words = clean_text(r).split()
        all_words.extend([w for w in words if len(w) > 4])
    return [word for word, count in Counter(all_words).most_common(n)]

# ---- PASTE YOUR PRE-SCRAPED REVIEWS HERE ----
# Format: { "Movie Title": ["review1", "review2", ...] }
# Split into thirds automatically for Early/Mid/Late

raw_reviews = {
    "Inception": [
        "A mind-bending masterpiece that keeps you on the edge.",
        "Confusing but brilliant filmmaking.",
        "One of the best films of the decade.",
        "Too complex for its own good but visually stunning.",
        "Christopher Nolan at his absolute best.",
        "The ending still haunts me years later.",
        "Overrated but enjoyable nonetheless.",
        "A true cinematic achievement.",
        "Gets better every time you watch it.",
        "The score alone makes it worth watching.",
        "Still thinking about it weeks later.",
        "Masterful storytelling and direction.",
    ],
    # Add all 20 movies here...
}

movies_data = {}
for title, reviews in raw_reviews.items():
    n = len(reviews)
    third = n // 3
    early = reviews[:third]
    mid = reviews[third:third*2]
    late = reviews[third*2:]
    
    early_stats = analyze_reviews(early)
    mid_stats = analyze_reviews(mid)
    late_stats = analyze_reviews(late)
    
    twist_score = abs(early_stats['score'] - late_stats['score'])
    
    # Mood scores for gradient (derived from sentiment)
    avg_score = (early_stats['score'] + mid_stats['score'] + late_stats['score']) / 3
    movies_data[title] = {
        "reviews": { "early": early, "mid": mid, "late": late },
        "sentiment": {
            "early": early_stats,
            "mid": mid_stats,
            "late": late_stats
        },
        "moodScores": {
            "excited": round(early_stats['positive_pct'] * 0.6),
            "angry": round(early_stats['negative_pct'] * 0.5),
            "nostalgic": round(late_stats['positive_pct'] * 0.4),
            "calm": round(late_stats['negative_pct'] * 0.3),
            "neutral": 10
        },
        "twistScore": twist_score,
        "overallScore": round(avg_score),
        "topWordsEarly": get_top_words(early),
        "topWordsLate": get_top_words(late)
    }

with open("movies.json", "w") as f:
    json.dump(movies_data, f, indent=2)

print(f"✅ movies.json generated with {len(movies_data)} movies!")