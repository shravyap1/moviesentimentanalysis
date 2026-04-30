from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import json
import re

app = Flask(__name__)
CORS(app)  # allows Lovable to call this

# Load model + vectorizer once on startup
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load pre-computed movie data
with open("movies.json", "r") as f:
    movies_data = json.load(f)

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)       # remove HTML
    text = re.sub(r'[^a-zA-Z\s]', '', text) # remove special chars
    return text.lower().strip()

def predict_sentiment(review):
    cleaned = clean_text(review)
    features = vectorizer.transform([cleaned])
    label = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    return {
        "label": label,
        "confidence": round(max(proba) * 100, 1)
    }

@app.route('/movie/<title>', methods=['GET'])
def get_movie(title):
    # Case-insensitive match
    for key in movies_data:
        if key.lower() == title.lower():
            return jsonify(movies_data[key])
    return jsonify({"error": "Movie not found"}), 404

@app.route('/movies', methods=['GET'])
def list_movies():
    return jsonify(list(movies_data.keys()))

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    reviews = data.get('reviews', [])
    results = [predict_sentiment(r) for r in reviews]
    
    positive = sum(1 for r in results if r['label'] == 'positive')
    negative = len(results) - positive
    score = round((positive / len(results)) * 100) if results else 0
    
    return jsonify({
        "results": results,
        "summary": {
            "total": len(results),
            "positive": positive,
            "negative": negative,
            "score": score
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(debug=True)