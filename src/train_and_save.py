"""
train_and_save.py
Run this ONCE to train your model on your CSV and save model.pkl + vectorizer.pkl.

Usage:
  python train_and_save.py --data data/reviews.csv
"""

import argparse
import pickle
from utils import parse_data
from preprocess import clean_text
from features import create_features
from models import train_logistic
from evaluation import evaluate
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data/reviews.csv", help="Path to reviews CSV (review, sentiment)")
args = parser.parse_args()

print(f"📂 Loading data from {args.data}...")
texts, labels = parse_data(args.data)

print(f"🧹 Cleaning {len(texts)} reviews...")
texts_clean = [clean_text(t) for t in texts]

print("🔢 Vectorizing...")
X, vectorizer = create_features(texts_clean)

print("✂️  Splitting train/test (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42, stratify=labels
)

print("🏋️  Training Logistic Regression...")
model = train_logistic(X_train, y_train)

print("\n📊 Evaluation on test set:")
evaluate(model, X_test, y_test)

print("\n💾 Saving model.pkl and vectorizer.pkl...")
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Done! Run `python app.py` to start the Flask server.")