from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


def get_tfidf_features(texts):
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(texts)
    return X, vectorizer


def encode_labels(labels):
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    return y, encoder
