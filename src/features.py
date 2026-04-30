from sklearn.feature_extraction.text import TfidfVectorizer
<<<<<<< HEAD
from sklearn.preprocessing import LabelEncoder


def get_tfidf_features(texts):
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(texts)
    return X, vectorizer


def encode_labels(labels):
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    return y, encoder
=======

def create_features(text_data):
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(text_data)
    return X, vectorizer




>>>>>>> 6a83cbc64a70c1dee6485aa0314f10f27192de0a
