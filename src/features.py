from sklearn.feature_extraction.text import TfidfVectorizer


def create_features(text_data):
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(text_data)
    return X, vectorizer




