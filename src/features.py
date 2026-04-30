from sklearn.feature_extraction.text import TfidfVectorizer


def create_features(text_data):
    vectorizer = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        stop_words="english",
        use_idf=True,
        smooth_idf=True,
        norm="l2",
        binary=False
    )
    X = vectorizer.fit_transform(text_data)
    return X, vectorizer




