from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


log_pipe = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression(max_iter=2000))
])

log_grid = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__min_df": [1, 2, 5],
    "tfidf__max_df": [0.85, 0.95, 1.0],
    "tfidf__max_features": [5000, 10000, 20000],
    "tfidf__sublinear_tf": [False, True],
    "clf__C": [0.01, 0.1, 1, 3, 10],
    "clf__solver": ["lbfgs"],
    "clf__class_weight": [None, "balanced"],
}

log_search = GridSearchCV(
    log_pipe,
    log_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

nb_pipe = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", MultinomialNB())
])

nb_grid = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__min_df": [1, 2, 5],
    "tfidf__max_df": [0.85, 0.95, 1.0],
    "tfidf__max_features": [5000, 10000, 20000],
    "tfidf__sublinear_tf": [False, True],
    "clf__alpha": [0.01, 0.05, 0.1, 0.5, 1.0],
    "clf__fit_prior": [True, False],
}
svm_pipe = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LinearSVC())
])

svm_grid = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__min_df": [1, 2, 5],
    "tfidf__max_df": [0.85, 0.95, 1.0],
    "tfidf__max_features": [5000, 10000, 20000],
    "tfidf__sublinear_tf": [False, True],
    "clf__C": [0.01, 0.1, 1, 3, 10],
    "clf__loss": ["hinge", "squared_hinge"],
    "clf__class_weight": [None, "balanced"],
    "clf__dual": ["auto"],
    "clf__max_iter": [3000, 5000],
}
