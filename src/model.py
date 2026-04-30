from sklearn.linear_model import LogisticRegression


def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    return model


def predict(model, X_test):
    return model.predict(X_test)