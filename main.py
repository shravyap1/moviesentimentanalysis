from matplotlib import cm
import pandas as pd
from src.features import create_features
from src.models import train_logistic, train_nb, train_svm
from src.evaluation import evaluate
from sklearn.model_selection import train_test_split
import pickle

# Load dataset
df = pd.read_csv("data/reviews.txt")
df = pd.read_csv("/Users/mrithika/Movie Sentinent Analysis/project/data/reviews.txt")
df = df.dropna()

# Features + labels
X, vectorizer = create_features(df['review'])
y = df['sentiment']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train best model (Logistic Regression wins for this task)
model = train_logistic(X_train, y_train)
model_nb = train_nb(X_train, y_train)
model_svm = train_svm(X_train, y_train)

print("Logistic Regression:")
evaluate(model, X_test, y_test)

print("\nMultinomial Naive Bayes:")
evaluate(model_nb, X_test, y_test)
print("\nLinear SVM:")
evaluate(model_svm, X_test, y_test)

# Save model + vectorizer so Flask can load them
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
   
print("✅ model.pkl and vectorizer.pkl saved!")