<<<<<<< HEAD
from src.utils import parse_data
from src.preprocess import clean_text
from src.features import get_tfidf_features, encode_labels
from src.model import train_model, predict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

file_path = "data/reviews.txt"

texts, labels = parse_data(file_path)
cleaned_texts = [clean_text(text) for text in texts]

X, vectorizer = get_tfidf_features(cleaned_texts)
y, encoder = encode_labels(labels)
=======
import pandas as pd
from src.evaluation import evaluate
from src.models import train_logistic, train_nb, train_svm
from src.features import create_features
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# load the dataset 
df = pd.read_csv("/Users/mrithika/Movie Sentinent Analysis/project/data/reviews.txt")

#clean 
df = df.dropna()  # drop rows with missing values

#print to check the data
print(df.head())
print(df.columns)

#create features and labels
X, vectorizer = create_features(df['review'])
y = df['sentiment']

print(type(X))
print(X.shape)

#check how many unique sentiments are there
print(df['sentiment'].unique())

>>>>>>> 6a83cbc64a70c1dee6485aa0314f10f27192de0a

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

<<<<<<< HEAD
model = train_model(X_train, y_train)
y_pred = predict(model, X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=encoder.classes_))
=======

log_model = train_logistic(X_train, y_train)
nb_model = train_nb(X_train, y_train)
svm_model = train_svm(X_train, y_train)

print("Logistic Regression Evaluation:")
evaluate(log_model, X_test, y_test)

print("Naive Bayes Evaluation:")
evaluate(nb_model, X_test, y_test)

print("SVM Evaluation:")
evaluate(svm_model, X_test, y_test)
>>>>>>> 6a83cbc64a70c1dee6485aa0314f10f27192de0a
