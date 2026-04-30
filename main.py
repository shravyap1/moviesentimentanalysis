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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = train_model(X_train, y_train)
y_pred = predict(model, X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=encoder.classes_))