"""Small, self-contained educational sentiment classifier."""
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

DATA = [
    ("excellent quality and very useful", "Positive"),
    ("I love this product", "Positive"),
    ("works perfectly and looks great", "Positive"),
    ("fast delivery and amazing service", "Positive"),
    ("very happy with my purchase", "Positive"),
    ("good value for money", "Positive"),
    ("easy to use and reliable", "Positive"),
    ("wonderful product highly recommended", "Positive"),
    ("the quality is fantastic", "Positive"),
    ("great experience with this item", "Positive"),
    ("battery lasts long and performance is great", "Positive"),
    ("beautiful design and excellent finish", "Positive"),
    ("I am satisfied with the product", "Positive"),
    ("superb quality at a good price", "Positive"),
    ("delivery was quick and packaging was nice", "Positive"),
    ("it exceeded my expectations", "Positive"),
    ("poor quality and stopped working", "Negative"),
    ("I hate this product", "Negative"),
    ("very bad experience", "Negative"),
    ("delivery was late and packaging was damaged", "Negative"),
    ("not worth the money", "Negative"),
    ("terrible customer service", "Negative"),
    ("the product broke after one day", "Negative"),
    ("waste of money do not buy", "Negative"),
    ("disappointed with the quality", "Negative"),
    ("slow performance and frequent errors", "Negative"),
    ("the item arrived damaged", "Negative"),
    ("uncomfortable and poorly designed", "Negative"),
    ("I want a refund this is awful", "Negative"),
    ("battery life is terrible", "Negative"),
    ("the product does not work", "Negative"),
    ("very frustrating and unreliable", "Negative"),
    ("the package arrived today", "Neutral"),
    ("the product is black", "Neutral"),
    ("it has two buttons and a cable", "Neutral"),
    ("the box contains one device", "Neutral"),
    ("I received the item yesterday", "Neutral"),
    ("the product weighs one kilogram", "Neutral"),
    ("the device has a screen", "Neutral"),
    ("it comes with a user manual", "Neutral"),
    ("the model number is 204", "Neutral"),
    ("the item is available in three sizes", "Neutral"),
    ("the package contains a charger", "Neutral"),
    ("the product was delivered on Monday", "Neutral"),
    ("the device has a blue indicator light", "Neutral"),
    ("the box is made of cardboard", "Neutral"),
]

def clean_text(text):
    """Basic normalization for demonstration and inspection."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def train_model():
    texts = [clean_text(text) for text, _ in DATA]
    labels = [label for _, label in DATA]
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(X_train_vec, y_train)
    predictions = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, zero_division=0)
    # Refit on all examples for the final demo model.
    vectorizer.fit(texts)
    model.fit(vectorizer.transform(texts), labels)
    return model, vectorizer, accuracy, report

def predict_sentiment(text, model, vectorizer):
    clean = clean_text(text)
    probabilities = model.predict_proba(vectorizer.transform([clean]))[0]
    labels = model.classes_
    scores = {label: float(prob) for label, prob in zip(labels, probabilities)}
    prediction = max(scores, key=scores.get)
    return prediction, scores
