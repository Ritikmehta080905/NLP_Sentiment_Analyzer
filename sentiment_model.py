"""Train and evaluate the review sentiment classifier."""

from pathlib import Path
from typing import Sequence

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

CLASSES = ("Negative", "Neutral", "Positive")
DATASET_PATH = Path(__file__).resolve().parent / "data" / "sentiment_labelled_sentences.csv"
NEUTRAL_DATASET_PATH = (
    Path(__file__).resolve().parent / "data" / "neutral_review_examples.csv"
)
LOW_CONFIDENCE_THRESHOLD = 0.6


def load_dataset(
    dataset_path: str | Path = DATASET_PATH,
    neutral_dataset_path: str | Path = NEUTRAL_DATASET_PATH,
) -> pd.DataFrame:
    """Load and validate the labeled and curated neutral review datasets."""
    path = Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(f"Sentiment dataset not found: {path}")

    data = pd.read_csv(path)
    required_columns = {"review", "sentiment"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if data.empty:
        raise ValueError("The sentiment dataset contains no reviews.")

    if "source" not in data.columns:
        data["source"] = "Provided dataset"
    data = data.loc[:, ["review", "sentiment", "source"]].copy()

    neutral_path = Path(neutral_dataset_path)
    if not neutral_path.is_file():
        raise FileNotFoundError(f"Curated neutral examples not found: {neutral_path}")
    neutral_data = pd.read_csv(neutral_path)
    missing_columns = required_columns.difference(neutral_data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Neutral examples are missing required columns: {missing}")
    if "source" not in neutral_data.columns:
        neutral_data["source"] = "Curated neutral examples"
    data = pd.concat(
        [
            data,
            neutral_data.loc[:, ["review", "sentiment", "source"]],
        ],
        ignore_index=True,
    )

    data["review"] = data["review"].fillna("").astype(str).str.strip()
    data["sentiment"] = data["sentiment"].fillna("").astype(str).str.strip()
    data["source"] = data["source"].fillna("").astype(str).str.strip()
    data = data.loc[data["review"].ne("")].reset_index(drop=True)
    invalid_labels = sorted(set(data["sentiment"]) - set(CLASSES))
    if invalid_labels:
        raise ValueError(f"Dataset contains unsupported sentiment labels: {invalid_labels}")
    missing_labels = set(CLASSES) - set(data["sentiment"])
    if missing_labels:
        missing = ", ".join(sorted(missing_labels))
        raise ValueError(f"Dataset is missing required sentiment classes: {missing}")
    return data


def train_model(dataset_path: str | Path = DATASET_PATH):
    """Tune on training folds, evaluate once on a holdout, then fit all data."""
    data = load_dataset(dataset_path)
    train_reviews, test_reviews, train_labels, test_labels = train_test_split(
        data["review"],
        data["sentiment"],
        test_size=0.2,
        random_state=42,
        stratify=data["sentiment"],
    )

    search = GridSearchCV(
        make_pipeline(
            TfidfVectorizer(analyzer="char_wb", min_df=2),
            LinearSVC(random_state=42),
        ),
        param_grid={
            "tfidfvectorizer__ngram_range": [(2, 5), (3, 6)],
            "linearsvc__C": [0.5, 1, 2],
            "linearsvc__class_weight": [None, "balanced"],
        },
        scoring={"accuracy": "accuracy", "macro_f1": "f1_macro"},
        refit="accuracy",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1,
    )
    search.fit(train_reviews, train_labels)

    best_vectorizer = search.best_estimator_.named_steps["tfidfvectorizer"]
    best_classifier = search.best_estimator_.named_steps["linearsvc"]
    model = make_pipeline(
        TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=best_vectorizer.ngram_range,
            min_df=best_vectorizer.min_df,
        ),
        CalibratedClassifierCV(
            estimator=LinearSVC(
                C=best_classifier.C,
                class_weight=best_classifier.class_weight,
                random_state=42,
            ),
            method="sigmoid",
            cv=3,
        ),
    )
    model.fit(train_reviews, train_labels)
    test_predictions = model.predict(test_reviews)

    metrics = {
        "accuracy": float(accuracy_score(test_labels, test_predictions)),
        "macro_f1": float(f1_score(test_labels, test_predictions, average="macro")),
        "cv_accuracy": float(search.best_score_),
        "best_parameters": search.best_params_,
        "report": classification_report(
            test_labels, test_predictions, labels=CLASSES, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            test_labels, test_predictions, labels=CLASSES
        ),
        "test_examples": len(test_reviews),
    }

    model.fit(data["review"], data["sentiment"])
    class_counts = data["sentiment"].value_counts().reindex(CLASSES, fill_value=0)
    return model, metrics, len(data), class_counts.to_dict()


def predict_sentiment(text: str, model):
    """Return the predicted class and estimated probabilities for one review."""
    if not isinstance(text, str):
        raise TypeError("Review text must be a string.")
    if not text.strip():
        raise ValueError("Review text cannot be empty.")

    probabilities = model.predict_proba([text])[0]
    scores = {
        label: float(probability)
        for label, probability in zip(model.classes_, probabilities)
    }
    return max(scores, key=scores.get), scores


def predict_reviews(reviews: Sequence[object], model) -> pd.DataFrame:
    """Predict a batch of reviews, marking blank rows instead of failing."""
    results = []
    for review in reviews:
        text = "" if pd.isna(review) else str(review).strip()
        if not text:
            results.append(
                {
                    "Predicted sentiment": "Not analyzed",
                    "Confidence": float("nan"),
                    "Status": "Skipped: empty review",
                }
            )
            continue

        label, probabilities = predict_sentiment(text, model)
        confidence = probabilities[label]
        results.append(
            {
                "Predicted sentiment": label,
                "Confidence": confidence,
                "Status": (
                    "Low confidence"
                    if confidence < LOW_CONFIDENCE_THRESHOLD
                    else "Analyzed"
                ),
            }
        )
    return pd.DataFrame(results)
