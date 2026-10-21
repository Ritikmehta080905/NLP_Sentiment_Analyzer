import tempfile
import unittest
from pathlib import Path

import pandas as pd

from sentiment_model import (
    CLASSES,
    load_dataset,
    predict_reviews,
    predict_sentiment,
    train_model,
)


class SentimentModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.metrics, cls.training_examples, cls.class_counts = train_model()

    def test_bundled_dataset_has_expected_size_and_labels(self):
        data = load_dataset()
        self.assertEqual(len(data), 3061)
        self.assertEqual(set(data["sentiment"]), set(CLASSES))
        self.assertEqual(data["source"].nunique(), 4)
        self.assertEqual(self.class_counts["Neutral"], 61)

    def test_holdout_metrics_are_present_and_bounded(self):
        self.assertGreaterEqual(self.metrics["accuracy"], 0.0)
        self.assertLessEqual(self.metrics["accuracy"], 1.0)
        self.assertGreaterEqual(self.metrics["macro_f1"], 0.0)
        self.assertLessEqual(self.metrics["macro_f1"], 1.0)
        self.assertGreaterEqual(self.metrics["cv_accuracy"], 0.0)
        self.assertLessEqual(self.metrics["cv_accuracy"], 1.0)
        self.assertEqual(self.metrics["test_examples"], 613)
        self.assertEqual(self.metrics["confusion_matrix"].shape, (3, 3))
        self.assertIn("precision", self.metrics["report"])

    def test_prediction_probabilities_sum_to_one(self):
        label, scores = predict_sentiment(
            "The quality is excellent and I love this purchase.", self.model
        )
        self.assertIn(label, CLASSES)
        self.assertAlmostEqual(sum(scores.values()), 1.0)

    def test_single_prediction_rejects_blank_text(self):
        with self.assertRaises(ValueError):
            predict_sentiment("  ", self.model)

    def test_neutral_review_can_be_predicted(self):
        label, scores = predict_sentiment(
            "The package contains one charger and a USB cable.", self.model
        )
        self.assertEqual(label, "Neutral")
        self.assertEqual(set(scores), set(CLASSES))

    def test_batch_predictions_preserve_rows_and_mark_empty_reviews(self):
        results = predict_reviews(
            ["Excellent product and fast delivery.", "", None], self.model
        )
        self.assertEqual(len(results), 3)
        self.assertEqual(results.loc[0, "Status"], "Analyzed")
        self.assertIn(results.loc[0, "Predicted sentiment"], CLASSES)
        self.assertEqual(results.loc[1, "Status"], "Skipped: empty review")
        self.assertEqual(results.loc[2, "Predicted sentiment"], "Not analyzed")
        self.assertTrue(pd.isna(results.loc[2, "Confidence"]))

    def test_dataset_validation_reports_missing_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.csv"
            pd.DataFrame({"review": ["Good"]}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "sentiment"):
                load_dataset(path)


if __name__ == "__main__":
    unittest.main()
