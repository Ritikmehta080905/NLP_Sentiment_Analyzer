# Smart Review Sentiment Analyzer (NLP Mini Project)

A Streamlit app for classifying product, movie, and restaurant review text as **Positive**, **Negative**, or **Neutral** using character-level TF-IDF and a calibrated linear SVM. It supports individual predictions and batch CSV analysis.

## Requirements
- Python 3.10 or newer
- Internet connection for the first installation of Python packages

## Run on Windows
1. Open Command Prompt in the project folder.
2. Run:

```bat
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`. If `py` is not recognized, install Python from https://www.python.org/downloads/ and select **Add Python to PATH** during setup.

## Features
- Individual review predictions with estimated class probabilities
- Sample positive and negative reviews
- CSV upload with selectable review-text column and downloadable results
- Empty CSV review rows are clearly marked as skipped
- Holdout accuracy, macro F1, per-class report, and confusion matrix
- Low-confidence predictions are flagged for caution

## NLP concepts used
1. Character-level TF-IDF feature extraction
2. Linear SVM classification with calibrated probability estimates
3. Hyperparameter selection with five-fold cross-validation on training data
4. Stratified holdout evaluation, kept separate from hyperparameter selection

## Important note
This is an educational model, not a production-grade sentiment service. Positive and Negative examples come from the 3,000-example UCI dataset (1,000 each from Amazon product, IMDb movie, and Yelp restaurant reviews). The original UCI data has no Neutral class, so Neutral is trained on a separate set of 61 manually curated factual examples in `data/neutral_review_examples.csv`. This class imbalance means Neutral performance may be weaker and its holdout score is based on a small sample. The model may misclassify sarcasm, mixed-language text, mixed sentiment, or reviews from unfamiliar domains. Confidence values are model probabilities, not calibrated guarantees.

## Dataset and attribution
The bundled `data/sentiment_labelled_sentences.csv` is derived from the UCI Machine Learning Repository's [Sentiment Labelled Sentences dataset](https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The dataset describes its source as Kotzias et al., “From Group to Individual Labels using Deep Features,” KDD 2015. The file retains each review's source domain and sentiment label. `data/neutral_review_examples.csv` contains manually curated factual examples to provide the third class; these are not from UCI.

## Tests
Run the model and CSV behavior tests from the project folder:

```bat
python -m unittest discover -s tests -v
```

## Project files
- `app.py` — Streamlit user interface
- `sentiment_model.py` — dataset validation, model training, evaluation, and prediction
- `data/sentiment_labelled_sentences.csv` — attributed labeled review data
- `data/neutral_review_examples.csv` — curated Neutral training examples
- `tests/` — automated model and batch-prediction tests
- `requirements.txt` — Python dependencies
- `LAB_REPORT.md` — report content aligned with the lab manual
