# Smart Review Sentiment Analyzer (NLP Mini Project)

A ready-to-run Python mini-project for Experiment No. 10. It classifies a product review as **Positive**, **Negative**, or **Neutral** using TF-IDF features and Logistic Regression, with a Streamlit web interface.

## Requirements
- Python 3.10 or newer
- Internet connection for the first installation of Python packages

## Run on Windows
1. Extract `NLP_Sentiment_Analyzer.zip`.
2. Open the extracted `NLP_Sentiment_Analyzer` folder.
3. Click the folder address bar, type `cmd`, and press Enter.
4. Run these commands:

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
streamlit run app.py
```

A browser tab should open with the app. If `python` is not recognized, install Python from https://www.python.org/downloads/ and select **Add Python to PATH** during setup.

## Features
- Review text input
- Positive / Negative / Neutral classification
- Estimated class probabilities
- Built-in sample reviews
- Held-out validation accuracy and classification report
- Explanation of the NLP pipeline

## NLP concepts used
1. Text normalization (lowercasing and punctuation removal)
2. TF-IDF feature extraction with unigrams and bigrams
3. Logistic Regression text classification
4. Train/test split and accuracy/report evaluation

## Important note
This is a small educational demo dataset, not a production-grade sentiment model. Its metrics may vary with package versions, and it can misclassify complex, sarcastic, mixed-language, or unfamiliar reviews. Explain this limitation in your viva.

## Project files
- `app.py` — Streamlit user interface
- `sentiment_model.py` — sample data, preprocessing, model training and prediction
- `requirements.txt` — Python dependencies
- `LAB_REPORT.md` — report content aligned with the lab manual
