# Experiment No. 10 — Mini Project

**Course:** Natural Language Processing  
**Project Title:** Smart Review Sentiment Analyzer  
**Name:** ____________________  
**Roll No.:** ____________________  
**Class/Batch:** ____________________  
**Date:** ____________________

## A.1 Aim
To develop a real-world Natural Language Processing application that classifies product reviews into Positive, Negative, or Neutral sentiment using Python and machine learning.

## A.2 Prerequisites
Python, basic text processing, and basic machine-learning concepts.

## A.3 Outcome
After completing the project, the student can demonstrate a basic NLP pipeline involving text preprocessing, feature extraction, text classification, and evaluation.

## Theory
Sentiment analysis is an NLP task used to identify the opinion or emotional polarity expressed in text. Businesses can analyze customer reviews to understand whether feedback is positive, negative, or neutral.

This project uses:
- **Character-level TF-IDF:** represents text numerically using character sequences.
- **Linear SVM:** learns to classify reviews as Positive, Negative, or Neutral. Its probability estimates are calibrated for the application interface.
- **Hyperparameter selection:** five-fold cross-validation is performed on the training split.
- **Model evaluation:** a stratified held-out split is used to calculate accuracy, macro F1, per-class precision and recall, and a confusion matrix.
- **Batch inference:** a CSV upload can be analyzed and downloaded with predictions and confidence scores.

The bundled UCI dataset contains 3,000 labeled review sentences from Amazon, IMDb, and Yelp, with positive and negative labels. Since the source dataset has no neutral class, this project includes a separate set of 61 manually curated factual review examples for the Neutral class. Results for Neutral should be interpreted cautiously because this class has fewer training examples.

## B.1 Software Code Written by Student
The complete source code is provided in `app.py` and `sentiment_model.py` in the project folder.

## B.2 Input and Output
**Input 1:** `I absolutely love this product. Great quality and fast delivery.`  
**Expected type of output:** Predicted sentiment: Positive

**Input 2:** `Very poor quality. It stopped working and customer service was terrible.`  
**Expected type of output:** Predicted sentiment: Negative

**Input 3:** `The package arrived today. The product is black and has two buttons.`  
**Expected type of output:** Predicted sentiment: Neutral

The actual prediction and probability scores are displayed in the running application. Results depend on the trained model; record the output you observe during execution.

## B.3 Observations and Learning
1. NLP allows computers to process and classify human language.
2. Text must be converted into numerical features before it can be used by a traditional machine-learning classifier.
3. TF-IDF gives greater weight to terms that are informative within the dataset.
4. A linear Support Vector Machine can classify text using numeric TF-IDF features.
5. A held-out test set provides an estimate of model performance on unseen examples.
6. The 3,000 UCI examples cover three review domains and provide Positive/Negative labels.
7. The Neutral class uses 61 manually curated factual examples and is less representative.

## B.4 Conclusion
The Smart Review Sentiment Analyzer was developed using Python, character-level TF-IDF, a calibrated linear SVM, and Streamlit. It predicts Positive, Negative, or Neutral sentiment for review text and supports batch CSV analysis. The project demonstrates feature extraction, classification, evaluation, and downloadable inference results.

## Viva Questions
1. **What is NLP?** Natural Language Processing is a field of AI that enables computers to process human language.
2. **What is sentiment analysis?** It identifies the opinion or sentiment expressed in text.
3. **What is TF-IDF?** It is a method for converting text into numerical features based on term frequency and inverse document frequency.
4. **Why is a linear SVM used?** It learns class boundaries between numeric text features and performs well on sparse, high-dimensional TF-IDF data.
5. **Why split data into training and testing sets?** To estimate how well the model performs on examples it did not train on.
6. **What is a limitation of this project?** The dataset is small and predictions may not generalize to real-world reviews.
