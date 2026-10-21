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
- **Text normalization:** converts text to lowercase and removes punctuation.
- **TF-IDF:** represents text numerically according to the importance of words and word combinations in the dataset.
- **Logistic Regression:** learns patterns from labeled examples and predicts a sentiment class.
- **Model evaluation:** a held-out test split is used to calculate accuracy and a classification report.

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
4. Logistic Regression can be used for multi-class text classification.
5. A held-out test set provides an estimate of model performance on unseen examples.
6. A small dataset limits generalization, so predictions can be incorrect for new or ambiguous reviews.

## B.4 Conclusion
The Smart Review Sentiment Analyzer was developed using Python, TF-IDF, Logistic Regression, and Streamlit. It demonstrates a basic NLP workflow by accepting review text and predicting whether its sentiment is Positive, Negative, or Neutral. The project helped demonstrate text preprocessing, feature extraction, classification, and evaluation.

## Viva Questions
1. **What is NLP?** Natural Language Processing is a field of AI that enables computers to process human language.
2. **What is sentiment analysis?** It identifies the opinion or sentiment expressed in text.
3. **What is TF-IDF?** It is a method for converting text into numerical features based on term frequency and inverse document frequency.
4. **Why is Logistic Regression used?** It is a machine-learning classifier that can predict classes from numerical features.
5. **Why split data into training and testing sets?** To estimate how well the model performs on examples it did not train on.
6. **What is a limitation of this project?** The dataset is small and predictions may not generalize to real-world reviews.
