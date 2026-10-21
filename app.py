import pandas as pd
import streamlit as st

from sentiment_model import (
    LOW_CONFIDENCE_THRESHOLD,
    predict_reviews,
    predict_sentiment,
    train_model,
)

MODEL_CACHE_VERSION = "char-svm-cv-1"

st.set_page_config(page_title="Smart Review Sentiment Analyzer", page_icon="💬", layout="wide")
st.title("💬 Smart Review Sentiment Analyzer")
st.write(
    "Analyze individual reviews or upload a CSV for batch predictions. "
    "The model predicts Positive, Negative, or Neutral sentiment."
)


@st.cache_resource
def load_model(model_version):
    # Include a version key so cached models refresh when the training pipeline changes.
    return train_model()


try:
    model, metrics, training_examples, class_counts = load_model(MODEL_CACHE_VERSION)
except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
    st.error(f"Could not load or train the sentiment model: {error}")
    st.stop()

single_tab, batch_tab = st.tabs(["Single review", "Batch CSV analysis"])

with single_tab:
    left, right = st.columns([2, 1])
    with left:
        st.subheader("Analyze a review")
        review = st.text_area(
            "Enter a product, movie, or restaurant review:",
            placeholder="Example: The product quality is excellent and delivery was fast.",
            height=130,
            key="review",
        )
        if st.button("Analyze sentiment", type="primary"):
            if not review.strip():
                st.warning("Please enter a review first.")
            else:
                label, scores = predict_sentiment(review, model)
                st.success(f"Predicted sentiment: {label}")
                confidence = scores[label]
                st.caption(
                    "Confidence is the model's estimated probability, not a guarantee "
                    "of correctness."
                )
                st.progress(confidence)
                if confidence < LOW_CONFIDENCE_THRESHOLD:
                    st.warning(
                        "Low-confidence result. Treat this prediction as uncertain."
                    )
                st.write("**Estimated class probabilities**")
                st.bar_chart(
                    pd.DataFrame({"Probability": scores}).sort_values(
                        "Probability", ascending=False
                    )
                )

    with right:
        st.subheader("Try a sample")
        samples = {
            "Positive": "I absolutely love this product. Great quality and fast delivery.",
            "Negative": "Very poor quality. It stopped working and customer service was terrible.",
            "Neutral": "The package arrived today. The product is black and has two buttons.",
        }

        def load_sample(text):
            st.session_state["review"] = text

        for name, text in samples.items():
            st.button(
                f"Load {name} example",
                key=f"load_{name}",
                on_click=load_sample,
                args=(text,),
            )

with batch_tab:
    st.subheader("Analyze a CSV of reviews")
    st.write(
        "Upload a CSV file, choose the column containing review text, and download "
        "the results with predictions and confidence scores."
    )
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            uploaded_data = pd.read_csv(uploaded_file)
        except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as error:
            st.error(f"Could not read this CSV file: {error}")
        else:
            if uploaded_data.empty:
                st.error("The uploaded CSV has no review rows.")
            elif len(uploaded_data.columns) == 0:
                st.error("The uploaded CSV has no columns.")
            else:
                text_column = st.selectbox(
                    "Select the review text column",
                    options=uploaded_data.columns.tolist(),
                )
                result_key = (uploaded_file.file_id, text_column)
                if st.session_state.get("batch_result_key") != result_key:
                    st.session_state.pop("batch_results", None)
                    st.session_state["batch_result_key"] = result_key

                if st.button("Analyze uploaded reviews", type="primary"):
                    if not uploaded_data[text_column].fillna("").astype(str).str.strip().ne("").any():
                        st.error("The selected column contains no non-empty reviews.")
                    else:
                        predictions = predict_reviews(
                            uploaded_data[text_column].tolist(), model
                        )
                        st.session_state["batch_results"] = pd.concat(
                            [uploaded_data.reset_index(drop=True), predictions], axis=1
                        )

                results = st.session_state.get("batch_results")
                if results is not None:
                    st.dataframe(results, width="stretch")
                    if (results["Status"] == "Skipped: empty review").any():
                        st.warning(
                            f"{int((results['Status'] == 'Skipped: empty review').sum())} "
                            "empty review(s) were skipped and marked in the results."
                        )
                    low_confidence = int((results["Status"] == "Low confidence").sum())
                    if low_confidence:
                        st.warning(
                            f"{low_confidence} prediction(s) have low confidence. "
                            "Review their confidence scores carefully."
                        )
                    st.download_button(
                        "Download results as CSV",
                        data=results.to_csv(index=False).encode("utf-8-sig"),
                        file_name="sentiment_results.csv",
                        mime="text/csv",
                        key="download_sentiment_results",
                        on_click="ignore",
                    )

st.divider()
st.subheader("Model evaluation")
metric_columns = st.columns(3)
metric_columns[0].metric("Holdout accuracy", f"{metrics['accuracy']:.1%}")
metric_columns[1].metric("Holdout macro F1", f"{metrics['macro_f1']:.1%}")
metric_columns[2].metric("Training examples", f"{training_examples:,}")
st.caption(
    f"Evaluation uses a stratified 80/20 split ({metrics['test_examples']:,} held-out "
    "examples). Model settings are selected with 5-fold CV on the training split only."
)
st.caption(f"Best training-fold CV accuracy: {metrics['cv_accuracy']:.1%}")
st.caption(
    "Training examples by class: "
    + " · ".join(f"{label}: {class_counts[label]:,}" for label in class_counts)
)
st.info(
    "The Positive and Negative classes use 3,000 labeled sentences from Amazon, IMDb, "
    "and Yelp. Neutral uses a smaller, separately curated set of factual review "
    "examples, so performance may be weaker for Neutral and mixed-sentiment text."
)
with st.expander("Classification report"):
    st.text(metrics["report"])
with st.expander("Confusion matrix"):
    st.dataframe(
        pd.DataFrame(
            metrics["confusion_matrix"],
            index=[f"Actual {label}" for label in ("Negative", "Neutral", "Positive")],
            columns=[f"Predicted {label}" for label in ("Negative", "Neutral", "Positive")],
        ),
        width="stretch",
    )
with st.expander("About the dataset and NLP pipeline"):
    st.markdown(
        """
        - **Positive and Negative data:** 3,000 labeled review sentences, split evenly
          across Amazon, IMDb, and Yelp.
        - **Neutral data:** a small, manually curated set of factual review sentences;
          these examples are not part of the original UCI dataset.
        - **Features:** character-level TF-IDF n-grams, selected through cross-validation.
        - **Classifier:** class-weight-tuned Linear SVM with calibrated probabilities.
        - **Evaluation:** a stratified holdout set, accuracy, macro F1, per-class
          precision/recall/F1, and a confusion matrix for all three classes.
        - **Limitations:** estimated probabilities are not calibrated guarantees.
          Neutral examples are fewer and curated, so the model may not generalize
          as well to varied neutral language.
        """
    )
