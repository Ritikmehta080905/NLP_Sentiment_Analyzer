import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sentiment_model import train_model, predict_sentiment

st.set_page_config(page_title="Smart Review Sentiment Analyzer", page_icon="💬", layout="wide")
st.title("💬 Smart Review Sentiment Analyzer")
st.write("An NLP mini-project that classifies product reviews as Positive, Negative, or Neutral.")

@st.cache_resource
def load_model():
    return train_model()

model, vectorizer, accuracy, report = load_model()

left, right = st.columns([2, 1])
with left:
    st.subheader("Analyze a review")
    review = st.text_area(
        "Enter a product review:",
        placeholder="Example: The product quality is excellent and delivery was fast.",
        height=130,
        key="review",
    )
    if st.button("Analyze Sentiment", type="primary"):
        if not review.strip():
            st.warning("Please enter a review first.")
        else:
            label, scores = predict_sentiment(review, model, vectorizer)
            st.success(f"Predicted sentiment: {label}")
            st.caption("The confidence score is the model's estimated probability, not a guarantee of correctness.")
            st.progress(float(max(scores.values())))
            st.write("**Estimated class probabilities**")
            st.bar_chart(pd.DataFrame({"Probability": scores}).sort_values("Probability", ascending=False))

with right:
    st.subheader("Try a sample")
    samples = {
        "Positive": "I absolutely love this product. Great quality and fast delivery.",
        "Negative": "Very poor quality. It stopped working and customer service was terrible.",
        "Neutral": "The package arrived today. The product is black and has two buttons."
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

st.divider()
st.subheader("Model information")
m1, m2 = st.columns(2)
m1.metric("Validation accuracy", f"{accuracy:.1%}")
m2.metric("Training examples", "Built-in sample dataset")
st.info("This is an educational demonstration trained on a small example dataset. Its accuracy and predictions are not suitable for production use.")
with st.expander("Classification report"):
    st.text(report)
with st.expander("About the NLP pipeline"):
    st.markdown("""
    1. **Text preprocessing:** lowercase text and remove punctuation through vectorizer settings.
    2. **Feature extraction:** TF-IDF converts words and word combinations into numeric features.
    3. **Classification:** Logistic Regression predicts sentiment classes.
    4. **Evaluation:** a held-out split is used to estimate accuracy and generate a classification report.
    """)
