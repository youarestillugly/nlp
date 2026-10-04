import math
from pathlib import Path

import joblib
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "imdb_sentiment_final_pipeline.joblib"

st.set_page_config(
    page_title="IMDB Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.write(
    "Enter a movie review and the classical NLP model will "
    "predict whether the sentiment is Positive or Negative."
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

if not MODEL_PATH.exists():
    st.error("The trained model could not be found.")
    st.code(f"Expected model location:\n{MODEL_PATH}")
    st.stop()

try:
    model = load_model()
except Exception as e:
    st.error("The trained model could not be loaded.")
    st.code(
        f"Expected model location:\n{MODEL_PATH}\n\n"
        f"Error:\n{e}"
    )
    st.stop()

review = st.text_area(
    "Movie Review",
    placeholder=(
        "Example: The acting was excellent and "
        "I really enjoyed this movie."
    ),
    height=180
)

if st.button("Predict Sentiment", type="primary"):
    if not review.strip():
        st.warning("Please enter a movie review before predicting.")
    elif len(review.strip()) < 3:
        st.warning(
            "Please enter a little more text so the model "
            "can make a meaningful prediction."
        )
    else:
        prediction = model.predict([review])[0]
        decision_score = float(model.decision_function([review])[0])

        confidence_like = (
            1 / (1 + math.exp(-abs(decision_score)))
        ) * 100

        if prediction == 1:
            st.success("### 😊 Predicted Sentiment: Positive")
        else:
            st.error("### 😞 Predicted Sentiment: Negative")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Decision Score", f"{decision_score:.3f}")
        with col2:
            st.metric("Confidence-like Score", f"{confidence_like:.1f}%")

        if decision_score > 0:
            st.info(
                "The positive decision score indicates that the "
                "model leans toward Positive sentiment."
            )
        else:
            st.info(
                "The negative decision score indicates that the "
                "model leans toward Negative sentiment."
            )

        st.caption(
            "The decision score is produced by the Linear SVM. "
            "The confidence-like score is derived from its magnitude "
            "and is not a calibrated probability."
        )

st.divider()
st.caption(
    "Model: classical NLP pipeline using the same text cleaning, "
    "TF-IDF feature extraction, and tuned Linear SVM classifier "
    "used during training."
)
