from __future__ import annotations
import io
import math
import pickle
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from scipy import sparse

# -------------------------------
# Page / App Config
# -------------------------------
st.set_page_config(
    page_title="Anti-India / Hate Speech Detector",
    page_icon="🚨",
    layout="centered",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = r"""
<style>
  /* App background */
  .stApp {
    background: radial-gradient(1200px 600px at 10% 10%, #0b1220 0%, #0b1220 40%, #0a0f1a 100%),
                linear-gradient(120deg, #0b1220 0%, #121a2a 50%, #0b1220 100%);
    color: #e5e7eb;
  }

  /* Nice title gradient */
  .hero-title {
    font-size: clamp(28px, 3.8vw, 44px);
    font-weight: 800;
    line-height: 1.1;
    text-align: center;
    background: linear-gradient(90deg, #60a5fa 0%, #a78bfa 50%, #34d399 100%);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    margin: 0.2rem 0 0.5rem 0;
  }
  .hero-subtitle {
    text-align: center;
    color: #9ca3af;
    margin-bottom: 1rem;
  }

  /* Card (glassmorphism) */
  .card {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255,255,255,0.12);
    box-shadow: 0 10px 30px rgba(0,0,0,0.25);
    backdrop-filter: blur(8px);
    border-radius: 18px;
    padding: clamp(14px, 3vw, 22px);
    margin: 8px 0 16px 0;
  }

  /* Result chip */
  .result-chip {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    font-weight: 700;
    font-size: clamp(16px, 2.5vw, 22px);
    padding: 12px 18px;
    border-radius: 9999px;
    border: 1px solid rgba(255,255,255,0.14);
  }
  .is-hate {
    background: linear-gradient(90deg, #ef4444, #f97316);
    color: #111827;
  }
  .not-hate {
    background: linear-gradient(90deg, #34d399, #10b981);
    color: #0b1220;
  }

  /* Buttons (primary) */
  .stButton>button {
    width: 100%;
    border-radius: 14px;
    font-weight: 700;
    padding: 0.8rem 1rem;
    border: 1px solid rgba(255,255,255,0.15);
    background: linear-gradient(90deg, #6366f1, #22d3ee);
    color: #0b1220;
  }
  .stButton>button:hover {
    transform: translateY(-1px);
    box-shadow: 0 10px 24px rgba(34,211,238,0.15);
  }

  /* Textarea */
  textarea {
    border-radius: 14px !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    background: rgba(255,255,255,0.07) !important;
    color: #e5e7eb !important;
  }

  /* Pills for examples */
  .pill {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 9999px;
    border: 1px solid rgba(255,255,255,0.18);
    background: rgba(255,255,255,0.06);
    color: #cbd5e1;
    margin: 2px 6px 8px 0;
    cursor: pointer;
    font-size: 13px;
  }
  .pill:hover { filter: brightness(1.15); }

  /* Footer */
  .muted { color: #9ca3af; font-size: 12px; }

  /* Progress */
  .prob-wrap { display: grid; grid-template-columns: 1fr auto; gap: 8px; align-items: center; }

  /* Small screens spacing tweaks */
  @media (max-width: 640px) {
    .card { padding: 12px; }
  }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -------------------------------
# Utilities
# -------------------------------
@st.cache_resource(show_spinner=False)
def load_artifacts() -> Tuple[object, object]:
    """Load the trained model and TF-IDF vectorizer once and cache them."""
    model = pickle.load(open("hate_speech_model.pkl", "rb"))
    vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
    return model, vectorizer


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def get_probability(model: object, X_vec: sparse.csr_matrix) -> Optional[float]:
    """Best-effort probability for the positive (hate) class.
    Tries predict_proba, otherwise falls back to a sigmoid(decision_function).
    Returns None if neither is available.
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_vec)
        # Find column for class '1' if available
        if hasattr(model, "classes_"):
            try:
                idx = list(model.classes_).index(1)
            except ValueError:
                idx = -1  # fallback last column
        else:
            idx = -1
        return float(proba[0, idx])
    if hasattr(model, "decision_function"):
        score = model.decision_function(X_vec)
        # ensure scalar
        score = np.asarray(score).ravel()[0]
        return float(sigmoid(score))
    return None


def explain_contributions(
    model, vectorizer, X_vec, top_k: int = 8, class_index: int | None = None
):
    """Return top (token, contribution) pairs based on linear coefficients.
    Works for models with .coef_. If unavailable, returns an empty list.
    """
    if not hasattr(model, "coef_"):
        return []

    feature_names = (
        vectorizer.get_feature_names_out()
        if hasattr(vectorizer, "get_feature_names_out")
        else vectorizer.get_feature_names()
    )

    coef = model.coef_

    if coef.ndim == 1:   # already a single vector
        weights = coef
    elif coef.shape[0] == 1:   # binary classification stored in one row
        weights = coef[0]
    else:
        # multi-class case
        if class_index is None and hasattr(model, "classes_"):
            try:
                class_index = list(model.classes_).index(1)
            except ValueError:
                class_index = 0
        elif class_index is None:
            class_index = 0
        weights = coef[class_index]

    contrib_sparse = X_vec.multiply(weights)
    contrib = contrib_sparse.toarray().ravel()

    idx = np.argsort(-np.abs(contrib))[: top_k * 2]
    pairs = [(feature_names[i], float(contrib[i])) for i in idx if contrib[i] != 0]

    positives = [(t, c) for (t, c) in pairs if c > 0]
    if len(positives) >= top_k:
        return positives[:top_k]
    return pairs[:top_k]


def highlight_tokens_html(text: str, tokens: List[str]) -> str:
    """Highlight given tokens inside the text using <mark> with soft tint."""
    def repl(match: re.Match) -> str:
        return f"<mark style='background:#fde68a33;padding:2px 4px;border-radius:6px'>{match.group(0)}</mark>"

    for tok in sorted(set(tokens), key=len, reverse=True):
        if not tok or tok.strip() == "":
            continue
        # Rough word-boundary match, case-insensitive
        pattern = re.compile(rf"(?i)(?<!\\w){re.escape(tok)}(?!\\w)")
        text = pattern.sub(repl, text)
    return text

# -------------------------------
# Simple abusive lexicon overlay
# -------------------------------
abuse_words = {"madar", "chod", "bc", "mc", "bhosdi", "chutiya", "harami","suar","chodu","bhadve","chut","lunf","gandu","bhenchod"}

def contains_abuse(text: str) -> bool:
    tokens = text.lower().split()
    return any(tok in abuse_words for tok in tokens)

# -------------------------------
# App UI
# -------------------------------
st.markdown("""
<div class="card" style="text-align:center">
  <div class="hero-title">🚨 Detecting Anti‑India Campaigns</div>
  <div class="hero-subtitle">Type or paste a message to check if it likely contains hate / abusive content.</div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Controls")
    st.caption("Tune the decision threshold and view options.")
    threshold = st.slider("Hate probability threshold", 0.05, 0.95, 0.50, 0.01,
                          help="If predicted probability ≥ threshold → labeled as HATE.")
    show_explanations = st.toggle("Show token contributions", value=True)
    st.divider()
    st.markdown(
        "**About**\n\nThis demo uses your trained model + TF‑IDF vectorizer. Predictions may be imperfect, especially for slang variants.")

# Load model/vectorizer once
try:
    model, vectorizer = load_artifacts()
except Exception as e:
    st.error("Couldn't load artifacts (hate_speech_model.pkl / tfidf_vectorizer.pkl). Ensure files exist.")
    st.exception(e)
    st.stop()


# Example pills (click to paste)
import streamlit as st

st.write("### Try Examples:")

examples = [
    "You are welcome. Have a nice day!",
    "Jhoot mat bolo yaar, tumhari baaton mein nafrat hai",
    "Tumhara dimaag kharab ho gaya kya?",
]

# Initialize session state
if "example_text" not in st.session_state:
    st.session_state.example_text = ""

# Show each example as a button
for ex in examples:
    if st.button(ex):
        st.session_state.example_text = ex

# # Input box with auto-filled example
# text = st.text_area(
#     "Enter text:", 
   
#     key="text_input"
# )

# Tabs for single vs batch
single_tab, batch_tab = st.tabs(["Single Check", "Batch (CSV)"])

with single_tab:
    text = st.text_area(
        "Enter text / tweet:",
         value=st.session_state.example_text,
        height=150,
        placeholder="Write or paste Hinglish/English text here...",
    )
    analyze = st.button("Analyze")

    if analyze and text.strip():
        if contains_abuse(text):
            # Force hate if lexicon match
            pred = 1
            proba_view = 1.0
            reason = "Lexicon match"
            X = vectorizer.transform([text])   # ✅ still define X for explanations
            
        else:
            X = vectorizer.transform([text])
            proba = get_probability(model, X)
            if proba is None:
                pred = int(model.predict(X)[0])
                proba_view = None
            else:
                pred = int(proba >= threshold)
                proba_view = proba
            reason = "Model prediction"
            
        # Result card
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            if pred == 1:
                st.markdown(
                    f"<div class='result-chip is-hate'>⚠️ HATE / Anti‑India detected</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div class='result-chip not-hate'>✅ Not classified as hate</div>",
                    unsafe_allow_html=True,
                )

            st.caption(f"Decision source: {reason}")

            if proba_view is not None:
                pct = int(round(proba_view * 100))
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<div class='prob-wrap'>" \
                            f"<div>Model probability (hate): <b>{pct}%</b></div>" \
                            f"<div style='opacity:.7'>threshold: {int(threshold*100)}%</div>" \
                            "</div>", unsafe_allow_html=True)
                st.progress(min(max(proba_view, 0.0), 1.0))

            # Explanations: top contributor tokens
            if show_explanations:
                feats = explain_contributions(model, vectorizer, X, top_k=8)
                if feats:
                    toks = [t for t, _ in feats]
                    st.markdown("<br><b>Top contributing tokens</b> (positive → towards hate):", unsafe_allow_html=True)
                    for t, c in feats:
                        bar = min(1.0, max(0.0, abs(c)))
                        st.write(f"• {t}")
                        st.progress(min(1.0, 0.5 + 0.5 * (abs(c) / (abs(feats[0][1]) + 1e-9))))

                    # Highlight tokens in original text
                    st.markdown("<br><b>Highlighted text</b>", unsafe_allow_html=True)
                    st.markdown(
                        f"<div class='card'>{highlight_tokens_html(text, toks)}</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.caption("Explanations unavailable for this model type.")

            st.markdown('</div>', unsafe_allow_html=True)

    elif analyze and not text.strip():
        st.warning("Please enter some text to analyze.")


with batch_tab:
    st.write("Upload a CSV with a column named **text**. We'll return predictions and probabilities.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"], accept_multiple_files=False)
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception:
            uploaded.seek(0)
            df = pd.read_csv(uploaded, encoding_errors="ignore")

        if "text" not in df.columns:
            st.error("No 'text' column found. Please include a column named 'text'.")
        else:
            X = vectorizer.transform(df["text"].astype(str).tolist())

            proba = None
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)
                if hasattr(model, "classes_") and 1 in set(model.classes_):
                    idx = list(model.classes_).index(1)
                else:
                    idx = probs.shape[1] - 1
                proba = probs[:, idx]
                pred = (proba >= threshold).astype(int)
            else:
                pred = model.predict(X)

            df_out = df.copy()
            df_out["prediction"] = pred
            if proba is not None:
                df_out["hate_probability"] = np.round(proba, 4)

            st.dataframe(df_out.head(30), use_container_width=True)

            # Download button
            csv_bytes = df_out.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download results CSV",
                data=csv_bytes,
                file_name="predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )

# Footer / Disclaimer
st.markdown(
    """
    <div class="card">
      <div class="muted">
        ⚠️ <b>Disclaimer:</b> This tool is a statistical approximation and may produce false positives/negatives.
        Always apply human review for moderation or enforcement decisions.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
