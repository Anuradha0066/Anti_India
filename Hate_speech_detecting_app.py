import os
import streamlit as st
import pickle
import pandas as pd
import joblib
import re, string, unicodedata

# ==============================
# Load model and vectorizer
# ==============================
# ==============================
# File paths
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "hate_speech_model.pkl")
VECTORIZER_FILE = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")

# ==============================
# Function to safely load pickle/joblib files
# ==============================
def load_model(path, use_joblib=False):
    try:
        if use_joblib:
            obj = joblib.load(path)
        else:
            with open(path, "rb") as f:
                obj = pickle.load(f)
        # st.success(f"✅ Loaded: {os.path.basename(path)}")
        return obj
    except FileNotFoundError:
        st.error(f"❌ File not found: {path}")
    except ModuleNotFoundError as e:
        st.error(f"❌ Missing module while loading pickle: {e.name}")
    except Exception as e:
        st.error(f"❌ Error loading file '{os.path.basename(path)}': {e}")
    return None

# ==============================
# Load model and vectorizer
# ==============================
# If your model/vectorizer is from scikit-learn, use_joblib=True is recommended
model = load_model(MODEL_FILE, use_joblib=True)
vectorizer = load_model(VECTORIZER_FILE, use_joblib=True)

# Stop the app if loading failed
if model is None or vectorizer is None:
    st.stop()

# ==============================
# Page Config
# ==============================
st.set_page_config(
    page_title="Anti-India / Hate Speech Detector",
    page_icon="🧠",
    layout="wide",
)

# ==============================
# Styling
# ==============================
st.markdown("""
    <style>
        body { background-color: #0e1117; color: #fafafa; }
        .main { background: linear-gradient(180deg, #0e1117 0%, #161a22 100%);
                color: white; font-family: "Poppins", sans-serif; }
        .title { text-align: center; color: #a2d2ff; font-size: 2.2rem; font-weight: 700; }
        .analyze-button > button {
            width: 100%; background: linear-gradient(90deg, #00b4db, #0083b0);
            color: white; border: none; border-radius: 8px; font-size: 1.1rem; padding: 10px;
        }
        .analyze-button > button:hover {
            background: linear-gradient(90deg, #36d1dc, #5b86e5); color: #fff;
        }
        textarea { border-radius: 10px !important; font-size: 1rem !important; }
    </style>
""", unsafe_allow_html=True)

# ==============================
# Sidebar
# ==============================
st.sidebar.header("⚙️ Controls")
threshold = st.sidebar.slider("Hate probability threshold", 0.05, 0.95, 0.5)
show_contrib = st.sidebar.toggle("Show token contributions", False)
st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.write(
    "This demo uses your trained model + TF-IDF vectorizer. "
    "Predictions may be imperfect, especially for slang variants."
)

# ==============================
# Header
# ==============================
st.markdown("<div class='title'>📢 Detecting Anti-India Campaigns</div>", unsafe_allow_html=True)
st.write("Type or paste a message to check if it likely contains hate / abusive content.")

# ==============================
# Text Cleaning Function
# ==============================
def clean_text(text):
    text = str(text).lower()
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# ==============================
# Tabs: Single & Batch
# ==============================
tab1, tab2 = st.tabs(["📝 Single Check", "📂 Batch (CSV)"])

# ==============================
# TAB 1: Single Prediction
# ==============================
with tab1:
    example_texts = [
        "You are welcome. Have a nice day!",
        "Jhoot mat bolo yaar, tumhari baaton mein nafrat hai",
        "Tumhara dimaag kharab ho gaya kya?"
    ]
    st.markdown("**Try examples:**")
    cols = st.columns(len(example_texts))
    for i, text in enumerate(example_texts):
        if cols[i].button(text, key=f"example_{i}"):
            st.session_state["user_input"] = text

    user_input = st.text_area(
        "Enter text / tweet:",
        value=st.session_state.get("user_input", ""),
        placeholder="Write or paste Hinglish/English text here..."
    )

    st.markdown("<div class='analyze-button'>", unsafe_allow_html=True)
    if st.button("Analyze", key="analyze_single"):
        if user_input.strip() == "":
            st.warning("⚠️ Please enter some text.")
        else:
            cleaned = clean_text(user_input)
            vec = vectorizer.transform([cleaned])
            prob = model.predict_proba(vec)[0][1]
            pred = 1 if prob >= threshold else 0
            label = "🚨 Hate Speech / Anti-India" if pred == 1 else "✅ Non-Hate Speech"
            st.subheader(f"Prediction: {label}")
            st.write(f"**Probability:** {prob:.2f}")
    st.markdown("</div>", unsafe_allow_html=True)

# ==============================
# TAB 2: Batch CSV Upload
# ==============================
with tab2:
    st.write("Upload a CSV file containing text data. Ensure it has a column named `text`.")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if "text" not in df.columns:
            st.error("CSV must contain a column named 'text'.")
        else:
            st.success(f"✅ Loaded {len(df)} rows from CSV.")
            df["cleaned"] = df["text"].apply(clean_text)
            X = vectorizer.transform(df["cleaned"])
            probs = model.predict_proba(X)[:, 1]
            df["probability"] = probs
            df["prediction"] = ["Hate Speech" if p >= threshold else "Non-Hate Speech" for p in probs]
            st.dataframe(df[["text", "prediction", "probability"]])
            csv_out = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Results CSV", csv_out, "hate_speech_predictions.csv", "text/csv")
