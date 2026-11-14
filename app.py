import streamlit as st
import tensorflow as tf
import numpy as np
import json
import sqlite3
from PIL import Image
import pandas as pd

# Page Configuration
st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿", layout="wide")

# Custom CSS for Modern UI Styling
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    /* Stylish Gradient Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #2e7d32 0%, #4caf50 100%);
        color: white;
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        border: none;
        box-shadow: 0 4px 10px rgba(46, 125, 50, 0.3);
        transition: 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 100%);
        box-shadow: 0 6px 15px rgba(46, 125, 50, 0.5);
    }
    /* Modern Metric Cards */
    div.stMetric {
        background-color: #161b22;
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        border-left: 5px solid #4caf50;
        border-top: 1px solid #30363d;
        border-right: 1px solid #30363d;
        border-bottom: 1px solid #30363d;
    }
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #111418;
        border-right: 1px solid #21262d;
    }
    </style>
""", unsafe_allow_html=True)

# Constants
MODEL_PATH = "models/best_model.keras"
CLASS_NAMES_PATH = "models/class_names.json"
DB_PATH = "history.db"

# Load Class Names
@st.cache_data
def load_class_names():
    with open(CLASS_NAMES_PATH, "r") as f:
        return json.load(f)

class_names = load_class_names()

# Load Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# Initialize SQLite Database
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            predicted_class TEXT,
            confidence REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def format_class_name(class_str):
    """ Converts 'Tomato___Late_blight' to Crop: Tomato, Disease: Late blight """
    if "___" in class_str:
        parts = class_str.split("___")
        crop = parts[0].replace("_", " ")
        disease = parts[1].replace("_", " ")
        return crop, disease
    return "Plant", class_str.replace("_", " ")

# Sidebar Navigation
st.sidebar.title("🌿 Navigation")
page = st.sidebar.radio("Go to", ["Predict", "History", "Statistics Dashboard"])
st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Upload a clear leaf photo for accurate disease detection.")

# --- PAGE 1: PREDICT ---
if page == "Predict":
    st.title("🌱 Plant Disease Detection System")
    st.markdown("Upload a leaf image below to instantly detect diseases using our advanced deep learning model.")
    st.markdown("")

    uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        col_img, col_space = st.columns([1, 2])
        with col_img:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Leaf", width=350)

        if st.button("🔍 Analyze Plant Health", type="primary"):
            with st.spinner("Analyzing image features... Please wait."):
                # Preprocessing
                img_resized = image.resize((224, 224))
                img_array = np.array(img_resized, dtype=np.float32)
                img_array = np.expand_dims(img_array, axis=0)

                # Prediction
                preds = model.predict(img_array)[0]
                pred_idx = np.argmax(preds)
                predicted_class = class_names[pred_idx]
                confidence = float(preds[pred_idx] * 100)

                crop, disease = format_class_name(predicted_class)
                is_healthy = "healthy" in disease.lower()

                # Save to Database
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO predictions (filename, predicted_class, confidence) VALUES (?, ?, ?)",
                    (uploaded_file.name, predicted_class, confidence)
                )
                conn.commit()
                conn.close()

            st.markdown("---")
            st.subheader("📋 Diagnostic Results")
            
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.metric(label="Detected Crop", value=crop)
                st.metric(label="Condition / Disease", value=disease)
            with res_col2:
                st.metric(label="Confidence Score", value=f"{confidence:.2f}%")
                if is_healthy:
                    st.success("Status: Healthy Plant 🟢")
                else:
                    st.error("Status: Disease Detected 🔴")

            # Low Confidence Warning (< 60%)
            if confidence < 60.0:
                st.warning("⚠️ **Low Confidence Warning (< 60%)**: The image might be unclear or out of distribution. Consider retaking under better lighting.")

            # Top-3 Predictions
            st.markdown("### 📊 Top-3 Probable Conditions")
            top3_indices = np.argsort(preds)[-3:][::-1]
            top3_data = []
            for idx in top3_indices:
                c_name, d_name = format_class_name(class_names[idx])
                top3_data.append({
                    "Crop & Disease": f"{c_name} — {d_name}",
                    "Confidence (%)": round(float(preds[idx] * 100), 2)
                })
            st.table(pd.DataFrame(top3_data))

# --- PAGE 2: HISTORY ---
elif page == "History":
    st.title("📜 Prediction History")
    st.markdown("Review all past scans and diagnostics stored securely in the local SQLite database.")

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT id, filename, predicted_class, confidence, timestamp FROM predictions ORDER BY id DESC", conn)
    conn.close()

    if not df.empty:
        df['Crop'] = df['predicted_class'].apply(lambda x: format_class_name(x)[0])
        df['Disease'] = df['predicted_class'].apply(lambda x: format_class_name(x)[1])
        df['Confidence (%)'] = df['confidence'].round(2)
        
        display_df = df[['id', 'filename', 'Crop', 'Disease', 'Confidence (%)', 'timestamp']]
        st.dataframe(display_df, use_container_width=True)
    else:
        st.info("No prediction history found yet. Head over to the **Predict** page to analyze your first leaf image!")

# --- PAGE 3: STATISTICS DASHBOARD ---
elif page == "Statistics Dashboard":
    st.title("📊 Statistics Dashboard")
    st.markdown("Overview of overall model usage and disease distribution trends.")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*), AVG(confidence) FROM predictions")
    total_count, avg_conf = cursor.fetchone()
    total_count = total_count or 0
    avg_conf = round(avg_conf, 2) if avg_conf else 0.0

    conn.close()

    stat_col1, stat_col2 = st.columns(2)
    with stat_col1:
        st.metric(label="Total Scans Performed", value=total_count)
    with stat_col2:
        st.metric(label="Average Confidence Rate", value=f"{avg_conf}%")

    st.markdown("---")
    st.subheader("📈 Disease Frequency Distribution")
    
    conn = sqlite3.connect(DB_PATH)
    chart_df = pd.read_sql("SELECT predicted_class, COUNT(*) as count FROM predictions GROUP BY predicted_class", conn)
    conn.close()

    if not chart_df.empty:
        chart_df['Label'] = chart_df['predicted_class'].apply(lambda x: f"{format_class_name(x)[0]} - {format_class_name(x)[1]}")
        chart_data = chart_df.set_index('Label')['count']
        st.bar_chart(chart_data)
    else:
        st.info("Not enough prediction data available yet to render charts.")