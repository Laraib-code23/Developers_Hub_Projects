# app_simplified.py

import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image
from io import BytesIO

# Page config
st.set_page_config(page_title="Heart Disease Predictor", layout="wide")

# ---------------- UI Styling ----------------
st.markdown("""
<style>
h1 { color: #d63447; }
h3 { color: #0b2545; }
.stButton>button { border-radius: 8px; padding: 8px 14px; }
</style>
""", unsafe_allow_html=True)

# ---------------- Load Model & Scaler ----------------
@st.cache_resource
def load_model(path='heart_disease_model.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_scaler(path='scaler.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)

try:
    model = load_model()
    scaler = load_scaler()
except FileNotFoundError:
    st.error("Model or scaler not found. Ensure 'heart_disease_model.pkl' and 'scaler.pkl' are in the app folder.")
    st.stop()

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Heart Disease Predictor")
    st.write("Polished UI • Risk Probability Chart Only")
    st.markdown("---")
    st.write("Author: Laraib Huma")
    st.write(datetime.now().strftime("%Y-%m-%d %H:%M"))

# ---------------- Main Layout ----------------
st.markdown('<h1>Heart Disease Prediction App ❤️</h1>', unsafe_allow_html=True)
st.write("Provide patient details and click **Predict** to see the risk probability.")

left_col, right_col = st.columns([2, 1])

with left_col:
    st.markdown('<h3>Patient Details</h3>', unsafe_allow_html=True)
    age = st.number_input("Age", 18, 100, 50)
    sex_label = st.selectbox("Sex", ["Female", "Male"])
    sex = 0 if sex_label == "Female" else 1

    chest_pain_map = {
        "Typical Angina (1)": 1,
        "Atypical Angina (2)": 2,
        "Non-anginal Pain (3)": 3,
        "Asymptomatic (4)": 4
    }
    chest_pain_label = st.selectbox("Chest Pain Type", list(chest_pain_map.keys()))
    chest_pain = chest_pain_map[chest_pain_label]

    bp = st.number_input("Resting Blood Pressure (mm Hg)", 60, 250, 130)
    cholesterol = st.number_input("Serum Cholesterol (mg/dl)", 100, 700, 250)

    fbs_label = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", ["No", "Yes"])
    fbs = 0 if fbs_label == "No" else 1

    ekg_map = {
        "Normal (0)": 0,
        "ST-T Wave Abnormality (1)": 1,
        "Left Ventricular Hypertrophy (2)": 2
    }
    ekg_label = st.selectbox("Resting ECG Results", list(ekg_map.keys()))
    ekg = ekg_map[ekg_label]

    max_hr = st.number_input("Max Heart Rate Achieved", 40, 250, 150)
    exercise_angina_label = st.selectbox("Exercise Induced Angina?", ["No", "Yes"])
    exercise_angina = 0 if exercise_angina_label == "No" else 1

    st_depression = st.number_input("ST Depression induced by exercise", 0.0, 12.0, 1.0, step=0.1)

    slope_map = {"Upsloping (1)": 1, "Flat (2)": 2, "Downsloping (3)": 3}
    slope_label = st.selectbox("Slope of ST Segment", list(slope_map.keys()))
    slope_st = slope_map[slope_label]

    num_vessels = st.number_input("Number of Major Vessels (0-3)", 0, 3, 0)

    thallium_map = {"Normal (3)": 3, "Fixed Defect (6)": 6, "Reversible Defect (7)": 7}
    thallium_label = st.selectbox("Thallium Heart Scan", list(thallium_map.keys()))
    thallium = thallium_map[thallium_label]

with right_col:
    st.markdown('<h3>Quick Summary</h3>', unsafe_allow_html=True)
    st.write(f"Age: **{age}**")
    st.write(f"Sex: **{sex_label}**")
    st.write(f"Chest Pain: **{chest_pain_label}**")
    st.write(f"BP: **{bp}**, Cholesterol: **{cholesterol}**")
    st.write(f"FBS>120: **{fbs_label}**, Exercise Angina: **{exercise_angina_label}**")

# ---------------- Prepare input ----------------
input_features = np.array([[age, sex, chest_pain, bp, cholesterol, fbs, ekg,
                            max_hr, exercise_angina, st_depression, slope_st,
                            num_vessels, thallium]])
input_scaled = scaler.transform(input_features)

# ---------------- Prediction Button ----------------
if st.button("Predict Heart Disease", type="primary"):
    with st.spinner("Predicting..."):
        prediction = model.predict(input_scaled)[0]
        try:
            prob = float(model.predict_proba(input_scaled)[0][1])
        except Exception:
            from scipy.special import expit
            try:
                score = model.decision_function(input_scaled)[0]
                prob = float(expit(score))
            except Exception:
                prob = 0.5

    st.markdown("---")
    if prediction == 1:
        st.error("Prediction: **Heart Disease Present**")
    else:
        st.success("Prediction: **No Heart Disease**")
    st.write(f"**Risk Probability:** {prob*100:.2f}%")

    # ---------------- Probability Chart ----------------
    fig, ax = plt.subplots(figsize=(6, 1.2))
    ax.barh([0], [prob], height=0.6, color="#e74c3c")
    ax.barh([0], [1-prob], left=[prob], height=0.6, color="#2ecc71")
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Probability")
    ax.set_title("Heart Disease Risk Probability", color="#0b2545")
    ax.grid(axis='x', linestyle='--', alpha=0.35)
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)
