# app.py

import streamlit as st
import numpy as np
import joblib

# ── Load Model & Scaler ────────────────────────────────────
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# ── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="Sales Prediction App",
    page_icon="📈",
    layout="centered"
)

# ── Header ─────────────────────────────────────────────────
st.title("📈 Advertising Sales Predictor")
st.markdown("Enter advertising budget details to predict sales.")
st.divider()

# ── Input Section ──────────────────────────────────────────
st.subheader("📋 Advertising Budget")

col1, col2 = st.columns(2)

with col1:
    tv = st.slider(
        "TV Advertising Budget ($)",
        0.0, 300.0, 150.0, 1.0
    )

    radio = st.slider(
        "Radio Advertising Budget ($)",
        0.0, 50.0, 25.0, 1.0
    )

with col2:
    newspaper = st.slider(
        "Newspaper Advertising Budget ($)",
        0.0, 120.0, 30.0, 1.0
    )

st.divider()

# ── Prediction ─────────────────────────────────────────────
if st.button("🔍 Predict Sales", use_container_width=True):

    features = np.array([[
        tv,
        radio,
        newspaper
    ]])

    # Scale Features
    features_scaled = scaler.transform(features)

    # Predict
    prediction = model.predict(features_scaled)[0]

    # ── Result ─────────────────────────────────────────────
    st.success(f"📊 Estimated Sales: **{prediction:.2f} units**")

    # ── Input Summary ──────────────────────────────────────
    st.markdown("#### 📋 Advertising Summary")

    st.dataframe({
        "Platform": ["TV", "Radio", "Newspaper"],
        "Budget ($)": [tv, radio, newspaper]
    }, use_container_width=True)

# ── Footer ─────────────────────────────────────────────────
st.markdown("---")
st.caption("Built with Scikit-learn + Streamlit · Advertising Sales Dataset")