# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.title("Stock Up/Down Prediction (LightGBM)")

# Load model (do this once)
@st.cache_resource
def load_model(path='stock_model_v1.joblib'):
    obj = joblib.load(path)
    return obj['model'], obj['preprocessor'], obj['features']

model, preprocessor, FEATURES = load_model()

st.markdown("Upload a CSV with the same feature columns used for training (or drop to try single-row mode).")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    st.write("Raw input:", df.head())

    # ensure columns exist & order them
    for f in FEATURES:
        if f not in df.columns:
            df[f] = 0   # or np.nan if you prefer impute
    X = df[FEATURES].copy()

    # preprocess and predict
    X_proc = preprocessor.transform(X)
    preds = model.predict(X_proc)
    probs = model.predict_proba(X_proc)[:,1]
    df['pred'] = preds
    df['prob_up'] = probs
    st.write("Predictions:", df[['pred','prob_up'] + FEATURES].head())

# Single-row manual input example (optional)
if st.checkbox("Single-row input"):
    sample = {f: 0.0 for f in FEATURES}
    for f in FEATURES:
        sample[f] = st.number_input(f, value=float(sample[f]))
    if st.button("Predict row"):
        import numpy as np
        row = pd.DataFrame([sample])[FEATURES]
        row_proc = preprocessor.transform(row)
        p = model.predict(row_proc)[0]
        prob = model.predict_proba(row_proc)[0,1]
        st.write(f"Pred: {int(p)}, Prob(up): {prob:.3f}")
