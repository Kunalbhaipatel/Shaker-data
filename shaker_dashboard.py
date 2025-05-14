# Streamlit Pipeline for Shaker Screen Monitoring (Optimized for Large Files)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import seaborn as sns
import joblib
import time

st.set_page_config(page_title="Shaker Screen Monitor", layout="wide")
st.title("🛠️ Shaker Performance Monitor")

uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])

if uploaded_file:
    with st.spinner("Preparing dataset..."):
        row_limit = st.number_input("Rows to load (latest)", min_value=1000, max_value=100000, value=20000, step=1000)
        required_cols = [
            "YYYY/MM/DD", "HH:MM:SS", "Rate Of Penetration (ft_per_hr)", "Mud Density (lbs_per_gal)",
            "SHAKER #1 (Units)", "Pump 1 strokes/min (SPM)", "Pump 2 strokes/min (SPM)", "Pump 3 strokes/min (SPM)"
        ]

        try:
            total_rows = sum(1 for _ in uploaded_file) - 1
            uploaded_file.seek(0)
            skip = max(1, total_rows - row_limit)
            df = pd.read_csv(uploaded_file, usecols=required_cols, skiprows=range(1, skip+1), low_memory=False)
            df.replace(-999.25, np.nan, inplace=True)
            df["Datetime"] = pd.to_datetime(df["YYYY/MM/DD"].astype(str) + " " + df["HH:MM:SS"].astype(str), errors="coerce")
            df.set_index("Datetime", inplace=True)
            df = df.sort_index()
        except Exception as e:
            st.error(f"Error loading file: {e}")
            st.stop()

    df["ROP"] = pd.to_numeric(df["Rate Of Penetration (ft_per_hr)"], errors='coerce')
    df["Mud Density"] = pd.to_numeric(df["Mud Density (lbs_per_gal)"], errors='coerce')
    df["Shaker"] = pd.to_numeric(df["SHAKER #1 (Units)"], errors='coerce')
    df["Pumps"] = pd.to_numeric(df["Pump 1 strokes/min (SPM)"], errors='coerce') + \
                  pd.to_numeric(df["Pump 2 strokes/min (SPM)"], errors='coerce') + \
                  pd.to_numeric(df["Pump 3 strokes/min (SPM)"], errors='coerce')

    df["Solids_Load"] = df["ROP"] * df["Mud Density"]
    df["SLI"] = df["Solids_Load"] / (df["Shaker"] + 1)

    st.line_chart(df[["Shaker", "SLI"]].dropna())

    df["Overload"] = (df["SLI"] > 1.2)
    if df["Overload"].any():
        st.error("⚠️ Overload Detected")

    # Add simple generic feedback from pattern analysis
    st.subheader("📝 ML-Based Operational Insight")
    with st.expander("View Auto Comments"):
        overload_pct = df["Overload"].mean() * 100
        sli_avg = df["SLI"].mean()
        if overload_pct > 5:
            st.warning(f"⚠️ High overload frequency detected: {overload_pct:.1f}% of samples")
        else:
            st.success(f"✅ Overload frequency is under control: {overload_pct:.1f}%")

        if sli_avg > 1.0:
            st.info(f"ℹ️ Average SLI suggests moderately high solid loading: SLI ≈ {sli_avg:.2f}")
        else:
            st.info(f"✅ SLI within safe operational range: SLI ≈ {sli_avg:.2f}")

    st.subheader("Train ML Model")
    with st.expander("Train model"):
        df["Target"] = df["Overload"].astype(int)
        model_df = df[["ROP", "Mud Density", "Pumps", "SLI", "Shaker", "Target"]].dropna()
        X = model_df.drop("Target", axis=1)
        y = model_df["Target"]

        if len(model_df) < 100:
            st.warning("Not enough data points to train model.")
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

            model = Pipeline([
                ("scale", StandardScaler()),
                ("rf", RandomForestClassifier(n_estimators=100, random_state=0))
            ])
            with st.spinner("Training model..."):
                time.sleep(1)
                model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.code(classification_report(y_test, y_pred), language="text")
            joblib.dump(model, "shaker_model.pkl")
            with open("shaker_model.pkl", "rb") as f:
                st.download_button("Download Model", f, "shaker_model.pkl")
else:
    st.info("Upload a CSV file to begin.")
