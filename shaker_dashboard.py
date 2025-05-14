# Streamlit Pipeline for Shaker Screen Monitoring
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

st.set_page_config(page_title="Shaker Screen Monitor", layout="wide")
st.title("🛠️ Shaker Performance Monitor")

uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, low_memory=False)
        df.replace(-999.25, np.nan, inplace=True)
        if "YYYY/MM/DD" in df.columns and "HH:MM:SS" in df.columns:
            df["Datetime"] = pd.to_datetime(df["YYYY/MM/DD"].astype(str) + " " + df["HH:MM:SS"].astype(str), errors="coerce")
        else:
            df["Datetime"] = pd.to_datetime(df.iloc[:, 0].astype(str) + " " + df.iloc[:, 1].astype(str), errors="coerce")
        df.set_index("Datetime", inplace=True)
        df = df.sort_index()
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

    df["ROP"] = pd.to_numeric(df.get('Rate Of Penetration (ft_per_hr)', np.nan), errors='coerce')
    df["Mud Density"] = pd.to_numeric(df.get('Mud Density (lbs_per_gal)', np.nan), errors='coerce')
    df["Shaker"] = pd.to_numeric(df.get('SHAKER #1 (Units)', np.nan), errors='coerce')
    df["Pumps"] = pd.to_numeric(df.get('Pump 1 strokes/min (SPM)', np.nan), errors='coerce') + \
                  pd.to_numeric(df.get('Pump 2 strokes/min (SPM)', np.nan), errors='coerce') + \
                  pd.to_numeric(df.get('Pump 3 strokes/min (SPM)', np.nan), errors='coerce')

    df["Solids_Load"] = df["ROP"] * df["Mud Density"]
    df["SLI"] = df["Solids_Load"] / (df["Shaker"] + 1)

    st.line_chart(df[["Shaker", "SLI"]].dropna())

    df["Overload"] = (df["SLI"] > 1.2)
    if df["Overload"].any():
        st.error("⚠️ Overload Detected")

    st.subheader("Train ML Model")
    with st.expander("Train model"):
        df["Target"] = df["Overload"].astype(int)
        model_df = df[["ROP", "Mud Density", "Pumps", "SLI", "Shaker", "Target"]].dropna()
        X = model_df.drop("Target", axis=1)
        y = model_df["Target"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

        model = Pipeline([
            ("scale", StandardScaler()),
            ("rf", RandomForestClassifier(n_estimators=100, random_state=0))
        ])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.code(classification_report(y_test, y_pred), language="text")
        joblib.dump(model, "shaker_model.pkl")
        with open("shaker_model.pkl", "rb") as f:
            st.download_button("Download Model", f, "shaker_model.pkl")
else:
    st.info("Upload a CSV file to begin.")
