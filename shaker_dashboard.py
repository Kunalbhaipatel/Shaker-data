# Streamlit Pipeline for Shaker Screen Monitoring and Prediction (Optimized for Large Files)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import joblib
import os

st.set_page_config(page_title="Shaker Screen Health Dashboard", layout="wide")
st.title("🛠️ Shaker Screen Performance Monitoring")

# Upload section
uploaded_file = st.file_uploader("Upload your sensor dataset CSV (large files supported)", type=["csv"])

if uploaded_file:
    st.info("Loading preview of the dataset to optimize performance...")

    try:
        preview_df = pd.read_csv(uploaded_file, low_memory=False)
        total_rows = len(preview_df)
        row_limit = st.number_input("Max rows to load", min_value=1000, max_value=min(100000, total_rows), step=1000, value=min(20000, total_rows))
        df = preview_df.tail(row_limit)
        st.success(f"Loaded last {row_limit} rows from total {total_rows} rows.")
    except Exception as e:
        st.error(f"Failed to load file: {e}")
        st.stop()

    df.replace(-999.25, np.nan, inplace=True)
    try:
        if "YYYY/MM/DD" in df.columns and "HH:MM:SS" in df.columns:
            df["Datetime"] = pd.to_datetime(df["YYYY/MM/DD"].astype(str) + " " + df["HH:MM:SS"].astype(str), errors="coerce")
        else:
            df["Datetime"] = pd.to_datetime(df.iloc[:, 0].astype(str) + " " + df.iloc[:, 1].astype(str), errors="coerce")
    except Exception as e:
        st.error(f"Datetime parsing failed: {e}")
        st.stop()

    df.set_index("Datetime", inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    df = df.sort_index()

    # --- Column Filter --- #
    st.subheader("📦 Select Data Fields")
    selected_columns = st.multiselect("Choose fields to analyze", df.columns.tolist(), default=[
        'Rate Of Penetration (ft_per_hr)', 'Mud Density (lbs_per_gal)', 'SHAKER #1 (Units)',
        'Pump 1 strokes/min (SPM)', 'Pump 2 strokes/min (SPM)', 'Pump 3 strokes/min (SPM)'
    ])
    df = df[selected_columns]

    nan_threshold = st.slider("Drop columns with more than X% missing values", 0, 100, 50)
    valid_cols = df.isnull().mean() < (nan_threshold / 100)
    df = df.loc[:, valid_cols]

    # --- Feature Engineering --- #
    st.subheader("📊 Feature Engineering")
    with st.spinner("Calculating derived metrics..."):
        df["ROP"] = pd.to_numeric(df.get('Rate Of Penetration (ft_per_hr)', np.nan), errors='coerce')
        df["Mud Density"] = pd.to_numeric(df.get('Mud Density (lbs_per_gal)', np.nan), errors='coerce')
        df["Shaker #1"] = pd.to_numeric(df.get('SHAKER #1 (Units)', np.nan), errors='coerce')
        df["Shaker #2"] = pd.to_numeric(df.get('SHAKER #2 (Units)', np.nan), errors='coerce')
        df["Shaker #3"] = pd.to_numeric(df.get('SHAKER #3 (PERCENT)', np.nan), errors='coerce')
        df["Pump1"] = pd.to_numeric(df.get('Pump 1 strokes/min (SPM)', np.nan), errors='coerce')
        df["Pump2"] = pd.to_numeric(df.get('Pump 2 strokes/min (SPM)', np.nan), errors='coerce')
        df["Pump3"] = pd.to_numeric(df.get('Pump 3 strokes/min (SPM)', np.nan), errors='coerce')

        df["Solids_Load"] = df["ROP"] * df["Mud Density"]
        df["Fluid_Flow"] = df["Pump1"] + df["Pump2"] + df["Pump3"]
        df["SLI"] = df["Solids_Load"] / (df["Shaker #1"] + 1)
        df["Shaker_Eff_Trend"] = df["Shaker #1"].rolling(window=30).mean()
        df["Screen_Degradation"] = df["Shaker_Eff_Trend"].diff()

    st.success("Features calculated.")

    # --- Visualizations --- #
    st.subheader("📈 Key Performance Indicators (Downsampled)")
    downsampled = df[["Shaker #1", "Shaker #2", "Shaker #3", "SLI"]].resample("5Min").mean()
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Avg SLI", f"{df['SLI'].mean():.2f}")
    kpi2.metric("Max Solids Load", f"{df['Solids_Load'].max():.2f}")
    kpi3.metric("Shaker Output Drop", f"{df['Screen_Degradation'].min():.2f}")
    st.line_chart(downsampled.dropna())

    # --- Alert Logic --- #
    st.subheader("🚨 Screen Health Alerts")
    df["Overload"] = (df["SLI"] > 1.2) & (df["Screen_Degradation"] < -0.5)
    alert_times = df[df["Overload"]].index

    if not alert_times.empty:
        st.error(f"⚠️ Screen Overload Detected at {alert_times[-1]}")
    else:
        st.success("✅ No overload detected in current window.")

    # --- Machine Learning Training --- #
    st.subheader("🤖 Train Predictive Model")
    with st.expander("Train classifier to predict screen change need"):
        df["Target"] = df["Overload"].astype(int)
        df_ml = df[["ROP", "Mud Density", "Fluid_Flow", "SLI", "Shaker #1", "Screen_Degradation", "Target"]].dropna()
        X = df_ml.drop("Target", axis=1)
        y = df_ml["Target"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.code(classification_report(y_test, y_pred), language="text")

        st.markdown("### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        st.markdown("### 💾 Download Trained Model")
        model_path = "trained_screen_model.pkl"
        joblib.dump(model, model_path)
        with open(model_path, "rb") as f:
            st.download_button(label="Download Model (.pkl)", data=f, file_name=model_path)

    st.caption("This dashboard is optimized for large files with row limits, field filters, and downsampling.")
else:
    st.warning("Please upload a CSV file to begin.")
