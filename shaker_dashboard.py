# Streamlit Pipeline for Shaker Screen Monitoring (Optimized for Large Files)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

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
    df["Overload"] = (df["SLI"] > 1.2)

    st.subheader("📊 Technical Visualizations")
    st.markdown("Use these visuals to understand performance and stress trends.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**1. Shaker Output vs Time**")
        import plotly.express as px
        fig_shaker = px.line(df.reset_index(), x="Datetime", y="Shaker", title="Shaker Output vs Time")
        fig_shaker.add_hline(y=250, line_dash="dash", line_color="red")
        fig_shaker.add_hline(y=30, line_dash="dash", line_color="orange")
        st.plotly_chart(fig_shaker, use_container_width=True)

    with col2:
        st.markdown("**2. SLI (Solids Loading Index) vs Time**")
        fig_sli = px.line(df.reset_index(), x="Datetime", y="SLI", title="SLI vs Time")
        fig_sli.add_hline(y=1.2, line_dash="dash", line_color="red")
        fig_sli.add_hline(y=1.5, line_dash="dash", line_color="purple")
        st.plotly_chart(fig_sli, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**3. Combined Load Effects (ROP × Mud Density)**")
        fig_solids = px.line(df.reset_index(), x="Datetime", y="Solids_Load", title="Combined Load Effects (ROP × Mud Density)")
        st.plotly_chart(fig_solids, use_container_width=True)
    with col4:
        st.markdown("**4. Pumping Load vs SLI**")
        fig_scatter = px.scatter(df, x="Pumps", y="SLI", title="Pumping Load vs SLI", opacity=0.6, color_discrete_sequence=['purple'])
        st.plotly_chart(fig_scatter, use_container_width=True)

    if df["Overload"].any():
        st.error("⚠️ Overload Detected")

    st.subheader("⚙️ Train Trigger Model")
    with st.expander("Train ML model for alert scoring"):
        df["Target"] = df["Overload"].astype(int)
        model_df = df[["ROP", "Mud Density", "Pumps", "SLI", "Shaker", "Target"]].dropna()
        X = model_df.drop("Target", axis=1)
        y = model_df["Target"]

        if len(model_df) >= 100:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
            model = Pipeline([
                ("scale", StandardScaler()),
                ("rf", RandomForestClassifier(n_estimators=100, random_state=0))
            ])
            model.fit(X_train, y_train)
            probs = model.predict_proba(X)
            if probs.shape[1] > 1:
                df.loc[model_df.index, "Alert_Score"] = probs[:, 1]
            else:
                df.loc[model_df.index, "Alert_Score"] = 0.0  # fallback in case of binary single column
            joblib.dump(model, "shaker_alert_model.pkl")
            st.success("Model trained and alert scores added to data.")
            st.code(classification_report(y_test, model.predict(X_test)), language="text")

    st.subheader("📋 Operational Recommendations Log")
    with st.expander("Real-time Suggestions per Timestamp"):
        comments = []
        for timestamp, row in df.iterrows():
            msg = []
            if pd.notna(row.get("Alert_Score")) and row["Alert_Score"] > 0.7:
                msg.append(f"🚨 Screen overload risk ({row['Alert_Score']:.2f}) — consider adjusting deck angle or screen tension. Notify mud engineer.")
            if pd.notna(row["SLI"]) and row["SLI"] > 1.5:
                msg.append("High SLI — possible solids overload.")
            if pd.notna(row["Shaker"]) and row["Shaker"] > 250:
                msg.append("Shaker vibration high — check for worn screen.")
            if pd.notna(row["Pumps"]) and row["Pumps"] > 90:
                msg.append("High GPM load — monitor flow rate.")
            if pd.notna(row["Shaker"]) and row["Shaker"] < 30:
                msg.append("Shaker output low — check tensioning or blinding.")
            if len(msg) == 0:
                msg.append("✔ Normal performance.")
            comments.append((timestamp, " | ".join(msg)))

        comments_df = pd.DataFrame(comments, columns=["Timestamp", "Recommendation"])
        st.dataframe(comments_df, use_container_width=True)
else:
    st.info("Upload a CSV file to begin.")
