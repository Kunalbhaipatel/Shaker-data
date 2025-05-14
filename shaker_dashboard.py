# Streamlit Pipeline for Shaker Screen Monitoring (Optimized for Large Files)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

    # Customized Insight Section
    st.subheader("📋 Operational Recommendations Log")
    with st.expander("Real-time Suggestions per Timestamp"):
        comments = []
        for timestamp, row in df.iterrows():
            msg = []
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
