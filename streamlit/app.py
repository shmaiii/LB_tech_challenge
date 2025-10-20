import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from ai_utils import summarize_recent_news

st.set_page_config(page_title=" Canada Interprovincial Migration (Quarterly)", layout="centered")

st.title(" Canada Interprovincial Migration Predictor (Quarterly)")
st.write("""
Predict quarterly Canadian interprovincial migration using Statistics Canada Table 17-10-0020-01 data.
""")

# --- Load & clean data ---
@st.cache_data
def load_quarterly_data(path="data/canada_migration_quarterly.csv"):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    # First column is "Interprovincial migration"
    df = df.set_index("Interprovincial migration")
    df = df.replace(",", "", regex=True)
    df = df.apply(pd.to_numeric, errors="coerce")

    # Transpose to make quarters rows
    df = df.T.reset_index().rename(columns={"index": "Quarter"})
    df["In_migrants"] = df.get("In-migrants", np.nan)
    df["Out_migrants"] = df.get("Out-migrants", np.nan)
    df["Net_migration"] = df["In_migrants"] - df["Out_migrants"]

    # Parse quarter strings like "Q1 2000" → 2000.0, 2000.25, etc.
    def parse_quarter(q):
        try:
            qnum, year = q.strip().split(" ")
            qn = int(qnum[-1])
            yr = int(year)
            return yr + (qn - 1) / 4
        except Exception:
            return np.nan

    df["Time"] = df["Quarter"].apply(parse_quarter)
    df = df.dropna(subset=["Time"])

    return df[["Quarter", "Time", "In_migrants", "Out_migrants", "Net_migration"]]

df = load_quarterly_data()

# Sidebar: choose variable & forecast horizon
metric = st.sidebar.selectbox("Metric to predict", ["In_migrants", "Out_migrants", "Net_migration"])
n_future = st.sidebar.slider("Quarters to forecast", 4, 16, 8)

if st.button("Run Prediction"):
    # Create lag features so the model can learn from previous quarters
    df["lag1"] = df[metric].shift(1)
    df["lag4"] = df[metric].shift(4)  # same quarter last year
    df = df.dropna()

    X = df[["Time", "lag1", "lag4"]]
    y = df[metric]

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)

    # Forecast next n_future quarters iteratively
    last_rows = df.tail(4).copy()
    preds = []

    for i in range(n_future):
        next_time = last_rows["Time"].iloc[-1] + 0.25
        lag1 = last_rows[metric].iloc[-1]
        lag4 = last_rows[metric].iloc[-4] if len(last_rows) >= 4 else lag1
        next_pred = model.predict([[next_time, lag1, lag4]])[0]
        preds.append(next_pred)

        # Append predicted value for next iteration
        last_rows = pd.concat([
            last_rows,
            pd.DataFrame({
                "Time": [next_time],
                metric: [next_pred],
                "lag1": [lag1],
                "lag4": [lag4]
            })
        ], ignore_index=True)

    # Save results so they persist after reruns
    st.session_state["metric"] = metric
    st.session_state["y"] = y
    st.session_state["preds"] = preds
    st.session_state["n_future"] = n_future

    # Optional: clear old AI summaries when new prediction runs
    for key in ["forecast_summary", "news_summary"]:
        st.session_state.pop(key, None)

    # --- Plot results ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["Time"], y, "o-", label="Historical")
    future_times = np.arange(df["Time"].max() + 0.25,
                             df["Time"].max() + 0.25 * (n_future + 1),
                             0.25)
    ax.plot(future_times, preds, "r--o", label="Random Forest Forecast")
    ax.set_xlabel("Time (Year + Quarter fraction)")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Quarterly {metric.replace('_', ' ').title()} Forecast (Random Forest)")
    ax.legend()
    st.pyplot(fig)

    # Display predictions
    pred_df = pd.DataFrame({
        "Future Quarter": np.arange(1, n_future + 1),
        "Predicted": [int(p) for p in preds]
    })
    st.write(f"### Random Forest Forecast for {metric.replace('_', ' ').title()}")
    st.dataframe(pred_df)


# --- If a forecast already exists, show the AI sections ---
if "preds" in st.session_state:
    st.divider()

    if st.button(" Get AI-Summarized Immigration News"):
        with st.spinner("Fetching latest news from Groq..."):
            news_summary = summarize_recent_news("Canada immigration")
            st.session_state["news_summary"] = news_summary

    if "news_summary" in st.session_state:
        st.markdown("### AI Summary of Immigration News")
        st.write(st.session_state["news_summary"])