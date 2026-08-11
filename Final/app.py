import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from models.GoldPricePredictor import GoldPricePredictor
import os
import kagglehub
# Configuration --------------------------

USD_TO_SAR = 3.75
TROY_OUNCE_GRAMS = 31.1034768
PREDICTION_DATE = "2026-07-01"

# Page Configuration --------------------------------------
st.set_page_config(
    page_title="Gold Price Forecast",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS -----------------------------------------------------
st.markdown(
    """
    <style>
    .main {
        padding-top: 2rem;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
    }

    .title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        text-align: center;
        color: #888888;
        margin-bottom: 35px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 600;
        margin-top: 35px;
        margin-bottom: 18px;
    }

    .metric-card {
        padding: 25px;
        border-radius: 15px;
        border: 1px solid rgba(128,128,128,0.25);
        text-align: center;
        min-height: 150px;
    }

    .metric-title {
        font-size: 16px;
        color: #888888;
        margin-bottom: 12px;
    }

    .metric-value {
        font-size: 30px;
        font-weight: 700;
    }

    .metric-date {
        font-size: 14px;
        color: #888888;
        margin-top: 8px;
    }

    .current-price {
        padding: 30px;
        border-radius: 15px;
        border: 1px solid rgba(128,128,128,0.25);
        text-align: center;
        margin-bottom: 25px;
    }

    .current-label {
        font-size: 17px;
        color: #888888;
    }

    .current-value {
        font-size: 42px;
        font-weight: 700;
        margin-top: 8px;
    }

    .current-unit {
        font-size: 14px;
        color: #888888;
        margin-top: 5px;
    }

    .prediction-card {
        padding: 25px;
        border-radius: 15px;
        border: 1px solid rgba(128,128,128,0.25);
        text-align: center;
        min-height: 140px;
    }

    .karat {
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 10px;
    }

    .prediction-price {
        font-size: 28px;
        font-weight: 700;
    }

    .signal-card {
        padding: 25px;
        border-radius: 15px;
        border: 1px solid rgba(128,128,128,0.25);
        text-align: center;
        margin-top: 20px;
    }

    .signal-title {
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .signal-text {
        font-size: 18px;
    }

    .footer {
        text-align: center;
        color: #888888;
        margin-top: 60px;
        padding: 20px;
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# Header -----------------------------------------------------------------
st.markdown('<div class="title">Gold Price Forecast</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">'
    'AI-powered gold price prediction based on historical market data'
    '</div>',
    unsafe_allow_html=True
)

# Forecast Selection -----------------------------------------------------------------

col1, col2, col3 = st.columns([1, 1.2, 1])
with col2:
    period = st.selectbox(
        "Forecast Period",
        options=[
            "7 Days",
            "30 Days",
            "90 Days"
        ],
        index=1
    )

period_map = {
    "7 Days": 7,
    "30 Days": 30,
    "90 Days": 90
}
days = period_map[period]

# Load Predictor -----------------------------------------------------------------
@st.cache_resource
def load_predictor():
    predictor = GoldPricePredictor()
    predictor.load_models(
        recursive_path="artifacts/recursive_gold_model.keras",
        direct_path="artifacts/direct_gold_model.keras"
    )
    return predictor

# Load Data -----------------------------------------------------------------
@st.cache_data
def load_data():
    path = kagglehub.dataset_download("hayaalwizrah1/gold-price-prediction-dataset-20002026/versions/2")
    df = pd.read_csv(os.path.join(path, "gold_data (1).csv"))

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    df = df.sort_index()
    return df

# Load Scaler -----------------------------------------------------------------
@st.cache_resource
def load_scaler():
    SCALER_PATH = "artifacts/scaler_X.pkl"
    scaler_X = joblib.load(SCALER_PATH)
    return scaler_X

# Prediction -----------------------------------------------------------------
if st.button("Generate Forecast", use_container_width=True):
    try:
        with st.spinner("Generating forecast..."):
            predictor = load_predictor()
            df = load_data()
            scaler_X = load_scaler()

            prediction_date = pd.to_datetime(PREDICTION_DATE)
            if prediction_date not in df.index:
                st.error(f"Prediction date {PREDICTION_DATE} was not found in the dataset.")
                st.stop()

            result = predictor.predict_gold(
                date=PREDICTION_DATE,
                df=df,
                scaler_X=scaler_X,
                days=days
            )

        st.session_state["result"] = result
        st.success("Forecast generated successfully.")

    except Exception as e:
        import traceback
        st.error(
        f"Error while generating forecast: {str(e)}"
        )

        st.code(
            traceback.format_exc(),
            language="text"
        )

        st.stop()


# Display Results -----------------------------------------------------------------

if "result" in st.session_state:
    result = st.session_state["result"]
    recursive_df = result["recursive"]
    direct = result["direct"]

    if recursive_df is None or recursive_df.empty:
        st.error("No recursive forecast results were returned.")
        st.stop()

    # Recursive Min / Max
    recursive_df = result["recursive"]
    direct = result["direct"]

    if recursive_df is None or recursive_df.empty:
        st.error("No recursive forecast results were returned.")
        st.stop()

    min_row = recursive_df.loc[recursive_df["Predicted_Close"].idxmin()]
    max_row = recursive_df.loc[recursive_df["Predicted_Close"].idxmax()]

    lowest_24k_usd =  min_row["Predicted_Close"] / TROY_OUNCE_GRAMS
    highest_24k_usd  = max_row["Predicted_Close"] / TROY_OUNCE_GRAMS

    # 21K
    lowest_21k_usd = lowest_24k_usd * 21 / 24
    highest_21k_usd = highest_24k_usd * 21 / 24

    # 18K
    lowest_18k_usd = lowest_24k_usd * 18 / 24
    highest_18k_usd = highest_24k_usd * 18 / 24

    # Convert USD -> SAR
    lowest_24k = lowest_24k_usd * USD_TO_SAR
    highest_24k = highest_24k_usd * USD_TO_SAR

    lowest_21k = lowest_21k_usd * USD_TO_SAR
    highest_21k = highest_21k_usd * USD_TO_SAR

    lowest_18k = lowest_18k_usd * USD_TO_SAR
    highest_18k = highest_18k_usd * USD_TO_SAR

    lowest_date = pd.to_datetime(min_row["Date"]).date()
    highest_date = pd.to_datetime(max_row["Date"]).date()

    # Direct Result
    current_price_usd = direct["current_24k"]
    predicted_24k_usd = direct["karat_forecast"]["24K"]
    predicted_21k_usd = direct["karat_forecast"]["21K"]
    predicted_18k_usd = direct["karat_forecast"]["18K"]
    change = direct["pct_change"]
    signal = direct["signal"]

    current_price = current_price_usd * USD_TO_SAR

    # Recursive Forecast
    st.markdown(
        f'<div class="section-title">'
        f'{days}-Day Gold Price Forecast'
        f'</div>',
        unsafe_allow_html=True
    )
    c1, c2, c3 = st.columns(3)

    # 24K
    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">24K Gold</div>
                <div class="metric-value">SAR {lowest_24k:,.2f}</div>
                <div class="metric-date">Lowest · {lowest_date}</div>
                <br>
                <div class="metric-value">SAR {highest_24k:,.2f}</div>
                <div class="metric-date">Highest · {highest_date}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # 21K
    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">21K Gold</div>
                <div class="metric-value">SAR {lowest_21k:,.2f}</div>
                <div class="metric-date">Lowest · {lowest_date}</div>
                <br>
                <div class="metric-value">SAR {highest_21k:,.2f}</div>
                <div class="metric-date">Highest · {highest_date}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # 18K
    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">18K Gold</div>
                <div class="metric-value">SAR {lowest_18k:,.2f}</div>
                <div class="metric-date">Lowest · {lowest_date}</div>
                <br>
                <div class="metric-value">SAR {highest_18k:,.2f}</div>
                <div class="metric-date">Highest · {highest_date}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Forecast Chart
    st.markdown(
        '<div class="section-title">Forecast Trend</div>',
        unsafe_allow_html=True
    )
    chart_df = recursive_df.copy()
    chart_df["Date"] = pd.to_datetime(
        chart_df["Date"]
    )

    chart_df["Predicted_24K_SAR"] = chart_df["Predicted_Close"] / TROY_OUNCE_GRAMS * USD_TO_SAR
    fig, ax = plt.subplots(
        figsize=(12, 4)
    )
    ax.plot(
        chart_df["Date"],
        chart_df["Predicted_24K_SAR"],
        linewidth=2
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("SAR / gram")
    ax.set_title(f"{days}-Day Recursive Gold Forecast")
    ax.grid(alpha=0.2)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Forecast Table
    with st.expander("View Forecast Data"):
        display_df = chart_df[[
            "Date",
            "Predicted_24K_SAR"
        ]].copy()

        display_df["Predicted_24K_SAR"] = display_df["Predicted_24K_SAR"].round(2)
        display_df = display_df.rename(
            columns={
                "Date": "Date",
                "Predicted_24K_SAR": "Predicted 24K (SAR/g)"
            }
        )

        st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Direct Forecast
    # Expected Change
    st.markdown('<div class="section-title">''Market Signal''</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="signal-card">
            <div class="signal-title">Expected Change: {change:+.2f}%</div>
            <div class="signal-text">{signal}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Footer
st.markdown("""<div class="footer">Gold Price Forecasting System · Prices displayed in SAR per gram</div>""", unsafe_allow_html=True)