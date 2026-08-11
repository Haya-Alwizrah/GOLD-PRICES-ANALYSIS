import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from models.GoldPricePredictor import GoldPricePredictor
import os
import kagglehub


# ============================================================
# Configuration
# ============================================================

USD_TO_SAR = 3.75
TROY_OUNCE_GRAMS = 31.1034768
PREDICTION_DATE = "2026-07-01"


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Gold Price Forecast",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# Custom CSS - Dark Gold Theme
# ============================================================

st.markdown(
    """
    <style>

    /* =========================
       Global
    ========================= */

    .stApp {
        background:
            radial-gradient(
                circle at 50% -20%,
                rgba(212, 175, 55, 0.12),
                transparent 35%
            ),
            #0b0d0f;
        color: #f5f5f5;
    }

    .main {
        padding-top: 1rem;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* =========================
       Header
    ========================= */

    .hero {
        text-align: center;
        padding: 25px 20px 35px 20px;
    }

    .hero-icon {
        font-size: 48px;
        margin-bottom: 5px;
    }

    .hero-title {
        font-size: 46px;
        font-weight: 800;
        letter-spacing: -1px;
        margin-bottom: 5px;

        background: linear-gradient(
            90deg,
            #f5d76e,
            #d4af37,
            #fff2a8,
            #d4af37
        );

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        font-size: 17px;
        color: #9b9b9b;
        margin-top: 5px;
    }


    /* =========================
       Section Titles
    ========================= */

    .section-title {
        font-size: 24px;
        font-weight: 700;
        margin-top: 35px;
        margin-bottom: 18px;

        color: #f5d76e;

        border-left: 4px solid #d4af37;
        padding-left: 12px;
    }


    /* =========================
       Forecast Selector
    ========================= */

    .selector-container {
        background: linear-gradient(
            145deg,
            rgba(212, 175, 55, 0.08),
            rgba(255, 255, 255, 0.02)
        );

        border: 1px solid rgba(212, 175, 55, 0.22);
        border-radius: 16px;

        padding: 18px 25px;
        margin: 0 auto 30px auto;

        max-width: 500px;

        box-shadow:
            0 10px 35px rgba(0, 0, 0, 0.25);
    }


    /* =========================
       Streamlit Selectbox
    ========================= */

    div[data-baseweb="select"] > div {
        background-color: #151719 !important;
        border: 1px solid rgba(212, 175, 55, 0.35) !important;
        border-radius: 10px !important;
    }

    div[data-baseweb="select"] > div:hover {
        border-color: #d4af37 !important;
    }


    /* =========================
       Button
    ========================= */

    .stButton > button {
        width: 100%;

        background: linear-gradient(
            135deg,
            #d4af37,
            #f5d76e,
            #d4af37
        );

        color: #111 !important;

        border: none;
        border-radius: 12px;

        padding: 14px 25px;

        font-size: 17px;
        font-weight: 700;

        box-shadow:
            0 5px 20px rgba(212, 175, 55, 0.20);

        transition: all 0.25s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);

        box-shadow:
            0 8px 28px rgba(212, 175, 55, 0.35);

        background: linear-gradient(
            135deg,
            #f5d76e,
            #fff0a0,
            #d4af37
        );
    }


    /* =========================
       Metric Cards
    ========================= */

    .metric-card {
        position: relative;

        padding: 25px;

        border-radius: 18px;

        background:
            linear-gradient(
                145deg,
                rgba(212, 175, 55, 0.10),
                rgba(255, 255, 255, 0.025)
            );

        border: 1px solid rgba(212, 175, 55, 0.25);

        text-align: center;

        min-height: 185px;

        box-shadow:
            0 12px 35px rgba(0, 0, 0, 0.25);

        overflow: hidden;
    }

    .metric-card::before {
        content: "";

        position: absolute;

        top: 0;
        left: 0;
        right: 0;

        height: 3px;

        background: linear-gradient(
            90deg,
            transparent,
            #d4af37,
            #fff0a0,
            #d4af37,
            transparent
        );
    }

    .metric-title {
        font-size: 17px;
        font-weight: 600;

        color: #d4af37;

        margin-bottom: 14px;
    }

    .metric-value {
        font-size: 29px;
        font-weight: 800;

        color: #f5f5f5;
    }

    .metric-date {
        font-size: 13px;

        color: #8f9397;

        margin-top: 5px;
    }


    /* =========================
       Chart Container
    ========================= */

    .chart-container {
        padding: 10px;

        border-radius: 18px;

        background:
            linear-gradient(
                145deg,
                rgba(255, 255, 255, 0.025),
                rgba(212, 175, 55, 0.04)
            );

        border: 1px solid rgba(212, 175, 55, 0.15);
    }


    /* =========================
       Signal Card
    ========================= */

    .signal-card {
        padding: 28px;

        border-radius: 18px;

        background:
            linear-gradient(
                135deg,
                rgba(212, 175, 55, 0.13),
                rgba(255, 255, 255, 0.025)
            );

        border: 1px solid rgba(212, 175, 55, 0.30);

        text-align: center;

        margin-top: 15px;

        box-shadow:
            0 10px 35px rgba(0, 0, 0, 0.25);
    }

    .signal-title {
        font-size: 25px;

        font-weight: 800;

        color: #f5d76e;

        margin-bottom: 10px;
    }

    .signal-text {
        font-size: 18px;

        color: #d7d7d7;
    }


    /* =========================
       Expander
    ========================= */

    div[data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.025);

        border: 1px solid rgba(212, 175, 55, 0.18);

        border-radius: 14px;
    }


    /* =========================
       Dataframe
    ========================= */

    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }


    /* =========================
       Footer
    ========================= */

    .footer {
        text-align: center;

        color: #777;

        margin-top: 60px;

        padding: 25px;

        font-size: 13px;

        border-top: 1px solid rgba(212, 175, 55, 0.12);
    }

    .footer-gold {
        color: #d4af37;
        font-weight: 600;
    }


    /* =========================
       Success Message
    ========================= */

    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# Header
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-icon">🪙</div>
        <div class="hero-title">Gold Price Forecast</div>
        <div class="hero-subtitle">AI-powered gold price prediction based on historical market data</div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# Forecast Selection
# ============================================================

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

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


period_map = {
    "7 Days": 7,
    "30 Days": 30,
    "90 Days": 90
}

days = period_map[period]


# ============================================================
# Load Predictor
# ============================================================

@st.cache_resource
def load_predictor():

    predictor = GoldPricePredictor()

    predictor.load_models(
        recursive_path="artifacts/recursive_gold_model.keras",
        direct_path="artifacts/direct_gold_model.keras"
    )

    return predictor


# ============================================================
# Load Data
# ============================================================

@st.cache_data
def load_data():

    path = kagglehub.dataset_download(
        "hayaalwizrah1/gold-price-prediction-dataset-20002026/versions/2"
    )

    df = pd.read_csv(
        os.path.join(
            path,
            "gold_data (1).csv"
        )
    )

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.set_index("Date")

    df = df.sort_index()

    return df


# ============================================================
# Load Scaler
# ============================================================

@st.cache_resource
def load_scaler():

    SCALER_PATH = "artifacts/scaler_X.pkl"

    scaler_X = joblib.load(SCALER_PATH)

    return scaler_X


# ============================================================
# Prediction
# ============================================================

if st.button(
    "Generate Forecast",
    use_container_width=True
):

    try:

        with st.spinner("Generating forecast..."):

            predictor = load_predictor()

            df = load_data()

            scaler_X = load_scaler()

            prediction_date = pd.to_datetime(
                PREDICTION_DATE
            )

            if prediction_date not in df.index:

                st.error(
                    f"Prediction date {PREDICTION_DATE} "
                    "was not found in the dataset."
                )

                st.stop()

            result = predictor.predict_gold(
                date=PREDICTION_DATE,
                df=df,
                scaler_X=scaler_X,
                days=days
            )

        st.session_state["result"] = result

        st.success(
            "Forecast generated successfully."
        )

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


# ============================================================
# Display Results
# ============================================================

if "result" in st.session_state:

    result = st.session_state["result"]

    recursive_df = result["recursive"]

    direct = result["direct"]


    if recursive_df is None or recursive_df.empty:

        st.error(
            "No recursive forecast results were returned."
        )

        st.stop()


    # ========================================================
    # Recursive Min / Max
    # ========================================================

    min_row = recursive_df.loc[
        recursive_df["Predicted_Close"].idxmin()
    ]

    max_row = recursive_df.loc[
        recursive_df["Predicted_Close"].idxmax()
    ]


    lowest_24k_usd = (
        min_row["Predicted_Close"]
        / TROY_OUNCE_GRAMS
    )

    highest_24k_usd = (
        max_row["Predicted_Close"]
        / TROY_OUNCE_GRAMS
    )


    # 21K

    lowest_21k_usd = (
        lowest_24k_usd * 21 / 24
    )

    highest_21k_usd = (
        highest_24k_usd * 21 / 24
    )


    # 18K

    lowest_18k_usd = (
        lowest_24k_usd * 18 / 24
    )

    highest_18k_usd = (
        highest_24k_usd * 18 / 24
    )


    # USD -> SAR

    lowest_24k = (
        lowest_24k_usd * USD_TO_SAR
    )

    highest_24k = (
        highest_24k_usd * USD_TO_SAR
    )

    lowest_21k = (
        lowest_21k_usd * USD_TO_SAR
    )

    highest_21k = (
        highest_21k_usd * USD_TO_SAR
    )

    lowest_18k = (
        lowest_18k_usd * USD_TO_SAR
    )

    highest_18k = (
        highest_18k_usd * USD_TO_SAR
    )


    lowest_date = pd.to_datetime(
        min_row["Date"]
    ).date()

    highest_date = pd.to_datetime(
        max_row["Date"]
    ).date()


    # ========================================================
    # Direct Result
    # ========================================================

    current_price_usd = direct["current_24k"]

    predicted_24k_usd = (
        direct["karat_forecast"]["24K"]
    )

    predicted_21k_usd = (
        direct["karat_forecast"]["21K"]
    )

    predicted_18k_usd = (
        direct["karat_forecast"]["18K"]
    )

    change = direct["pct_change"]

    signal = direct["signal"]

    current_price = (
        current_price_usd * USD_TO_SAR
    )


    # ========================================================
    # Recursive Forecast
    # ========================================================

    st.markdown(
        f"""
        <div class="section-title">
            {days}-Day Gold Price Forecast
        </div>
        """,
        unsafe_allow_html=True
    )


    c1, c2, c3 = st.columns(3)


    # ========================================================
    # 24K
    # ========================================================

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


    # ========================================================
    # 21K
    # ========================================================

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


    # ========================================================
    # 18K
    # ========================================================

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


    # ========================================================
    # Forecast Chart
    # ========================================================

    st.markdown("""<div class="section-title">Forecast Trend</div>""", unsafe_allow_html=True)

    chart_df = recursive_df.copy()
    chart_df["Date"] = pd.to_datetime(chart_df["Date"])


    chart_df["Predicted_24K_SAR"] = (
        chart_df["Predicted_Close"]
        / TROY_OUNCE_GRAMS
        * USD_TO_SAR
    )

    fig, ax = plt.subplots(figsize=(12, 4.5))


    # Dark chart background

    fig.patch.set_facecolor("#0b0d0f")
    ax.set_facecolor("#111416")


    # Gold line

    ax.plot(
        chart_df["Date"],
        chart_df["Predicted_24K_SAR"],
        linewidth=2.5,
        color="#d4af37",
        marker="o",
        markersize=3
    )


    ax.fill_between(
        chart_df["Date"],
        chart_df["Predicted_24K_SAR"],
        alpha=0.08,
        color="#d4af37"
    )


    ax.set_xlabel(
        "Date",
        color="#999999"
    )

    ax.set_ylabel(
        "SAR / gram",
        color="#999999"
    )


    ax.set_title(
        f"{days}-Day Recursive Gold Forecast",
        color="#f5d76e",
        fontsize=14,
        fontweight="bold"
    )


    ax.tick_params(
        colors="#888888"
    )


    ax.grid(
        alpha=0.12,
        color="#888888"
    )


    for spine in ax.spines.values():

        spine.set_color(
            "#333333"
        )


    plt.xticks(
        rotation=45
    )

    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


    # ========================================================
    # Forecast Table
    # ========================================================

    with st.expander(
        "📋 View Forecast Data"
    ):

        display_df = chart_df[
            [
                "Date",
                "Predicted_24K_SAR"
            ]
        ].copy()


        display_df[
            "Predicted_24K_SAR"
        ] = display_df[
            "Predicted_24K_SAR"
        ].round(2)


        display_df = display_df.rename(
            columns={
                "Date": "Date",
                "Predicted_24K_SAR":
                    "Predicted 24K (SAR/g)"
            }
        )


        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )


    # ========================================================
    # Market Signal
    # ========================================================

    st.markdown("""<div class="section-title">Market Signal</div>""", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="signal-card">
            <div class="signal-title">Expected Change: {change:+.2f}%</div>
            <div class="signal-text">{signal}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# Footer
# ============================================================

st.markdown(
    """
    <div class="footer">
        🪙 <span class="footer-gold">Gold Price Forecasting System</span>
        <br>
        AI-powered forecasting · Prices displayed in SAR per gram
    </div>
    """,
    unsafe_allow_html=True
)