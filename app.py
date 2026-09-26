import streamlit as st
import pandas as pd
import joblib

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Airbnb Room Type Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# DARK THEME UI
# =========================================================
st.markdown("""
<style>
    .stApp {
        background: #0b1120;
        color: #f8fafc;
    }

    [data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #263244;
    }

    [data-testid="stHeader"] {
        background: rgba(11, 17, 32, 0);
    }

    .hero {
        padding: 30px 34px;
        border: 1px solid #263244;
        border-radius: 22px;
        background: linear-gradient(135deg, #111827 0%, #172033 100%);
        margin-bottom: 24px;
        box-shadow: 0 10px 35px rgba(0, 0, 0, .25);
    }

    .hero h1 {
        margin: 0;
        color: #f8fafc;
        font-size: 38px;
        font-weight: 750;
    }

    .hero p {
        color: #94a3b8;
        font-size: 16px;
        margin: 8px 0 0 0;
    }

    .section {
        color: #e2e8f0;
        font-size: 21px;
        font-weight: 700;
        margin: 12px 0 10px 0;
    }

    div[data-testid="stMetric"] {
        background: #111827;
        border: 1px solid #263244;
        padding: 15px;
        border-radius: 15px;
    }

    div[data-testid="stMetricLabel"] {
        color: #94a3b8;
    }

    div[data-testid="stMetricValue"] {
        color: #f8fafc;
    }

    .result-card {
        padding: 24px;
        border-radius: 18px;
        background: #111827;
        border: 1px solid #334155;
        margin-top: 20px;
        text-align: center;
    }

    .result-title {
        color: #94a3b8;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1.2px;
    }

    .result-value {
        color: #f8fafc;
        font-size: 32px;
        font-weight: 800;
        margin-top: 6px;
    }

    .hint {
        color: #94a3b8;
        font-size: 13px;
    }

    .stButton > button {
        width: 100%;
        border-radius: 12px;
        min-height: 48px;
        font-weight: 700;
    }

    /* Dropdown text */
    div[data-baseweb="select"] {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# MODEL
# =========================================================
MODEL_FILE = "model_pipeline.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


try:
    model = load_model()
except Exception as e:
    st.error(
        "Model load nahi ho pa raha. Confirm karein ki "
        "`model_pipeline.pkl` isi folder me hai."
    )
    st.code(str(e))
    st.stop()

# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class="hero">
    <h1>🏠 Airbnb Room Type Predictor</h1>
    <p>
        Enter property details below and predict the expected room type
        using the trained Machine Learning model.
    </p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## ⚙️ Prediction Settings")

    st.markdown(
        '<p class="hint">Enter the property information and click Predict.</p>',
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("### 📌 Model Information")

    st.markdown(
        '<p class="hint">10 features are used for prediction.</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="hint">Model: ML Pipeline (.pkl)</p>',
        unsafe_allow_html=True,
    )

# =========================================================
# INPUT SECTION
# =========================================================
st.markdown(
    '<div class="section">📍 Property & Location Details</div>',
    unsafe_allow_html=True,
)

with st.form("prediction_form"):

    col1, col2, col3 = st.columns(3)

    # -----------------------------------------------------
    # COLUMN 1
    # -----------------------------------------------------
    with col1:

        latitude = st.number_input(
            "Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=40.7128,
            step=0.0001,
            format="%.4f",
        )

        longitude = st.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=-74.0060,
            step=0.0001,
            format="%.4f",
        )

        price = st.number_input(
            "Price per Night ($)",
            min_value=0.01,
            value=150.0,
            step=5.0,
        )

        minimum_nights = st.number_input(
            "Minimum Nights",
            min_value=1,
            max_value=365,
            value=2,
            step=1,
        )

    # -----------------------------------------------------
    # COLUMN 2
    # -----------------------------------------------------
    with col2:

        number_of_reviews = st.number_input(
            "Number of Reviews",
            min_value=0,
            value=25,
            step=1,
        )

        reviews_per_month = st.number_input(
            "Reviews per Month",
            min_value=0.0,
            value=1.5,
            step=0.1,
        )

        calculated_host_listings_count = st.number_input(
            "Host Listings Count",
            min_value=0,
            value=1,
            step=1,
        )

        availability_365 = st.number_input(
            "Availability (365 Days)",
            min_value=0,
            max_value=365,
            value=200,
            step=1,
        )

    # -----------------------------------------------------
    # COLUMN 3 - DROPDOWN
    # -----------------------------------------------------
    with col3:

        neighbourhood_group = st.selectbox(
            "Neighbourhood Group",
            options=[
                "Bronx",
                "Brooklyn",
                "Manhattan",
                "Queens",
                "Staten Island",
            ],
            index=2,
        )

        neighbourhood = st.selectbox(
            "Neighbourhood",
            options=[
                "Midtown",
                "Upper East Side",
                "Upper West Side",
                "Chelsea",
                "Harlem",
                "Hell's Kitchen",
                "Crown Heights",
                "Bushwick",
                "Bedford-Stuyvesant",
                "Astoria",
                "Long Island City",
                "Flushing",
            ],
            index=0,
        )

    st.markdown("")

    submitted = st.form_submit_button(
        "🔮 Predict Room Type",
        type="primary",
    )

# =========================================================
# PREDICTION
# =========================================================
if submitted:

    # IMPORTANT:
    # Keep exactly the same column names expected by the model.
    row = pd.DataFrame([{
        "latitude": latitude,
        "longitude": longitude,
        "price": price,
        "minimum_nights": minimum_nights,
        "number_of_reviews": number_of_reviews,
        "reviews_per_month": reviews_per_month,
        "calculated_host_listings_count": calculated_host_listings_count,
        "availability_365": availability_365,
        "neighbourhood_group": neighbourhood_group,
        "neighbourhood": neighbourhood,
    }])

    try:

        prediction = model.predict(row)

        predicted_class = prediction[0]

        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-title">Predicted Room Type</div>
                <div class="result-value">{predicted_class}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # -------------------------------------------------
        # PROBABILITY
        # -------------------------------------------------
        probability = None

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(row)[0]

        if probability is not None:

            classes = getattr(model, "classes_", None)

            # For Pipeline objects
            if classes is None and hasattr(model, "steps"):
                final_estimator = model.steps[-1][1]
                classes = getattr(final_estimator, "classes_", None)

            if classes is not None:

                st.markdown(
                    '<div class="section">📊 Prediction Confidence</div>',
                    unsafe_allow_html=True,
                )

                prob_df = pd.DataFrame({
                    "Room Type": [str(x) for x in classes],
                    "Probability": probability,
                }).sort_values(
                    "Probability",
                    ascending=False
                )

                for _, item in prob_df.iterrows():

                    st.progress(
                        float(item["Probability"]),
                        text=(
                            f'{item["Room Type"]}: '
                            f'{item["Probability"]:.1%}'
                        ),
                    )

        # -------------------------------------------------
        # INPUT PREVIEW
        # -------------------------------------------------
        with st.expander("🔎 View Input Sent to Model"):

            st.dataframe(
                row,
                use_container_width=True,
                hide_index=True,
            )

    except Exception as e:

        st.error("Prediction ke time error aaya.")

        st.exception(e)

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")

st.caption(
    "🏠 Airbnb Room Type Predictor • "
    "Built with Streamlit + Machine Learning"
)
