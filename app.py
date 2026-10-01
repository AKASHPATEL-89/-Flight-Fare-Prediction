import streamlit as st
import pandas as pd
import joblib
import os
import random


# ============================================================
# PAGE CONFIGURATION (must be the first Streamlit call)
# ============================================================

st.set_page_config(
    page_title="Flight Fare Prediction",
    page_icon="✈️",
    layout="wide"
)


# ============================================================
# UNSPLASH BACKGROUND IMAGES
# ============================================================

UNSPLASH_URLS = [
    "https://images.unsplash.com/photo-1542281286-9e0a16bb7366?w=1920&q=80",
    "https://images.unsplash.com/photo-1436491865332-7a61a109cc05?w=1920&q=80",
    "https://images.unsplash.com/photo-1569154941061-e231b4725ef1?w=1920&q=80",
    "https://images.unsplash.com/photo-1474302770737-173ee21bab63?w=1920&q=80",
    "https://images.unsplash.com/photo-1517479149777-5f3b1511d5ad?w=1920&q=80",
    "https://images.unsplash.com/photo-1476514525535-07fb3b4ae5f1?w=1920&q=80",
    "https://images.unsplash.com/photo-1503220317375-aaad61436b1b?w=1920&q=80",
    "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=1920&q=80",
]


def set_background(image_url: str):
    """Inject a full-page background image via CSS."""

    st.markdown(
        f"""
        <style>

        /* ---- Main app background ---- */
        .stApp,
        [data-testid="stAppViewContainer"] {{
            background-image: url("{image_url}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        /* ---- Transparent header ---- */
        [data-testid="stHeader"] {{
            background: rgba(0, 0, 0, 0);
        }}

        /* ---- Dark overlay for readability ---- */
        [data-testid="stAppViewContainer"]::before {{
            content: "";
            position: fixed;
            inset: 0;
            background: linear-gradient(
                rgba(5, 15, 30, 0.65),
                rgba(5, 15, 30, 0.82)
            );
            z-index: 0;
            pointer-events: none;
        }}

        /* ---- Keep content above the overlay ---- */
        [data-testid="stAppViewContainer"] > .main,
        section.main,
        .block-container {{
            position: relative;
            z-index: 1;
        }}

        /* ---- Sidebar ---- */
        section[data-testid="stSidebar"] {{
            background: rgba(3, 12, 28, 0.92);
        }}

        section[data-testid="stSidebar"] * {{
            color: white;
        }}

        /* ---- Text colors ---- */
        label, .stMarkdown, h1, h2, h3, h4, h5, h6, p {{
            color: white !important;
        }}

        /* ---- Metric card ---- */
        div[data-testid="stMetric"] {{
            background: rgba(255, 255, 255, 0.12);
            border: 1px solid rgba(255, 255, 255, 0.20);
            padding: 22px;
            border-radius: 18px;
            backdrop-filter: blur(10px);
        }}

        div[data-testid="stMetricLabel"] {{
            color: #dbeafe !important;
        }}

        div[data-testid="stMetricValue"] {{
            color: white !important;
            font-weight: 800;
        }}

        /* ---- Button ---- */
        .stButton > button {{
            border-radius: 12px;
            font-weight: 700;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )


# Pick a random background ONCE per session
if "bg_url" not in st.session_state:
    st.session_state.bg_url = random.choice(UNSPLASH_URLS)


# ============================================================
# SIDEBAR — top (background shuffle button)
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            text-align:center;
            padding:15px;
            margin-bottom:20px;
            border-radius:15px;
            background:rgba(255,255,255,0.08);
        ">
            <div style="font-size:42px;">✈️</div>
            <div style="font-size:22px; font-weight:800;">Flight Fare Prediction
</div>
            <div style="font-size:13px; color:#bfdbfe;">
                Machine Learning Prediction
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button("🎲 Change Background", use_container_width=True):
        st.session_state.bg_url = random.choice(UNSPLASH_URLS)
        st.rerun()

    st.divider()


# Apply background (must come AFTER session_state is set)
set_background(st.session_state.bg_url)


# ============================================================
# TITLE
# ============================================================

st.title("✈️ Flight Fare Prediction")

st.markdown(
    """
    Predict flight ticket prices using Machine Learning.

    **Available Models:**
    - 🌳 Random Forest Regressor
    - 📦 Bagging Regressor
    """
)


# ============================================================
# FILE PATHS
# ============================================================

DATA_FILE = "Clean_Dataset.csv"

MODEL_FILES = {
    "Random Forest Regressor": "random_forest_regressor_model.joblib",
    "Bagging Regressor": "bagging_regressor_model.joblib"
}

ENCODER_FILE = "flight_fare_encoders.joblib"
SCALER_FILE = "flight_fare_scaler.joblib"


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

required_files = [
    DATA_FILE,
    ENCODER_FILE,
    SCALER_FILE,
    "random_forest_regressor_model.joblib",
    "bagging_regressor_model.joblib"
]

missing_files = [f for f in required_files if not os.path.exists(f)]

if missing_files:
    st.error("❌ Required files are missing.")
    for f in missing_files:
        st.write(f"- `{f}`")
    st.info("Keep all required files in the same folder as app.py.")
    st.stop()


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():
    data = pd.read_csv(DATA_FILE)
    if "Unnamed: 0" in data.columns:
        data = data.drop("Unnamed: 0", axis=1)
    return data


df = load_dataset()


# ============================================================
# LOAD ENCODERS / SCALER / MODELS
# ============================================================

@st.cache_resource
def load_encoders():
    return joblib.load(ENCODER_FILE)


@st.cache_resource
def load_scaler():
    return joblib.load(SCALER_FILE)


@st.cache_resource
def load_models():
    return {name: joblib.load(f) for name, f in MODEL_FILES.items()}


encoders = load_encoders()
scaler = load_scaler()
models = load_models()


# ============================================================
# FEATURE ORDER
# ============================================================

feature_order = [
    "airline",
    "flight",
    "source_city",
    "departure_time",
    "stops",
    "arrival_time",
    "destination_city",
    "class",
    "duration",
    "days_left"
]


# ============================================================
# SIDEBAR — MODEL SELECTION
# ============================================================

with st.sidebar:

    st.header("⚙️ Model Information")
    st.write("🌳 Random Forest Regressor")
    st.write("📦 Bagging Regressor")

    st.divider()

    st.header("🎯 Model Selection")

    model_choice = st.selectbox(
        "Select Model",
        [
            "Random Forest Regressor",
            "Bagging Regressor",
            "Compare All Models"
        ]
    )


# ============================================================
# FLIGHT INPUT
# ============================================================

st.subheader("🛫 Enter Flight Details")

col1, col2, col3 = st.columns(3)

with col1:
    airline = st.selectbox("Airline", list(encoders["airline"].classes_))

with col2:
    flight = st.selectbox("Flight", list(encoders["flight"].classes_))

with col3:
    source_city = st.selectbox("Source City", list(encoders["source_city"].classes_))

with col1:
    departure_time = st.selectbox("Departure Time", list(encoders["departure_time"].classes_))

with col2:
    stops = st.selectbox("Stops", list(encoders["stops"].classes_))

with col3:
    arrival_time = st.selectbox("Arrival Time", list(encoders["arrival_time"].classes_))

with col1:
    destination_city = st.selectbox("Destination City", list(encoders["destination_city"].classes_))

with col2:
    flight_class = st.selectbox("Class", list(encoders["class"].classes_))

with col3:
    duration = st.number_input(
        "Duration (hours)",
        min_value=0.0, max_value=100.0, value=2.5, step=0.1
    )

with col1:
    days_left = st.number_input(
        "Days Left",
        min_value=1, max_value=50, value=10, step=1
    )


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

predict = st.button(
    "🔮 Predict Flight Fare",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict:
    try:
        encoded_values = {
            "airline": encoders["airline"].transform([airline])[0],
            "flight": encoders["flight"].transform([flight])[0],
            "source_city": encoders["source_city"].transform([source_city])[0],
            "departure_time": encoders["departure_time"].transform([departure_time])[0],
            "stops": encoders["stops"].transform([stops])[0],
            "arrival_time": encoders["arrival_time"].transform([arrival_time])[0],
            "destination_city": encoders["destination_city"].transform([destination_city])[0],
            "class": encoders["class"].transform([flight_class])[0],
            "duration": duration,
            "days_left": days_left
        }

        input_data = pd.DataFrame([encoded_values])[feature_order]
        input_scaled = scaler.transform(input_data)

        # ---------- Single model ----------
        if model_choice != "Compare All Models":
            model = models[model_choice]
            price = float(model.predict(input_scaled)[0])

            st.success("✅ Prediction Successful!")
            st.metric("💰 Predicted Flight Fare", f"₹ {price:,.2f}")
            st.info(f"🤖 Model Used: **{model_choice}**")

        # ---------- Compare all models ----------
        else:
            st.subheader("📊 Prediction Comparison")
            results = []
            for name, model in models.items():
                price = float(model.predict(input_scaled)[0])
                results.append({
                    "Model": name,
                    "Predicted Fare": f"₹ {price:,.2f}"
                })

            st.dataframe(
                pd.DataFrame(results),
                use_container_width=True,
                hide_index=True
            )

    except Exception as e:
        st.error(f"❌ Prediction Error: {str(e)}")