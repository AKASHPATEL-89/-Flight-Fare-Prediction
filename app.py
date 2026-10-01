
import streamlit as st
import pandas as pd
import joblib
import os
import base64


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Flight Fare Prediction",
    page_icon="✈️",
    layout="wide"
)


# ============================================================
# BACKGROUND IMAGE
# ============================================================
import streamlit as st
import os
import base64


# ============================================================
# BACKGROUND IMAGE
# ============================================================

def set_background(image_file):

    if not os.path.exists(image_file):
        st.error(f"❌ Image not found: {image_file}")
        return

    with open(image_file, "rb") as file:
        encoded_image = base64.b64encode(file.read()).decode()

    st.markdown(
        f"""
        <style>

        /* Main Streamlit background */
        [data-testid="stAppViewContainer"] {{
            background-image: url(
                "data:image/jpeg;base64,{encoded_image}"
            );

            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        /* Remove default header background */
        [data-testid="stHeader"] {{
            background: rgba(0,0,0,0);
        }}

        /* Main content transparent */
        [data-testid="stMain"] {{
            background: transparent;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# SET BACKGROUND
# ============================================================

set_background("background1.jpg")


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

missing_files = [
    file for file in required_files
    if not os.path.exists(file)
]

if missing_files:

    st.error("❌ Required files are missing.")

    for file in missing_files:
        st.write(f"- `{file}`")

    st.info(
        "Keep all required files in the same folder as app.py."
    )

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
# LOAD ENCODERS
# ============================================================

@st.cache_resource
def load_encoders():
    return joblib.load(ENCODER_FILE)


encoders = load_encoders()


# ============================================================
# LOAD SCALER
# ============================================================

@st.cache_resource
def load_scaler():
    return joblib.load(SCALER_FILE)


scaler = load_scaler()


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    models = {}

    for name, file in MODEL_FILES.items():
        models[name] = joblib.load(file)

    return models


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
# SIDEBAR - MODEL INFORMATION
# ============================================================

st.sidebar.header("⚙️ Model Information")


# ============================================================
# SIDEBAR - MODEL SELECTION
# ============================================================

st.sidebar.header("🎯 Model Selection")

model_choice = st.sidebar.selectbox(
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


# Airline
with col1:

    airline = st.selectbox(
        "Airline",
        list(encoders["airline"].classes_)
    )


# Flight
with col2:

    flight = st.selectbox(
        "Flight",
        list(encoders["flight"].classes_)
    )


# Source City
with col3:

    source_city = st.selectbox(
        "Source City",
        list(encoders["source_city"].classes_)
    )


# Departure Time
with col1:

    departure_time = st.selectbox(
        "Departure Time",
        list(encoders["departure_time"].classes_)
    )


# Stops
with col2:

    stops = st.selectbox(
        "Stops",
        list(encoders["stops"].classes_)
    )


# Arrival Time
with col3:

    arrival_time = st.selectbox(
        "Arrival Time",
        list(encoders["arrival_time"].classes_)
    )


# Destination City
with col1:

    destination_city = st.selectbox(
        "Destination City",
        list(encoders["destination_city"].classes_)
    )


# Class
with col2:

    flight_class = st.selectbox(
        "Class",
        list(encoders["class"].classes_)
    )


# Duration
with col3:

    duration = st.number_input(
        "Duration (hours)",
        min_value=0.0,
        max_value=100.0,
        value=2.5,
        step=0.1
    )


# Days Left
with col1:

    days_left = st.number_input(
        "Days Left",
        min_value=1,
        max_value=50,
        value=10,
        step=1
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

        # ----------------------------------------------------
        # ENCODE INPUT VALUES
        # ----------------------------------------------------

        encoded_values = {

            "airline":
                encoders["airline"].transform(
                    [airline]
                )[0],

            "flight":
                encoders["flight"].transform(
                    [flight]
                )[0],

            "source_city":
                encoders["source_city"].transform(
                    [source_city]
                )[0],

            "departure_time":
                encoders["departure_time"].transform(
                    [departure_time]
                )[0],

            "stops":
                encoders["stops"].transform(
                    [stops]
                )[0],

            "arrival_time":
                encoders["arrival_time"].transform(
                    [arrival_time]
                )[0],

            "destination_city":
                encoders["destination_city"].transform(
                    [destination_city]
                )[0],

            "class":
                encoders["class"].transform(
                    [flight_class]
                )[0],

            "duration": duration,

            "days_left": days_left
        }


        # ----------------------------------------------------
        # CREATE INPUT DATAFRAME
        # ----------------------------------------------------

        input_data = pd.DataFrame(
            [encoded_values]
        )

        input_data = input_data[
            feature_order
        ]


        # ----------------------------------------------------
        # FEATURE CHECK
        # ----------------------------------------------------

        if input_data.shape[1] != 10:

            st.error(
                f"❌ Expected 10 features, "
                f"but got {input_data.shape[1]}."
            )

            st.stop()


        # ----------------------------------------------------
        # SCALE INPUT
        # ----------------------------------------------------

        input_scaled = scaler.transform(
            input_data
        )


        # ----------------------------------------------------
        # SCALE CHECK
        # ----------------------------------------------------

        if input_scaled.shape[1] != 10:

            st.error(
                f"❌ Scaler generated "
                f"{input_scaled.shape[1]} features."
            )

            st.stop()


        # ====================================================
        # SINGLE MODEL PREDICTION
        # ====================================================

        if model_choice != "Compare All Models":

            model = models[model_choice]

            expected_features = getattr(
                model,
                "n_features_in_",
                10
            )


            if expected_features != 10:

                st.error(
                    f"❌ {model_choice} expects "
                    f"{expected_features} features."
                )

                st.stop()


            prediction = model.predict(
                input_scaled
            )

            price = float(
                prediction[0]
            )


            # ------------------------------------------------
            # SHOW RESULT
            # ------------------------------------------------

            st.success(
                "✅ Prediction Successful!"
            )

            st.metric(
                "💰 Predicted Flight Fare",
                f"₹ {price:,.2f}"
            )

            st.info(
                f"🤖 Model Used: **{model_choice}**"
            )


        # ====================================================
        # COMPARE BOTH MODELS
        # ====================================================

        else:

            st.subheader(
                "📊 Prediction Comparison"
            )

            results = []


            for name, model in models.items():

                expected_features = getattr(
                    model,
                    "n_features_in_",
                    10
                )


                if expected_features != 10:

                    st.error(
                        f"❌ {name} expects "
                        f"{expected_features} features."
                    )

                    st.stop()


                prediction = model.predict(
                    input_scaled
                )

                price = float(
                    prediction[0]
                )


                results.append(
                    {
                        "Model": name,
                        "Predicted Fare":
                            f"₹ {price:,.2f}"
                    }
                )


            result_df = pd.DataFrame(
                results
            )


            st.dataframe(
                result_df,
                use_container_width=True,
                hide_index=True
            )


    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as e:

        st.error(
            f"❌ Prediction Error: {str(e)}"
        )
