# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import os

# st.set_page_config(
#     page_title="Flight Fare Prediction",
#     page_icon="✈️",
#     layout="wide"
# )

# st.title("✈️ Flight Fare Prediction")

# st.markdown(
#     """
#     Predict flight ticket prices using Machine Learning.

#     **Available Models:**
#     - 🌲 Extra Trees Regressor
#     - 🌳 Random Forest Regressor
#     - 📦 Bagging Regressor
#     """
# )

# DATA_FILE = "Clean_Dataset.csv"

# MODEL_FILES = {
#     "Extra Trees Regressor": "extra_trees_regressor_model.joblib",
#     "Random Forest Regressor": "random_forest_regressor_model.joblib",
#     "Bagging Regressor": "bagging_regressor_model.joblib"
# }

# ENCODER_FILE = "flight_fare_encoders.joblib"
# SCALER_FILE = "flight_fare_scaler.joblib"

# required_files = [
#     DATA_FILE,
#     ENCODER_FILE,
#     SCALER_FILE,
#     "extra_trees_regressor_model.joblib",
#     "random_forest_regressor_model.joblib",
#     "bagging_regressor_model.joblib"
# ]

# missing_files = [
#     file for file in required_files
#     if not os.path.exists(file)
# ]

# if missing_files:
#     st.error("❌ Required files are missing.")
#     for file in missing_files:
#         st.write(f"- `{file}`")
#     st.info("Keep all required files in the same folder as app.py.")
#     st.stop()

# @st.cache_data
# def load_dataset():
#     data = pd.read_csv(DATA_FILE)

#     if "Unnamed: 0" in data.columns:
#         data = data.drop("Unnamed: 0", axis=1)

#     return data

# df = load_dataset()

# @st.cache_resource
# def load_encoders():
#     return joblib.load(ENCODER_FILE)

# encoders = load_encoders()

# @st.cache_resource
# def load_scaler():
#     return joblib.load(SCALER_FILE)

# scaler = load_scaler()

# @st.cache_resource
# def load_models():
#     models = {}

#     for name, file in MODEL_FILES.items():
#         models[name] = joblib.load(file)

#     return models

# models = load_models()

# feature_order = [
#     "airline",
#     "flight",
#     "source_city",
#     "departure_time",
#     "stops",
#     "arrival_time",
#     "destination_city",
#     "class",
#     "duration",
#     "days_left"
# ]

# st.sidebar.header("⚙️ Model Information")

# for name, model in models.items():
#     expected_features = getattr(
#         model,
#         "n_features_in_",
#         None
#     )

#     if expected_features == 10:
#         st.sidebar.success(
#             f"✅ {name}: 10 features"
#         )
#     else:
#         st.sidebar.warning(
#             f"⚠️ {name}: {expected_features} features"
#         )

# st.sidebar.header("🎯 Model Selection")

# model_choice = st.sidebar.selectbox(
#     "Select Model",
#     [
#         "Extra Trees Regressor",
#         "Random Forest Regressor",
#         "Bagging Regressor",
#         "Compare All Models"
#     ]
# )

# st.subheader("🛫 Enter Flight Details")

# col1, col2, col3 = st.columns(3)

# with col1:
#     airline = st.selectbox(
#         "Airline",
#         list(encoders["airline"].classes_)
#     )

# with col2:
#     flight = st.selectbox(
#         "Flight",
#         list(encoders["flight"].classes_)
#     )

# with col3:
#     source_city = st.selectbox(
#         "Source City",
#         list(encoders["source_city"].classes_)
#     )

# with col1:
#     departure_time = st.selectbox(
#         "Departure Time",
#         list(encoders["departure_time"].classes_)
#     )

# with col2:
#     stops = st.selectbox(
#         "Stops",
#         list(encoders["stops"].classes_)
#     )

# with col3:
#     arrival_time = st.selectbox(
#         "Arrival Time",
#         list(encoders["arrival_time"].classes_)
#     )

# with col1:
#     destination_city = st.selectbox(
#         "Destination City",
#         list(encoders["destination_city"].classes_)
#     )

# with col2:
#     flight_class = st.selectbox(
#         "Class",
#         list(encoders["class"].classes_)
#     )

# with col3:
#     duration = st.number_input(
#         "Duration (hours)",
#         min_value=0.0,
#         max_value=100.0,
#         value=2.5,
#         step=0.1
#     )

# with col1:
#     days_left = st.number_input(
#         "Days Left",
#         min_value=1,
#         max_value=50,
#         value=10,
#         step=1
#     )

# st.divider()

# predict = st.button(
#     "🔮 Predict Flight Fare",
#     type="primary",
#     use_container_width=True
# )

# if predict:

#     try:

#         encoded_values = {
#             "airline": encoders["airline"].transform([airline])[0],
#             "flight": encoders["flight"].transform([flight])[0],
#             "source_city": encoders["source_city"].transform([source_city])[0],
#             "departure_time": encoders["departure_time"].transform([departure_time])[0],
#             "stops": encoders["stops"].transform([stops])[0],
#             "arrival_time": encoders["arrival_time"].transform([arrival_time])[0],
#             "destination_city": encoders["destination_city"].transform([destination_city])[0],
#             "class": encoders["class"].transform([flight_class])[0],
#             "duration": duration,
#             "days_left": days_left
#         }

#         input_data = pd.DataFrame([encoded_values])

#         input_data = input_data[feature_order]

#         if input_data.shape[1] != 10:
#             st.error(
#                 f"❌ Expected 10 features, but got {input_data.shape[1]}."
#             )
#             st.stop()

#         input_scaled = scaler.transform(input_data)

#         if input_scaled.shape[1] != 10:
#             st.error(
#                 f"❌ Scaler generated {input_scaled.shape[1]} features."
#             )
#             st.stop()

#         if model_choice != "Compare All Models":

#             model = models[model_choice]

#             expected_features = getattr(
#                 model,
#                 "n_features_in_",
#                 10
#             )

#             if expected_features != 10:
#                 st.error(
#                     f"❌ {model_choice} expects {expected_features} features."
#                 )
#                 st.stop()

#             prediction = model.predict(input_scaled)

#             price = float(prediction[0])

#             st.success("✅ Prediction Successful!")

#             st.metric(
#                 "💰 Predicted Flight Fare",
#                 f"₹ {price:,.2f}"
#             )

#             st.info(
#                 f"🤖 Model Used: **{model_choice}**"
#             )

#         else:

#             st.subheader("📊 Prediction Comparison")

#             results = []

#             for name, model in models.items():

#                 expected_features = getattr(
#                     model,
#                     "n_features_in_",
#                     10
#                 )

#                 if expected_features != 10:
#                     st.error(
#                         f"❌ {name} expects {expected_features} features."
#                     )
#                     st.stop()

#                 prediction = model.predict(input_scaled)

#                 price = float(prediction[0])

#                 results.append({
#                     "Model": name,
#                     "Predicted Fare": f"₹ {price:,.2f}"
#                 })

#             result_df = pd.DataFrame(results)

#             st.dataframe(
#                 result_df,
#                 use_container_width=True,
#                 hide_index=True
#             )

#         with st.expander("🔎 View Encoded Input"):
#             st.dataframe(
#                 input_data,
#                 use_container_width=True
#             )

#         with st.expander("📐 View Scaled Input"):

#             scaled_df = pd.DataFrame(
#                 input_scaled,
#                 columns=feature_order
#             )

#             st.dataframe(
#                 scaled_df,
#                 use_container_width=True
#             )

#     except Exception as e:

#         st.error(
#             f"❌ Prediction Error: {str(e)}"
#         )





































































import streamlit as st
import pandas as pd
import joblib
import os

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Flight Fare Prediction",
    page_icon="✈️",
    layout="wide"
)

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
