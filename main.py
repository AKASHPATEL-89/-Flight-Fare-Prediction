# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
    BaggingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv("Clean_Dataset.csv")

print("Original Dataset Shape:")
print(df.shape)

print("\nOriginal Columns:")
print(df.columns.tolist())


# ============================================================
# 3. REMOVE UNNECESSARY INDEX COLUMN
# ============================================================

if "Unnamed: 0" in df.columns:
    df = df.drop("Unnamed: 0", axis=1)

print("\nAfter Removing Unnamed: 0:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 4. CHECK MISSING VALUES
# ============================================================

print("\nMissing Values:")
print(df.isnull().sum())


# ============================================================
# 5. LABEL ENCODING
# ============================================================

categorical_columns = [
    "airline",
    "flight",
    "source_city",
    "departure_time",
    "stops",
    "arrival_time",
    "destination_city",
    "class"
]

encoders = {}

for col in categorical_columns:

    le = LabelEncoder()

    df[col] = le.fit_transform(df[col])

    encoders[col] = le


print("\nCategorical Encoding Completed.")


# ============================================================
# 6. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop("price", axis=1)

y = df["price"]


print("\nFeature Columns:")
print(X.columns.tolist())

print("\nNumber of Features:")
print(X.shape[1])

print("\nTarget:")
print("price")


# ============================================================
# 7. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)


print("\nTrain Shape:")
print(X_train.shape)

print("\nTest Shape:")
print(X_test.shape)


# ============================================================
# 8. FEATURE SCALING
# ============================================================

scaler = MinMaxScaler(
    feature_range=(0, 1)
)

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


print("\nScaling Completed.")


# ============================================================
# 9. CREATE MODELS
# ============================================================

modelETR = ExtraTreesRegressor(
    random_state=42,
    n_jobs=-1
)

modelRFR = RandomForestRegressor(
    random_state=42,
    n_jobs=-1
)

modelBAG = BaggingRegressor(
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 10. TRAIN EXTRA TREES
# ============================================================

print("\nTraining Extra Trees...")

modelETR.fit(
    X_train_scaled,
    y_train
)

print("Extra Trees Training Completed.")


# ============================================================
# 11. TRAIN RANDOM FOREST
# ============================================================

print("\nTraining Random Forest...")

modelRFR.fit(
    X_train_scaled,
    y_train
)

print("Random Forest Training Completed.")


# ============================================================
# 12. TRAIN BAGGING
# ============================================================

print("\nTraining Bagging...")

modelBAG.fit(
    X_train_scaled,
    y_train
)

print("Bagging Training Completed.")


# ============================================================
# 13. PREDICTIONS
# ============================================================

y_pred_ETR = modelETR.predict(X_test_scaled)

y_pred_RFR = modelRFR.predict(X_test_scaled)

y_pred_BAG = modelBAG.predict(X_test_scaled)


# ============================================================
# 14. EVALUATION FUNCTION
# ============================================================

def evaluate_model(model_name, y_test, y_pred):

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    mse = mean_squared_error(
        y_test,
        y_pred
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        y_pred
    )

    print("\n" + "=" * 50)

    print(model_name)

    print("=" * 50)

    print(f"MAE  : {mae:.2f}")

    print(f"MSE  : {mse:.2f}")

    print(f"RMSE : {rmse:.2f}")

    print(f"R²   : {r2:.4f}")

    return {
        "Model": model_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }


# ============================================================
# 15. EVALUATE ALL MODELS
# ============================================================

results = []

results.append(
    evaluate_model(
        "Extra Trees Regressor",
        y_test,
        y_pred_ETR
    )
)

results.append(
    evaluate_model(
        "Random Forest Regressor",
        y_test,
        y_pred_RFR
    )
)

results.append(
    evaluate_model(
        "Bagging Regressor",
        y_test,
        y_pred_BAG
    )
)


# ============================================================
# 16. COMPARISON TABLE
# ============================================================

results_df = pd.DataFrame(results)

print("\n\nMODEL COMPARISON")
print("=" * 80)
print(
    results_df.sort_values(
        by="R2",
        ascending=False
    ).to_string(index=False)
)
# ============================================================
# 17. VERIFY NUMBER OF FEATURES
# ============================================================

print("\n\nMODEL FEATURE CHECK")
print("=" * 50)

print(
    "Extra Trees:",
    modelETR.n_features_in_
)

print(
    "Random Forest:",
    modelRFR.n_features_in_
)

print(
    "Bagging:",
    modelBAG.n_features_in_
)


# ============================================================
# 18. SAVE MODELS
# ============================================================

joblib.dump(
    modelETR,
    "extra_trees_regressor_model.joblib"
)

joblib.dump(
    modelRFR,
    "random_forest_regressor_model.joblib"
)

joblib.dump(
    modelBAG,
    "bagging_regressor_model.joblib"
)


# ============================================================
# 19. SAVE SCALER
# ============================================================

joblib.dump(
    scaler,
    "flight_fare_scaler.joblib"
)


# ============================================================
# 20. SAVE ENCODERS
# ============================================================

joblib.dump(
    encoders,
    "flight_fare_encoders.joblib"
)


# ============================================================
# 21. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)

print("✅ TRAINING COMPLETED")

print("=" * 70)

print("\nSaved Files:")

print("1. extra_trees_regressor_model.joblib")

print("2. random_forest_regressor_model.joblib")

print("3. bagging_regressor_model.joblib")

print("4. flight_fare_scaler.joblib")

print("5. flight_fare_encoders.joblib")

print("\nNumber of Features:")

print("Extra Trees:", modelETR.n_features_in_)

print("Random Forest:", modelRFR.n_features_in_)

print("Bagging:", modelBAG.n_features_in_)