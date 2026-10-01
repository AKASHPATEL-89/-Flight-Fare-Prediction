# ✈️ Flight Fare Prediction

> **An end-to-end Machine Learning project for predicting flight ticket fares using multiple regression algorithms and an interactive Streamlit web application.**

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?logo=numpy)](https://numpy.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployment-FF4B4B?logo=streamlit)](https://streamlit.io/)

---

## 🚀 Live Demo

### 🌐 Try the Flight Fare Prediction App

👉 **[Open Live Demo](https://flightfareprediction89.streamlit.app/)**

The application provides an interactive interface where users can enter flight-related information and receive an estimated flight fare.

---

## 📂 GitHub Repository

👉 **[Flight Fare Prediction — GitHub](https://github.com/AKASHPATEL-89/Flight-Fare-Prediction)**

---

# 📌 Project Overview

Flight ticket prices depend on several factors such as airline, source city, destination city, departure time, arrival time, number of stops, travel class, and other flight-related information.

This project uses **Machine Learning regression algorithms** to learn patterns from historical flight data and predict the estimated price of a flight.

The complete project follows an end-to-end Machine Learning workflow:

```text
Dataset
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Feature Engineering
   ↓
Categorical Encoding
   ↓
Feature Scaling
   ↓
Train-Test Split
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Comparison
   ↓
Best Model Selection
   ↓
Model Saving
   ↓
Streamlit Deployment
   ↓
Flight Fare Prediction
```

---

# 🎯 Problem Statement

The objective of this project is to build a Machine Learning regression system that can predict the price of a flight based on its available features.

### Input

The model uses flight-related features such as:

* ✈️ Airline
* 📍 Source City
* 📍 Destination City
* 🕐 Departure Time
* 🕐 Arrival Time
* 🛑 Number of Stops
* 🎫 Travel Class
* 📅 Journey-related information
* Other relevant flight features

### Output

The system predicts:

> 💰 **Estimated Flight Fare**

---

# 💡 Project Objectives

The main objectives of this project are:

* Understand the factors affecting flight fares
* Perform data cleaning and preprocessing
* Analyze the dataset using EDA
* Encode categorical variables
* Scale numerical features
* Train multiple regression algorithms
* Compare model performance
* Select the best-performing model
* Save trained models for deployment
* Build an interactive Streamlit application
* Deploy the application online

---

# 📊 Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the dataset and identify relationships between input features and flight prices.

### EDA includes:

* Dataset shape
* Data types
* Missing-value analysis
* Duplicate-value analysis
* Statistical summary
* Categorical feature analysis
* Numerical feature analysis
* Distribution analysis
* Outlier analysis
* Correlation analysis
* Feature vs. target analysis

### Example Analysis

```text
Airline → Flight Fare
Source → Flight Fare
Destination → Flight Fare
Stops → Flight Fare
Departure Time → Flight Fare
Arrival Time → Flight Fare
Class → Flight Fare
```

---

# 🧹 Data Preprocessing

The following preprocessing steps were performed:

### 1. Data Cleaning

* Checked missing values
* Checked duplicate records
* Examined incorrect data types
* Removed unnecessary information

### 2. Feature and Target Separation

The dataset was divided into:

```python
X = features
y = target
```

where `y` represents the flight fare.

### 3. Categorical Encoding

Categorical features were converted into numerical representations using appropriate encoding techniques.

### 4. Feature Scaling

Numerical features were scaled where required to make them suitable for machine learning algorithms.

---

# 🤖 Machine Learning Models

A total of **10 regression models** were trained and evaluated.

### Models Used

1. Extra Trees Regressor
2. Random Forest Regressor
3. Bagging Regressor
4. XGBoost Regressor
5. Decision Tree Regressor
6. K-Nearest Neighbors Regressor
7. Gradient Boosting Regressor
8. Linear Regression
9. Ridge Regression
10. Lasso Regression

---

# 📈 Evaluation Metrics

The models were evaluated using the following regression metrics:

### MAE — Mean Absolute Error

Measures the average absolute difference between actual and predicted values.

```text
MAE = (1/n) Σ |yᵢ - ŷᵢ|
```

**Lower is better.**

---

### MSE — Mean Squared Error

```text
MSE = (1/n) Σ(yᵢ - ŷᵢ)²
```

**Lower is better.**

---

### RMSE — Root Mean Squared Error

```text
RMSE = √MSE
```

**Lower is better.**

---

### MAPE — Mean Absolute Percentage Error

```text
MAPE = (100/n) Σ |(yᵢ - ŷᵢ) / yᵢ|
```

**Lower is better.**

---

### R² Score

R² measures how much variance in the target variable is explained by the model.

```text
R² = 1 - SSres / SStot
```

**Higher is better.**

---

### Adjusted R²

Adjusted R² accounts for the number of predictors used by the model.

**Higher is better.**

---

# 🏆 Model Performance

The following results were obtained on the evaluation dataset:

| Rank | Model                     |  Adjusted R² |          MAE |         RMSE |      MAPE |           MSE |        RMSLE |     R² Score |
| ---: | ------------------------- | -----------: | -----------: | -----------: | --------: | ------------: | -----------: | -----------: |
| 🥇 1 | **ExtraTreesRegressor**   | **0.984841** | **1,147.23** | **2,793.04** | **7.65%** | **7,801,088** | **7.934887** | **0.984842** |
| 🥈 2 | **RandomForestRegressor** |     0.984092 |     1,170.96 |     2,861.26 |     7.92% |     8,186,830 |     7.959019 |     0.984093 |
| 🥉 3 | **BaggingRegressor**      |     0.982645 |     1,216.11 |     2,988.53 |     8.27% |     8,931,315 |     8.002537 |     0.982646 |
|    4 | XGBRegressor              |     0.977895 |     1,831.84 |     3,372.85 |    14.03% |    11,376,100 |     8.123512 |     0.977896 |
|    5 | DecisionTreeRegressor     |     0.973272 |     1,263.24 |     3,708.85 |     8.63% |    13,755,550 |     8.218477 |     0.973273 |
|    6 | KNeighborsRegressor       |     0.971645 |     1,854.47 |     3,820.07 |    11.11% |    14,592,970 |     8.248025 |     0.971646 |
|    7 | GradientBoostingRegressor |     0.956823 |     2,808.07 |     4,713.93 |    20.67% |    22,211,200 |     8.458277 |     0.956824 |
|    8 | LinearRegression          |     0.904653 |     4,630.30 |     7,005.02 |    43.89% |    49,070,240 |     8.854382 |     0.904656 |
|    9 | Ridge Regression          |     0.904653 |     4,630.31 |     7,005.01 |    43.89% |    49,070,190 |     8.854381 |     0.904656 |
|   10 | Lasso Regression          |     0.904653 |     4,630.18 |     7,005.01 |    43.89% |    49,070,110 |     8.854380 |     0.904656 |

---

# 🥇 Best Model — ExtraTreesRegressor

Among all evaluated models, **ExtraTreesRegressor** achieved the best overall performance.

### Performance

| Metric          |        Result |
| --------------- | ------------: |
| **R² Score**    |  **0.984842** |
| **Adjusted R²** |  **0.984841** |
| **MAE**         | **₹1,147.23** |
| **RMSE**        | **₹2,793.04** |
| **MAPE**        |     **7.65%** |
| **MSE**         | **7,801,088** |
| **RMSLE**       |  **7.934887** |

### Key Result

The model achieved an **R² score of 0.984842**, meaning it explains approximately **98.48% of the variance** in flight fare on the evaluated dataset.

The **MAE of ₹1,147.23** indicates an average absolute prediction error of approximately ₹1,147 on the evaluation data.

---

# 🌲 Why ExtraTreesRegressor?

ExtraTreesRegressor performed particularly well because flight prices can have complex and non-linear relationships with features such as:

* Airline
* Route
* Destination
* Source
* Number of stops
* Departure time
* Arrival time
* Travel class

Tree-based ensemble methods can capture these non-linear relationships effectively without requiring a purely linear relationship between the features and target.

---

# 🖥️ Streamlit Web Application

The trained Machine Learning models are integrated into an interactive **Streamlit application**.

## Application Workflow

```text
User Input
    ↓
Flight Details
    ↓
Data Preprocessing
    ↓
Categorical Encoding
    ↓
Feature Scaling
    ↓
Trained Model
    ↓
Prediction
    ↓
Estimated Flight Fare
```

### Application Features

* ✈️ Flight information input
* 🏢 Airline selection
* 📍 Source city selection
* 📍 Destination city selection
* 🕐 Departure time
* 🕐 Arrival time
* 🛑 Stops selection
* 🎫 Class selection
* 🔎 Encoded input inspection
* 📐 Scaled input inspection
* 🤖 Multiple model predictions
* 💰 Estimated fare output

---

# 🌐 Live Application

## Try it now

### 👉 [✈️ Flight Fare Prediction — Live Demo](https://flight-fareprediction.streamlit.app/)

The application is deployed using **Streamlit Community Cloud**.

---

# 🛠️ Technology Stack

## Programming

* Python

## Data Analysis

* Pandas
* NumPy

## Data Visualization

* Matplotlib
* Seaborn

## Machine Learning

* Scikit-learn
* XGBoost

## Model Persistence

* Joblib

## Web Application

* Streamlit

## Development

* Jupyter Notebook
* VS Code
* Git
* GitHub

## Deployment

* Streamlit Community Cloud

---

# 📦 Installation

## 1. Clone the repository

```bash
git clone https://github.com/AKASHPATEL-89/Flight-Fare-Prediction.git
```

## 2. Navigate to the project

```bash
cd Flight-Fare-Prediction
```

## 3. Create a virtual environment

```bash
python -m venv venv
```

## 4. Activate the virtual environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

## 5. Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Locally

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

# 📁 Project Structure

```text
Flight-Fare-Prediction/
│
├── app.py
├── README.md
├── requirements.txt
│
├── dataset/
│   └── ...
│
├── models/
│   ├── ...
│   └── ...
│
├── notebooks/
│   └── ...
│
└── ...
```

> The exact structure depends on the files included in the repository.

---

# 🔬 Machine Learning Pipeline

```text
                 ┌──────────────────┐
                 │  Flight Dataset  │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │  Data Cleaning   │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │      EDA         │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │ Feature Engineer │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │    Encoding      │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │     Scaling      │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │ Model Training   │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │ Model Evaluation │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │ ExtraTrees Model │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │ Streamlit App    │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │ Fare Prediction  │
                 └──────────────────┘
```

---

# 📊 Key Results

### Best Model

**ExtraTreesRegressor**

### Best R² Score

**98.48%**

### Adjusted R²

**98.48%**

### Mean Absolute Error

**₹1,147.23**

### MAPE

**7.65%**

These results demonstrate strong predictive performance on the evaluated dataset.

---

# 🚀 Future Improvements

The project can be further improved by adding:

* Hyperparameter optimization
* Cross-validation
* Feature importance analysis
* SHAP explainability
* Automated model retraining
* Real-time flight data integration
* Price trend analysis
* Airline-wise price comparison
* Route-wise fare analysis
* Advanced boosting algorithms
* MLflow experiment tracking
* Docker deployment
* CI/CD pipeline

---

# 📚 Learning Outcomes

This project helped implement an end-to-end Machine Learning workflow including:

* Data Cleaning
* Exploratory Data Analysis
* Feature Engineering
* Categorical Encoding
* Feature Scaling
* Regression
* Ensemble Learning
* Model Evaluation
* Model Comparison
* Model Selection
* Model Serialization
* Streamlit Development
* Cloud Deployment
* Git & GitHub

---

# 👨‍💻 Author

## Akash Singh

**B.Tech Student | AI & Machine Learning | Data Science**

Interested in:

* 🤖 Artificial Intelligence
* 🧠 Machine Learning
* 👁️ Computer Vision
* 📊 Data Science
* 🐍 Python
* ☁️ Cloud & Deployment

### 🔗 Connect

* 💻 **GitHub:** [AKASHPATEL-89](https://github.com/AKASHPATEL-89)
* 🌐 **Live Demo:** [Flight Fare Prediction](https://flight-fareprediction.streamlit.app/)

---

# ⭐ Support

If you found this project useful or interesting, please consider giving the repository a ⭐.

Your support is appreciated!

---

## 📄 License

This project is developed for **educational, learning, and portfolio purposes**.
