import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Title and Sidebar
st.title("Machine Learning Models Presentation")
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select Model", ["Insurance Charges Prediction", "Titanic Survival Prediction", "Diabetes Prediction"])

# Load data function
@st.cache_data  # Updated caching for data
def load_data(file_path):
    return pd.read_csv(file_path)

# Insurance Charges Prediction Model
if page == "Insurance Charges Prediction":
    st.subheader("Insurance Charges Prediction Model")
    
    # Load and display the dataset
    data = load_data("insurance.csv")
    st.write("Dataset preview:")
    st.write(data.head())
    
    # Input from user
    age = st.slider("Age", 18, 100)
    bmi = st.slider("BMI", 10.0, 50.0)
    smoker = st.selectbox("Smoker", ["yes", "no"])
    smoker_value = 1 if smoker == "yes" else 0
    
    # Train the model
    X = data[['age', 'bmi', 'smoker']].replace({"yes": 1, "no": 0})
    y = data['charges']
    model = LinearRegression()
    model.fit(X, y)
    
    # Prediction
    prediction = model.predict([[age, bmi, smoker_value]])
    st.write(f"Predicted Charges: ${prediction[0]:.2f}")

# Titanic Survival Prediction Model
elif page == "Titanic Survival Prediction":
    st.subheader("Titanic Survival Prediction Model")
    
    # Load and display the dataset
    train_data = load_data("train.csv")
    st.write("Train dataset preview:")
    st.write(train_data.head())
    
    # Input from user
    pclass = st.selectbox("Passenger Class", [1, 2, 3])
    age = st.slider("Age", 1, 80)
    sibsp = st.slider("Number of Siblings/Spouses", 0, 8)
    fare = st.number_input("Fare Paid", min_value=0.0)
    
    # Train the model (RandomForest)
    X_train = train_data[['Pclass', 'Age', 'SibSp', 'Fare']].fillna(0)
    y_train = train_data['Survived']
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    # Prediction
    prediction = model.predict([[pclass, age, sibsp, fare]])
    result = "Survived" if prediction[0] == 1 else "Did Not Survive"
    st.write(f"Prediction: {result}")

# Diabetes Prediction Model
elif page == "Diabetes Prediction":
    st.subheader("Pima Indian Diabetes Prediction Model")
    
    # Load and display the dataset
    data = load_data("diabetes.csv")
    st.write("Dataset preview:")
    st.write(data.head())
    
    # Input from user
    pregnancies = st.slider("Pregnancies", 0, 20)
    glucose = st.slider("Glucose", 0, 200)
    bp = st.slider("Blood Pressure", 0, 150)
    bmi = st.slider("BMI", 10.0, 50.0)
    
    # Train the model (KNN)
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = KNeighborsClassifier()
    model.fit(X_scaled, y)
    
    # Prediction
    input_data = scaler.transform([[pregnancies, glucose, bp, bmi, 0, 0, 0, 0]])  # add appropriate inputs
    prediction = model.predict(input_data)
    result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
    st.write(f"Prediction: {result}")

# To run this Streamlit app, save it in a Python file (e.g., app.py) and run:
# `streamlit run app.py`
