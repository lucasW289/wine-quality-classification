# app.py

import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Project Name: VinClassification: Predicting Wine Quality
st.title('VinClassification: Wine Quality Predictor')

st.write("""
Enter the physiochemical properties of the wine to predict its quality category (Low, Medium, or High).
""")

# Load the trained model and feature names
# Make sure these files are in the same directory as your app.py file,
# or provide the correct path.
try:
    model = joblib.load('wine_quality_classifier_model.kpl')
    feature_names = joblib.load('wine_quality_features.kpl')
    st.success("Model and feature names loaded successfully!")
except FileNotFoundError:
    st.error("Error: Model or feature names file not found.")
    st.info("Please ensure 'wine_quality_classifier_model.kpl' and 'wine_quality_features.kpl' are in the same directory.")
    st.stop() # Stop the app if files are not found
except Exception as e:
    st.error(f"An error occurred while loading the model or feature names: {e}")
    st.stop()


# Create input fields for each feature
# You can adjust the min_value, max_value, and step based on your data's characteristics
input_data = {}
for feature in feature_names:
    # Customize input widget based on feature name if needed
    if feature == 'pH':
        input_data[feature] = st.number_input(f'{feature.replace("_", " ").title()}', min_value=2.5, max_value=4.5, value=3.3, step=0.01)
    elif feature == 'alcohol':
         input_data[feature] = st.number_input(f'{feature.replace("_", " ").title()}', min_value=8.0, max_value=15.0, value=9.5, step=0.1)
    elif feature in ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'sulphates']:
         # Provide a wider range and smaller step for other numerical features
         input_data[feature] = st.number_input(f'{feature.replace("_", " ").title()}', min_value=0.0, value=input_data.get(feature, 0.5), step=0.001, format="%.4f") # Use format for better precision
    else:
        input_data[feature] = st.number_input(f'{feature.replace("_", " ").title()}', value=0.0) # Default for unexpected features


# Create a button to make predictions
if st.button('Predict Quality'):
    # Prepare the input data for the model
    # Ensure the order of features matches the training data
    input_df = pd.DataFrame([input_data])

    # Make prediction
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    # Display the prediction
    st.subheader('Prediction:')
    predicted_class = prediction[0]
    st.write(f'The predicted wine quality category is: **{predicted_class}**')

    # Display prediction probabilities
    st.subheader('Prediction Probabilities:')
    proba_df = pd.DataFrame(prediction_proba, columns=model.classes_)
    proba_df = proba_df.transpose().reset_index()
    proba_df.columns = ['Quality Category', 'Probability']
    st.dataframe(proba_df)

    st.balloons()
