import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the pre-trained classifier and scaler using joblib
classifier = joblib.load('knn_model.joblib')  
scaler = joblib.load('scaler.joblib')  

# Define the prediction function
def predict_heart_disease(d):
    sample_data = pd.DataFrame([d])
    scaled_data = scaler.transform(sample_data)
    pred = classifier.predict(scaled_data)[0]
    prob = np.max(classifier.predict_proba(scaled_data)[0])
    return pred, prob

# Streamlit UI components
st.title("Heart Disease Prediction")

# Input fields for each parameter
age = st.number_input("Age", min_value=29, max_value=77, value=50, step=1)
sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")  
cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])  
trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=94, max_value=200, value=130, step=1)
chol = st.number_input("Serum Cholesterol (chol)", min_value=126, max_value=564, value=250, step=1)
fbs = st.selectbox("Fasting Blood Sugar (fbs)", [0, 1], format_func=lambda x: "False" if x == 0 else "True")  
restecg = st.selectbox("Resting Electrocardiographic Results (restecg)", [0, 1, 2])  
thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", min_value=71, max_value=202, value=160, step=1)
exang = st.selectbox("Exercise-Induced Angina (exang)", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")  
oldpeak = st.number_input("ST Depression Induced by Exercise Relative to Rest (oldpeak)", min_value=0.0, max_value=6.2, value=1.0, step=0.1)
slope = st.selectbox("Slope of the Peak Exercise ST Segment (slope)", [0, 1, 2])  
ca = st.selectbox("Number of Major Vessels (ca)", [0, 1, 2, 3, 4])  
thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])  

# Create the input dictionary for prediction
input_data = {
    'age': age,
    'sex': sex,
    'cp': cp,
    'trestbps': trestbps,
    'chol': chol,
    'fbs': fbs,
    'restecg': restecg,
    'thalach': thalach,
    'exang': exang,
    'oldpeak': oldpeak,
    'slope': slope,
    'ca': ca,
    'thal': thal
}

# When the user clicks the "Predict" button
if st.button("Predict"):
    with st.spinner('Making prediction...'):
        pred, prob = predict_heart_disease(input_data)

        if pred == 1:
            st.error(f"Prediction: Heart Disease with probability {prob:.2f}") 
        else:
            st.success(f"Prediction: No Heart Disease with probability {prob:.2f}")
