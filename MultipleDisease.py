# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 18:29:56 2023

@author: Rahul Kumar
"""
import pickle 
import streamlit as st
import sklearn
from streamlit_option_menu import option_menu

## loading the saved models
diabetes_model = pickle.load(open("diabetes_model.sav", 'rb'))

heart_disease_model = pickle.load(open("heart_disease_model.sav", 'rb'))



# slidebar for navigation
with st.sidebar:
    selected = option_menu("Multiple Disease Prediction Systems", [
        'Diabetes Prediction',
        'Heart Disease Prediction'],
        icons = ['activity','heart'],
        default_index=1)
  
    
# Diabetes Prediction Page
if (selected == 'Diabetes Prediction'):
    
    # page title
    st.title('Diabetes Prediction using ML with probability in percentage')
    
    
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.number_input('Number of Pregnancies')
        
    with col2:
        Glucose = st.number_input('Glucose Level')
    
    with col3:
        BloodPressure = st.number_input('Blood Pressure value')
    
    with col1:
        SkinThickness = st.number_input('Skin Thickness value')
    
    with col2:
        Insulin = st.number_input('Insulin Level')
    
    with col3:
        BMI = st.number_input('BMI value')
    
    with col1:
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function value')
    
    with col2:
        Age = st.number_input('Age of the Person')
    
    # Prediction section
    diab_diagnosis = '' # this will store end result
    
    # Creating a button 
    if st.button('Diabetes Test Result'):
        diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        diab_prediction_prob = diabetes_model.predict_proba([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        
        
        if diab_prediction[0] == 1:
            diab_diagnosis = f"This person is Diabetic, with probability of {round(diab_prediction_prob[0][1]*100, 2)} %" 
        else:
            diab_diagnosis = f"This person is not Diabetic, with probability of {round(diab_prediction_prob[0][0]*100 , 2)} %"
        
        st.success(diab_diagnosis)
    




# Heart Disease Prediction Page
if (selected == 'Heart Disease Prediction'):
    
    # page title
    st.title('Heart Disease Prediction using ML with probability in percentage')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input('Age')
        
    with col2:
        sex = st.number_input('Sex')
        
    with col3:
        cp = st.number_input('Chest Pain types')
        
    with col1:
        trestbps = st.number_input('Resting Blood Pressure')
        
    with col2:
        chol = st.number_input('Serum Cholestoral in mg/dl')
        
    with col3:
        fbs = st.number_input('Fasting Blood Sugar > 120 mg/dl')
        
    with col1:
        restecg = st.number_input('Resting Electrocardiographic results')
        
    with col2:
        thalach = st.number_input('Maximum Heart Rate achieved')
        
    with col3:
        exang = st.number_input('Exercise Induced Angina')
        
    with col1:
        oldpeak = st.number_input('ST depression induced by exercise')
        
    with col2:
        slope = st.number_input('Slope of the peak exercise ST segment')
        
    with col3:
        ca = st.number_input('Major vessels colored by flourosopy')
        
    
    thal = st.number_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
        
     
    # code for Prediction
    heart_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Heart Disease Test Result'):
        heart_prediction = heart_disease_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg,thalach,exang,oldpeak,slope,ca,thal]])                          
        heart_prediction_prob = heart_disease_model.predict_proba([[age, sex, cp, trestbps, chol, fbs, restecg,thalach,exang,oldpeak,slope,ca,thal]])

        if (heart_prediction[0] == 1):
          heart_diagnosis = f'The person is having heart disease, with probability of {round(heart_prediction_prob[0][1]*100 , 2)} %'
        else:
          heart_diagnosis = f'The person does not have any heart disease, with probability of {round(heart_prediction_prob[0][0]*100 , 2)} %'
        
    st.success(heart_diagnosis)
        
    
    