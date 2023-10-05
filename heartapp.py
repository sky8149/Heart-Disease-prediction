import streamlit as st
import pandas as pd
import pickle

model_filename = 'model.pkl'

with open(model_filename, 'rb') as file:
    model = pickle.load(file)
st.title('Heart Disease Prediction') 
st.write('To input or change the values of attributes kindly use the sidebar accessible from the top left side of the page')   
st.sidebar.header('User Input Features')
    

def main():
    age = st.sidebar.slider('Age', 18, 100, 50)
    sex_options = ['Male', 'Female']
    sex = st.sidebar.selectbox('Sex', sex_options)
    sex_num = 1 if sex == 'Male' else 0 
    cp_options = ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic']
    cp = st.sidebar.selectbox('Chest Pain Type', cp_options)
    cp_num = cp_options.index(cp)
    trestbps = st.sidebar.slider('Resting Blood Pressure', 90, 200, 120)
    chol = st.sidebar.slider('Cholesterol', 100, 600, 250)
    fbs_options = ['False', 'True']
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl', fbs_options)
    fbs_num = fbs_options.index(fbs)
    restecg_options = ['Normal', 'ST-T Abnormality', 'Left Ventricular Hypertrophy']
    restecg = st.sidebar.selectbox('Resting Electrocardiographic Results', restecg_options)
    restecg_num = restecg_options.index(restecg)
    thalach = st.sidebar.slider('Maximum Heart Rate Achieved', 70, 220, 150)
    exang_options = ['No', 'Yes']
    exang = st.sidebar.selectbox('Exercise Induced Angina', exang_options)
    exang_num = exang_options.index(exang)
    oldpeak = st.sidebar.slider('ST Depression Induced by Exercise Relative to Rest', 0.0, 6.2, 1.0)
    slope_options = ['Upsloping', 'Flat', 'Downsloping']
    slope = st.sidebar.selectbox('Slope of the Peak Exercise ST Segment', slope_options)
    slope_num = slope_options.index(slope)
    ca = st.sidebar.slider('Number of Major Vessels Colored by Fluoroscopy', 0, 4, 1)
    thal_options = ['Normal', 'Fixed Defect', 'Reversible Defect']
    thal = st.sidebar.selectbox('Thalassemia', thal_options)
    thal_num = thal_options.index(thal)
    user_input = pd.DataFrame(data={
            'age': [age],
            'sex': [sex_num],  
            'cp': [cp_num],
            'trestbps': [trestbps],
            'chol': [chol],
            'fbs': [fbs_num],
            'restecg': [restecg_num],
            'thalach': [thalach],
            'exang': [exang_num],
            'oldpeak': [oldpeak],
            'slope': [slope_num],
            'ca': [ca],
            'thal': [thal_num]
        })
    st.write(user_input)    
    if st.button('Predict'):
        user_input = pd.DataFrame(data={
            'age': [age],
            'sex': [sex_num],  
            'cp': [cp_num],
            'trestbps': [trestbps],
            'chol': [chol],
            'fbs': [fbs_num],
            'restecg': [restecg_num],
            'thalach': [thalach],
            'exang': [exang_num],
            'oldpeak': [oldpeak],
            'slope': [slope_num],
            'ca': [ca],
            'thal': [thal_num]
        })
        prediction = model.predict(user_input)
        prediction_proba = model.predict_proba(user_input)

        if prediction[0] == 1:
            bg_color = 'red'
            prediction_result = 'Positive'
        else:
            bg_color = 'green'
            prediction_result = 'Negative'
        
        confidence = prediction_proba[0][1] if prediction[0] == 1 else prediction_proba[0][0]

        st.markdown(f"<p style='background-color:{bg_color}; color:white; padding:10px;'>Prediction: {prediction_result}<br>Confidence: {((confidence*10000)//1)/100}%</p>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
st.write('This is a simple web application that allows users to predict the likelihood of heart disease based on input features')
