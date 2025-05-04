import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('titanic_model.pkl')

st.title("🛳️ Titanic Survival Prediction App")

# Sidebar for user input
st.sidebar.header("Enter Passenger Details")

def user_input_features():
    Pclass = st.sidebar.selectbox('Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)', [1, 2, 3])
    Sex = st.sidebar.selectbox('Sex', ['male', 'female'])
    Age = st.sidebar.slider('Age', 0, 80, 30)
    SibSp = st.sidebar.slider('Number of Siblings/Spouses aboard', 0, 8, 0)
    Fare = st.sidebar.slider('Fare ($)', 0, 500, 32)
    Embarked = st.sidebar.selectbox('Port of Embarkation', ['S', 'C', 'Q'])

    # Encode categorical values
    sex_encoded = 0 if Sex == 'male' else 1
    embarked_encoded = {'S': 2, 'C': 0, 'Q': 1}[Embarked]

    data = {
        'Pclass': Pclass,
        'Sex': sex_encoded,
        'Age': Age,
        'SibSp': SibSp,
        'Fare': Fare,
        'Embarked': embarked_encoded
    }

    return pd.DataFrame([data])

input_df = user_input_features()

st.subheader('Passenger Data')
st.write(input_df)

if st.button('Predict'):
    prediction = model.predict(input_df)
    result = "🎉 Survived!" if prediction[0] == 1 else "☠️ Did not survive."
    st.subheader('Prediction:')
    st.success(result)
