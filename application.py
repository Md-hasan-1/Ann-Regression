import streamlit as st
from prediction import Pred
from data_transformation import data_transformation_config
import pickle
import bentoml

# Load the trained model
model = bentoml.keras.get("kerasregressor:latest").to_runner()
model.init_local()

# Load the encoders and scaler
with open(data_transformation_config.preprocessor_path, 'rb') as file_obj:
    preprocessor = pickle.load(file_obj)

## streamlit app
st.title('Customer Churn PRediction')

# User input
geography = st.selectbox('Geography', ['France', 'Spain', 'Germany'])
gender = st.selectbox('Gender', ['Female', 'Male'])
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
Exited = st.selectbox('Exited', [0, 1])
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

prediction = Pred()
prediction_proba = prediction.predict(model, preprocessor, geography, gender, age, balance, credit_score, 
                Exited, tenure, num_of_products, has_cr_card, 
                is_active_member)

st.write(f'Predicted Salary: {prediction_proba:.2f}')
