import streamlit as st
import pickle
st.title('MPG ML Project')
#'displacement', 'horsepower', 'weight', 'acceleration'
displacement = st.number_input('Displacement',value=100, placeholder='enter a value for displacemment')
horsepower = st.number_input('Horsepower',value=130, placeholder='enter a value for placeholder')
weight = st.number_input('Weight',value=300, placeholder='enter a value for weight')
acceleration = st.number_input('Acceleration',value=12, placeholder='enter a value for Acceleration')

loaded_model = pickle.load(open('mpg_regression.sav', 'rb'))
prediction=loaded_model.predict([[displacement,horsepower,weight,acceleration]])
st.subheader(f'Predicted mpg value for the above parammeter is {prediction[0]}')
st.write(prediction)