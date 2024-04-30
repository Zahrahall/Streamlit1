#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import joblib

# Load your trained model
model = joblib.load('xgb_model.pkl')

# Creating a simple form
st.title('Cart Abandonment Prediction App')
st.write("This app predicts whether a cart will be abandoned.")
def front_page():
    st.header("Crafting Sweet Moments")
    st.image('"C:\Users\z_hal\Downloads\e42f7cc3-19bf-4232-be16-bf3df3bf85b8.jpg"', use_column_width=True)
# Input bars
cart_quantity = st.number_input('Insert the cart quantity', min_value=0.0)
total = st.number_input('Insert the cart total', min_value=0.0)
is_campaign = st.selectbox('Is it a campaign period?', options=[0, 1])

# When 'Predict' is clicked, make the prediction and store it under the variable result
if st.button('Predict'):
    features = pd.DataFrame({
        'Cart Quantity': [cart_quantity],
        'Total': [total],
        'Is_Campaign': [is_campaign]
    })
    
    # Assume the model uses these features to predict
    result = model.predict(features)
    
    # Display the prediction
    st.write('The prediction is:')
    st.write('Abandoned' if result[0] == 1 else 'Completed')
