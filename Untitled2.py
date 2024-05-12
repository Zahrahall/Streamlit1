streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# Load the trained model
model = joblib.load('xgb (1).pkl')  # Make sure the path is correct

# Sidebar for navigation
page = st.sidebar.selectbox("Choose a page", ["Single Prediction", "Bulk Prediction"])

if page == "Single Prediction":
    st.title('Cart Abandonment Prediction App')
    st.write("This app predicts whether a cart will be abandoned based on individual order inputs.")

    # Display an image
    st.image("/workspaces/Streamlit/55eb8cd8-ff03-4fcd-b85d-5638f35839d0.jpg", use_column_width=True)  # Update the path to your image

    # Display a header
    st.header("Crafting Sweet Moments")

    # Input fields for single prediction
    countries = ["USA", "Canada", "UK", "Germany", "France", "Australia", "Brazil", "China", "India", "Russia", "Japan"]
    european_countries = ["UK", "Germany", "France", "Italy", "Spain", "Sweden", "Netherlands", "Belgium", "Poland", "Switzerland"]
    
    cart_quantity = st.number_input('Insert the cart quantity', min_value=0.0)
    total = st.number_input('Insert the cart total', min_value=0.0)
    is_campaign = st.selectbox('Is it a campaign period?', options=[0, 1])
    location = st.selectbox('Select your location', countries)
    store_type_options = ['International Store', 'European Store']
    store_type_index = 1 if location in european_countries else 0
    store_type = st.selectbox('Select the store type:', store_type_options, index=store_type_index)
    store_type_bool = 1 if store_type == 'International Store' else 0
    new_customer = st.selectbox('Is this a new customer?', ['Yes', 'No'])
    new_customer_bool = 1 if new_customer == 'Yes' else 0
    now = datetime.now().hour
    if 6 <= now < 12:
        time_of_day = 'Morning'
    elif 12 <= now < 17:
        time_of_day = 'Afternoon'
    elif 17 <= now < 21:
        time_of_day = 'Evening'
    else:
        time_of_day = 'Night'

    if st.button('Predict Single Order'):
        features = pd.DataFrame({
            'Cart Quantity': [cart_quantity],
            'Total': [total],
            'Is_Campaign': [is_campaign],
            'Location': [location],
            'Website': [store_type_bool],
            'Customer_Type': [new_customer_bool],
            'Time of Day': [time_of_day]
        })

        result = model.predict(features)
        st.write('The prediction is:', 'will be Abandoned' if result[0] == 1 else 'will be Completed')

elif page == "Bulk Prediction":
    st.title("Bulk Prediction for Cart Abandonment")
    st.write("This page allows for bulk predictions from a CSV file.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        
        # Assuming 'data' DataFrame has the necessary columns for prediction
        # Adjust preprocessing as needed based on your model's requirements
        predictions = model.predict(data)
        data['Prediction'] = predictions
        data['Prediction'] = data['Prediction'].apply(lambda x: 'will be Abandoned' if x == 1 else 'will be Completed')
        
        st.write("Prediction Results:")
        st.dataframe(data)
