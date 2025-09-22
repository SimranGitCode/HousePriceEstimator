import os
import numpy as np
import pandas as pd
import pickle
import streamlit as st
import json
import math
 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
 
with open(os.path.join(BASE_DIR, "../house_price_model.pkl"), 'rb') as f:
    __model = pickle.load(f)
 
with open(os.path.join(BASE_DIR, "../Columns.json"), 'r') as obj:
    __data_columns = json.load(obj)["Columns"]
    __area_types = __data_columns[4:8]   
    __locations = __data_columns[8:]    

def get_predicted_price(area_type, location, sqft, balcony, bathroom, BHK):
    """
    Predicts the price of a house based on input features.
    """
    try:
        area_index = __data_columns.index(area_type.lower())
        loc_index = __data_columns.index(location.lower())
    except ValueError:
        area_index = -1
        loc_index = -1

    lis = np.zeros(len(__data_columns))
    lis[0] = sqft
    lis[1] = bathroom
    lis[2] = balcony
    lis[3] = BHK

    if loc_index >= 0 and area_index >= 0:
        lis[area_index] = 1
        lis[loc_index] = 1

    price = round(__model.predict([lis])[0], 2)
 
    strp = ' lakhs'
    if math.log10(price) >= 2:
        price = price / 100
        price = round(price, 2)
        strp = " crores"

    return str(price) + strp

def main():
    """
    Runs the Streamlit web app.
    """
    st.title("Bangalore House Price Predictor")
 
    total_sqft = st.text_input("Total Sqft")
    balcony = st.text_input("Number of Balconies")
    bathroom = st.text_input("Number of Bathrooms")
    BHK = st.text_input("BHK")
    area_type = st.selectbox("Area Type", __area_types)
    location = st.selectbox("Location", __locations)
 
    if st.button("Predict"):
        result = get_predicted_price(area_type, location, total_sqft, balcony, bathroom, BHK)
        st.success(f"Price = {result}")

if __name__ == "__main__":
    main()
