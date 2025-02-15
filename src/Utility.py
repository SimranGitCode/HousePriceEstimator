import os
import json
import pickle
import numpy as np
import math

# Get the base directory of the script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Global variables
__locations = None
__area_types = None
__data_columns = None
__model = None

def load_artifacts():
    """
    Load the trained model and column details from JSON.
    """
    global __data_columns
    global __area_types
    global __locations
    global __model

    # Load columns
    with open(os.path.join(BASE_DIR, "../artifacts/Columns.json"), 'r') as obj:
        __data_columns = json.load(obj)["Columns"]
        __area_types = __data_columns[4:8]
        __locations = __data_columns[8:]

    # Load trained model
    with open(os.path.join(BASE_DIR, "../artifacts/bangalore_home_prices_model.pickle"), 'rb') as f:
        __model = pickle.load(f)

def get_area_types():
    """
    Returns the list of available area types.
    """
    return __area_types

def get_locations():
    """
    Returns the list of available locations.
    """
    return __locations

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

    # Convert to crores if value is large
    strp = ' lakhs'
    if math.log10(price) >= 2:
        price = price / 100
        price = round(price, 2)
        strp = " crores"

    return str(price) + strp

if __name__ == "__main__":
    load_artifacts()  # Load model and data files
    print(f"Area Types: {get_area_types()}")
    print(f"Locations: {get_locations()}")
    print(get_predicted_price('Carpet Area', 'Varthur', 1000, 3, 3, 3))
    print(get_predicted_price('Carpet Area', '1st phase jp nagar', 1000, 2, 2, 2))
