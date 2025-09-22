from flask import Flask, request, jsonify
from src import utility

app = Flask(__name__)

@app.route("/get_area_types")
def get_area_types():
    response = jsonify({'area_types': utility.get_area_types()})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route("/get_locations")
def get_locations():
    response = jsonify({'locations': utility.get_locations()})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route("/predict_home_price", methods=['POST'])
def predict_home_price():
    area_type = request.form['area_type']
    location = request.form['location']
    total_sqft = float(request.form['total_sqft'])
    balcony = int(request.form['balcony'])
    bathroom = int(request.form['bathroom'])
    BHK = int(request.form['BHK'])

    response = jsonify({
        'predicted_price': utility.get_predicted_price(area_type, location, total_sqft, balcony, bathroom, BHK)
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

if __name__ == "__main__":
    print("Bangalore House Price Prediction")
    utility.load_artifacts()
    app.run(debug=True)
