from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # السماح للفرونت إند يتواصل مع الـ API

# تحميل الموديل و الـ scaler
model = joblib.load('house_price_model.pkl')
scaler = joblib.load('scaler.pkl')

# قائمة أنواع البيوت
home_types = ['Apartment', 'Condo', 'Single_Family', 'Townhouse', 'Lot', 'Manufactured', 'Multi_Family', 'Home_Type_Unknown']

def predict_house_price(livingArea, bedrooms, bathrooms, latitude, longitude, yearBuilt, homeType,
                        propertyTaxRate=0.0, annualHomeownersInsurance=0.0):
    home_type_encoded = [1 if homeType == ht else 0 for ht in home_types]
    features = np.array([livingArea, bedrooms, bathrooms, latitude, longitude, yearBuilt,
                         propertyTaxRate, annualHomeownersInsurance] + home_type_encoded)
    features_scaled = scaler.transform(features.reshape(1, -1))
    predicted_price = model.predict(features_scaled)
    return float(predicted_price[0])

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # استخراج البيانات من الطلب
        livingArea = data['livingArea']
        bedrooms = data['bedrooms']
        bathrooms = data['bathrooms']
        latitude = data['latitude']
        longitude = data['longitude']
        yearBuilt = data['yearBuilt']
        homeType = data['homeType']
        propertyTaxRate = data.get('propertyTaxRate', 0.0)
        annualHomeownersInsurance = data.get('annualHomeownersInsurance', 0.0)

        prediction = predict_house_price(livingArea, bedrooms, bathrooms, latitude, longitude,
                                         yearBuilt, homeType, propertyTaxRate, annualHomeownersInsurance)

        return jsonify({'predicted_price': prediction})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)



