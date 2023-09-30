import numpy as np
import joblib

scaler = joblib.load('housepriceprediction/models/scaler.pkl')
rf_reg = joblib.load('housepriceprediction/models/rf_reg_model.pkl')


def predict(values=[3,1,5,3,2,2,4]):
    values = scaler.transform(np.array(values).reshape(1,-1))
    price = rf_reg.predict(values)
    return price
    

# Bedroom = 2
# Bathroom = 5
# Floors = 1.0
# Parking = 10.0
# Area = 3
# Road=10.0
# Amenities = 2

# print(predict([Bedroom,Bathroom,Floors,Parking,Area,Road,Amenities]))