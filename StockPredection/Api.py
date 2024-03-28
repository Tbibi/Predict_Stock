from flask import Flask, request, jsonify
import numpy as np
from keras.models import load_model

from StockPredection.Predection import sc

app = Flask(__name__)


# Load the trained model
model = load_model('AAPLModel.h5')

# Endpoint for stock price prediction
@app.route('/predict')
def predict():
    data = request.get_json(force=True)
    inputs = np.array(data['inputs']).reshape(1, 60, 1)

    # Perform prediction
    predicted_stock_price = model.predict(inputs)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)

    return jsonify({'prediction': predicted_stock_price.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

