from flask import Flask, jsonify
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load('iris_model.pkl')

from flask import request

@app.route('/predict', methods=['GET'])
def predict():
    # Get the input array from the request
    # try:
    #     iris_input = [float(request.args.get(f'input{i}')) for i in range(4)]
    # except (TypeError, ValueError):
    #     return jsonify({'error': 'Invalid input'}), 400
    #try:
        # Manually parse the JSON body from the request data
    get_json = request.get_json()
    iris_input = get_json['input']
    # except (json.JSONDecodeError, TypeError, KeyError):
    #     return jsonify({'error': 'Invalid input'}), 400
    #
    # TODO: Import trained model
    # model = ...
    
    # TODO: Make prediction using the model 
    # HINT: use np.array().reshape(1, -1) to convert input to 2D array
    iris_input = np.array(iris_input).reshape(1, -1)
    prediction = model.predict(iris_input)

    # TODO: Return the prediction as a response
    return jsonify({'prediction': prediction[0]})

@app.route('/')
def hello():
    return 'Welcome to Docker Lab'

if __name__ == '__main__':
    app.run(debug=True, port=8080, host='0.0.0.0')
