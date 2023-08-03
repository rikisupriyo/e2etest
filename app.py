import numpy as np
import pandas as pd
import pickle
from flask import Flask, render_template, request, app, jsonify, url_for

app = Flask(__name__)
model = pickle.load(open('xgregressorhousing.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods = ['POST'])
def predict_api():
    data = request.json['data']
    new_data = np.array(list(data.values())).reshape(1, -1)
    output = model.predict(new_data)
    return jsonify(float(output[0]))

@app.route('/predict', methods = ['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_data = np.array(np.array(data).reshape(1, -1))
    output = model.predict(final_data)
    return render_template('home.html', prediction = 'The price is $ {:.2f}'.format(output[0] * 10000))

if __name__ == "__main__":
    app.run(debug = True)