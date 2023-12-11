from flask import Flask, request, jsonify
import numpy as np
import pickle

model = pickle.load(open('model_lr.sav', 'rb'))

app = Flask(__name__)


@app.route('/')
def hello():
    return 'Hello apna kaam kr'


@app.route('/post', methods=["POST"])
def predict():
    input_data = request.get_data(as_text=True)
    input_data = input_data.replace("[", "")
    input_data = input_data.replace("]", "")
    pred_vals = input_data.split(",")
    for i in range(len(pred_vals)):
        pred_vals[i] = int(pred_vals[i])
    val = [[0, 0, 0, 0, 0, 0, 0, 0], pred_vals]
    pred = model.predict_proba(val)
    prob = round(pred[0][1] * 100, 2)
    return str("Probability of you having a heart disease is " + str(prob) + "%")


if __name__ == '__main__':
    app.run(debug=True)
