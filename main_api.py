from flask import Flask, request, jsonify
import numpy as np
import pickle

model = pickle.load(open('model_clf.sav', 'rb'))

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
    val = [[0, 0, 0, 0, 0,0, 0, 0, 0], pred_vals]
    pred = model.predict(val)
    pred_prob = model.predict_proba(val)
    print(pred[1])
    # converting it to yes or no
    if pred[1]==1:
        ret_str = "You are prone to have heart disease,"+"the chances of having are "+str(round(pred_prob[1][1]*100,2)) + "%"
    else:
        ret_str = "You are not prone to have heart disease,"+"the chances of not having are "+str(round(pred_prob[1][0]*100,2))+ "%"
    return str(ret_str)


if __name__ == '__main__':
    app.run(debug=True)
