import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open(r"model.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    platform=request.form['platform']
    genre=request.form['genre']
    publisher=request.form['publisher']
    na_sales=float(request.form['na_sales'])
    eu_sales=float(request.form['eu_sales'])
    jp_sales=float(request.form['jp_sales'])
    other_sales=float(request.form['other_sales'])
    tot_sales=na_sales+eu_sales+jp_sales+other_sales
    response_data=pd.DataFrame([[platform, genre, publisher, tot_sales]], columns=['Platform','Genre','Publisher','Total_Sales'])
    prediction = model.predict(response_data)
    final_prediction=prediction.item()
    return render_template("index.html", predicted_value = "The Predicted global sales is {:.2f}".format(final_prediction))

if __name__ == "__main__":
    flask_app.run(debug=True)