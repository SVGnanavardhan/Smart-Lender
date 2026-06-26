from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import List

app = Flask(__name__)
base_dir = Path(__file__).resolve().parent
model = pickle.load(open(base_dir / 'model.pkl', 'rb'))
scale = pickle.load(open(base_dir / 'scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict', methods=['POST',"GET"])
def prediction():
    return render_template('predict.html')

@app.route('/result', methods=['POST','GET'])
def result():
    input_features_list: List[int] = [int(x) for x in request.form.values()]
    input_features = [np.array(input_features_list)]
    print(input_features)

    names =['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area']
    data = pd.DataFrame(input_features, columns=names)

    print(data)
    data_scaled = scale.transform(data)
    prediction = model.predict(data_scaled)
    print(prediction)
    prediction = int(prediction)
    print(type(prediction))
    if (prediction == 0):
        return render_template("result.html", result = "Loan will Not be Approved")
    else:
        return render_template("result.html", result = "Loan will be Approved")


if __name__ == '__main__':
    app.run(debug=True)

    