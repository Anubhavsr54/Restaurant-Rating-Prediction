import numpy as np
from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import os

from flask_cors import CORS, cross_origin

'''Load pickel file'''
file = os.listdir('./bestmodel/')[0]
model = pickle.load(open('./bestmodel/'+file, 'rb'))
scaler = pickle.load(open('std_scaler.pkl','rb'))

app = Flask(__name__)


@app.route('/')
@cross_origin()
def home():
    return render_template('Zomato.html')

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    if (request.method=='POST'):
            online_order = int(request.form['Online_Order'])
            book_table = int(request.form['Table_Booking'])
            votes = int(request.form['Votes'])
            rest_type= int(request.form['Restaurant_Type'])
            location= int(request.form['Location'])
            cost= int(request.form['Cost'])
            type_=int(request.form['Type'])

            final_features = [np.array((online_order,book_table,votes,rest_type,location,cost,type_))]
            std_data = scaler.transform(final_features)

            prediction = model.predict(std_data)
            output = round(prediction[0], 2)

            return render_template('result.html', output=f"Predicted Rating is: {str(output)}")
    else:
            return render_template('Zomato.html')

if __name__ == "__main__":
    app.run(port=5000, debug=True)

