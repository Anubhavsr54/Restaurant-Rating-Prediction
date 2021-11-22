from flask import Flask, request, jsonify
from Modules.preprocessing import Preprocessing
from Modules.Model_tuning import Parameter_tuning

app = Flask(__name__)
@app.route('/train',methods=['POST'])
def training():
    if (request.method == 'POST'):
        operation = request.json['operation']
        
        if (operation.lower() == 'training'):

            Preprocessing_obj = Preprocessing()
            df = Preprocessing_obj.preprocess_data('./zomato.csv')

            trainmodel_obj = Parameter_tuning(df)
            result = trainmodel_obj.model_result()

            best_model_name = result[0]
            r2_score = result[1]
            rmse = result[2]

            return jsonify(f'''Best_model: {best_model_name} with evaluation Score:{round(r2_score, 2)} and RMSE: {rmse}''')
if __name__ == '__main__':
    app.run(port=5000, debug=True)