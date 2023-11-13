import numpy as np
from flask import Flask, request, render_template
import pickle

application = Flask(__name__)
model = pickle.load(open('models/Savedmodel.pkl', 'rb'))

def scaler(data):
    mean_list = [36.45003079516354, 231971.29206781418, 0.08408700444098674, 0.6410058819108129, 5.3068711033312805, -8.422581715236584, 0.6312468259370915, 0.08659007315202005, 0.31843934541487023, 0.17499633233535392, 0.21890214496418034, 0.4630629720142198, 122.47310274779304, 3.928868574886274, 10.16195014425103]
    var_list = [396.244239871765, 12734158518.315853, 0.0770163801251282, 0.06483784457539407, 12.646383574458293, 26.26860119715149, 0.23277427068143886, 0.012164690101788332, 0.1121076532850726, 0.1046948417008579, 0.038926248234284234, 0.06752104977096093, 898.6653046193176, 0.11906103740236142, 46.6282504158748]
    for i in range(len(data)):
        data[i] = (data[i] - mean_list[i])/var_list[i]
    return data

@application.route('/')
def home():
    return render_template('index.html')

@application.route('/predict',methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = scaler(features)
    prediction = model.predict([final_features])

    return render_template('index.html', prediction_text='Danceability is: {}'.format(prediction[0]))

if __name__ == "__main__":
    application.run(debug=True)
