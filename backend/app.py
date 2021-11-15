import numpy as np
from flask import Flask, request, jsonify, render_template
from keras.models import load_model

app = Flask(__name__, template_folder = 'template')
model = load_model('C:\\Users\\alfred5063\\decodevic\\model\\model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(int_features)
    
    output = round(prediction[0], 2)
    
    return render_template('index.html',
                           prediction_test = 'Prediction Percentage $ {}'.format(output))

if __name__ == "__main__":
    app.run(debug = True)