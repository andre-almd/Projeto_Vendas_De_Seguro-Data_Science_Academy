# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 13:45:10 2022

@author: André Almeida
"""

import torch
from flask import Flask, request, jsonify
import numpy as np
from pickle import load

import loadNetwork

# Defining the app according to the Flask documentation
app = Flask(__name__)

# Creating my neural network model object
model = loadNetwork.loadNet()
model.load_state_dict(torch.load('modelos/modelo_Pytorch.pth'))
model.eval()

# Loading the scalers for data pre process
X_scaler = load(open('X_scaler.pkl', 'rb'))
Y_scaler = load(open('Y_scaler.pkl', 'rb'))
    
# Function to perform the prediction
def get_prediction(inputs):
    
    # Pre process
    inputs = X_scaler.transform(inputs)        
    inputs = torch.from_numpy(inputs)
    
    # Make the prediction    
    with torch.no_grad():
        outputs = model(inputs)
        
    outputs = Y_scaler.inverse_transform(outputs)
    
    return outputs


# Run the app with flask
@app.route('/predict', methods=['POST'])
def predict():
    
    if request.method == 'POST':
        # Defining the request input data
        inputs = np.array(request.form.getlist('data'), dtype=np.float32).reshape(1,-1)
        try:
            outputs = get_prediction(inputs)
            
            return jsonify({'Sales amount prediction': outputs.item()})
        
        except:
            return 'Erro durante a predição.'


if __name__ == '__main__':
    app.run()