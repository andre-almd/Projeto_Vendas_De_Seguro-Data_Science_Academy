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

# Definindo o app conforme documentação Flask
app = Flask(__name__)

# Criando o objeto do meu modelo da rede neural
model = loadNetwork.loadNet()
model.load_state_dict(torch.load('modelos/modelo_Pytorch.pth'))
model.eval()

# Carregando os scalers para transformação dos dados
X_scaler = load(open('X_scaler.pkl', 'rb'))
Y_scaler = load(open('Y_scaler.pkl', 'rb'))
    
# Função que realizará a predição 
def get_prediction(inputs):
    
    # Tranformação do dado de entrada com o scaler
    inputs = X_scaler.transform(inputs)
    # Transformação do input para tensor        
    inputs = torch.from_numpy(inputs)
    
    # Executando a predição        
    with torch.no_grad():
        outputs = model(inputs)
    # Tranformando o valor de y para a escala real  
    outputs = Y_scaler.inverse_transform(outputs)
    return outputs


# Método de execução do app com flask
@app.route('/predict', methods=['POST'])
def predict():
    
    if request.method == 'POST':
        # Definindo o dado de entrada do request
        inputs = np.array(request.form.getlist('data'), dtype=np.float32).reshape(1,-1)
        try:
            # Chamando a função que realiza a predição
            outputs = get_prediction(inputs)
            
            # Return com valor da predição 
            return jsonify({'Sales amount prediction': outputs.item()})
        except:
            return 'Erro durante a predição.'


if __name__ == '__main__':
    app.run()