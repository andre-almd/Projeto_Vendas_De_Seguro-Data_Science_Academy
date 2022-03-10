# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 15:36:40 2022

@author: André Almeida
"""

'''
# ------------------ Módulo para carregar a rede neural na execução do app ------------------
'''

import torch.nn as nn
import torch.nn.functional as F

# Classe do modelo
class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, layer_1_nodes, layer_2_nodes, layer_3_nodes):
        super(Net, self).__init__()
        
        # Camadas da rede
        self.fc1 = nn.Linear(num_inputs, layer_1_nodes)
        self.fc2 = nn.Linear(layer_1_nodes, layer_2_nodes)
        self.fc3 = nn.Linear(layer_2_nodes, layer_3_nodes)
        self.fc4 = nn.Linear(layer_3_nodes, num_outputs)
        
        # Dropout
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x
   
# Definindo inputs e outputs
num_inputs = 9
num_outputs = 1

# Camadas
layer_1_nodes = 50
layer_2_nodes = 75
layer_3_nodes = 15

# Função para criar o objeto da rede
def loadNet():
    net = Net(num_inputs=num_inputs, num_outputs=num_outputs, layer_1_nodes=layer_1_nodes,
              layer_2_nodes=layer_2_nodes, layer_3_nodes=layer_3_nodes)
    return net