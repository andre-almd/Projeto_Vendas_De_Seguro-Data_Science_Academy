"""
# Versão do projeto 1 do capítulo 7 do curso Deep Learning II da DSA usando pytorch
@author: André Almeida
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from pickle import dump

'''
# ------------------------- Section 1 - Loading and Normalizing Data -------------------------
'''

# Load training data
dados_treino = pd.read_csv("datasets/vendas_data_training.csv", dtype = np.float32)

# Set X and Y of training
X_treino = dados_treino.drop("total_vendas", axis= 1).values
Y_treino = dados_treino[['total_vendas']].values

# Load test data
dados_teste = pd.read_csv("datasets/vendas_data_test.csv", dtype = np.float32)

# Set X and Y of test
X_teste = dados_teste.drop("total_vendas", axis = 1).values
Y_teste = dados_teste[['total_vendas']].values


# Creating scale operators
X_scaler = MinMaxScaler(feature_range=(0, 1))
Y_scaler = MinMaxScaler(feature_range=(0, 1))

# Scaling training data
X_scaler = X_scaler.fit(X_treino)
Y_scaler = Y_scaler.fit(Y_treino)

'''
# ------------------------- Section 1.1 - Creating datasets with pytorch -------------------------
'''

# Transformer to convert data into tensor when creating dataset
transformer = transforms.Compose([transforms.ToTensor()])

# batch size
batch_size = 128

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None, scaler=False, X_scaler=None, Y_scaler=None):
        self.X = data
        self.y = labels
        self.transform = transform
        
        # pre process to scale the data
        if scaler:
            self.X = X_scaler.transform(self.X)
            self.y = Y_scaler.transform(self.y)
            
        # Transformer to tensor
        self.X = self.transform(self.X).squeeze()
        self.y = self.transform(self.y).reshape(-1, 1)
            

    def __len__(self):
        return (len(self.X))

    def __getitem__(self, i):
        return self.X[i, :], self.y[i]


# Creating the training dataset
train_data = CustomDataset(X_treino, Y_treino, transform=transformer,
                           scaler=True, X_scaler=X_scaler, Y_scaler=Y_scaler)

# Creating the test dataset
test_data = CustomDataset(X_teste, Y_teste, transform=transformer,
                               scaler=True, X_scaler=X_scaler, Y_scaler=Y_scaler)

# Print de um dado no dataset
#print(train_data[0])

# Creating training and testing dataloaders
trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=0)
testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)

# Saving the scalers
dump(X_scaler, open('X_scaler.pkl', 'wb'))
dump(Y_scaler, open('Y_scaler.pkl', 'wb'))

'''
# ------------------------- Section 2 - Model Structure -------------------------
'''

# Hyperparameters
learning_rate = 0.001
num_epochs = 250
#display_step = 5

# Defining inputs and outputs
num_inputs = train_data.X.shape[1]
num_outputs = train_data.y.shape[1]

# layers
layer_1_nodes = 50
layer_2_nodes = 75
layer_3_nodes = 15

'''
# ------------------------- Section 3 - Building Neural Network Layers -------------------------
'''

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # layers
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
    
# Net object
net = Net()

'''
# ------------------------- Section 4 - Cost and Optimization -------------------------
'''

# Loss function
criterion = nn.MSELoss()

# optimizer
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

# Creating summary object for tensorboard
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter(f'runs/treino_curso_DSA/{timestamp}')

# Adding the network on the tensorboard
writer.add_graph(net, train_data.X)

'''
# --------------- Section 5 - Training and Validation with Tensorboard ---------------
'''

# TRain loop
for epoch in range(num_epochs):
    
    '''
    # ---------------------- TRAINING STAGE ----------------------
    '''
    
    # Train mode
    net.train(True)
    
    # variables to store error
    running_loss = 0.
    last_loss = 0.
    
    # loop through dataloader batches
    for i, data in enumerate(trainloader):
        
        #print(f'Batch {i}')
        
        # Input and output data
        inputs, labels = data
        
        # 
        optimizer.zero_grad()
        
        # Make the prediction
        outputs = net(inputs)
        
        # Adding a histogram of predicted training values to the tensorboard
        writer.add_histogram('Predicted Value/Train', outputs, epoch)
        
        # Loss
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Adding the last layer's bias values to the tensorboard
        writer.add_histogram('Bias/Output_Layer', net.fc4.bias, epoch)
        
        # Accumulate batch errors
        running_loss += loss.item()
        
        #print(f'Batch {i} | Loss: {loss.item()}')
        
    epoch_loss = running_loss / len(trainloader)
    
    # Writing training loss value to tensorboard
    writer.add_scalar('Loss/Train', epoch_loss, epoch)
        
    '''
    # ---------------------- VALIDATION STAGE ----------------------
    '''
    
    net.train(False)
    running_vloss = 0.0
    
    with torch.no_grad():
        for i, data in enumerate(testloader):
            
            inputs, labels = data
            outputs = net(inputs)
            
            # Adding to the tensoboard a histogram of the predicted values in the test
            writer.add_histogram('Predicted Value/Test', outputs, epoch)
            
            vloss = criterion(outputs, labels)
            running_vloss += vloss
        
        epoch_vloss = running_vloss / len(testloader)
        
        # Writing test loss value to tensorboard
        writer.add_scalar('Loss/Test', epoch_vloss, epoch)
        
        print(f'Epoch: {epoch} | Train Loss: {epoch_loss:.10f} | Valid Loss: {epoch_vloss:.10f}')
    
    
    
# Making predictions for the test data
net.train(False)

with torch.no_grad():
    y_predicted_scaled = net(test_data.X)

# Rremove the scale
Y_predicted = Y_scaler.inverse_transform(y_predicted_scaled)

# Collect actual data and predicted values
total_vendas_real = dados_teste['total_vendas'].values
total_vendas_previsto = Y_predicted

# Create DataFrame with the results
dados_previsao = pd.DataFrame()
dados_previsao['Real'] = total_vendas_real
dados_previsao['Previsto'] = total_vendas_previsto

# Ensure that all tensoboard events are written to disk
writer.flush()

# Closing tensorboard writer
writer.close()

# Saving the model to the disk for inference
# ---------------------------------------------------------------

# Defining path
PATH = './modelos/modelo_Pytorch.pth'

# saving the model
torch.save(net.state_dict(), PATH)
