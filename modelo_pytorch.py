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
# ------------------------- Seção 1 - Carregando e Normalizando os Dados -------------------------
'''

# Carrega dados de treino
dados_treino = pd.read_csv("datasets/vendas_data_training.csv", dtype = np.float32)

# Define X e Y de treino
X_treino = dados_treino.drop("total_vendas", axis= 1).values
Y_treino = dados_treino[['total_vendas']].values

# Carrega dados de teste
dados_teste = pd.read_csv("datasets/vendas_data_test.csv", dtype = np.float32)

# Define X e Y de teste
X_teste = dados_teste.drop("total_vendas", axis = 1).values
Y_teste = dados_teste[['total_vendas']].values


# Criando operadores de escala
X_scaler = MinMaxScaler(feature_range=(0, 1))
Y_scaler = MinMaxScaler(feature_range=(0, 1))

# Aplicando escala aos dados de treino
X_scaler = X_scaler.fit(X_treino)
Y_scaler = Y_scaler.fit(Y_treino)

'''
# ------------------------- Seção 1.1 - Criando os dataset com pytorch -------------------------
'''

# Transformer para converter os dados em tensor ao criar o dataset
transformer = transforms.Compose([transforms.ToTensor()])

# Tamanho de batch
batch_size = 128

# Classe para o dataset customizado
class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None, scaler=False, X_scaler=None, Y_scaler=None):
        self.X = data
        self.y = labels
        self.transform = transform
        
        # Realizando o pre processamento com o scaler
        if scaler:
            self.X = X_scaler.transform(self.X)
            self.y = Y_scaler.transform(self.y)
            
        # Trasformando o tipo de dado para tensor
        self.X = self.transform(self.X).squeeze()
        self.y = self.transform(self.y).reshape(-1, 1)
            

    def __len__(self):
        return (len(self.X))

    def __getitem__(self, i):
        return self.X[i, :], self.y[i]


# Criando o dataset de treino
train_data = CustomDataset(X_treino, Y_treino, transform=transformer,
                           scaler=True, X_scaler=X_scaler, Y_scaler=Y_scaler)

# Criando o dataset de teste
test_data = CustomDataset(X_teste, Y_teste, transform=transformer,
                               scaler=True, X_scaler=X_scaler, Y_scaler=Y_scaler)

# Print de um dado no dataset
#print(train_data[0])

# Criando os dataloaders de treino e teste
trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=0)
testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)

# Salvando os scalers
dump(X_scaler, open('X_scaler.pkl', 'wb'))
dump(Y_scaler, open('Y_scaler.pkl', 'wb'))

'''
# ------------------------- Seção 2 - Estrutura do Modelo -------------------------
'''

# Hiperparâmetros
learning_rate = 0.001
num_epochs = 250
#display_step = 5

# Definindo inputs e outputs
num_inputs = train_data.X.shape[1]
num_outputs = train_data.y.shape[1]

# Camadas
layer_1_nodes = 50
layer_2_nodes = 75
layer_3_nodes = 15

'''
# ------------------------- Seção 3 - Construindo Camadas da Rede Neural -------------------------
'''

class Net(nn.Module):
    def __init__(self):
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
    
# Criando objeto da rede
net = Net()

'''
# ------------------------- Seção 4 - Custo e Otimização -------------------------
'''

# Função loss
criterion = nn.MSELoss()

# optimizer
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

# Criando objeto do summary para o tensorboard
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter(f'runs/treino_curso_DSA/{timestamp}')

# Adicionando a rede no tensorboard
writer.add_graph(net, train_data.X)

'''
# --------------- Seção 5 - Treinamento e Validação com Tensorboard---------------
'''

# Loop para treino
for epoch in range(num_epochs):
    
    '''
    # ---------------------- ETAPA DE TREINO ----------------------
    '''
    
    # Entra no modo de treino
    net.train(True)
    
    # variáveis para armazenar erro
    running_loss = 0.
    last_loss = 0.
    
    # loop pelos batchs do dataloader
    for i, data in enumerate(trainloader):
        
        #print(f'Batch {i}')
        
        # Par de dados X e y
        inputs, labels = data
        
        # Zerar gradiante para o batch
        optimizer.zero_grad()
        
        # Realizar as predições
        outputs = net(inputs)
        
        # Adicionando ao tensoboard um histograma dos valores preditos no treino
        writer.add_histogram('Predicted Value/Train', outputs, epoch)
        
        # Calcular erro e seus gradientes
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Atualizar os pesos 
        optimizer.step()
        
        # Adicionando ao tensorboard os valores de bias da última camada
        writer.add_histogram('Bias/Output_Layer', net.fc4.bias, epoch)
        
        # Acumular erros batchs
        running_loss += loss.item()
        
        #print(f'Batch {i} | Loss: {loss.item()}')
        
    epoch_loss = running_loss / len(trainloader)
    
    # Escrevendo o valor do loss de treino no tensoboard
    writer.add_scalar('Loss/Train', epoch_loss, epoch)
        
    '''
    # ---------------------- ETAPA DE VALIDAÇÃO ----------------------
    '''
    
    net.train(False)
    running_vloss = 0.0
    
    with torch.no_grad():
        for i, data in enumerate(testloader):
            
            inputs, labels = data
            outputs = net(inputs)
            
            # Adicionando ao tensoboard um histograma dos valores preditos no teste
            writer.add_histogram('Predicted Value/Test', outputs, epoch)
            
            vloss = criterion(outputs, labels)
            running_vloss += vloss
        
        epoch_vloss = running_vloss / len(testloader)
        
        # Escrevendo o valor do loss de teste no tensoboard
        writer.add_scalar('Loss/Test', epoch_vloss, epoch)
        
        print(f'Epoch: {epoch} | Train Loss: {epoch_loss:.10f} | Valid Loss: {epoch_vloss:.10f}')
    
    
    
# Realizando predição para os dados de teste
net.train(False)

with torch.no_grad():
    y_predicted_scaled = net(test_data.X)

# Remove a escala
Y_predicted = Y_scaler.inverse_transform(y_predicted_scaled)

# Coleta os dados reais e os valores previstos
total_vendas_real = dados_teste['total_vendas'].values
total_vendas_previsto = Y_predicted

# Cria DataFrame com os resultados acima
dados_previsao = pd.DataFrame()
dados_previsao['Real'] = total_vendas_real
dados_previsao['Previsto'] = total_vendas_previsto

# Garantir que todos os eventos do tensoboard sejam gravados no disco
writer.flush()

# Fechando writer do tensoboard
writer.close()

# Salvando o modelo no disco para inferência
# ---------------------------------------------------------------

# Definindo o caminho
PATH = './modelos/modelo_Pytorch.pth'

# Gravando no disco
torch.save(net.state_dict(), PATH)



'''
# Etapa desconsiderada durante o processo
# ------------------------------------------------------------------------------
# Testando o onnx para exportar o modelo

dummy_input = train_data.X[0]

input_names = [ "actual_input"]
output_names = [ "output" ]

torch.onnx.export(net, dummy_input, "net.onnx", verbose=True, export_params=True,
                  input_names=input_names, output_names=output_names)

'''