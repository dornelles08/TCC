import pandas as pd
import torch
import numpy as np
import seaborn as sns
from torch import nn, optim
import torch.nn.functional as F

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

base = pd.read_csv('testeSk.csv')

base = base.drop('tipoveiculo', axis=1)

previsores = base.iloc[:, 0:20].values
preco_real = base.iloc[:, 21].values

onehotencoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(),
                                                 [0, 1, 5, 6, 7, 8])],
                                  remainder='passthrough')

previsores = onehotencoder.fit_transform(previsores).toarray()

previsores = torch.tensor(previsores, dtype=torch.float)
preco_real = torch.tensor(preco_real, dtype=torch.float).view(-1, 1)

print(previsores.shape)
print(preco_real.shape)

regressor = nn.Sequential(nn.Linear(2746, 1373),
                          nn.ReLU(),
                          nn.Linear(1373, 1373),
                          nn.ReLU(),
                          nn.Linear(1373, 1))

criterion = nn.L1Loss()

optimizer = optim.Adam(regressor.parameters())

dataset = torch.utils.data.TensorDataset(previsores, preco_real)
train_loader = torch.utils.data.DataLoader(
    dataset, batch_size=300, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

regressor.to(device)

for epoch in range(100):
    running_loss = 0.
    running_mae = 0.

    for i, data in enumerate(train_loader):
        inputs, labels = data

        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = regressor.forward(inputs)

        mae = F.l1_loss(outputs, labels).item()

        running_mae += mae

        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        print('\rÉpoca {:3d} - Loop {:3d} de {:3d}: perda {:06.2f} - MAE {:06.2f}'.format(epoch+1,
                                                                                          i + 1,
                                                                                          len(
                                                                                              preco_real)//300,
                                                                                          loss, mae))
    print('ÉPOCA {:3d} finalizada: perda {:0.5f} - MAE {:0.5f} '.format(epoch+1,
                                                                        running_loss /
                                                                        len(train_loader),
                                                                        running_mae/len(train_loader)))


regressor.eval()

print(preco_real.mean() - preco_real.mean())
