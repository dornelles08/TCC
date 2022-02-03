import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


def returnTesteAndTrainBataset(base):
    batchSize = int(base.shape[0]/10)

    indices = torch.randperm(len(base)).tolist()

    train_size = int(0.8*len(base))
    df_train = base.iloc[indices[:train_size]]
    df_test = base.iloc[indices[train_size:]]

    onehotencoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(),
                                                     [0, 1, 5, 6, 7, 8])],
                                      remainder='passthrough')

    # Criando Dataset De Treino
    previsores_train = df_train.iloc[:, 0:20].values
    preco_real_train = df_train.iloc[:, 21].values

    previsores_train = onehotencoder.fit_transform(previsores_train).toarray()

    previsores_train = torch.tensor(previsores_train, dtype=torch.float)
    preco_real_train = torch.tensor(
        preco_real_train, dtype=torch.float).view(-1, 1)

    dataset_train = TensorDataset(previsores_train, preco_real_train)

    train_loader = DataLoader(
        dataset_train, batch_size=batchSize, shuffle=True)

    # Criando Dataset De Teste
    previsores_test = df_test.iloc[:, 0:20].values
    preco_real_test = df_test.iloc[:, 21].values

    previsores_test = onehotencoder.fit_transform(previsores_test).toarray()

    previsores_test = torch.tensor(previsores_test, dtype=torch.float)
    preco_real_test = torch.tensor(
        preco_real_test, dtype=torch.float).view(-1, 1)

    dataset_test = TensorDataset(previsores_test, preco_real_test)

    test_loader = DataLoader(
        dataset_test, batch_size=batchSize, shuffle=False)

    return previsores_train.shape[1], train_loader, previsores_test.shape[1], test_loader


base = pd.read_csv('peugeot.csv')

batchSize = int(base.shape[0]/10)

previsores = base.iloc[:, 0:20].values
preco_real = base.iloc[:, 21].values

# onehotencoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(),
#                                                  [0, 1, 5, 6, 7, 8])],
#                                   remainder='passthrough')

# previsores = onehotencoder.fit_transform(previsores).toarray()

# previsores = torch.tensor(previsores, dtype=torch.float)
# preco_real = torch.tensor(preco_real, dtype=torch.float).view(-1, 1)

inputData, train_loader, inputTestData, test_loader = returnTesteAndTrainBataset(
    base)

# inputData = previsores.shape[1]

regressor = nn.Sequential(nn.Linear(inputData, int((inputData+1)/2)),
                          nn.ReLU(),
                          nn.Linear(int((inputData+1)/2),
                                    int((inputData+1)/2)),
                          nn.ReLU(),
                          nn.Linear(int((inputData+1)/2), 1))

criterion = nn.L1Loss()

optimizer = optim.Adam(regressor.parameters())

# dataset = torch.utils.data.TensorDataset(
#     previsores, preco_real)

# train_loader = torch.utils.data.DataLoader(
#     dataset, batch_size=batchSize, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

regressor.to(device)

for epoch in range(2):
    running_loss = 0.
    running_loss_test = 0.

    # Train
    for i, data in enumerate(train_loader):
        inputs, labels = data

        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = regressor.forward(inputs)

        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        print(
            f'\rTRAIN -> Época {epoch+1} - Loop {i + 1} de {len(preco_real)//300}: perda {round(loss.item(), 2)}')

    # Test
    for i, data in enumerate(test_loader):
        inputs, labels = data

        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = regressor.forward(inputs)

        loss = criterion(outputs, labels)

        optimizer.step()

        running_loss_test += loss.item()

        print(
            f'\TEST -> Época {epoch+1} - Loop {i + 1} de {len(preco_real)//300}: perda {loss}')

    print(
        f'ÉPOCA {epoch+1} finalizada: perda {round(running_loss/len(train_loader), 2)}')
