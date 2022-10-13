from os import listdir
from os.path import isfile, join
from pathlib import Path
import random
import time

import torch
from torch import nn, optim
from torch.nn.functional import normalize
from torch.utils.data import DataLoader, Dataset

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


class Car(Dataset):
    def __init__(self, csv_path, columns):
        self.dados = pd.read_csv(csv_path).to_numpy()
        self.columns = columns

    def __getitem__(self, idx):
        sample = self.dados[idx][:self.columns]
        label = self.dados[idx][-1:]

        sample = torch.from_numpy(sample.astype(np.float32))
        label = torch.from_numpy(label.astype(np.float32))

        # sample = normalize(sample, p=2.0, dim=0)

        return sample, label

    def __len__(self):
        return len(self.dados)

    def getRotulo(self):
        rotulos = []
        for dado in self.dados:
            rotulos.append(dado[-1:][0])
        return rotulos


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super(MLP, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, out_size),
            nn.ReLU(),
        )

    def forward(self, X):

        hidden = self.features(X)
        output = self.classifier(hidden)

        return output


def train(train_loader, net, epoch):
    # Training mode
    net.train()
    epoch_loss = []

    for batch in train_loader:
        dado, rotulo = batch

        # Cast do dado na GPU
        dado = dado.to(args['device'])
        rotulo = rotulo.to(args['device'])

        optimizer.zero_grad()

        # Forward
        ypred = net(dado)

        loss = criterion(ypred, rotulo)

        percent = (abs(rotulo-ypred) / rotulo)*100

        for i in range(len(rotulo)):
            abs_dif_train.append(abs((rotulo[i].item())-(ypred[i].item())))

        for p in percent:
            percent_train.append(p.item())

        dif = diferenca(ypred, rotulo)
        dif_train.append(dif.cpu().data)

        epoch_loss.append(loss.cpu().data)

        # Backpropagation
        loss.backward()
        optimizer.step()

    epoch_loss = np.asarray(epoch_loss)

    return epoch_loss.mean()


def validate(test_loader, net, epoch):
    # Evaluation mode
    net.eval()
    epoch_loss = []
    epoch_dif = []

    with torch.no_grad():
        for batch in test_loader:
            dado, rotulo = batch

            # Cast do dado na GPU
            dado = dado.to(args['device'])
            rotulo = rotulo.to(args['device'])

            optimizer.zero_grad()

            # Forward
            ypred = net(dado)
            loss = criterion(ypred, rotulo)

            percent = (abs(rotulo-ypred) / rotulo)*100

            for i in range(len(rotulo)):
                abs_dif_test.append(abs((rotulo[i].item())-(ypred[i].item())))

            for p in percent:
                percent_test.append(p.item())

            dif = diferenca(ypred, rotulo)
            dif_test.append(dif.cpu().data)

            epoch_dif.append(dif.cpu().data)
            epoch_loss.append(loss.cpu().data)

    epoch_loss = np.asarray(epoch_loss)
    epoch_dif = np.asarray(epoch_dif)

    return epoch_loss.mean()


args = {
    'batch_size': 100,
    'num_workers': 16,
    'epoch_num': 300,
}

if torch.cuda.is_available():
    args['device'] = torch.device('cuda')
else:
    args['device'] = torch.device('cpu')

# mypath = './dados'

# files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

files = [
    {"file": "se_corolla.csv", "minValue": 13000},
    {"file": "sp_ka.csv", "minValue": 4500},
    {"file": "sp_hb20.csv", "minValue": 5500},
    {"file": "sp_fit.csv", "minValue": 11000},
    {"file": "se_onix.csv", "minValue": 11000},
    {"file": "se_completo.csv", "minValue": 23000},
    {"file": "sp_completo.csv", "minValue": 21000},
]

for item in files:
    print(item)
    file = item["file"]
    print(file)

    prep_data_inicio = time.time()
    df = pd.read_csv(f'dados/{file}')
    indices = torch.randperm(len(df)).tolist()

    train_size = int(0.8*len(df))
    print('Separando em dados de treino e de teste')
    df_train = df.iloc[indices[:train_size]]
    df_test = df.iloc[indices[train_size:]]

    df_train.to_csv('car_train.csv', index=False)
    df_test.to_csv('car_test.csv', index=False)

    print('Criando DataLoaders')
    train_set = Car('car_train.csv', df.shape[1]-1)
    test_set = Car('car_test.csv', df.shape[1]-1)

    train_loader = DataLoader(train_set,
                              args['batch_size'],
                              num_workers=args['num_workers'],
                              shuffle=True)

    test_loader = DataLoader(test_set,
                             args['batch_size'],
                             num_workers=args['num_workers'],
                             shuffle=False)

    prep_data_fim = time.time()
    print(f"Tempo preparando dados: {prep_data_fim-prep_data_inicio}")

    config_inicio = time.time()
    print("Criando Rede")
    input_size = train_set[0][0].shape[0]
    hidden_size = (train_set[0][0].shape[0] + 1)
    out_size = 1

    net = MLP(input_size, hidden_size, out_size).to(args['device'])

    criterion = nn.L1Loss().to(args['device'])
    diferenca = nn.L1Loss().to(args['device'])

    optimizer = optim.Adadelta(net.parameters())

    dif_train, dif_test = [], []
    percent_train, percent_test = [], []
    abs_dif_train, abs_dif_test = [], []

    config_fim = time.time()
    print(f"Tempo Configurando Parametros: {config_fim-config_inicio}")

    print('Treinamento')
    train_losses, test_losses = [], []
    start = time.time()

    for epoch in range(args['epoch_num']):
        # print(f"Epoca: {epoch}")
        # Train
        train_losses.append(train(train_loader, net, epoch))
        # Validate
        test_losses.append(validate(test_loader, net, epoch))

    end = time.time()
    print(f"Tempo de Treinamento e Testes: {end-start}")

    start = time.time()

    train_losses = np.asarray(train_losses)
    test_losses = np.asarray(test_losses)

    percent_train = np.asarray(percent_train)
    percent_test = np.asarray(percent_test)

    abs_dif_train = np.asarray(abs_dif_train)
    abs_dif_test = np.asarray(abs_dif_test)

    dif_train = np.asarray(dif_train)
    dif_test = np.asarray(dif_test)

    print(f''' 
Tamanho List  -> {len(df)}
Percentual    -> {percent_train.mean()}
LossMedia     -> {train_losses.mean()}
LossAbsMedia  -> {dif_train.mean()}
    ''')

    fileName = file.split('.')[0]
    Path(f'./resultados/{fileName}').mkdir(exist_ok=True)

    if not isfile(f'./resultados/{fileName}.txt'):
        Path(f'./resultados/{fileName}.txt').touch(exist_ok=True)

    with open(f'./resultados/{fileName}.txt', 'a') as arq:
        arq.write(f'{train_losses.mean()}\n')

    if train_losses.mean() > item["minValue"]:
        print("Repetindo...")
        print("---------------------------------------------------------------------------")
        print()
        files.append(item)
        continue

    pd.Series(train_losses).to_csv(
        f'./resultados/{fileName}/train_loss_epoch_{fileName}.csv', index=False, header=False)
    pd.Series(test_losses).to_csv(
        f'./resultados/{fileName}/test_loss_epoch_{fileName}.csv', index=False, header=False)

    pd.Series(dif_train).to_csv(
        f'./resultados/{fileName}/train_loss_{fileName}.csv', index=False, header=False)
    pd.Series(dif_test).to_csv(
        f'./resultados/{fileName}/test_loss_{fileName}.csv', index=False, header=False)

    pd.Series(abs_dif_train).to_csv(
        f'./resultados/{fileName}/abs_train_loss_{fileName}.csv', index=False, header=False)
    pd.Series(abs_dif_test).to_csv(
        f'./resultados/{fileName}/abs_test_loss_{fileName}.csv', index=False, header=False)

    torch.save(net, f'./resultados/{fileName}/modelo_{fileName}')

    results = f'''
Quantidade de registros no Dataset: {len(df)}

Porcentagem Média de Treino: {percent_train.mean()}
Porcentagem Média de Teste: {percent_test.mean()}

Menor Valor de Loss por Época de Treino: {min(train_losses)}
Maior Valor de Loss por Época de Treino: {max(train_losses)}
Valor Médio de Loss por Época de Treino: {train_losses.mean()}

Menor Valor de Loss por Época de Teste: {min(test_losses)}
Maior Valor de Loss por Época de Teste: {max(test_losses)}
Valor Médio de Loss por Época de Teste: {test_losses.mean()}

Menor Valor de Loss por Registro de Treino: {min(dif_train)}
Maior Valor de Loss por Registro de Treino: {max(dif_train)}
Valor Médio de Loss por Registro de Treino: {dif_train.mean()}

Menor Valor de Loss por Registro de Teste: {min(dif_test)}
Maior Valor de Loss por Registro de Teste: {max(dif_test)}
Valor Médio de Loss por Registro de Teste: {dif_test.mean()}

Menor Valor de Abs Loss por Registro de Treino: {min(abs_dif_train)}
Maior Valor de Abs Loss por Registro de Treino: {max(abs_dif_train)}
Valor Médio de Abs Loss por Registro de Treino: {abs_dif_train.mean()}

Menor Valor de Abs Loss por Registro de Teste: {min(abs_dif_test)}
Maior Valor de Abs Loss por Registro de Teste: {max(abs_dif_test)}
Valor Médio de Abs Loss por Registro de Teste: {abs_dif_test.mean()}
'''

    if not isfile(f'./resultados/{fileName}/{fileName}.txt'):
        Path(f'./resultados/{fileName}/{fileName}.txt').touch(exist_ok=True)

    with open(f'./resultados/{fileName}/{fileName}.txt', 'w') as arq:
        arq.write(results)

    end = time.time()
    print(f"Tempo de Salvar informações: {end-start}")
