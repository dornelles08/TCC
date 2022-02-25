from os import listdir
from os.path import isfile, join
from pathlib import Path
import random
import time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


class Car(Dataset):
    def __init__(self, csv_path, columns):
        self.dados = pd.read_csv(csv_path).to_numpy()
        self.columns = columns

    def __getitem__(self, idx):
        sample = self.dados[idx][:self.columns]
        label = self.dados[idx][-1:]

        sample = torch.from_numpy(sample.astype(np.float32))
        label = torch.from_numpy(label.astype(np.float32))

        return sample, label

    def __len__(self):
        return len(self.dados)


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
    epoch_dif = []

    for batch in train_loader:
        dado, rotulo = batch

        # Cast do dado na GPU
        dado = dado.to(args['device'])
        rotulo = rotulo.to(args['device'])

        optimizer.zero_grad()

        # Forward
        ypred = net(dado)
        loss = criterion(ypred, rotulo)

        dif = diferenca(ypred, rotulo)
        dif_train.append(dif.cpu().data)

        epoch_dif.append(dif.cpu().data)
        epoch_loss.append(loss.cpu().data)

        # Backpropagation
        loss.backward()
        optimizer.step()

    epoch_loss = np.asarray(epoch_loss)
    epoch_dif = np.asarray(epoch_dif)

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
    'epoch_num': 10,
}

if torch.cuda.is_available():
    args['device'] = torch.device('cuda')
else:
    args['device'] = torch.device('cpu')


mypath = './dados'

files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

files = files[0:1]

for file in files:
    print(file)
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

    print("Criando Rede")
    input_size = train_set[0][0].shape[0]
    hidden_size = int((train_set[0][0].shape[0] + 1) / 2)
    out_size = 1

    net = MLP(input_size, hidden_size, out_size).to(args['device'])

    criterion = nn.L1Loss().to(args['device'])
    diferenca = nn.L1Loss().to(args['device'])

    optimizer = optim.Adam(net.parameters())

    dif_train, dif_test = [], []

    print('Treinamento')
    train_losses, test_losses = [], []
    start = time.time()

    for epoch in range(args['epoch_num']):
        # Train
        train_losses.append(train(train_loader, net, epoch))

        # Validate
        test_losses.append(validate(test_loader, net, epoch))

    end = time.time()
    print(end-start)

    file = file.split('.')[0]

    Path(f'./resultados/{file}').mkdir(exist_ok=True)

    plt.figure(figsize=(20, 10))
    plt.plot(train_losses, label='Train')
    plt.plot(test_losses, label='Test', linewidth=3, alpha=0.5)
    plt.xlabel('Épocas', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title('Convergência', fontsize=16)
    plt.legend()
    plt.savefig(f'./resultados/{file}/epochs_loss_{file}.png', format='png')

    plt.figure(figsize=(20, 10))
    plt.plot(random.sample(dif_train, 100), label='Train')
    plt.xlabel('Testes', fontsize=16)
    plt.ylabel('Diferença', fontsize=16)
    plt.title('Convergence Treino', fontsize=16)
    plt.legend()
    plt.savefig(
        f'./resultados/{file}/car_loss_train_{file}.png', format='png')

    plt.figure(figsize=(20, 10))
    plt.plot(random.sample(dif_test, 100),
             label='Test', linewidth=3, alpha=0.5)
    plt.xlabel('Testes', fontsize=16)
    plt.ylabel('Diferença', fontsize=16)
    plt.title('Convergence Teste', fontsize=16)
    plt.legend()
    plt.savefig(
        f'./resultados/{file}/car_loss_test_{file}.png', format='png')
