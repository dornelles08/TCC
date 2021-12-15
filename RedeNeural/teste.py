import torch
from torch import nn, optim

from torch.utils.data import DataLoader

from os import path, mkdir

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

from Car import Car
from MLP import MLP


def train(train_loader, net, epoch):
    # Training mode
    net.train()

    start = time.time()

    epoch_loss = []
    for batch in train_loader:

        dado, rotulo = batch

        # Cast do dado na GPU
        dado = dado.to(args['device'])
        rotulo = rotulo.to(args['device'])

        # Forward
        ypred = net(dado)
        loss = criterion(ypred, rotulo)
        epoch_loss.append(loss.cpu().data)

        # Backpropagation
        loss.backward()
        optimizer.step()

    epoch_loss = np.asarray(epoch_loss)

    end = time.time()
    print('Epoch %d, Loss: %.4f +/- %.4f, Time: %.2f' %
          (epoch, epoch_loss.mean(), epoch_loss.std(), end-start))

    return epoch_loss.mean()


args = {
    'batch_size': 20,
    'num_workers': 16,
    'lr': 0.0003,
    'weight_decay': 0.3,
    'num_epochs': 1
}

if torch.cuda.is_available():
    args['device'] = torch.device('cuda')
else:
    args['device'] = torch.device('cpu')

print(f"Device: {args['device']}")

file = "testeReduzido.csv"
# file = "testeSk.csv"
# file = "result2.csv"

df = pd.read_csv(file)

train_set = Car(file, df.shape[1]-1)

print(f"Shape: {train_set[0][0].shape}")

train_loader = DataLoader(train_set,
                          args['batch_size'],
                          num_workers=args['num_workers'],
                          shuffle=True)

input_size = train_set[0][0].shape[0]
hidden_size = int((train_set[0][0].shape[0] + 1) / 2)
out_size = 1

net = MLP(input_size, hidden_size, out_size).to(args['device'])

criterion = nn.L1Loss().to(args['device'])

optimizer = optim.Adam(
    net.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

train_losses, test_losses = [], []

# for epoch in range(args['num_epochs']):
#     # Train
#     train_losses.append(train(train_loader, net, epoch))
