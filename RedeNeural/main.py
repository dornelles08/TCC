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

class RN:
  def __init__(self, lr, wd, epocs, index, prefix, columns):
    self.args={
        'batch_size': 20,
        'num_workers': 8,        
        'lr': lr,
        'weight_decay': wd,
        'num_epochs': epocs
    }
    self.index = index
    self.prefix = prefix
    self.columns = columns
  
  def train(self, train_loader, net, epoch):

    # Training mode
    net.train()
    
    start = time.time()
    
    epoch_loss  = []
    for batch in train_loader:
      
      dado, rotulo = batch
      
      # Cast do dado na GPU
      dado = dado.to(self.args['device'])
      rotulo = rotulo.to(self.args['device'])
      
      # Forward
      ypred = net(dado)    
      loss = self.criterion(ypred, rotulo)
      epoch_loss.append(loss.cpu().data)
      
      # Backpropagation
      loss.backward()
      self.optimizer.step()
    
    epoch_loss = np.asarray(epoch_loss)
    
    end = time.time()
    # print('#################### Train ####################')
    # print('Epoch %d, Loss: %.4f +/- %.4f, Time: %.2f' % (epoch, epoch_loss.mean(), epoch_loss.std(), end-start))
    
    return epoch_loss.mean()

  def validate(self, test_loader, net, epoch):

    # Evaluation mode
    net.eval()
    
    start = time.time()
    
    epoch_loss  = []
    
    with torch.no_grad(): 
      for batch in test_loader:

        dado, rotulo = batch

        # Cast do dado na GPU
        dado = dado.to(self.args['device'])
        rotulo = rotulo.to(self.args['device'])

        # Forward
        ypred = net(dado)
        loss = self.criterion(ypred, rotulo)
        epoch_loss.append(loss.cpu().data)

    epoch_loss = np.asarray(epoch_loss)
    
    end = time.time()
    # print('********** Validate **********')
    # print('Epoch %d, Loss: %.4f +/- %.4f, Time: %.2f\n' % (epoch, epoch_loss.mean(), epoch_loss.std(), end-start))
    
    return epoch_loss.mean()

  def run(self, arq):
    #Definição do dispositivo de execução, cpu ou gpu
    if torch.cuda.is_available():
      self.args['device'] = torch.device('cuda')
    else:
      self.args['device'] = torch.device('cpu')

    df = pd.read_csv(arq)

    indices = torch.randperm(len(df)).tolist()

    train_size = int(0.8*len(df))
    df_train = df.iloc[indices[:train_size]]
    df_test = df.iloc[indices[train_size:]]

    df_train.to_csv('car_train.csv', index=False)
    df_test.to_csv('car_test.csv', index=False)

    train_set = Car('car_train.csv', self.columns)
    test_set  = Car('car_test.csv', self.columns)

    # Criando dataloader
    train_loader = DataLoader(train_set,
                              self.args['batch_size'],
                              num_workers=self.args['num_workers'],
                              shuffle=True)
    test_loader = DataLoader(test_set,
                            self.args['batch_size'],
                            num_workers=self.args['num_workers'],
                            shuffle=False)

    input_size  = train_set[0][0].size(0)
    hidden_size = 128
    out_size    = 1

    net = MLP(input_size, hidden_size, out_size).to(self.args['device'])

    self.criterion = nn.L1Loss().to(self.args['device'])

    self.optimizer = optim.Adam(net.parameters(), lr=self.args['lr'], weight_decay=self.args['weight_decay'])

    train_losses, test_losses = [], []

    start = time.time()

    for epoch in range(self.args['num_epochs']):    
      # Train
      train_losses.append(self.train(train_loader, net, epoch))
      
      # Validate
      test_losses.append(self.validate(test_loader, net, epoch))

    end = time.time()

    Xtest = torch.stack([tup[0] for tup in test_set])
    Xtest = Xtest.to(self.args['device'])

    ytest = torch.stack([tup[1] for tup in test_set])
    ypred = net(Xtest).cpu().data

    data = torch.cat((ytest, ypred), axis=1)

    df_results = pd.DataFrame(data, columns=['ypred', 'ytest'])
    df_results.to_csv('pred.csv', index=False)

    print(end-start, str(min(train_losses)))

    if not path.exists("results-"+str(self.prefix)):
      mkdir("results-"+str(self.prefix))

    f = open(f"results-{str(self.prefix)}/results.txt", "a+")
    string = f"{str(self.index)} {str(min(train_losses))}"
    f.write(string+"\n")
    f.close()

    path = f"results-{str(self.prefix)}/result-{str(self.index)}.txt"
    f = open(path, "w+")
    f.write(str(self.args)+"\n")
    f.write(df_results.to_string()+"\n")
    f.close()

lrs = [
      1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5, 9e-5, 
      1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 
      1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3, 9e-3, 
      1e-2, 2e-2, 3e-2, 4e-2, 5e-2, 6e-2, 7e-2, 8e-2, 9e-2, 
      1e-1, 2e-1, 3e-1, 4e-1, 5e-1, 6e-1, 7e-1, 8e-1, 9e-1
      ]

wds = [
      1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5, 9e-5, 
      1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 
      1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3, 9e-3, 
      1e-2, 2e-2, 3e-2, 4e-2, 5e-2, 6e-2, 7e-2, 8e-2, 9e-2, 
      1e-1, 2e-1, 3e-1, 4e-1, 5e-1, 6e-1, 7e-1, 8e-1, 9e-1
      ]


# lrs = [1e-05, 2e-05, 3e-05, 4e-05, 5e-05, 6e-05, 7e-05, 8e-05, 9e-05, 0.0001] 
# wds = [0.0008, 0.01, 5e-05, 8e-05, 9e-05, 0.0002, 0.0004, 0.007, 0.02, 0.7, 1e-05, 4e-05, 0.06, 0.07, 0.08, 0.1, 0.5, 0.003, 0.004, 0.005, 0.006, 0.03, 0.05, 0.4, 6e-05, 0.009, 0.001, 0.0006, 2e-05]

# lrs = [7e-05]
# wds = [0.02]
total = 0

prefix = input("Pre fixo da execução: ")

df = pd.read_csv("result.csv")
print(df.shape)

for lr in lrs:
  for wd in wds:      
    rn = RN(lr, wd, 150, total, prefix, df.shape[1])
    rn.run("result.csv")
    total += 1
