import torch
import torch.nn as nn
import torch.nn.functional as functional
from MLP import MLP

modelo = torch.load('modelo_carros')
modelo.eval()

#305,37,4,0,1,9,2017,70000,2,4,1,1,1,1,1,1,1,1,1,1,0,88900

tensor = torch.FloatTensor([305,37,4,0,1,9,2017,70000,2,4,1,1,1,1,1,1,1,1,1,1,0])

valor = modelo.forward(tensor)

print(valor)