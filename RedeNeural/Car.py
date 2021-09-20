import torch
from torch.utils.data import Dataset

import numpy as np
import pandas as pd

class Car(Dataset):
  def __init__(self, csv_path, columns):
    self.dados = pd.read_csv(csv_path).to_numpy()
    self.columns = columns
    # self.dados = self.dados/np.amax(self.dados, axis=0)


  def __getitem__(self, idx):    
    sample = self.dados[idx][:self.columns]
    label = self.dados[idx][-1:]

    #converte pea tensor
    sample = torch.from_numpy(sample.astype(np.float32))
    label = torch.from_numpy(label.astype(np.float32))

    return sample, label

  def __len__(self):
    return len(self.dados)