import torch
from torch.utils.data import Dataset

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


class Car(Dataset):
    def __init__(self, csv_path, columns):
        self.dados = pd.read_csv(csv_path).to_numpy()
        # self.dados = pd.read_csv(csv_path)
        self.columns = columns

    def __getitem__(self, idx):
        # sample = self.dados.iloc[:, 0:20].values
        # label = self.dados.iloc[:, 21].values

        sample = self.dados[idx][:self.columns]
        label = self.dados[idx][-1:]

        onehotencoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(),
                                                         [0, 1, 5, 6, 7, 8])],
                                          remainder='passthrough')

        sample = onehotencoder.fit_transform(sample).toarray()

        # converte pea tensor
        # sample = torch.tensor(sample, dtype=torch.float)
        # label = torch.tensor(label, dtype=torch.float).view(-1, 1)

        sample = torch.from_numpy(sample.astype(np.float32))
        label = torch.from_numpy(label.astype(np.float32))

        return sample, label

    def __len__(self):
        return len(self.dados)
