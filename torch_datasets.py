import torch
import torch.utils.data
import pandas as pd
import os
from sklearn.model_selection import train_test_split

class Dataset:
    def __init__(self, type:str, batch_size:int):
        path = os.path.join('./UCI_data/txt', type+'.txt')
        data = pd.read_csv(path).values
        X, y = data[:, 1:], data[:, 0]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        #dataset = torch.utils.data.TensorDataset(X, y)
        train_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X_train).float(), torch.from_numpy(y_train).int())
        test_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X_test).float(), torch.from_numpy(y_test).int())
        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    def get_dataset(self):
        '''
        for data, label in self.train_loader:
            print(data, label)
        '''
        return self.train_loader, self.test_loader




print(Dataset("Iris", 64).get_dataset())
