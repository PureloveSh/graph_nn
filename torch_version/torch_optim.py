import torch.optim as optim

class Optimizer:
    def __init__(self, optimizer:str):
        if optimizer == 'SGD':
            self.optimizer = optim.SGD
        if optimizer == 'Adam':
            self.optimizer = optim.Adam
        if optimizer == 'Adadelta':
            self.optimizer = optim.Adadelta
        if optimizer == 'RMSprop':
            self.optimizer = optim.RMSprop
        if optimizer == 'Adagrad':
            self.optimizer = optim.Adagrad
        if optimizer == 'Adamax':
            self.optimizer = optim.Adamax