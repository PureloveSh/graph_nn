import torch.nn as nn


class Loss:
    def __init__(self, loss_str):
        if loss_str == 'MSE':
            self.loss = nn.MSELoss
        if loss_str == 'CrossEntropy':
            self.loss = nn.CrossEntropyLoss
        if loss_str == 'L1Loss':
            self.loss = nn.L1Loss