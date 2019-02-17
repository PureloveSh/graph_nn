import torch.nn.functional as F

class Activator:
    def __init__(self, type):
        if type == 'Sigmoid':
            self.acitvator = F.sigmoid
        elif type == 'Tanh':
            self.acitvator = F.tanh
        elif type == 'ReLu':
            self.acitvator = F.relu
        elif type == 'Softmax':
            self.acitvator = F.softmax
        else:
            self.acitvator = None

    def change_func(self, type):
        if type == 'Sigmoid':
            self.acitvator = F.sigmoid
        elif type == 'Tanh':
            self.acitvator = F.tanh
        elif type == 'ReLu':
            self.acitvator = F.relu
        elif type == 'Softmax':
            self.acitvator = F.softmax
        elif type == 'None':
            self.acitvator = None

    def get_activator(self):
        return self.acitvator