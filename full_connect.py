import torch
from torch.autograd import Variable
from torch_activators import Activator

class Layer:
    def __init__(self, num: int, activatior: str):
        self.num_node = num
        self.str_activator = activatior
        self.activator = Activator(activatior).get_activator()
        self.output = None
        self.next_connection = None
        self.pre_connection = None

    def set_activator(self, activator:str):
        self.activator = Activator(activator).get_activator()

    def forward(self):
        if self.activator and self.next_connection:
            self.output = self.activator(self.pre_connection.linear(self.pre_connection.pre_layer.output))
        elif (self.str_activator == 'Softmax' and self.next_connection is None) or self.str_activator == 'Linear':
            self.output = self.next_connection.linear(self.pre_connection.pre_layer.output)


class Connection:
    def __init__(self, keep_prob=1.0):
        self.pre_layer = None
        self.next_layer = None
        self.keep_prob = keep_prob
        self.linear = torch.nn.Linear(self.pre_layer.num_node, self.next_layer.num_node)


class Model():
    def __init__(self, layer_list, epochs, optimizer):
        self.epochs = epochs
        self.layer_list = layer_list
        self.optimizer = optimizer
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self):
        for layer in self.layer_list:
            layer.forward()
        return self.layer_list[-1].output


    def train(self, Y):
        loss = self.criterion(self.layer_list[-1].output, Y)
        loss.backward()


