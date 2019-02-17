import numpy as np


class SigmoidActivator:
    def forward(self, input):
        return 1.0/(1.0+np.exp(-input))

    def backward(self, output):
        return output*(1-output)


class ReluActivator:
    def forward(self, input):
        return np.maximum(0, input)

    def backward(self, output):
        return 1.0 * (output > 0)


class TanhActivator:
    def forward(self, input):
        return 2.0/(1.0+np.exp(-2.0*input))-1.0

    def backward(self, output):
        return 1 - output*output


class LinearActivator:
    def forward(self, input):
        return input

    def backward(self, output):
        return 1.0


class SoftmaxActivator:
    def forward(self, input):
        input = input-max(input)
        return np.exp(input)/np.sum(np.exp(input), axis=0)

    def backward(self, output):
        return 1.0


class Activator_factory:
    def __init__(self, type):
        self.activator = None
        if type == 'Sigmoid':
            self.activator = SigmoidActivator()
        elif type == 'Tanh':
            self.activator = TanhActivator()
        elif type == 'ReLu':
            self.activator = ReluActivator()
        elif type == "Linear":
            self.activator = LinearActivator()
        elif type == 'Softmax':
            self.activator = SoftmaxActivator()

    def get_activator(self):
        return self.activator