import numpy as np
from activators import *
import generator_point

class FullConnectionLayer:
    def __init__(self, input_size, output_size, activator):
        self.input_size = input_size
        self.output_size = output_size
        self.activator = activator

        #self.W = np.random.uniform(-0.1, 0.1, (output_size, input_size))
        self.W = self.xvarier_init(output_size, input_size)
        self.b = np.zeros((output_size, 1))
        self.output = np.zeros((output_size, 1))

    def xvarier_init(self, fan_in, fan_out, constant=1):
        low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
        high = constant * np.sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(low, high, size=(fan_in, fan_out))

    def forward(self, input):
        self.input = input
        self.output = self.activator.forward(np.dot(self.W, input)+self.b)

    def backward(self, delta):
        self.delta = np.dot(self.W.T, delta) * self.activator.backward(self.input)
        self.W_grad = np.dot(delta, self.input.T)
        self.b_grad = delta

    def update(self, lr):
        self.W -= lr*self.W_grad
        self.b -= lr*self.b_grad


class Network:
    def __init__(self, layers, lr, epochs):
        self.layers = []
        self.learning_rate = lr
        self.epochs = epochs
        for i in range(len(layers)-1):
            self.layers.append(FullConnectionLayer(layers[i], layers[i+1], ReluActivator()))

    def predict(self, sample):
        output = np.array(sample).reshape((len(sample), 1))
        for layer in self.layers:
            layer.forward(output)
            output = layer.output
        return output

    def cal_gradient(self, label):
        delta = self.layers[-1].activator.backward(self.layers[-1].output)*(self.layers[-1].output-label)
        for layer in self.layers[::-1]:
            layer.backward(delta)
            delta = layer.delta
        return delta

    def update_weight(self):
        for layer in self.layers:
            layer.update(self.learning_rate)

    def train_one_sample(self, sample, label):
        output = self.predict(sample)
        label = np.array(label).reshape((2, 1))
        self.cal_gradient(label)
        self.update_weight()
        print(np.add.reduce(self.loss(output, label)))
        print(np.argmax(output) == np.argmax(label))
        print(output)

    def train(self, samples, labels):
        for i in range(self.epochs):
            for d in range(len(samples)):
                self.train_one_sample(samples[d], labels[d])

    def loss(self, output, label):
        return 0.5 * np.square(output-label)

if __name__ == '__main__':
    net = Network([2, 6, 4, 2], 1e-2, 100)
    data, label = generator_point.Generator_Point().simple_two_class()
    #data, label = generator_point.Generate_csv().get_samples()
    net.train(data, label)
