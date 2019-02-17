import numpy as np
import random


class Connection:

    def __init__(self, upstream_node, downstream_node):
        '''

        :param upstream_node:该连接得上游节点
        :param downstream_node:该连接的下游节点
        '''
        self.upstream_node = upstream_node
        self.downstream_node = downstream_node
        self.weight = random.uniform(-1, 1)
        self.gradient = 0.0

    def cal_gradient(self):
        self.gradient = self.upstream_node.delta*self.downstream_node.output

    def update_weight(self, lr):
        self.cal_gradient()
        self.weight -= np.float64(lr)*self.gradient

    def cal_gradient_batch(self):
        self.gradient += self.upstream_node.delta*self.downstream_node.output

    def update_weigh_batch(self, lr, batch_size, optimizer):
        self.gradient /= batch_size
        if optimizer == 'momentum':
            if hasattr(self, 'v'):
                self.v = 0.9 * self.v - np.float(lr) * self.gradient
                self.weight += self.v
            else:
                self.v = 0
        elif optimizer == 'adam':
            if hasattr(self, 'v'):
                self.v = 0.9 * self.v + 0.1 * self.gradient
                self.r = 0.999 * self.r + 0.001 * np.square(self.gradient)
                v_hat = self.v/(1 - np.power(0.9, self.step))
                r_hat = self.r/(1 - np.power(0.999, self.step))
                self.weight -= np.float64(lr) * v_hat/(np.square(r_hat)+1e-8)
                self.step += 1
            else:
                self.v = 0.0
                self.r = 0.0
                self.step = 0
        elif optimizer == 'RMSprop':
            if hasattr(self, 'r'):
                self.r = 0.9 * self.r + 0.1 * np.square(self.gradient)
                self.weight -= np.float64(lr) * self.gradient / np.sqrt(1e-6+self.r)
            else:
                self.r = 0.0
        elif optimizer == 'AdaGrad':
            if hasattr(self, 'r'):
                self.r = self.r + np.square(self.gradient)
                self.weight -= np.float64(lr) * self.gradient / (1e-7+np.sqrt(self.r))
            else:
                self.r = 0.0
        else:
            self.weight -= np.float64(lr)*self.gradient
        self.gradient = 0.0

    def get_gradient(self):
        return self.gradient


class Connections:
    def __init__(self):
        self.connections = []

    def add_connection(self, connection):
        self.connections.append(connection)