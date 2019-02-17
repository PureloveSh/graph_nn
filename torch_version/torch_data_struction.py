import torch_activators
import torch
import numpy as np

class Node:
    def __init__(self, activation_func):
        self.op = 0.0
        self.upstream = []
        self.downstream = []
        self.func = torch_activators.Activator(activation_func).get_activator()


class PartConnection:
    def __init__(self, upstream_node, downstream_node, keep_prob=1.0):
        self.upstream_node = upstream_node
        self.downstream_node = downstream_node
        self.keep_prob = keep_prob

    def late_init(self, up_num, down_num):
        self.up_num = up_num
        self.down_num = down_num
        self.mask = torch.from_numpy(np.random.binomial(1, self.keep_prob, size=(up_num, down_num))).float()


class FullConnnection:
    def __init__(self, upstream_node, downstream_node):
        self.upstream_node = upstream_node
        self.downstream_node = downstream_node


    def late_init(self, up_num, down_num):
        pass
