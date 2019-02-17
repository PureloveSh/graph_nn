import numpy as np
from functools import reduce
from connection import Connection
import activators


class Node(object):
    def __init__(self, node_index, activation_func):
        self.node_index = node_index
        self.downstream = []
        self.upstream = []
        self.shortcut_stream = []
        self.output = 0.0
        self.delta = 0.0
        self.down_output = []
        self.shortcut_output = []
        self.upNodeS = []
        self.downNodeS = []
        self.func = activators.Activator_factory(activation_func).get_activator()

    def set_activation_func(self, activation_func):
        self.func = activators.Activator_factory(activation_func).get_activator()

    #添加一个偏置神经元
    def set_biasNode(self):
        self.bias_node = ConstNode()
        conn = Connection(self, self.bias_node)
        self.downstream.append(conn)
        self.bias_node.append_upstream(conn)

    def set_output(self, output):
        self.output = output

    def add_UpNodeS(self, conn):
        self.upNodeS.extend(conn)

    def get_UpNodeS(self):
        return self.upNodeS

    def add_DownNodeS(self, conn):
        self.downNodeS.extend(conn)

    def get_DownNodeS(self):
        return self.downNodeS

    def append_downstream(self, conn):
        '''
        :param conn: 节点
        :return: 该节点所有下游节点的列表
        '''
        self.downstream.append(conn)

    def get_downstream(self):
        return self.downstream

    def get_downstream_output(self):
        self.down_output = []
        for node in self.downstream:
            self.down_output.append(node.output)
        return self.down_output

    def append_upstream(self, conn):
        self.upstream.append(conn)

    def get_upstream(self):
        return self.upstream

    def append_shortcut_stream(self, conn):
        self.shortcut_stream.append(conn)

    def get_shortcut_stream(self):
        return self.shortcut_stream

    def get_shortcut_output(self):
        for node in self.shortcut_stream:
            self.shortcut_output.append(node.output)
        return self.shortcut_output

    def cal_output(self, activator):
        output = reduce(lambda ret, conn: ret + conn.downstream_node.output * conn.weight, self.downstream, 0.0)
        self.output = self.func.forward(output)

    def cal_not_output_delta(self, activator):
        downstream_delta = reduce(lambda ret, conn: ret+conn.upstream_node.delta * conn.weight, self.upstream, 0.0)
        self.delta = self.func.backward(self.output) * downstream_delta
        self.bias_node.cal_not_output_delta(self.func)

    def cal_output_delta(self, label, activator):
        self.delta = activators.backward(self.output) * (self.output-label)
        self.bias_node.cal_not_output_delta(activator)


class ConstNode(object):
    def __init__(self):
        self.upstream = []
        self.output = 1.0
        self.delta = 0.0

    def append_upstream(self, conn):
        self.upstream.append(conn)

    def cal_not_output_delta(self, activator):
        upstream_delta = reduce(lambda ret, conn: ret + conn.upstream_node.delta*conn.weight, self.upstream, 0.0)
        self.delta = activator.backward(self.output)*upstream_delta
