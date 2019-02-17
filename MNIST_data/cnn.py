import numpy as np
from functools import reduce
import copy
import activators
import executor
from tensorflow.examples.tutorials.mnist import input_data


def element_wise(tensor, op):
    for i in np.nditer(tensor, op_flags=['readwrite']):
        i[...] = op(i)


def get_max_index(matrix):
    max_i = 0
    max_j = 0
    max_value = matrix[0, 0]
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] > max_value:
                max_i = i
                max_j = j
    return max_i, max_j


def padding(input_tensor, zero_padding):
    if zero_padding == 0:
        return input_tensor
    else:
        if input_tensor.ndim == 3:
            input_width = input_tensor.shape[2]
            input_height = input_tensor.shape[1]
            input_depth = input_tensor.shape[0]
            padding_tensor = np.zeros([input_depth, input_height+2*zero_padding, input_width+2*zero_padding])
            padding_tensor[:, zero_padding:input_height+zero_padding, zero_padding:input_width+zero_padding] = input_tensor
            return padding_tensor
        elif input_tensor.ndim == 2:
            input_width = input_tensor.shape[1]
            input_height = input_tensor.shape[0]
            padding_tensor = np.zeros([input_height+2*zero_padding, input_width+2*zero_padding])
            padding_tensor[zero_padding:input_height+zero_padding, zero_padding:input_width+zero_padding] = input_tensor
            return padding_tensor


def conv(input_tenor, filter_tensor, output_tensor, stride, bias):
    channel_num = input_tenor.ndim
    output_height = output_tensor.shape[0]
    output_width = output_tensor.shape[1]
    filter_width = filter_tensor.shape[-1]
    filter_height = filter_tensor.shape[-2]
    for i in range(output_height):
        for j in range(output_width):
            output_tensor[i][j] = np.sum((get_patch(input_tenor, i, j, filter_width, filter_height, stride)*filter_tensor))+bias


def get_patch(input_tensor, i, j, filter_width, filter_height, stride):
    start_i = i*stride
    start_j = j*stride
    if input_tensor.ndim == 2:
        return input_tensor[start_i:start_i+filter_height, start_j:start_j+filter_width]
    elif input_tensor.ndim == 3:
        return input_tensor[:, start_i:start_i+filter_height, start_j:start_j+filter_width]


def expand_sensitivity_map(conn):
    depth = conn.upstream_node.delta.shape[0]
    input_tensor = conn.downstream_node.output
    output_width = ConvLayer.cal_output_size(input_tensor.shape[2], conn.filter_width, 0, conn.stride)
    output_height = ConvLayer.cal_output_size(input_tensor.shape[1], conn.filter_height, 0, conn.stride)
    expand_width = conn.downstream_node.output.shape[2] - conn.filter_width + 1
    expand_height = conn.downstream_node.output.shape[1] - conn.filter_height + 1
    expand_tensor = np.zeros((depth, expand_height, expand_width))
    for i in range(int(output_height)):
        for j in range(int(output_width)):
            i_pos = i * conn.stride
            j_pos = j * conn.stride
            expand_tensor[:, i_pos, j_pos] = conn.upstream_node.delta[:, i, j]
    return expand_tensor


class ConvLayer(object):
    @staticmethod
    def cal_output_size(input_size, filter_size, zero_padding, stride):
        return (input_size - filter_size + 2*zero_padding)/stride + 1

    @staticmethod
    def forward(conn):
        input_tensor = conn.downstream_node.output
        output_width = ConvLayer.cal_output_size(input_tensor.shape[2], conn.filter_width, 0, conn.stride)
        output_height = ConvLayer.cal_output_size(input_tensor.shape[1], conn.filter_height, 0, conn.stride)
        output_tensor = np.zeros([conn.filter_num, output_height, output_width])
        for i in range(conn.filter_num):
            conv(input_tensor, conn.weights[i], output_tensor[i], conn.stride, conn.bias[i])
        return output_tensor
        #element_wise(self.output_tensor, self.activator.forward)

    @staticmethod
    def backward(conn, activator):
        expand_sensitivity_tensor = expand_sensitivity_map(conn)
        expand_width = expand_sensitivity_tensor.shape[2]
        zp = (conn.downstream_node.output.shape[2] + conn.filter_width - 1 - expand_width) / 2
        padded_tensor = padding(expand_sensitivity_tensor, zp)
        total_delta_tensor = ConvLayer.create_delta_tensor(conn)
        for i in range(conn.filter_num):
            flipped_weights = np.array(map(lambda i: np.rot90(i, 2), conn.weights[i]))
            delta_tensor = ConvLayer.create_delta_tensor(conn)
            for d in range(delta_tensor.shape[0]):
                conv(padded_tensor[i], flipped_weights[d], delta_tensor[d], 1, 0)
            total_delta_tensor += delta_tensor
            derivative_array = np.array(conn.downstream_node.output)
            element_wise(derivative_array, activator.backward)
            total_delta_tensor *= derivative_array
        return total_delta_tensor

    @staticmethod
    def create_delta_tensor(conn):
        input_tensor = conn.downstream_node.output
        return np.zeros(input_tensor.shape)


class Filter(object):
    def __init__(self, filter_width, filter_height, filter_depth, filter_num, stride, downstream_node, upstream_node):
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.filter_depth = filter_depth
        self.weights = []
        self.bias = []
        self.weights_grad = []
        self.bias_grad = []
        self.filter_num = filter_num
        self.stride = stride
        self.downstream_node = downstream_node
        self.upstream_node = upstream_node
        for i in range(filter_num):
            self.weights.append(np.random.uniform(-1e-3, 1e-3, (filter_depth, filter_height, filter_width)))
            self.bias.append(0.1)
            self.weights_grad.append(np.zeros(self.weights[i].shape))
            self.bias_grad.append(0.0)

        self.weights = np.array(self.weights)
        self.bias = np.array(self.bias)
        self.weights_grad = np.array(self.weights_grad)
        self.bias_grad = np.array(self.bias_grad)

    def get_weights(self):
        return self.weights

    def get_bias(self):
        return self.bias

    def bp_gradient(self):
        expanded_tensor = expand_sensitivity_map(self.upstream_node.delta)
        for f in range(self.filter_num):
            # 计算每个权重的梯度
            for d in range(self.weights.shape[0]):
                conv(self.downstream_node.output[d], expanded_tensor[f], self.weights_grad[d], 1, 0)
            # 计算偏置项的梯度
            self.bias_grad = expanded_tensor[f].sum()

    def update_weight(self, learning_rate):
        self.bp_gradient()
        self.weights -= learning_rate*self.weights_grad
        self.bias -= learning_rate*self.bias_grad


class MaxPooling:
    @staticmethod
    def forward(conn):
        input_tensor = conn.downstream_node.output
        channel_num = input_tensor[0]
        output_height = (input_tensor[1]-conn.filter_height)/conn.stride + 1
        output_width = (input_tensor[2]-conn.filter_width)/conn.stride + 1
        output_tensor = np.zeros((channel_num, output_height, output_width))
        for d in range(channel_num):
            for i in range(output_height):
                for j in range(output_width):
                    output_tensor[d, i, j] = np.max(get_patch(input_tensor[d], i, j, conn.filter_width, conn.filter_height, conn.stride))
        return output_tensor

    @staticmethod
    def backward(conn):
        delta_tensor = np.zeros(conn.downstream_node.output.shape)
        input_tensor = conn.downstream_node.output
        channel_num = input_tensor[0]
        output_height = (input_tensor[1] - conn.filter_height) / conn.stride + 1
        output_width = (input_tensor[2] - conn.filter_width) / conn.stride + 1
        for d in range(channel_num):
            for i in range(output_height):
                for j in range(output_width):
                    new_tensor = get_patch(input_tensor[d], i, j, conn.filter_width, conn.filter_height, conn.stride)
                    row, col = get_max_index(new_tensor)
                    delta_tensor[d, i*conn.stride+row, j*conn.stride+col] = conn.upstream_node.delta[d, i, j]
        return delta_tensor


class PoolingConnection:
    def __init__(self, filter_height, filter_width, stride, upstream_node, downstream_node):
        self.filter_height = filter_height
        self.filter_width = filter_width
        self.stride = stride
        self.upstream_node = upstream_node
        self.downstream_node = downstream_node

    def update_weight(self, lr):
        pass


class Reshape:
    @staticmethod
    def forward(conn):
        return conn.downstream_node.output.reshape((-1, 1))

    @staticmethod
    def backward(conn):
        height = conn.downstream_node.output.shape[1]
        width = conn.downstream_node.output.shape[2]
        channel_num = conn.downstream_node.output.shape[0]
        return np.reshape(conn.upstream_node.delta, (channel_num, height, width))


class ReshapeConnection:
    def __init__(self, upstream_node, downstream_node):
        self.upstream_node = upstream_node
        self.downstream_node = downstream_node

    def update_weight(self, lr):
        pass


class Concat:
    @staticmethod
    def forward(conn):
        if conn.upstream_node.output == 0.0:
            conn.upstream_node.output = conn.downstream_node.output
        else:
            conn.upstream_node.output = np.concatenate((conn.upstream_node.output, conn.downstream_node.output), axis=0)

    @staticmethod
    def backward(conn):
        node = conn.upstream_node
        index = node.downstream.find(conn)
        depth = 0
        depth1 = conn.downstream_node.delta.shape[0]
        for i in range(index):
            depth += node.downstream[i].downstream_node.output.shape[0]
        return conn.upstream_node.delta[depth+1:depth1, :, :]


class ConcatConnection:
    def __init__(self, upstream_node, downstream_node):
        self.upstream_node = upstream_node
        self.downstream_node = downstream_node

    def update_weight(self, lr):
        pass


class FullConnectionLayer:
    @staticmethod
    def forward(conn):
        return np.dot(conn.W, conn.downstream_node.ouput) + conn.b

    @staticmethod
    def backward(conn, activator):
        return np.dot(conn.W.T, conn.upstream_node.delta) * activator.backward(conn.downstream_node.output)


class TensorConnection:
    def __init__(self, up_node_num, down_node_num, upstream_node, downstream_node):
        self.upstream_node = upstream_node
        self.downstream_node = downstream_node
        self.W = self.xvarier_init(up_node_num, down_node_num)
        self.b = np.zeros((up_node_num, 1))
        self.W_grad = np.zeros(self.W.shape)
        self.b_grad = np.zeros(self.b.shape)

    def xvarier_init(self, fan_in, fan_out, constant=1):
        low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
        high = constant * np.sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(low, high, size=(fan_in, fan_out))

    def cal_gradient(self):
        self.W_grad = np.dot(self.upstream_node.delta, self.downstream_node.output.T)
        self.b_grad = self.upstream_node.delta

    def update_weight(self, lr):
        self.cal_gradient()
        self.W -= lr*self.W_grad
        self.b -= lr*self.b_grad


class Node:
    def __init__(self, activator):
        self.downstream = []
        self.upstream = []
        self.activator = activator
        self.output = 0.0
        self.delta = 0.0

    def append_downstream(self, conn):
        '''
        :param conn: 节点
        :return: 该节点所有下游节点的列表
        '''
        self.downstream.append(conn)

    def append_upstream(self, conn):
        self.upstream.append(conn)

    def cal_output(self):
        for conn in self.downstream:
            if isinstance(conn, TensorConnection):
                self.output += FullConnectionLayer.forward(conn)
            elif isinstance(conn, ReshapeConnection):
                self.output += Reshape.forward(conn)
            elif isinstance(conn, PoolingConnection):
                self.output += MaxPooling.forward(conn)
            elif isinstance(conn, Filter):
                self.output += ConvLayer.forward(conn)
            elif isinstance(conn, ConcatConnection):
                self.output += Concat.forward(conn)
        self.output = self.activator.forward(self.output)

    def cal_not_output_delta(self):
        for conn in self.upstream:
            if isinstance(conn, TensorConnection):
                self.delta += FullConnectionLayer.backward(conn, self.activator)
            elif isinstance(conn, ReshapeConnection):
                self.delta += Reshape.backward(conn)
            elif isinstance(conn, PoolingConnection):
                self.delta += MaxPooling.backward(conn)
            elif isinstance(conn, Filter):
                self.delta += ConvLayer.backward(conn, self.activator)

    def cal_output_delta(self, label):
        self.delta = self.activator.backward(self.output) * (self.output - label)


class Network:
    def __init__(self, v, e, activation_func, problem_type, epochs, learning_rate):
        self.node_collection = v
        self.topo_node_list = executor.Executor(copy.copy(self.node_collection.node_list()), e).topoSort()
        self.func = activators.Activator_factory(activation_func).get_activator()
        self.problem_type = problem_type
        self.epochs = epochs
        self.learning_rate = learning_rate

    def get_output_nodelist(self):
        node_list = []
        for node in self.topo_node_list:
            if not node.upstream:
                node_list.append(node)
        return node_list

    def get_not_input_nodelist(self):
        node_list = []
        for node in self.topo_node_list:
            if node.downstream:
                node_list.append(node)
        return node_list

    def get_not_output_nodelist(self):
        node_list = []
        for node in self.topo_node_list:
            if node.upstream:
                node_list.append(node)
        return node_list

    def get_hidden_nodelist(self):
        node_list = []
        for node in self.topo_node_list:
            if node.upstream and node.downstream:
                node_list.append(node)
        return node_list

    def set_input(self, data):
        node_list = []
        for node in self.topo_node_list:
            if not node.downstream:
                node_list.append(node)
        #assert len(node_list) == len(data)
        for i in range(len(node_list)):
            node_list[i].set_output(data)

    def cal_delta(self, label):
        output_nodes = self.get_output_nodelist()
        assert len(output_nodes) == len(label)
        loss = 0.0
        for i in range(len(output_nodes)):
            output_nodes[i].cal_output_delta(label, self.func)
            #loss += 0.5*np.square(output_nodes[i].output - label[i])
        for node in self.get_hidden_nodelist()[::-1]:
            node.cal_not_output_delta(self.func)
        return loss

    def update_weight(self):
        #node_list = self.get_not_output_nodelist()
        node_list = self.get_not_input_nodelist()
        for node in node_list:
            #for conn in node.upstream:
            for conn in node.downstream:
                conn.update_weight(self.learning_rate)

    def cal_gradient(self):
        for node in self.get_not_output_nodelist():
            for conn in node.upstream:
                conn.cal_gradient()

    def train_one_sample(self, data, label):
        output = list(self.predict(data))
        loss = self.cal_delta(label)
        self.update_weight()
        #print(data, output, label)
        #print(np.argmax(np.array(output))==np.argmax(label))
        return loss

    def predict(self, data):
        self.set_input(data)
        for node in self.get_not_input_nodelist():
            node.cal_output(self.func)
        return map(lambda neru: neru.output, self.get_output_nodelist())

    def train(self):
        mnist = input_data.read_data_sets('/MNIST_data', one_hot=True)
        for i in range(self.epochs):
            for d in range(60000):
                loss = self.train_one_sample(mnist.train.images[d].reshape((28, 28)), mnist.train.labels[d])
                print('Iter: {0} data:{1}  Train loss: {2}'.format(i, d, loss))



if __name__ == "__main__":
    # mnist = input_data.read_data_sets('/MNIST_data', one_hot=True)
    #print(mnist.train.images[0].reshape((28,28)[0]))
    pass

