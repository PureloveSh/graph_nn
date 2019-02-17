import numpy as np
import copy
import activators
import executor
from tensorflow.examples.tutorials.mnist import input_data
import generator_point


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
                max_value = matrix[i, j]
                max_i = i
                max_j = j
    return max_i, max_j


# 将tensor填0到指定形状
def padding(input_tensor, zero_padding):
    if zero_padding == 0:
        return input_tensor
    else:
        if input_tensor.ndim == 3:
            input_width = input_tensor.shape[2]
            input_height = input_tensor.shape[1]
            input_depth = input_tensor.shape[0]
            padding_tensor = np.zeros([input_depth, input_height + 2 * zero_padding, input_width + 2 * zero_padding])
            padding_tensor[:, zero_padding:input_height + zero_padding,
            zero_padding:input_width + zero_padding] = input_tensor
            return padding_tensor
        elif input_tensor.ndim == 2:
            input_width = input_tensor.shape[1]
            input_height = input_tensor.shape[0]
            padding_tensor = np.zeros([input_height + 2 * zero_padding, input_width + 2 * zero_padding])
            padding_tensor[zero_padding:input_height + zero_padding,
            zero_padding:input_width + zero_padding] = input_tensor
            return padding_tensor


# 实现卷积运算
def conv(input_tenor, filter_tensor, output_tensor, stride, bias):
    output_height = output_tensor.shape[0]
    output_width = output_tensor.shape[1]
    filter_width = filter_tensor.shape[-1]
    filter_height = filter_tensor.shape[-2]
    for i in range(output_height):
        for j in range(output_width):
            output_tensor[i][j] = np.sum(
                (get_patch(input_tenor, i, j, filter_width, filter_height, stride) * np.rot90(filter_tensor, 2))) + bias


def conv_slice(input_tenor, filter_tensor, output_tensor, stride, bias):
    channel_num = input_tenor.ndim
    output_height = output_tensor.shape[0]
    output_width = output_tensor.shape[1]
    # filter_width = filter_tensor.shape[-1]
    # filter_height = filter_tensor.shape[-2]
    filter_width = filter_tensor.shape[-1]
    filter_height = filter_tensor.shape[-2]

    for i in range(output_height):
        for j in range(output_width):
            output_tensor[i][j] = np.sum(
                (get_patch(input_tenor, i, j, filter_width, filter_height, stride) * np.rot90(filter_tensor, 2))) + bias
    return output_tensor


# 获取卷积运算的区域
def get_patch(input_tensor, i, j, filter_width, filter_height, stride):
    start_i = i * stride
    start_j = j * stride
    if input_tensor.ndim == 2:
        return input_tensor[start_i:start_i + filter_height, start_j:start_j + filter_width]
    elif input_tensor.ndim == 3:
        return input_tensor[:, start_i:start_i + filter_height, start_j:start_j + filter_width]


def expand_sensitivity_map(conn):
    depth = conn.upstream_node.delta.shape[0]
    input_tensor = conn.downstream_node.output
    output_width = ConvLayer.cal_output_size(input_tensor.shape[2], conn.filter_width, 0, conn.stride)
    output_height = ConvLayer.cal_output_size(input_tensor.shape[1], conn.filter_height, 0, conn.stride)
    # 确定扩展后sensitivity map的大小
    # 计算stride为1时sensitivity map的大小
    expand_width = input_tensor.shape[2] - conn.filter_width + 1
    expand_height = input_tensor.shape[1] - conn.filter_height + 1
    # 构建新的sensitivity_map
    expand_tensor = np.zeros((depth, expand_height, expand_width))
    # 从原始sensitivity map拷贝误差值
    for i in range(int(output_height)):
        for j in range(int(output_width)):
            i_pos = i * conn.stride
            j_pos = j * conn.stride
            expand_tensor[:, i_pos, j_pos] = conn.upstream_node.delta[:, i, j]
    return expand_tensor


class ConvLayer(object):
    @staticmethod
    def cal_output_size(input_size, filter_size, zero_padding, stride):
        return int((input_size - filter_size + 2 * zero_padding) / stride + 1)

    @staticmethod
    def forward(conn):
        input_tensor = conn.downstream_node.output
        output_width = ConvLayer.cal_output_size(input_tensor.shape[2], conn.filter_width, 0, conn.stride)
        output_height = ConvLayer.cal_output_size(input_tensor.shape[1], conn.filter_height, 0, conn.stride)
        output_tensor = np.zeros([conn.filter_num, output_height, output_width])
        for i in range(conn.filter_num):
            conv(input_tensor, conn.weights[i], output_tensor[i], conn.stride, conn.bias[i])

        '''
        z = 0.0
        for j in range(conn.filter_num):
            for i in range(conn.downstream_node.output.shape[0]):
                #conv(input_tensor, conn.weights[i], output_tensor[i], conn.stride, conn.bias[i])
                z += conv_slice(input_tensor[i], conn.weights[j][i], output_tensor[j], conn.stride, conn.bias[j])
            output_tensor[j] = z
        '''
        # element_wise(self.output_tensor, self.activator.forward)
        return output_tensor

    @staticmethod
    def backward(conn, activator):
        # 处理卷积步长，对原始sensitivity map进行扩展
        expand_sensitivity_tensor = expand_sensitivity_map(conn)
        # full卷积，对sensitivitiy map进行zero padding
        # 虽然原始输入的zero padding单元也会获得残差
        # 但这个残差不需要继续向上传递，因此就不计算了
        expand_width = expand_sensitivity_tensor.shape[2]
        zp = int((conn.downstream_node.output.shape[2] + conn.filter_width - 1 - expand_width) / 2)
        padded_tensor = padding(expand_sensitivity_tensor, zp)
        # 初始化delta_tensor，用于保存传递到上一层的
        # sensitivity map
        total_delta_tensor = ConvLayer.create_delta_tensor(conn)
        # 对于具有多个filter的卷积层来说，最终传递到上一层的
        # sensitivity map相当于所有的filter的
        # sensitivity map之和
        '''
        z = 0.0

        for d in range(conn.filter_num):
            # 将filter权重翻转180度
            flipped_weights = np.array(list(map(lambda i: np.rot90(i, 2), conn.weights[d])))
            # 计算与一个filter对应的delta_tensor
            #delta_tensor = ConvLayer.create_delta_tensor(conn)
            for i in range(total_delta_tensor.shape[0]):
                #z += conv_slice(padded_tensor[d], flipped_weights[i][d], total_delta_tensor[i], 1, 0)
                z += conv_slice(padded_tensor[d], flipped_weights[i], total_delta_tensor[i], 1, 0)
            # 将计算结果与激活函数的偏导数做element-wise乘法操作
                total_delta_tensor[i] = z
            #total_delta_tensor += delta_tensor
        '''

        for d in range(conn.filter_num):
            flipped_weights = np.array(list(map(lambda i: np.rot90(i, 2), conn.weights[d])))
            # 计算与一个filter对应的delta_tensor
            delta_tensor = ConvLayer.create_delta_tensor(conn)
            for i in range(delta_tensor.shape[0]):
                conv(padded_tensor[d], flipped_weights[i], delta_tensor[i], 1, 0)
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
    def __init__(self, filter_width=None, filter_height=None, filter_depth=None, filter_num=None, stride=None,
                 downstream_node=None, upstream_node=None):
        self.downstream_node = downstream_node
        self.upstream_node = upstream_node
        # for i in range(filter_num):
        #     self.weights.append(np.random.uniform(-1e-3, 1e-3, (filter_depth, filter_height, filter_width)))
        #     self.bias.append(0.1)
        #     self.weights_grad.append(np.zeros(self.weights[i].shape))
        #     self.bias_grad.append(0.0)

    def late_init(self, filter_width, filter_height, filter_depth, filter_num, stride):
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.filter_depth = filter_depth
        self.weights = []
        self.bias = []
        self.weights_grad = []
        self.bias_grad = []
        self.filter_num = filter_num
        self.stride = stride
        for i in range(filter_num):
            self.weights.append(np.random.uniform(-1e-3, 1e-3, (filter_depth, filter_height, filter_width)))
            self.bias.append(0.0)
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

        # expanded_tensor = expand_sensitivity_map(self)
        for f in range(self.filter_num):
            # 计算每个权重的梯度
            for d in range(self.weights[f].shape[0]):
                # z = conv_slice(np.rot90(self.downstream_node.output[d], 2), self.upstream_node.delta[f], self.weights_grad[f][d], 1, 0)
                conv(np.rot90(self.downstream_node.output[d], 2), self.upstream_node.delta[f], self.weights_grad[f][d],
                     1, 0)
                # conv(np.rot90(self.downstream_node.output[d], 2), expanded_tensor[f], self.weights_grad[f][d], 1, 0)
                # self.weights_grad[f][d] = z
            # 计算偏置项的梯度
            self.bias_grad = self.upstream_node.delta[f].sum()

    def update_weight(self):
        self.bp_gradient()
        self.weights -= float(self.upstream_node.learning_rate) * self.weights_grad
        self.bias -= float(self.upstream_node.learning_rate) * self.bias_grad


class MaxPooling:
    @staticmethod
    def forward(conn):
        input_tensor = conn.downstream_node.output
        channel_num = input_tensor.shape[0]
        output_height = int((input_tensor.shape[1] - conn.filter_height) / conn.stride + 1)
        output_width = int((input_tensor.shape[2] - conn.filter_width) / conn.stride + 1)
        output_tensor = np.zeros((channel_num, output_height, output_width))
        for d in range(channel_num):
            for i in range(output_height):
                for j in range(output_width):
                    output_tensor[d, i, j] = np.max(
                        get_patch(input_tensor[d], i, j, conn.filter_width, conn.filter_height, conn.stride))
        return output_tensor

    @staticmethod
    def backward(conn):
        delta_tensor = np.zeros(conn.downstream_node.output.shape)
        input_tensor = conn.downstream_node.output
        channel_num = input_tensor.shape[0]
        output_height = int((input_tensor.shape[1] - conn.filter_height) / conn.stride + 1)
        output_width = int((input_tensor.shape[2] - conn.filter_width) / conn.stride + 1)
        for d in range(channel_num):
            for i in range(output_height):
                for j in range(output_width):
                    new_tensor = get_patch(input_tensor[d], i, j, conn.filter_width, conn.filter_height, conn.stride)
                    row, col = get_max_index(new_tensor)
                    delta_tensor[d, i * conn.stride + row, j * conn.stride + col] = conn.upstream_node.delta[d, i, j]
                    # new_tensor = conn.upstream_node.delta[d, i, j]
                    # print('new_tensor is:', new_tensor.shape)
        return delta_tensor


# 池化线
class PoolingConnection:
    def __init__(self, filter_height=2, filter_width=2, stride=2, upstream_node=None, downstream_node=None):
        self.filter_height = filter_height
        self.filter_width = filter_width
        self.stride = stride
        self.upstream_node = upstream_node
        self.downstream_node = downstream_node

    def bp_gradient(self):
        pass

    def update_weight(self):
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


# Reshape线
class ReshapeConnection:
    def __init__(self, upstream_node=None, downstream_node=None):
        self.upstream_node = upstream_node
        self.downstream_node = downstream_node

    def bp_gradient(self):
        pass

    def update_weight(self):
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
        return conn.upstream_node.delta[depth + 1:depth1, :, :]


# 合并的线
class ConcatConnection:
    def __init__(self, upstream_node, downstream_node):
        self.upstream_node = upstream_node
        self.downstream_node = downstream_node

    def bp_gradient(self):
        pass

    def update_weight(self):
        pass


class FullConnectionLayer:
    @staticmethod
    def forward(conn):
        return np.dot(conn.W * conn.mask, conn.downstream_node.output) + conn.b

    @staticmethod
    def backward(conn, activator):
        return np.dot(conn.W.T, conn.upstream_node.delta) * activator.backward(conn.downstream_node.output)


class TensorConnection:
    def __init__(self, upstream_node=None, downstream_node=None):
        self.upstream_node = upstream_node
        self.downstream_node = downstream_node

    def late_init(self, up_node_num, down_node_num, keep_prob):
        self.W = self.xvarier_init(up_node_num, down_node_num)
        self.b = np.zeros((up_node_num, 1))
        self.keep_prob = keep_prob
        self.control_mask()
        self.W_grad = np.zeros(self.W.shape)
        self.b_grad = np.zeros(self.b.shape)
        # print('init W is ...', self.W)

    def control_mask(self):
        # np.where返回非0的下标但形式为(array(...), array(...),type),取第一维的
        self.mask = np.random.binomial(1, self.keep_prob, size=self.W.shape)
        not_zero_idx = np.where(np.sum(self.mask, axis=1))[0]
        all_idx = np.arange(self.mask.shape[0])
        zero_idx = list(set(not_zero_idx) ^ set(all_idx))
        if not zero_idx:
            for i in zero_idx:
                j = np.random.randint(0, self.mask.shape[1])
                self.mask[i][j] = 1.0

    # 权重初始化器，服从U[-sqrt(6/(输入维度+输出维度)), sqrt(6/(输入维度+输出维度))]
    def xvarier_init(self, fan_in, fan_out, constant=1):
        low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
        high = constant * np.sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(low, high, size=(fan_in, fan_out))

    def bp_gradient(self):
        self.W_grad += np.dot(self.upstream_node.delta, self.downstream_node.output.T)
        self.b_grad += self.upstream_node.delta

    def clear_gradient(self):
        self.W_grad = 0
        self.b_grad = 0

    def update_weight(self, batch_size=1):
        '''
        1- 计算梯度
        2- 更新权重
        '''
        self.bp_gradient()
        self.W -= float(self.upstream_node.learning_rate) * (self.mask * self.W_grad) / batch_size
        self.b -= float(self.upstream_node.learning_rate) * self.b_grad / batch_size


class Node:
    def __init__(self, activator="Linear", learning_rate=1e-3):
        self.downstream = []
        self.upstream = []
        self.activator = activators.Activator_factory(activator).get_activator()
        self.output = 0.0
        self.delta = 0.0
        self.learning_rate = learning_rate

    def late_init(self, activator, learning_rate):
        self.activator = activators.Activator_factory(activator).get_activator()
        self.learning_rate = learning_rate

    def set_output(self, output):
        self.output = output

    def append_downstream(self, conn):
        '''
        :param conn: 边
        :return: 该节点所有下游节点的列表
        '''
        self.downstream.append(conn)

    def append_upstream(self, conn):
        self.upstream.append(conn)

    def cal_output(self):
        self.output = 0.0
        for conn in self.downstream:
            if isinstance(conn, TensorConnection):
                self.output += FullConnectionLayer.forward(conn)
                # print('Full connection:...', self.output)
            elif isinstance(conn, ReshapeConnection):
                self.output += Reshape.forward(conn)
                # print('Reshape connection:...', self.output)
            elif isinstance(conn, PoolingConnection):
                self.output += MaxPooling.forward(conn)
                # print('pooing connection:...', self.output)
            elif isinstance(conn, Filter):
                self.output += ConvLayer.forward(conn)
            elif isinstance(conn, ConcatConnection):
                self.output += Concat.forward(conn)
        self.output = self.activator.forward(self.output)

    def cal_not_output_delta(self):
        self.delta = 0.0
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
        # print('label is ...', label.reshape(10,))
        if isinstance(self.activator, activators.SoftmaxActivator):
            self.delta = self.output - label
        else:
            self.delta = self.activator.backward(self.output) * (self.output - label)


class Network:
    def __init__(self, v, e, problem_type, epochs):
        self.node_collection = v
        self.topo_node_list = executor.Executor(copy.copy(self.node_collection.node_list()), e).topoSort()
        self.problem_type = problem_type
        self.epochs = epochs

    def get_output_node(self):
        for node in self.topo_node_list:
            if not node.upstream:
                return node

    def get_not_input_nodelist(self):
        node_list = [node for node in self.topo_node_list if node.downstream]
        return node_list

    def get_not_output_nodelist(self):
        node_list = [node for node in self.topo_node_list if node.upstream]
        return node_list

    # 获取中间节点，返回列表
    def get_hidden_nodelist(self):
        node_list = [node for node in self.topo_node_list if node.upstream and node.downstream]
        return node_list

    def set_input(self, data):
        '''
        1- 获取输入胶囊（判断的依据是输入胶囊没有边指向该胶囊节点）
        2- 将输入数据设置为胶囊的output值
        '''
        for node in self.topo_node_list:
            if not node.downstream:
                node.set_output(data)
                break

    def cal_delta(self, label):
        # 获取输出胶囊节点
        output_node = self.get_output_node()

        # 断言，判断输出值的维度是否与标签的维度相同
        assert output_node.output.shape[0] == len(label)

        # 计算最后一个胶囊的delta
        output_node.cal_output_delta(label)

        # 计算中间隐层的delta，其中输入胶囊的delta对更新权重是没有用的，所以不去计算输入胶囊的delta
        for node in self.get_hidden_nodelist()[::-1]:
            node.cal_not_output_delta()

    # 更新整个网络的权重
    def update_weight(self):
        node_list = self.get_not_input_nodelist()
        for node in node_list:
            for conn in node.downstream:
                conn.update_weight()

    def train_one_sample(self, data, label):
        output = self.predict(data)
        self.cal_delta(label)
        self.update_weight()
        loss = np.sum(np.square(output - label))
        acc = (np.argmax(output) == np.argmax(label)).astype(np.float16)
        return loss, acc

    def sumup_gradient(self):
        node_list = self.get_not_input_nodelist()
        for node in node_list:
            for conn in node.downstream:
                conn.bp_gradient()

    def train_batch_sample(self, data, label):
        for i in range(data.shape[0]):
            output = self.predict(data[i])
            self.cal_delta(label[i])
            self.sumup_gradient()

            loss = np.sum(np.square(output - label))
            acc = (np.argmax(output) == np.argmax(label)).astype(np.float16)

            return loss, acc

    def get_batch(self, batch_size):
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        xs, ys = mnist.train.next_batch(batch_size)
        return xs, ys

    def test_one_sample(self, data, label):
        output = self.predict(data)
        loss = np.sum(np.square(output - label))
        acc = (np.argmax(output) == np.argmax(label)).astype(np.float16)
        return loss, acc

    # 输入一个输入数据，遍历整个网络计算得到预测值
    def predict(self, data):
        self.set_input(data)
        for node in self.get_not_input_nodelist():
            node.cal_output()
        return self.get_output_node().output

    def train(self):
        mnist = input_data.read_data_sets('/MNIST_data', one_hot=True)
        # data, label = generator_point.Generator_Point().simple_two_class()
        batch_size = 32
        num_batch = mnist.train.num_examples / batch_size
        loss = 0.0
        accuracy = 0.0
        print_epochs = 50
        test_epochs = 100
        for i in range(10):
            for d in range(num_batch):
                l, a = self.train_one_sample(mnist.train.images[d].reshape((1, 28, 28)),
                                             mnist.train.labels[d].reshape((10, 1)))
                # loss = self.train_one_sample(mnist.train.images[d].reshape((784,1)), mnist.train.labels[d].reshape((10,1)))
                loss += l
                accuracy += a
                if (d + 1) % print_epochs == 0:
                    print('Iter: {0} data:{1}  Train loss: {2}  Accuracy:{3}'.format(i + 1, d + 1, loss / print_epochs,
                                                                                     accuracy / print_epochs))
                    loss, accuracy = 0, 0

        for d in range(10000):
            l, a = self.test_one_sample(mnist.test.images[d].reshape((1, 28, 28)), mnist.test.labels.reshape((10, 1)))
            loss += l
            accuracy += a
            if (d + 1) % test_epochs == 0:
                print(
                    'data:{0}  Test loss: {1}  Accuracy:{2}'.format(d + 1, loss / test_epochs, accuracy / test_epochs))
                loss, accuracy = 0, 0

            '''
            for d in range(len(data)):
                #print(data[d], label[d])
                loss = self.train_one_sample(np.array(data[d]).reshape((2, 1)), label[d].reshape((2, 1)))
                print('Iter: {0} data:{1}  Train loss: {2}'.format(i, d, loss))
            '''


if __name__ == "__main__":
    # mnist = input_data.read_data_sets('/MNIST_data', one_hot=True)
    # print(mnist.train.images[0].reshape((28,28)[0]))
    pass
