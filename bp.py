import numpy as np
import executor
import copy
import activators
from node import *
import generator_point
from sklearn.model_selection import train_test_split, KFold


class Network:
    def __init__(self, v, e, activation_func, problem_type, epochs, learning_rate):
        self.node_collection = v
        self.topo_node_list = executor.Executor(
            copy.copy(self.node_collection.node_list()), e).topoSort()
        self.func = activators.Activator_factory(activation_func).get_activator()
        self.problem_type = problem_type
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.loss = []
        self.accuracy = []
        for node in self.get_not_input_nodelist():
            node.set_biasNode()

    def csv_data(self):
        data, label = generator_point.Generate_csv().read_csv()
        return data, label

    def get_output_nodelist(self):
        node_list = [node for node in self.topo_node_list if not node.upstream]
        return node_list

    def get_not_input_nodelist(self):
        node_list = [node for node in self.topo_node_list if node.downstream]
        return node_list

    def get_not_output_nodelist(self):
        node_list = [node for node in self.topo_node_list if node.upstream]
        return node_list

    def get_hidden_nodelist(self):
        node_list = [node for node in self.topo_node_list if node.upstream and node.downstream]
        return node_list

    def set_input(self, data):
        # 获取输入节点神经元，放入列表中
        node_list = [node for node in self.topo_node_list if not node.downstream]
        # 断言输入节点数量和输入数据的维度是否一致
        assert len(node_list) == len(data)
        # 数量和维度一一对应之后，为每一个节点设置输入数据
        for i in range(len(data)):
            node_list[i].set_output(data[i])

    def cal_delta(self, label):
        # 获取输出节点神经元
        output_nodes = self.get_output_nodelist()
        # 判断维度和神经元个数是否一致
        assert len(output_nodes) == len(label)
        loss = 0.0
        # 输出节点的delta计算
        for i in range(len(label)):
            output_nodes[i].cal_output_delta(label[i], self.func)
            loss += 0.5 * np.square(output_nodes[i].output - label[i])
        # 非输出节点的delta计算
        for node in self.get_hidden_nodelist()[::-1]:
            node.cal_not_output_delta(self.func)
        return loss

    def update_weight(self):
        # node_list = self.get_not_output_nodelist()
        node_list = self.get_not_input_nodelist()
        for node in node_list:
            # for conn in node.upstream:
            for conn in node.downstream:
                conn.update_weight(self.learning_rate)

    def cal_gradient(self):
        for node in self.get_not_output_nodelist():
            for conn in node.upstream:
                conn.cal_gradient()

    def predict(self, data):
        self.set_input(data)
        for node in self.get_not_input_nodelist():
            node.cal_output(self.func)
        return map(lambda neru: neru.output, self.get_output_nodelist())

    def train_one_sample(self, data, label):
        output = list(self.predict(data))
        loss = self.cal_delta(label)
        self.update_weight()
        return loss

    def update_weight_batch(self, batch_size):
        node_list = self.get_not_input_nodelist()
        for node in node_list:
            for conn in node.downstream:
                conn.update_weigh_batch(self.learning_rate, batch_size)

    def cal_gradient_batch(self):
        for node in self.get_not_output_nodelist():
            for conn in node.upstream:
                conn.cal_gradient_batch()

    def train_batch_samples(self, data_set, labels):
        # data_set和labels是二维的ndarry数据
        batch_size = data_set.shape[0]
        for i in range(batch_size):
            self.predict(data_set[i])
            self.cal_delta(labels[i])
            self.cal_gradient_batch()
        self.update_weight_batch(batch_size)

    def test(self, data_set, labels):
        test_size = data_set.shape[0]
        loss = 0.0
        accuracy = 0.0
        for i in range(test_size):
            y_hat = list(self.predict(data_set[i]))
            accuracy += (np.argmax(y_hat) == np.argmax(labels[i])).astype(np.float32)
            loss += self.cal_delta(labels[i])
        print('Test loss is {:.6f} Test accuracy is {:.5f}'.format(loss/test_size, accuracy/test_size))

    def k_cross_validation(self, k=10, percent=0.2):
        data, label = self.csv_data()
        train_valid_data, test_data = train_test_split(data, label, test_size=percent)
        kf = KFold(n_splits=k)
        for i in range(self.epochs):
            for train, valid in kf.split(train_valid_data):
                train_features, train_labels = train_valid_data.iloc[train, :8], train_valid_data.iloc[train, 8]
                valid_features, valid_labels = train_valid_data.iloc[valid, :8], train_valid_data.iloc[valid, 8]
                self.train_batch_samples(train_features, train_labels)
                self.test(valid_features, valid_labels)
        self.test(test_data.iloc[:, :8], test_data.iloc[:, 8])

    # 在线学习（一次训练单个样本）
    def online_study(self, percent=0.2):
        display_step = 50
        data, label = self.csv_data()
        #data, label = generator_point.Generator_Point().simple_two_class()
        #data = np.array(data)
        train_data, test_data, train_label, test_label = train_test_split(
            data, label, test_size=percent)
        for i in range(self.epochs):
            for d in range(len(train_data)):
                loss = self.train_one_sample(train_data[d], train_label[d])
                if d % display_step == 0:
                    print("Iter: {0} data:{1}  Train loss: {2}".format(i + 1, d + 1, loss))
        self.test(test_data, test_label)

    def mini_batch_study(self, batch_size, percent=0.2):
        # data, label = self.csv_data()
        data, label = generator_point.Generator_Point().simple_two_class()
        data = np.array(data)
        train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=percent)
        batch_num = train_data.shape[0]//batch_size
        for i in range(self.epochs):
            for d in range(batch_num):
                self.train_batch_samples(train_data[d*batch_size:(d+1)*batch_size,], train_label[d*batch_size:(d+1)*batch_size,])
        self.test(test_data, test_label)