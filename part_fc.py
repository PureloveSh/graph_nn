import numpy as np
from activators import *
import generator_point
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class FullConnectionLayer:
    def __init__(self, input_size, output_size, activator, keep_prob=0.7):
        self.input_size = input_size
        self.output_size = output_size
        self.activator = activator

        #self.W = np.random.uniform(-0.1, 0.1, (output_size, input_size))
        self.W = self.xvarier_init(output_size, input_size)
        self.keep_prob = keep_prob
        self.control_mask()
        self.b = np.zeros((output_size, 1))
        self.output = np.zeros((output_size, 1))
        self.W_gard = np.zeros_like(self.W)
        self.b_grad = np.zeros_like(self.b)

    def control_mask(self):
        '''该函数主要是用来控制前一层至少有一个节点与后一层相连'''
        # np.where返回非0的下标但形式为(array(...), array(...),type),取第一维的
        self.mask = np.random.binomial(1, self.keep_prob, size=self.W.shape)
        not_zero_idx = np.where(np.sum(self.mask, axis=0))[0]
        all_idx = np.arange(self.mask.shape[1])
        zero_idx = list(set(not_zero_idx) ^ set(all_idx))
        if not zero_idx:
            for i in zero_idx:
                j = np.random.randint(0, self.mask.shape[1])
                self.mask[i][j] = 1.0


    def define_mask(self, percent):
        '''
        该函数主要用来生成前一层与后一层每个节点相连的比例
        :param percent: 链接比例,数值为小于1.0的小数
        :return:
        '''
        assert percent<1.0, '比例系数不能大于1'
        self.mask = np.zeros_like(self.W)
        num_connect = int(self.mask.shape[1] * percent)
        for col in range(self.mask.shape[0]):
            idx = self.generate_randint(num_connect, self.mask.shape[1])
            self.mask[col, idx] = 1.0

    def define_mask1(self):
        '''
        :param percent1:
        :return:
        '''
        pass


    def generate_randint(self, num, max_range):
        '''
        生产一个不重复的随机数列表
        :param num: 随机数列表的大小
        :param max_range: 随机数的数值范围0-max_range
        :return: 返回没有重复值的随机数列表
        '''
        length = 0
        list_ = []
        while length < num:
            # random.randint 函数生成的随机数区间两侧的数都包括
            random_num = random.randint(0, max_range-1)
            if random_num not in list_:
                list_.append(random_num)
                length += 1
        return list_


    def xvarier_init(self, fan_in, fan_out, constant=1):
        low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
        high = constant * np.sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(low, high, size=(fan_in, fan_out))

    def forward(self, input):
        self.input = input
        self.output = self.activator.forward(np.dot(self.mask * self.W, input)+self.b)

    def backward(self, delta):
        self.delta = np.dot(self.W.T, delta) * self.activator.backward(self.input)
        self.W_grad = self.mask * np.dot(delta, self.input.T)
        self.b_grad = delta

    def backward_batch(self, delta, batch_size):
        self.delta = np.dot(self.W.T, delta) * self.activator.backward(self.input)
        self.W_grad = self.W_gard + (self.mask * np.dot(delta, self.input.T))/batch_size
        self.b_grad += delta/batch_size

    def update(self, lr):
        self.W -= lr*self.W_grad
        self.b -= lr*self.b_grad


class Network:
    def __init__(self, layers, lr, epochs, percent=0.7):
        self.layers = []
        self.learning_rate = lr
        self.epochs = epochs
        for i in range(len(layers)-2):
            self.layers.append(FullConnectionLayer(layers[i], layers[i+1], ReluActivator(), percent))
        self.layers.append(FullConnectionLayer(layers[-2], layers[-1], SoftmaxActivator(), percent))

    def predict(self, sample):
        output = np.array(sample).reshape((len(sample), 1))
        for layer in self.layers:
            layer.forward(output)
            output = layer.output
        return output

    def cal_gradient(self, label):
        #delta = self.layers[-1].activator.backward(self.layers[-1].output)*(self.layers[-1].output-label)
        delta = self.layers[-1].output - label
        for layer in self.layers[::-1]:
            layer.backward(delta)
            delta = layer.delta
        return delta

    def update_weight(self):
        for layer in self.layers:
            layer.update(self.learning_rate)

    def train_one_sample(self, sample, label):
        output = self.predict(sample)
        label = np.array(label).reshape((len(label), 1))
        self.cal_gradient(label)
        self.update_weight()
        #print(np.add.reduce(self.loss(output, label)))
        acc = np.argmax(output) == np.argmax(label)
        return acc

    def cal_batch_gradient(self, label, batch_size):
        delta = (self.layers[-1].output - label)/batch_size
        for layer in self.layers[::-1]:
            layer.backward_batch(delta, batch_size)
            delta = layer.delta


    def train_batch_size(self, sample, label, batch_size):
        label_shape = label.shape[1]
        acc = 0.
        for i in range(len(sample)):
            output = self.predict(sample[i])
            print(label[i], label_shape)
            label = np.array(label[i]).reshape((label_shape, 1))
            self.cal_batch_gradient(label, batch_size)
            acc += np.argmax(output) == np.argmax(label)
        self.update_weight()
        # print(np.add.reduce(self.loss(output, label)))
        return acc

    def train(self, samples, labels):
        acc_list = []
        for i in range(self.epochs):
            acc = 0.0
            for d in range(len(samples)):
                acc += self.train_one_sample(samples[d], labels[d])
            acc_list.append(acc/len(samples))
            #print('Epoch [{}/{}], accuracy:{:.5f}'.format(i+1, self.epochs, acc/len(samples)))
        #self.draw(acc_list)

    def trainN(self, samples, labels, batch_size):
        num_batch = samples.shape[0]//batch_size
        for i in range(self.epochs):
            for d in range(num_batch-1):
                self.train_batch_size(samples[d*batch_size:(d+1)*batch_size], labels[d*batch_size:(d+1)*batch_size], batch_size)

    def draw(self, points, smooth_factor=0.9):
        smooth_points = []
        for point in points:
            if smooth_points:
                previous = smooth_points[-1]
                smooth_points.append(previous*smooth_factor+point*(1-smooth_factor))
            else:
                smooth_points.append(point)
        epochs = range(1, len(points)+1)
        plt.plot(epochs, smooth_points, label='accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(loc='best')
        plt.show()

    def test(self, data, labels):
        acc = 0.
        label_shape = labels.shape[1]
        for i in range(len(data)):
            output = self.predict(data[i])
            label = np.array(labels[i]).reshape((label_shape, 1))
            acc += np.argmax(output) == np.argmax(label)
        print('test accuracy:', acc/len(data))
        return acc/len(data)

    def loss(self, output, label):
        return 0.5 * np.square(output-label)


if __name__ == '__main__':

    # data, label = generator_point.Generator_Point().simple_two_class()
    data, label = generator_point.Generate_csv().read_csv()

    #data, label = generator_point.Generate_csv().iris_csv()
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.3, random_state=3)

    '''
    net = Network([4, 100, 3], 1e-3, 500)
    net.trainN(X_train, y_train, 32)
    test_acc = net.test(X_test, y_test)
    print(test_acc)
    '''

    percent = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    percent_result = []
    for p in percent:
        test_acc = 0.
        print('percent', p)
        for _ in range(10):
            net = Network([8, 100, 2], 1e-3, 500, p)
            net.train(X_train, y_train)
            test_acc += net.test(X_test, y_test)
            del net
        percent_result.append(test_acc/10)

    print(percent_result)

