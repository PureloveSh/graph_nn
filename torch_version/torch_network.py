from torch.autograd import Function
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch_datasets
import torch_loss
import torch_optim
import executor
import copy


class Network:
    def __init__(self, v, e, dataset, batch_size, learning_rate, epochs, loss, optimizer):
        self.node_collection = v
        self.topo_node_list = executor.Executor(copy.copy(self.node_collection.node_list()), e).topoSort()
        self.train_loader, self.test_loader = torch_datasets.Dataset(dataset, batch_size).get_dataset()
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.criterion = torch_loss.Loss(loss).loss
        self.optimizer = torch_optim.Optimizer(optimizer).optimizer
        self.model = self.build_model()

    def get_input_node(self):
        nodes = [node for node in self.topo_node_list if not node.downstream]
        return nodes

    def get_output_node(self):
        nodes = [node for node in self.topo_node_list if not node.upstream]
        return nodes

    def get_not_input_node(self):
        nodes = [node for node in self.topo_node_list if node.downstream]
        return nodes

    def set_input(self, input):
        self.get_input_node().set_output(input)

    def build_model(self):
        model = torch.nn.Sequential()
        for index, node in enumerate(self.get_not_input_node(), 1):
            linear = node.downstream.linear
            activation_func = node.activation_func
            model.add_module('linear'+index, linear)
            if activation_func is not None:
                model.add_module('activation'+index, activation_func)
        return model

    def save_model(self):
        torch.save(self.model.static_dict(), 'params.ckpt')

    def load_model(self):
        self.model.load_state_dict(torch.load('params.ckpt'))

    def train(self):
        optimizer = self.optimizer(self.model.parameters(), lr=self.learning_rate)
        total_step = len(self.train_loader)
        for epoch in range(1, self.epochs+1):
            for index, (data, label) in enumerate(self.train_loader, 1):
                output = self.model(data)
                loss = self.criterion(output, label)
                print('Epoch [{}/{}] Step [{}/{}] Loss:{:.4f}'.format(epoch, self.epochs, index, total_step, loss.item()))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def test(self):
        with torch.no_grad():
            correct = 0
            total = 0
            for data, label in self.test_loader:
                output = self.model(data)
                # torch.max的返回值第一个值是值，第二个是下标
                _, predicted = torch.max(output.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
            print('Accuracy on test data is {:.4f}%'.format(100*correct/total))








    def start(self):
        self.train()
        self.test()




#Network('Iris', 32, 0.01, 1000, 'MSE', 'Adam')

#'''
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)


'''
y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

model = MyModel()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(2000):
    inputs = torch.from_numpy(x_train)
    targets = torch.from_numpy(y_train)

    outputs = model(inputs)
    #print(inputs, outputs)
    loss = criterion(targets, outputs)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1)%5 == 0:
        print('Epoch [{}/{}], Loss:{:.4f}'.format(epoch+1, 100, loss.item()))

predicted = model(torch.from_numpy(x_train)).detach().numpy()
print(y_train, predicted)
plt.plot(x_train, y_train, 'ro', label='Original data')
plt.plot(x_train, predicted, label='Fitted line')
plt.legend()
plt.show()
'''
