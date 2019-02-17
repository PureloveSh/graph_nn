from torch.autograd import Function
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch_datasets
import torch_activators
import torch_loss
import torch_optim

class LinearFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, mask, bias=None):
        #用ctx把该存的存起来，留着backward的时候用
        ctx.save_for_backward(input, weight, bias, mask)
        output = input.mm(weight.t()*mask.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, mask = ctx.saved_variables
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight*mask)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)*mask
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, None, grad_bias


class PartLinear(torch.nn.Module):
    def __init__(self, input_features, out_features, keep_prob=1.0, bias=True):
        super(PartLinear, self).__init__()
        self.input_features = input_features
        self.output_features = out_features
        self.keep_prob = keep_prob
        self.mask = torch.from_numpy(np.random.binomial(1, self.keep_prob, size=(out_features, input_features))).float()
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, input_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.weight.data.uniform_(-0.1, 0.1)
        if bias is not None:
            self.bias.data.uniform_(-0.1, 0.1)

    def forward(self, input):
        return LinearFunction.apply(input, self.weight, self.mask, self.bias)


class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.part_linear = PartLinear(1, 10, 0.5)
        self.linear = torch.nn.Linear(10, 1)

    def forward(self, input):
        x = self.part_linear(input)
        x = F.relu(x)
        x = self.linear(x)

        #for node in node_list:


        return x

class Network:
    def __init__(self, dataset, batch_size, learning_rate, epochs, loss, optimizer):
        self.train_loader, self.test_loader = torch_datasets.Dataset(dataset, batch_size).get_dataset()
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.criterion = torch_loss.Loss(loss).loss
        self.optimizer = torch_optim.Optimizer(optimizer).optimizer
        train_size = len(self.train_loader)
        print(self.criterion, self.optimizer)


Network('Iris', 32, 0.01, 1000, 'MSE', 'Adam')

#'''
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

inputs = torch.from_numpy(x_train)
y = F.sigmoid(inputs)
print(y)

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
