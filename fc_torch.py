import torch
from torch import optim
import generator_point
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn

class MyModel(torch.nn.Module):
    def __init__(self, model_shape):
        super(MyModel, self).__init__()
        assert model_shape != [], "输入不合法，请重新输入"
        self.model = torch.nn.Sequential()
        for i in range(1, len(model_shape) - 1):
            self.model.add_module("linear_" + str(i), torch.nn.Linear(model_shape[i - 1], model_shape[i]))
            self.model.add_module('activation_' + str(i), torch.nn.Sigmoid())
        self.model.add_module('linear_' + str(len(model_shape)), torch.nn.Linear(model_shape[-2], model_shape[-1]))
        #self.model.add_module('sigmoid', torch.nn.Sigmoid())

    def forward(self, x):
        #x = F.softmax(self.model(x), dim=1)
        return self.model(x)

class NeuralNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out




def main(model_shape):
    X, label =generator_point.Generate_csv().read_not_onehot()
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(label).long()
    torch_dataset = Data.TensorDataset(X, y)
    batch_size = 64
    train_loader = torch.utils.data.DataLoader(dataset=torch_dataset, batch_size=batch_size, shuffle=True)

    #train_dataset = torchvision.datasets.MNIST(root='data', train=True, transform=transforms.ToTensor(), download=False)
    #train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    model = MyModel(model_shape)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    total_step = len(train_loader)
    num_epochs = 1000
    for epoch in range(num_epochs):
        for i, (data, labels) in enumerate(train_loader):
            correct = 0.
            #data = data.reshape(-1, 28*28)
            outputs = model(data)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            if (i+1) % 5 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss:{:.4f}, acc:{:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item(), correct/batch_size))


if __name__ == '__main__':
    main([8, 50, 2])
