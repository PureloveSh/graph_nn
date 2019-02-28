import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch_version.pruning import MLP
from torch.autograd import Variable
import numpy as np

params = {
    'batch_size':64,
    'num_epochs':5,
    'learning_rate':1e-3,
    'weight_decay':5e-4,
    'pruning_perc':15.
}


def train(model, loss_fn, optimizer, params, train_loader):
    model.train()
    for epoch in range(params['num_epochs']):
        print('Starting epoch %d / %d' % (epoch+1, params['num_epochs']))
        for index, (x, y) in enumerate(train_loader):
            x, y = Variable(x), Variable(y)

            predict = model(x)
            loss = loss_fn(predict, y)

            if (index+1)%100 == 0:
                print('Batch = %d, loss=%.8f'%(index+1, loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def test(model, loader):
    model.eval()
    num_correct, num_samples = 0, len(loader.dataset)
    with torch.no_grad():
        for x, y in loader:
            predict = model(x)
            _, preds = predict.data.max(1)
            num_correct += (preds == y).sum()

    acc = float(num_correct) / num_samples

    print('Test accuracy: {:.2f}% ({}/{})'.format(100*acc, num_correct, num_samples))
    return acc


def prune_rate(model, verbose=True):
    '''
    打印出每一层以及整个网络的裁剪比例
    :param model:
    :param verbose:
    :return:
    '''
    total_param_num = 0
    zero_param_num = 0

    layer_index = 0

    for parameter in model.parameters():
        param_this_layer = 1
        for dim in parameter.data.size():
            param_this_layer *= dim
        total_param_num += param_this_layer

        if len(parameter.data.size()) != 1:
            layer_index += 1
            zero_param_this_layer = np.count_nonzero(parameter.cpu().data.numpy()==0)
            zero_param_num += zero_param_this_layer

            if verbose:
                print('Layer {} | {} layer | {:.2f}%parameters pruned'.format(layer_index, 'Linear', 100*zero_param_this_layer/param_this_layer))

    pruning_perc = 100.*zero_param_num/total_param_num
    if verbose:
        print('Final pruning rate:{:.2f}%'.format(pruning_perc))
    return pruning_perc

def weight_prune(model, pruning_perc):
    '''
    :param pruning_perc: 90%的90
    :return:
    '''

    # 根据裁剪比例确定裁剪的阈值
    all_weights = []
    for p in model.parameters():
        if len(p.data.size()) != 1:
            all_weights += list(p.data.abs().numpy().flatten())

    threshold = np.percentile(np.array(all_weights), pruning_perc)

    masks = []
    for p in model.parameters():
        if len(p.data.size()) != 1:
            pruned_indexs = p.data.abs() > threshold
            masks.append(pruned_indexs.float())
    return masks

train_dataset = datasets.MNIST('../data/', train=True, download=False, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)

test_dataset = datasets.MNIST('../data/', train=False, download=False, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)


net = MLP()

print('--------Pretrained network loader---------')
masks = weight_prune(net, params['pruning_perc'])
net.set_mask(masks)
print('------{}% parameters pruned -------'.format(params['pruning_perc']))
test(net, test_loader)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(net.parameters(), lr=params['learning_rate'],weight_decay=params['weight_decay'])
train(net, criterion, optimizer, params, train_loader)

print('----After retraining-----')
test(net, test_loader)
prune_rate(net)

torch.save(net.state_dict(), 'models/mlp_pruned.pkl')
