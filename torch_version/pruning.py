import torch
import torch.nn.functional as F
from torch.autograd import Variable


class MaskedLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskedLinear, self).__init__(in_features, out_features, bias)
        self.mask_flag = False

    def create_mask(self, mask):
        self.mask = Variable(mask, requires_grad=True)
        self.weight.data = self.weight.data * self.mask.data
        self.mask_flag = True

    def get_mask(self):
        print(self.mask_flag)
        return self.mask

    def forward(self, input):
        '''
        F.linear中的self.bias是通过父类继承得到，是out_features维度的向量，不是bool值
        :param input:
        :return:
        '''
        if self.mask_flag:
            weight = self.weight * self.mask
            return F.linear(input, weight, self.bias)
        else:
            return F.linear(input, self.weight, self.bias)


class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = MaskedLinear(28*28, 200)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.linear2 = MaskedLinear(200, 200)
        self.relu2 = torch.nn.ReLU(inplace=True)
        self.linear3 = MaskedLinear(200, 10)

    def forward(self, input):
        out = input.view(input.size(0), -1)
        out = self.relu1(self.linear1(out))
        out = self.relu2(self.linear2(out))
        out = self.linear3(out)
        return out

    def set_mask(self, masks):
        self.linear1.create_mask(masks[0])
        self.linear2.create_mask(masks[1])
        self.linear3.create_mask(masks[2])
