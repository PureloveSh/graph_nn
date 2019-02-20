import torch_activators
import torch
import numpy as np
from torch.autograd import Function


class LinearFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, mask, bias=None):
        # 用ctx把该存的存起来，留着backward的时候用
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
    def __init__(self, input_features, out_features, keep_prob, bias):
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


class Node:
    def __init__(self, activation_func):
        self.output = 0.0
        self.upstream = None
        self.downstream = None
        self.activation_func = torch_activators.Activator(activation_func).get_activator()

    def activate_output(self):
        self.output = self.activation_func(self.output)

    def set_output(self, output):
        self.output = output

    def cal_output(self):
        if self.downstream:
            self.output = self.downstream.linear(self.downstream.downstream_node.output)


class PartConnection:
    def __init__(self, upstream_node, downstream_node, keep_prob=1.0, bias=True):
        self.upstream_node = upstream_node
        self.downstream_node = downstream_node
        self.keep_prob = keep_prob
        self.bias = bias

    def late_init(self, down_num, up_num):
        self.down_num = down_num
        self.up_num = up_num
        self.linear = PartLinear(down_num, up_num, self.keep_prob, self.bias)



class FullConnnection:
    def __init__(self, upstream_node, downstream_node):
        self.upstream_node = upstream_node
        self.downstream_node = downstream_node

    def late_init(self, up_num, down_num):
        self.linear = torch.nn.Linear(down_num, up_num)

