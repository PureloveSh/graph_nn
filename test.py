import torch


a = torch.randn(1, 3)
print(a)

b = torch.max(a, 1)
print(b)