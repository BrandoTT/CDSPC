import imp


import torch

a = torch.rand([3,4])
print(a)
b = torch.rand([3,4])
print(b)
#c = torch.mm(a,b)
c = a*b
print(c)
d = torch.mul(a,b)
print(d)