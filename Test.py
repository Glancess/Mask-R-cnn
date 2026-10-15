import torch

a = torch.tensor([1.0, 2.0])
b = torch.tensor([3.0, 4.0])

x = torch.stack([a, b], dim=1)
y = torch.cat([a, b], dim=0)
z = x.sum()
