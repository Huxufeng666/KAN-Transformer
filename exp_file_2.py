import torch

device = 'cpu'

h = 28
w = 28
batch = 2
a = torch.rand((batch, h*w))
print('Random input shape: ', a.shape)

grid = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
# grid = torch.einsum('i,j->ij', torch.ones(h*w, device=device), torch.linspace(grid_range[0], grid_range[1], steps=num + 1, device=device))

a = torch.einsum('ij,k->ikj', a, torch.ones(128, device=device)).reshape(batch, h*w*128).permute(1, 0)
print('-- replicated input: ', a.shape)

a = torch.unsqueeze(a, dim = 1)
grid = torch.unsqueeze(grid, dim = 1)
print('unsqueezed  shapes: ', a.shape, grid.shape)
print(grid)
out1 = torch.sin(a*grid)
print('sin out shape: ', out1.shape)

coef = torch.rand((h*w*128, 9))

y_eval = torch.einsum('ij,ijk->ik', coef, out1)
print('y eval: ', y_eval.shape)
y_eval = y_eval.permute(1, 0)
print('y eval: ', y_eval.shape)

y = torch.sum(y_eval.reshape(batch, 128, h*w), dim=2)
print('final output shape: ', y.shape)