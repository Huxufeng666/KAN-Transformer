import torch
import torch.nn as nn
import torch.nn.functional as F

from KANLayer import KANLayer
from Symbolic_KANLayer import *
from SLayer import Slayer

class MLP_Net(nn.Module):
    def __init__(self):
        super(MLP_Net, self).__init__()

        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):

        x = torch.flatten(x, 1)
        # print('flattened shape: ', x.shape)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
    
class KAN_Net(nn.Module):
    def __init__(self, device = 'cuda'):
        super(KAN_Net, self).__init__()

        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = KANLayer(784, 128)
        self.fc2 = KANLayer(128, 10)
        
        self.symbolic_enabled = False # default value in the original code
        self.symbolic_fun_1 = Symbolic_KANLayer(in_dim = 784, out_dim = 128, device = device)
        self.symbolic_fun_2 = Symbolic_KANLayer(in_dim = 128, out_dim = 10, device = device)
        
        self.bias1 = nn.Linear(128, 1, bias = False)
        self.bias2 = nn.Linear(10, 1, bias = False)

    def forward(self, x):
        self.acts = []  # shape ([batch, n0], [batch, n1], ..., [batch, n_L])
        self.spline_preacts = []
        self.spline_postsplines = []
        self.spline_postacts = []
        self.acts_scale = []
        self.acts_scale_std = []
        
        x = torch.flatten(x, 1)
        
        x_numerical, preacts, postacts_numerical, postspline = self.fc1(x)
        
        if self.symbolic_enabled == True:
            x_symbolic, postacts_symbolic = self.symbolic_fun_1(x)
        else:
            x_symbolic = 0.
            postacts_symbolic = 0.

        x = x_numerical + x_symbolic
        postacts = postacts_numerical + postacts_symbolic

        # self.neurons_scale.append(torch.mean(torch.abs(x), dim=0))
        grid_reshape = self.fc1.grid.reshape(128, 784, -1)
        input_range = grid_reshape[:, :, -1] - grid_reshape[:, :, 0] + 1e-4
        output_range = torch.mean(torch.abs(postacts), dim=0)
        self.acts_scale.append(output_range / input_range)
        self.acts_scale_std.append(torch.std(postacts, dim=0))
        self.spline_preacts.append(preacts.detach())
        self.spline_postacts.append(postacts.detach())
        self.spline_postsplines.append(postspline.detach())

        x = x + self.bias1.weight
        
        
        x = self.dropout2(x)
        
        
        x_numerical, preacts, postacts_numerical, postspline = self.fc2(x)
        if self.symbolic_enabled == True:
            x_symbolic, postacts_symbolic = self.symbolic_fun_2(x)
        else:
            x_symbolic = 0.
            postacts_symbolic = 0.

        x = x_numerical + x_symbolic
        postacts = postacts_numerical + postacts_symbolic

        # self.neurons_scale.append(torch.mean(torch.abs(x), dim=0))
        grid_reshape = self.fc2.grid.reshape(10, 128, -1)
        input_range = grid_reshape[:, :, -1] - grid_reshape[:, :, 0] + 1e-4
        output_range = torch.mean(torch.abs(postacts), dim=0)
        self.acts_scale.append(output_range / input_range)
        self.acts_scale_std.append(torch.std(postacts, dim=0))
        self.spline_preacts.append(preacts.detach())
        self.spline_postacts.append(postacts.detach())
        self.spline_postsplines.append(postspline.detach())

        x = x + self.bias2.weight
        
        output = F.log_softmax(x, dim=1)
        return output