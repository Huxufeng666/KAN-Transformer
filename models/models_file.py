import torch
import torch.nn as nn
import torch.nn.functional as F

from  models.KANLayer import KANLayer
from models.Symbolic_KANLayer import *
from SLayer import Slayer, Slayer_grouped

from efficient_KAN import KANLinear
from Efficient_SLayer import Efficient_SKAN, Efficient_SKAN_grouped, Efficient_SKAN_Base



class MLP_Net(nn.Module):
    def __init__(self):
        super(MLP_Net, self).__init__()

        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(3072, 128) # 784, 128 for MNIST
        self.fc4 = nn.Linear(128, 10)

    def forward(self, x):

        x = torch.flatten(x, 1)
        # print('flattened shape: ', x.shape)
        x = self.fc1(x)
        x = F.relu(x)
        
        x = self.fc4(x)
        output = F.log_softmax(x, dim=1)
        return output
    
class KAN_Net(nn.Module):
    def __init__(self, device = 'cuda'):
        super(KAN_Net, self).__init__()

        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = KANLayer(3072, 128) # 784 for MNIST
        # self.fc2 = KANLayer(512, 256) # 784 for MNIST
        # self.fc3 = KANLayer(256, 128) # 784 for MNIST
        self.fc4 = KANLayer(128, 5)
        
        self.symbolic_enabled = False # default value in the original code
        self.symbolic_fun_1 = Symbolic_KANLayer(in_dim = 3072, out_dim = 128, device = device)
        self.symbolic_fun_2 = Symbolic_KANLayer(in_dim = 128, out_dim = 5, device = device)
        
        self.bias1 = nn.Linear(128, 1, bias = False)
        # self.bias2 = nn.Linear(256, 1, bias = False)
        # self.bias3 = nn.Linear(128, 1, bias = False)
        self.bias4 = nn.Linear(5, 1, bias = False)

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
        grid_reshape = self.fc1.grid.reshape(128, 3072, -1)
        input_range = grid_reshape[:, :, -1] - grid_reshape[:, :, 0] + 1e-4
        output_range = torch.mean(torch.abs(postacts), dim=0)
        self.acts_scale.append(output_range / input_range)
        self.acts_scale_std.append(torch.std(postacts, dim=0))
        self.spline_preacts.append(preacts.detach())
        self.spline_postacts.append(postacts.detach())
        self.spline_postsplines.append(postspline.detach())

        x = x + self.bias1.weight
        #########################################################ss
        # x_numerical, preacts, postacts_numerical, postspline = self.fc2(x)
        
        # if self.symbolic_enabled == True:
        #     x_symbolic, postacts_symbolic = self.symbolic_fun_1(x)
        # else:
        #     x_symbolic = 0.
        #     postacts_symbolic = 0.

        # x = x_numerical + x_symbolic
        # postacts = postacts_numerical + postacts_symbolic

        # # self.neurons_scale.append(torch.mean(torch.abs(x), dim=0))
        # grid_reshape = self.fc2.grid.reshape(256, 512, -1)
        # input_range = grid_reshape[:, :, -1] - grid_reshape[:, :, 0] + 1e-4
        # output_range = torch.mean(torch.abs(postacts), dim=0)
        # self.acts_scale.append(output_range / input_range)
        # self.acts_scale_std.append(torch.std(postacts, dim=0))
        # self.spline_preacts.append(preacts.detach())
        # self.spline_postacts.append(postacts.detach())
        # self.spline_postsplines.append(postspline.detach())

        # x = x + self.bias2.weight
        
        ########################################################ee
        x = self.dropout2(x)
        ########################################################ss
        
        # x_numerical, preacts, postacts_numerical, postspline = self.fc3(x)
        
        # if self.symbolic_enabled == True:
        #     x_symbolic, postacts_symbolic = self.symbolic_fun_1(x)
        # else: 
        #     x_symbolic = 0.
        #     postacts_symbolic = 0.

        # x = x_numerical + x_symbolic
        # postacts = postacts_numerical + postacts_symbolic

        # # self.neurons_scale.append(torch.mean(torch.abs(x), dim=0))
        # grid_reshape = self.fc3.grid.reshape(128, 256, -1)
        # input_range = grid_reshape[:, :, -1] - grid_reshape[:, :, 0] + 1e-4
        # output_range = torch.mean(torch.abs(postacts), dim=0)
        # self.acts_scale.append(output_range / input_range)
        # self.acts_scale_std.append(torch.std(postacts, dim=0))
        # self.spline_preacts.append(preacts.detach())
        # self.spline_postacts.append(postacts.detach())
        # self.spline_postsplines.append(postspline.detach())

        # x = x + self.bias3.weight
        #########################################################ee
        
        x_numerical, preacts, postacts_numerical, postspline = self.fc4(x)
        if self.symbolic_enabled == True:
            x_symbolic, postacts_symbolic = self.symbolic_fun_2(x)
        else:
            x_symbolic = 0.
            postacts_symbolic = 0.

        x = x_numerical + x_symbolic
        postacts = postacts_numerical + postacts_symbolic

        # self.neurons_scale.append(torch.mean(torch.abs(x), dim=0))
        grid_reshape = self.fc4.grid.reshape(5, 128, -1)
        input_range = grid_reshape[:, :, -1] - grid_reshape[:, :, 0] + 1e-4
        output_range = torch.mean(torch.abs(postacts), dim=0)
        self.acts_scale.append(output_range / input_range)
        self.acts_scale_std.append(torch.std(postacts, dim=0))
        self.spline_preacts.append(preacts.detach())
        self.spline_postacts.append(postacts.detach())
        self.spline_postsplines.append(postspline.detach())

        x = x + self.bias4.weight
        
        output = F.log_softmax(x, dim=1)
        return output
    

class Sin_Net(nn.Module):
    def __init__(self):
        super(Sin_Net, self).__init__()

        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # self.fc1 = Slayer_grouped(3072, 128, groups = 8) # 784 for MNIST
        # self.fc2 = Slayer_grouped(128, 10, groups = 1)
        
        self.fc1 = Slayer(3072, 128)
        self.fc2 = Slayer(128, 10)


    def forward(self, x):

        x = torch.flatten(x, 1)
        # print('flattened shape: ', x.shape)
        x = self.fc1(x)
        # x = self.fc11(x)
        # x = self.fc12(x)
        
        # x = self.dropout1(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
    
    
    
class Efficient_KAN_Net(nn.Module):
    def __init__(self):
        super(Efficient_KAN_Net, self).__init__()

        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = KANLinear(3072, 128) # 784, 128 for MNIST
        self.fc4 = KANLinear(128, 10)

    def forward(self, x):

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        # x = F.relu(x)
        
        x = self.fc4(x)
        output = F.log_softmax(x, dim=1)
        return output
    
    
class Efficient_SKAN_Net(nn.Module):
    def __init__(self):
        super(Efficient_SKAN_Net, self).__init__()

        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        self.fc1 = Efficient_SKAN_Base(3072, 128) # 784, 128 for MNIST
        self.fc4 = Efficient_SKAN_Base(128, 10)
        
        # self.fc1 = Efficient_SKAN_grouped(3072, 128, groups = 64) # 784, 128 for MNIST
        # self.fc4 = Efficient_SKAN_grouped(128, 10, groups = 1)

    def forward(self, x):

        x = torch.flatten(x, 1)
        # print('flattened shape: ', x.shape)
        x = self.fc1(x)
        # x = F.relu(x)
        
        x = self.fc4(x)
        output = F.log_softmax(x, dim=1)
        return output