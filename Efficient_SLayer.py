import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

class Efficient_SKAN_Base(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
        num_f = 8,
    ):
        super(Efficient_SKAN_Base, self).__init__()
        
        self.in_dim = in_features
        self.out_dim = out_features
        self.num = in_features * out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.num_f = num_f
        
        
        self.grid = nn.Parameter((torch.arange(num_f).float() + 1), requires_grad = True)
        
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        # self.inter_weights = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        
        self.scale_sp = nn.Parameter(torch.ones(self.out_dim)).requires_grad_(True)
        
        self.coef = nn.Parameter(torch.rand((self.in_dim, self.num_f), requires_grad = True))
        
        # self.fc1 = nn.Linear(self.in_dim, self.out_dim)
        
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps
        
        self.conv_layer = nn.Conv1d(self.in_dim*num_f, self.out_dim*num_f, kernel_size = 1, groups = num_f)#, bias = False)
        # self.fin_linear = nn.Linear(self.out_dim*num_f, self.out_dim)


    def forward(self, x: torch.Tensor):
        # print('*** SLayer input shape: ', x.shape)
        
        mult_dim = x.shape[0]
        mult_chan = x.shape[1]
        # mult_feats = x.shape[2]
        # x = x.reshape(mult_dim*mult_chan, mult_feats)
        
        
        # print('reshaped: ', x.shape, self.num_patches, self.in_dim, self.out_dim)
        assert x.dim() == 2 and x.size(1) == self.in_dim
        
        batch = x.shape[0]
        dims = x.shape[1]
        
        base_output = F.linear(self.base_activation(x), self.base_weight) # Compute the output of the base linear layer
        
        
        x = torch.unsqueeze(x, dim = 1)
        grid = torch.unsqueeze(self.grid, dim = 1)

        f_out = torch.sin(grid*x).permute(2, 1, 0)
        f_out = torch.einsum('ij,ijk->ijk', self.coef, f_out).permute(2, 1, 0)
        
        # print('f_out shape: ', f_out.shape)
        f_out = f_out.reshape(x.size(0), -1).unsqueeze(dim = 2)
        # print('flattened f_out shape: ', f_out.shape)
        
        spline_output = self.conv_layer(f_out)
        spline_output = spline_output.reshape(batch, self.num_f, self.out_dim)
        spline_output = torch.sum(spline_output, dim = 1)*self.scale_sp

        
        out = base_output + spline_output
        out = out.reshape(mult_dim, self.out_dim)
        
        return out


class Efficient_SKAN_Batch(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        num_patches,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
        num_f = 8,
    ):
        super(Efficient_SKAN_Batch, self).__init__()
        
        self.in_dim = in_features
        self.out_dim = out_features
        self.num = in_features * out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.num_f = num_f
        self.num_patches = num_patches
        
        
        self.grid = nn.Parameter((torch.arange(num_f).float() + 1), requires_grad = True)
        
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        # self.inter_weights = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        
        self.scale_sp = nn.Parameter(torch.ones(self.out_dim)).requires_grad_(True)
        
        self.coef = nn.Parameter(torch.rand((self.in_dim, self.num_f), requires_grad = True))
        
        # self.fc1 = nn.Linear(self.in_dim, self.out_dim)
        
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps
        
        self.conv_layer = nn.Conv1d(self.in_dim*num_f, self.out_dim*num_f, kernel_size = 1, groups = num_f)#, bias = False)
        # self.fin_linear = nn.Linear(self.out_dim*num_f, self.out_dim)


    def forward(self, x: torch.Tensor):
        # print('*** SLayer input shape: ', x.shape)
        
        mult_dim = x.shape[0]
        mult_chan = x.shape[1]
        mult_feats = x.shape[2]
        x = x.reshape(mult_dim*mult_chan, mult_feats)
        
        
        # print('reshaped: ', x.shape, self.num_patches, self.in_dim, self.out_dim)
        assert x.dim() == 2 and x.size(1) == self.in_dim
        
        batch = x.shape[0]
        dims = x.shape[1]
        
        base_output = F.linear(self.base_activation(x), self.base_weight) # Compute the output of the base linear layer
        
        
        x = torch.unsqueeze(x, dim = 1)
        grid = torch.unsqueeze(self.grid, dim = 1)

        f_out = torch.sin(grid*x).permute(2, 1, 0)
        f_out = torch.einsum('ij,ijk->ijk', self.coef, f_out).permute(2, 1, 0)
        
        # print('f_out shape: ', f_out.shape)
        f_out = f_out.reshape(x.size(0), -1).unsqueeze(dim = 2)
        # print('flattened f_out shape: ', f_out.shape)
        
        spline_output = self.conv_layer(f_out)
        spline_output = spline_output.reshape(batch, self.num_f, self.out_dim)
        spline_output = torch.sum(spline_output, dim = 1)*self.scale_sp

        
        out = base_output + spline_output
        out = out.reshape(mult_dim, mult_chan, self.out_dim)
        
        return out

class Efficient_SKAN(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        num_patches,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
        num_f = 8,
    ):
        super(Efficient_SKAN, self).__init__()
        
        self.in_dim = in_features
        self.out_dim = out_features
        self.num = in_features * out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.num_f = num_f
        self.num_patches = num_patches
        
        self.downsample = nn.Linear(self.in_dim*self.num_patches, self.in_dim*2)
        self.upsample = nn.Linear(2*self.out_dim, self.out_dim*self.num_patches)
        
        
        
        self.grid = nn.Parameter((torch.arange(num_f).float() + 1), requires_grad = True)
        
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features*self.num_patches, in_features*self.num_patches))
        # self.inter_weights = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        
        self.scale_sp = nn.Parameter(torch.ones(2*self.out_dim)).requires_grad_(True)
        
        self.coef = nn.Parameter(torch.rand((self.in_dim*2, self.num_f), requires_grad = True))
        
        # self.fc1 = nn.Linear(self.in_dim, self.out_dim)
        
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps
        self.act = nn.GELU()
        
        self.conv_layer = nn.Conv1d(2*self.in_dim*num_f, 2*self.out_dim*num_f, kernel_size = 1, groups = num_f)#, bias = False)
        # self.fin_linear = nn.Linear(self.out_dim*num_f, self.out_dim)

    def scaled_spline_weight(self):
        # print('mask bool: ', self.enable_standalone_scale_spline)
        # print('return mask shape: ', (self.spline_weight * self.spline_scaler.unsqueeze(-1)).shape)
        
        return self.spline_weight * (self.spline_scaler.unsqueeze(-1) if self.enable_standalone_scale_spline else 1.0)

    def forward(self, x: torch.Tensor):
        # print('*** SLayer input shape: ', x.shape)
        
        mult_dim = x.shape[0]
        mult_chan = x.shape[1]
        mult_feats = x.shape[2]
        x = x.reshape(mult_dim, mult_chan*mult_feats)
        
        
        # print('reshaped: ', x.shape, self.num_patches, self.in_dim, self.out_dim)
        assert x.dim() == 2 and x.size(1) == self.in_dim*self.num_patches
        
        batch = x.shape[0]
        dims = x.shape[1]
        
        base_output = F.linear(self.base_activation(x), self.base_weight) # Compute the output of the base linear layer
        
        x = self.act(self.downsample(x))
        
        x = torch.unsqueeze(x, dim = 1)
        grid = torch.unsqueeze(self.grid, dim = 1)

        f_out = torch.sin(grid*x).permute(2, 1, 0)
        f_out = torch.einsum('ij,ijk->ijk', self.coef, f_out).permute(2, 1, 0)
        
        # print('f_out shape: ', f_out.shape)
        f_out = f_out.reshape(x.size(0), -1).unsqueeze(dim = 2)
        # print('flattened f_out shape: ', f_out.shape)
        
        spline_output = self.conv_layer(f_out)
        spline_output = spline_output.reshape(batch, self.num_f, 2*self.out_dim)
        spline_output = torch.sum(spline_output, dim = 1)*self.scale_sp

        
        spline_output = self.act(self.upsample(spline_output))
        out = base_output + spline_output
        out = out.reshape(mult_dim, mult_chan, self.out_dim)
        
        return out
        # else:
            # return base_output + spline_output  # Returns the sum of the output of the base linear layer and the output of the piecewise polynomial linear layer



class Efficient_SKAN_grouped(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        num_patches,
        groups,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
        num_f = 8,
    ):
        super(Efficient_SKAN_grouped, self).__init__()
        
        self.in_dim = in_features
        self.out_dim = out_features
        self.num = in_features * out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.num_f = num_f
        self.num_patches = num_patches

        
        self.groups = groups
        self.group_size = int(self.in_dim / self.groups)


        self.grid = nn.Parameter((torch.arange(num_f).float() + 1), requires_grad = True)
        
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.scale_sp = nn.Parameter(torch.ones(self.out_dim)).requires_grad_(True)
        
        
        self.coef = nn.Parameter(torch.rand((self.groups, 1, self.num_f), requires_grad = True))
        
        self.conv_layer = nn.Conv1d(self.in_dim*num_f, self.out_dim*num_f, kernel_size = 1, groups = num_f)#, bias = False)
        

        self.base_activation = base_activation()


    def forward(self, x: torch.Tensor): # Pass the input data through each layer of the model, undergo linear transformation and activation function processing, and finally obtain the output result of the model.
        
        mult_dim = x.shape[0]
        mult_chan = x.shape[1]
        mult_feats = x.shape[2]
        x = x.reshape(mult_dim*mult_chan, mult_feats)
        
        assert x.dim() == 2 and x.size(1) == self.in_dim
        
        batch = x.shape[0]
        dims = x.shape[1]
        
        base_output = F.linear(self.base_activation(x), self.base_weight) # Compute the output of the base linear layer
        
        x = torch.unsqueeze(x, dim = 1)
        grid = torch.unsqueeze(self.grid, dim = 1)
        
        f_out = torch.sin(grid*x).permute(2, 1, 0)
        f_out  = f_out.reshape(self.groups, self.group_size, self.num_f, batch)
        # print('coef/f_out shape: ', self.coef.shape, f_out.shape)
        
        
        f_out = torch.einsum('ijk,ijkl->ijkl', self.coef, f_out)
        f_out  = f_out.reshape(self.groups*self.group_size, self.num_f, batch).permute(2, 1, 0)
        
        
        f_out = f_out.reshape(x.size(0), -1).unsqueeze(dim = 2)
        
        spline_output = self.conv_layer(f_out)
        spline_output = spline_output.reshape(batch, self.num_f, self.out_dim)
        spline_output = torch.sum(spline_output, dim = 1)*self.scale_sp
        
        out = base_output + spline_output
        out = out.reshape(mult_dim, mult_chan, self.out_dim)
        return out  # Returns the sum of the output of the base linear layer and the output of the piecewise polynomial linear layer

