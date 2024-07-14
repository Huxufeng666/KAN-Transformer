import torch
import torch.nn as nn

import matplotlib.pyplot as plt

class Slayer_base(nn.Module):
    def __init__(self, in_dim, out_dim, num_f = 8, base_fun = nn.SiLU(), device = 'cuda'):
        
        super(Slayer_base, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num = in_dim * out_dim
        self.num_f = num_f
        self.device = device
        self.base_fun = base_fun
        
        self.grid = nn.Parameter((torch.arange(num_f).float() + 1), requires_grad = True)
        self.coef = nn.Parameter(torch.rand((self.num, self.num_f), requires_grad = True))
        self.bias = nn.Linear(out_dim, 1, bias = False)
        
        self.scale_base = nn.Parameter(torch.ones(self.num)).requires_grad_(True)
        self.scale_sp = nn.Parameter(torch.ones(self.num)).requires_grad_(True)
        
        self.act = nn.ReLU()
        
        
    def forward(self, x, plot = False):
        # print('input shape: ', x.shape) # [1000, 784]
        batch = x.shape[0]
        
        
        x = torch.einsum('ij,k->ikj', x, torch.ones(self.out_dim, device = self.device)).reshape(batch, self.num).permute(1, 0)
        # print('replicated input: ', x.shape) # [100352, 1000]
        
        base = self.base_fun(x)
        # print('base function output: ', base.shape) # [100352, 1000]
        
        x = torch.unsqueeze(x, dim = 1)
        grid = torch.unsqueeze(self.grid, dim = 1)
        # print('unsqueezed input and grid: ', x.shape, ' ', grid.shape) # [100352, 1, 1000], [8, 1]
        
        f_out = torch.sin(grid*x)
        # print('sin output shape: ', f_out.shape) # [100352, 8, 1000]
        
        f_out = torch.einsum('ij,ijk->ik', self.coef, f_out).permute(1, 0)
        # print('coef shape: ', self.coef.shape) # [100352, 8]
        # print('coef multipled output: ', f_out.shape) # [1000, 100352]
        
        f_out = self.scale_sp*f_out + self.scale_base*base.permute(1, 0)
        
        y = torch.sum(f_out.reshape(batch, self.out_dim, self.in_dim), dim=2)
        # print('final output: ', y.shape) # [1000, 128]
        
        y = y + self.bias.weight
        
        
        return y

class Slayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_patches, num_f = 8, base_fun = nn.SiLU(), device = 'cuda'):
        
        super(Slayer, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num = in_dim * out_dim
        self.num_f = num_f
        self.device = device
        self.base_fun = base_fun
        
        self.grid = nn.Parameter((torch.arange(num_f).float() + 1), requires_grad = True)
        self.coef = nn.Parameter(torch.rand((self.num, self.num_f), requires_grad = True))
        self.bias = nn.Linear(out_dim, 1, bias = False)
        
        self.scale_base = nn.Parameter(torch.ones(self.num)).requires_grad_(True)
        self.scale_sp = nn.Parameter(torch.ones(self.num)).requires_grad_(True)
        
        self.act = nn.ReLU()
        
        
    def forward(self, x, plot = False):
        # print('input shape: ', x.shape) # [1000, 784]
        batch = x.shape[0]
        
        mult_dim = x.shape[0]
        mult_chan = x.shape[1]
        mult_feats = x.shape[2]
        x = x.reshape(mult_dim, mult_chan*mult_feats)
        
        x = torch.einsum('ij,k->ikj', x, torch.ones(self.out_dim, device = self.device)).reshape(batch, self.num).permute(1, 0)
        # print('replicated input: ', x.shape) # [100352, 1000]
        
        base = self.base_fun(x)
        # print('base function output: ', base.shape) # [100352, 1000]
        
        x = torch.unsqueeze(x, dim = 1)
        grid = torch.unsqueeze(self.grid, dim = 1)
        # print('unsqueezed input and grid: ', x.shape, ' ', grid.shape) # [100352, 1, 1000], [8, 1]
        
        f_out = torch.sin(grid*x)
        # print('sin output shape: ', f_out.shape) # [100352, 8, 1000]
        
        f_out = torch.einsum('ij,ijk->ik', self.coef, f_out).permute(1, 0)
        # print('coef shape: ', self.coef.shape) # [100352, 8]
        # print('coef multipled output: ', f_out.shape) # [1000, 100352]
        
        f_out = self.scale_sp*f_out + self.scale_base*base.permute(1, 0)
        
        y = torch.sum(f_out.reshape(batch, self.out_dim, self.in_dim), dim=2)
        # print('final output: ', y.shape) # [1000, 128]
        
        y = y + self.bias.weight
        
        
        return y
    
    def show_act(self):
        x = torch.unsqueeze(torch.linspace(0, 1, self.num), dim = 0).to(self.device)
        grid = torch.unsqueeze(self.grid, dim = 1)
        # print('x shape: ', x.shape)
        # print('grid shape: ', grid.shape)
        
        mult = x*grid
        # print('mult shape: ', mult.shape)
        
        with torch.no_grad():
            f_out = torch.sin(mult)
            sel_coef = torch.unsqueeze(self.coef[0, :], dim = 1)
            sel_coef2 = torch.unsqueeze(self.coef[100, :], dim = 1)
            sel_coef3 = torch.unsqueeze(self.coef[200, :], dim = 1)
            sel_coef4 = torch.unsqueeze(self.coef[500, :], dim = 1)
            # print('coef shape: ', sel_coef.shape)
    
            f_out1 = f_out * sel_coef
            f_out1 = torch.sum(f_out1, dim = 0)
            # print('out after sum: ', f_out.shape)
            
            f_out2 = f_out * sel_coef2
            f_out2 = torch.sum(f_out2, dim = 0)
            
            f_out3 = f_out * sel_coef3
            f_out3 = torch.sum(f_out3, dim = 0)
            
            f_out4 = f_out * sel_coef4
            f_out4 = torch.sum(f_out4, dim = 0)
        
        x = torch.squeeze(x).cpu().numpy()
        f_out1 = f_out1.cpu().numpy()
        
        f_out2 = f_out2.cpu().numpy()
        
        f_out3 = f_out3.cpu().numpy()
        
        f_out4 = f_out4.cpu().numpy()
        
        plt.plot(x, f_out1, color = 'b')
        plt.plot(x, f_out2, color = 'g')
        plt.plot(x, f_out3, color = 'r')
        plt.plot(x, f_out4, color = 'c')
        plt.show()




class Slayer_grouped(nn.Module):
    def __init__(self, in_dim, out_dim, groups, num_f = 8, base_fun = nn.SiLU(), device = 'cuda'):
        
        super(Slayer_grouped, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num = in_dim * out_dim
        self.num_f = num_f
        self.device = device
        self.base_fun = base_fun
        self.groups = groups
        self.group_size = int(self.num / self.groups)
        
        self.grid = nn.Parameter((torch.arange(num_f).float() + 1), requires_grad = True)
        self.coef = nn.Parameter(torch.rand((self.groups, 1, self.num_f), requires_grad = True))
        self.bias = nn.Linear(out_dim, 1, bias = False)
        
        self.scale_base = nn.Parameter(torch.ones(self.num)).requires_grad_(True)
        self.scale_sp = nn.Parameter(torch.ones(self.num)).requires_grad_(True)
        
        self.act = nn.ReLU()
        
        
    def forward(self, x, plot = False):
        # print('input shape: ', x.shape) # [1000, 784]
        batch = x.shape[0]
        # print('starting shapes: ', x.shape, ' ', torch.ones(self.out_dim, device = self.device).shape)
        x = torch.einsum('ij,k->ikj', x, torch.ones(self.out_dim, device = self.device)).reshape(batch, self.num).permute(1, 0)
        # print('replicated input: ', x.shape) # [100352, 1000]
        
        base = self.base_fun(x)
        # print('base function output: ', base.shape) # [100352, 1000]
        
        x = torch.unsqueeze(x, dim = 1)
        grid = torch.unsqueeze(self.grid, dim = 1)
        # print('unsqueezed input and grid: ', x.shape, ' ', grid.shape) # [100352, 1, 1000], [8, 1]
        
        f_out = torch.sin(grid*x)
        # print('sin output shape: ', f_out.shape) # [100352, 8, 1000]
        f_out  = f_out.reshape(self.groups, self.group_size, self.num_f, batch)
        # print('reshaped f_out: ', f_out.shape) # [64, 1568, 8, 1000]
        
        f_out = torch.einsum('ijk,ijkl->ijl', self.coef, f_out)
        # print('coef multiplied shape: ', f_out.shape) # [64, 1568, 1000]
        f_out = f_out.reshape(self.groups*self.group_size, batch).permute(1, 0)
        # print('coef shape: ', self.coef.shape) # [64, 1, 8]
        # print('coef multipled reshaped output: ', f_out.shape) # [1000, 100352]
        
        f_out = self.scale_sp*f_out + self.scale_base*base.permute(1, 0)
        y = torch.sum(f_out.reshape(batch, self.out_dim, self.in_dim), dim=2)
        # print('final output: ', y.shape) # [1000, 128]
        
        y = y + self.bias.weight
        
        return y
    
    def show_act(self):
        x = torch.unsqueeze(torch.linspace(0, 1, self.num), dim = 0).to(self.device)
        grid = torch.unsqueeze(self.grid, dim = 1)
        # print('x shape: ', x.shape)
        # print('grid shape: ', grid.shape)
        
        mult = x*grid
        # print('mult shape: ', mult.shape)
        
        with torch.no_grad():
            f_out = torch.sin(mult)
            sel_coef = torch.unsqueeze(self.coef[0, 0, :], dim = 1)
            sel_coef2 = torch.unsqueeze(self.coef[1, 0, :], dim = 1)
            # sel_coef3 = torch.unsqueeze(self.coef[30, 0, :], dim = 1)
            # sel_coef4 = torch.unsqueeze(self.coef[60, 0, :], dim = 1)
            # print('coef shape: ', sel_coef.shape)
    
            f_out1 = f_out * sel_coef
            f_out1 = torch.sum(f_out1, dim = 0)
            # print('out after sum: ', f_out.shape)
            
            f_out2 = f_out * sel_coef2
            f_out2 = torch.sum(f_out2, dim = 0)
            
            # f_out3 = f_out * sel_coef3
            # f_out3 = torch.sum(f_out3, dim = 0)
            
            # f_out4 = f_out * sel_coef4
            # f_out4 = torch.sum(f_out4, dim = 0)
        
        x = torch.squeeze(x).cpu().numpy()
        f_out1 = f_out1.cpu().numpy()
        
        f_out2 = f_out2.cpu().numpy()
        
        # f_out3 = f_out3.cpu().numpy()
        
        # f_out4 = f_out4.cpu().numpy()
        
        plt.plot(x, f_out1, color = 'b')
        plt.plot(x, f_out2, color = 'g')
        # plt.plot(x, f_out3, color = 'r')
        # plt.plot(x, f_out4, color = 'c')
        plt.show()



'''
def forward(self, x, plot = False):
    batch = x.shape[0]
    # print('starting shapes: ', x.shape, ' ', torch.ones(self.out_dim, device = self.device).shape)
    x = torch.einsum('ij,k->ikj', x, torch.ones(self.out_dim, device = self.device)).reshape(batch, self.num)
    
    base = self.base_fun(x)
    
    x = torch.unsqueeze(x, dim = 1)
    grid = torch.unsqueeze(self.grid, dim = 1)
    
    f_out = torch.sin(grid*x)
    
    f_out = torch.einsum('ij,ijk->ik', self.coef, f_out.permute(2, 1, 0)).permute(1, 0)
    
    f_out = self.scale_sp*f_out + self.scale_base*base
    y = torch.sum(f_out.reshape(batch, self.out_dim, self.in_dim), dim=2)
    
    y = y + self.bias.weight
    
    return y
'''