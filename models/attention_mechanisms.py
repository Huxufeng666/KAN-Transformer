import torch
import torch.nn as nn
import torch.nn.functional as F

# from spline import *

class KA_attention_heads_mod(nn.Module):
    def __init__(self, out_dim, patches, heads, head_dim, reduce_ratio = 8, num_f = 8, base_fun = nn.SiLU(), device = 'cuda'):
        super(KA_attention_heads_mod, self).__init__()
        
        self.num_f = num_f
        self.num = patches*heads*head_dim
        self.base_fun = base_fun
        self.device = device
        self.out_dim = out_dim
        self.reduced_dim = int(head_dim/reduce_ratio)
        
        self.head_dim = head_dim
        
        
        # self.grid = nn.Parameter((torch.arange(int(num_f)).float() + 1), requires_grad = True)
        # self.grid_outer = nn.Parameter((torch.arange(int(num_f)).float() + 1), requires_grad = True)
        
        grid = torch.arange(num_f).unsqueeze(0).float() + 1
        grid = grid.repeat(5, 1)
        self.grid = nn.Parameter(grid, requires_grad = True)
        
        self.base_weight_qk = nn.Parameter(torch.rand((head_dim, head_dim), requires_grad = True))
        
        
        self.coef_q = nn.Parameter(torch.rand((head_dim*heads*patches, int(self.num_f), 5), requires_grad = True))
        self.coef_k = nn.Parameter(torch.rand((head_dim*heads*patches, int(self.num_f), 5), requires_grad = True))
        
        # self.coef_outer = nn.Parameter(torch.rand((patches*head_dim, int(self.num_f)), requires_grad = True))

        self.shift_q = nn.Parameter(torch.rand(1, heads, 5, patches, self.head_dim))
        self.shift_k = nn.Parameter(torch.rand(1, heads, 5, patches, self.head_dim))

        self.scale_base = nn.Parameter(torch.ones(1, 1, 1, 1, head_dim)).requires_grad_(True)
        self.scale_sp = nn.Parameter(torch.ones(1, 1, 1, 1, head_dim)).requires_grad_(True)
        # self.scale_sp = nn.Parameter(torch.ones(1, heads, patches, self.out_dim)).requires_grad_(True)
        
        self.lin_q = nn.Linear(head_dim, head_dim*5)
        self.lin_k = nn.Linear(head_dim, head_dim*5)
        
        self.lin_out_q = nn.Linear(head_dim, head_dim)
        self.lin_out_k = nn.Linear(head_dim, head_dim)
        

    def inner_function(self, q, k, batch, heads, patches):
        
        q = self.lin_q(q).reshape(batch, 5, self.head_dim*heads*patches)
        k = self.lin_k(k).reshape(batch, 5, self.head_dim*heads*patches)
        
        grid = torch.unsqueeze(self.grid, dim = 2)
        
        q = torch.unsqueeze(q, dim = 2)
        k = torch.unsqueeze(k, dim = 2)
        
        f_out_q = torch.sin(grid*q).permute(3, 2, 1, 0)
        f_out_q = torch.einsum('ijk,ijkl->ijkl', self.coef_q, f_out_q).permute(3, 2, 1, 0)
        f_out_q = f_out_q.reshape(batch, heads, 5, self.num_f, patches, self.head_dim)
        f_out_q = torch.sum(f_out_q, dim = 3) + self.shift_q
        
        f_out_k = torch.sin(grid*k).permute(3, 2, 1, 0)
        f_out_k = torch.einsum('ijk,ijkl->ijkl', self.coef_k, f_out_k).permute(3, 2, 1, 0)
        f_out_k = f_out_k.reshape(batch, heads, 5, self.num_f, patches, self.head_dim)
        f_out_k = torch.sum(f_out_k, dim = 3) + self.shift_k
        
        return f_out_q, f_out_k
    
    # def outward_function(self, in_tensor, batch, heads, patches):
    #     grid = torch.unsqueeze(self.grid, dim = 1)
    #     in_tensor = torch.unsqueeze(in_tensor, dim = 2)
    #     in_tensor = in_tensor.reshape(batch*heads, 5, 1, patches*self.head_dim)

    #     f_out = torch.sin(grid*in_tensor).permute(3, 2, 1, 0)
    #     f_out = torch.einsum('ij,ijkl->ijkl', self.coef_outer, f_out).permute(3, 2, 1, 0)
        
    #     f_out = torch.sum(f_out, dim = 2)
        
    #     return f_out.reshape(batch, heads, 5, patches, self.head_dim)
        
    def forward(self, q, k, scale):
        batch = q.shape[0]
        heads = q.shape[1]
        patches = q.shape[2]
        dim = q.shape[3]
        
        qk = torch.cat((q, k), dim = 0)
        # qk = qk.reshape(int(2*batch)*heads*patches, 1, dim)
        
        base_qk = F.linear(self.base_fun(qk), self.base_weight_qk)
        base_qk = torch.unsqueeze(base_qk, dim = 2)
        
        #qq = q.reshape(batch*heads*patches, dim)
        #kk = k.reshape(batch*heads*patches, dim)
        
        f_out_q, f_out_k = self.inner_function(q, k, batch, heads, patches) #[batch*heads, 5, patches, dim]
        
        # print('feats and base shape: ', f_out_q.shape, base_qk[0:batch, :, :, :].shape)
        # print('scales shape: ', self.scale_sp.shape, self.scale_base.shape)
        f_out_q = f_out_q*self.scale_sp + base_qk[0:batch, :, :, :, :]*self.scale_base
        f_out_k = f_out_k*self.scale_sp + base_qk[batch:int(2*batch), :, :, :, :]*self.scale_base
        # f_out_q = self.outward_function(f_out_q, batch, heads, patches)
        # f_out_k = self.outward_function(f_out_k, batch, heads, patches)
        f_out_q = self.lin_out_q(f_out_q)
        f_out_k = self.lin_out_k(f_out_k)
        
        f_out_q = torch.sum(f_out_q, dim = 2)
        f_out_k = torch.sum(f_out_k, dim = 2)
        
        # f_out_q = torch.unsqueeze(f_out_q, dim = 3)
        # f_out_k = torch.unsqueeze(f_out_k, dim = 2)
        # f_out_q = torch.sigmoid(f_out_q)
        # f_out_k = torch.sigmoid(f_out_k)
        
        # f_out = (f_out_q + f_out_k) #[batch, heads, patches, patches, dim]
        # f_out = torch.sum(f_out, dim = 4)
        
        
        # attn = f_out.softmax(dim=-1)
        # print('shapes: ', q.shape, f_out_q.shape)
        # q = q*f_out_q
        # k = k*f_out_k
        
        return f_out_q, f_out_k


from scipy.linalg import expm, sinm, cosm
class KA_attention_crossinf_scaling(nn.Module):
    def __init__(self, out_dim, patches, heads, head_dim, reduce_ratio = 8, num_f = 8, base_fun = nn.SiLU(), device = 'cuda'):
        super(KA_attention_crossinf_scaling, self).__init__()
        
        self.num_f = num_f
        self.num = patches*heads*head_dim
        self.base_fun = base_fun
        self.device = device
        self.out_dim = out_dim
        self.reduced_dim = int(head_dim/reduce_ratio)
        
        
        self.grid = nn.Parameter((torch.arange(num_f).float() + 1), requires_grad = True)
        
        self.base_weight = nn.Parameter(torch.rand((head_dim, head_dim), requires_grad = True))
        #self.base_weight_k = nn.Parameter(torch.rand((head_dim, head_dim), requires_grad = True))
        
        
        self.coef = nn.Parameter(torch.rand((head_dim, self.num_f), requires_grad = True))
        #self.coef_k = nn.Parameter(torch.rand((head_dim, self.num_f), requires_grad = True))

        self.scale_base = nn.Parameter(torch.ones(1, heads, patches, head_dim)).requires_grad_(True)
        self.scale_sp = nn.Parameter(torch.ones(1, heads, patches, head_dim)).requires_grad_(True)
        
        self.lin_qk = nn.Linear(patches, patches)


    def inner_function(self, q, k):
        grid = torch.unsqueeze(self.grid, dim = 1)
        q = torch.unsqueeze(q, dim = 1)
        k = torch.unsqueeze(k, dim = 1)
        
        f_out_q = torch.sin(grid*q).permute(2, 1, 0)
        f_out_q = torch.einsum('ij,ijk->ijk', self.coef, f_out_q).permute(2, 1, 0)
        
        f_out_k = torch.sin(grid*k).permute(2, 1, 0)
        f_out_k = torch.einsum('ij,ijk->ijk', self.coef, f_out_k).permute(2, 1, 0)
        
        f_out_q = torch.sum(f_out_q, dim = 1)
        f_out_k = torch.sum(f_out_k, dim = 1)
        
        return f_out_q, f_out_k
        
    def forward(self, q, k, scale):
        batch = q.shape[0]
        heads = q.shape[1]
        patches = q.shape[2]
        dim = q.shape[3]
        
        
        base_q = F.linear(self.base_fun(q), self.base_weight)
        base_k = F.linear(self.base_fun(k), self.base_weight)
        
        qq = q.reshape(batch*heads*patches, dim)
        kk = k.reshape(batch*heads*patches, dim)
        
        f_out_q, f_out_k = self.inner_function(qq, kk)
        

        f_out_q = f_out_q.reshape(batch, heads, patches, dim)*self.scale_sp + base_q*self.scale_base
        f_out_k = f_out_k.reshape(batch, heads, patches, dim)*self.scale_sp + base_k*self.scale_base
        
        f_out_q = torch.sigmoid(torch.unsqueeze(f_out_q, dim = 3))
        f_out_k = torch.sigmoid(torch.unsqueeze(f_out_k, dim = 2))
        

        f_out = (f_out_q + f_out_k)
        f_out = torch.sum(f_out, dim = 4)
        
        f_out = self.lin_qk(f_out)
        # print('f_out shape: ', f_out.shape)
        # print('Exp shape: ', torch.matrix_exp(f_out).shape)
        # print('k shape: ', k.shape)
        k = torch.matrix_exp(f_out)@k
        # print('translated shape: ', k.shape)
        
        # attn = f_out.softmax(dim=-1)
        
        
        return q, k

class KA_attention_crossinf(nn.Module):
    def __init__(self, out_dim, patches, heads, head_dim, reduce_ratio = 8, num_f = 12, base_fun = nn.SiLU(), device = 'cuda'):
        super(KA_attention_crossinf, self).__init__()
        
        self.num_f = num_f
        self.num = patches*heads*head_dim
        self.base_fun = base_fun
        self.device = device
        self.out_dim = out_dim
        self.reduced_dim = int(head_dim/reduce_ratio)
        
        
        self.grid = nn.Parameter((torch.arange(num_f).float() + 1), requires_grad = True)
        
        self.base_weight = nn.Parameter(torch.rand((head_dim, head_dim), requires_grad = True))
        #self.base_weight_k = nn.Parameter(torch.rand((head_dim, head_dim), requires_grad = True))
        
        
        self.coef_q = nn.Parameter(torch.rand((head_dim, self.num_f), requires_grad = True))
        self.coef_k = nn.Parameter(torch.rand((head_dim, self.num_f), requires_grad = True))

        self.scale_base = nn.Parameter(torch.ones(1, heads, patches, head_dim)).requires_grad_(True)
        self.scale_sp = nn.Parameter(torch.ones(1, heads, patches, head_dim)).requires_grad_(True)
        
        self.lin_qk = nn.Linear(head_dim, head_dim)


    def inner_function(self, q, k):
        grid = torch.unsqueeze(self.grid, dim = 1)
        q = torch.unsqueeze(q, dim = 1)
        k = torch.unsqueeze(k, dim = 1)
        
        f_out_q = torch.sin(grid*q).permute(2, 1, 0)
        f_out_q = torch.einsum('ij,ijk->ijk', self.coef_q, f_out_q).permute(2, 1, 0)
        
        f_out_k = torch.sin(grid*k).permute(2, 1, 0)
        f_out_k = torch.einsum('ij,ijk->ijk', self.coef_k, f_out_k).permute(2, 1, 0)
        
        f_out_q = torch.sum(f_out_q, dim = 1)
        f_out_k = torch.sum(f_out_k, dim = 1)
        
        return f_out_q, f_out_k
        
    def forward(self, q, k, scale):
        batch = q.shape[0]
        heads = q.shape[1]
        patches = q.shape[2]
        dim = q.shape[3]
        
        
        base_q = F.linear(self.base_fun(q), self.base_weight)
        base_k = F.linear(self.base_fun(k), self.base_weight)
        
        q = q.reshape(batch*heads*patches, dim)
        k = k.reshape(batch*heads*patches, dim)
        
        f_out_q, f_out_k = self.inner_function(q, k)
        

        f_out_q = f_out_q.reshape(batch, heads, patches, dim)*self.scale_sp + base_q*self.scale_base
        f_out_k = f_out_k.reshape(batch, heads, patches, dim)*self.scale_sp + base_k*self.scale_base
        
        f_out_q = torch.sigmoid(torch.unsqueeze(f_out_q, dim = 3))
        f_out_k = torch.sigmoid(torch.unsqueeze(f_out_k, dim = 2))
        

        f_out = (f_out_q + f_out_k)
        f_out = torch.sum(f_out, dim = 4)
        f_out = self.lin_qk(f_out)
        
        attn = f_out.softmax(dim=-1)
        
        
        return attn

class KA_attention_reduced(nn.Module):
    def __init__(self, out_dim, patches, heads, head_dim, num_f = 8, base_fun = nn.SiLU(), device = 'cuda'):
        super(KA_attention_reduced, self).__init__()
        
        self.num_f = num_f
        self.num = patches*heads*head_dim
        self.base_fun = base_fun
        self.device = device
        self.out_dim = out_dim
        
        self.grid = nn.Parameter((torch.arange(num_f).float() + 1), requires_grad = True)
        
        # self.base_weight_qk = nn.Parameter(torch.rand((patches*heads*self.out_dim, self.num), requires_grad = True))
        self.base_weight_qk = nn.Parameter(torch.rand((head_dim, head_dim), requires_grad = True))
        
        self.coef_q = nn.Parameter(torch.rand((self.num, self.num_f), requires_grad = True))
        self.coef_k = nn.Parameter(torch.rand((self.num, self.num_f), requires_grad = True))

        self.scale_base = nn.Parameter(torch.ones(1, heads, patches, head_dim)).requires_grad_(True)
        # self.scale_sp = nn.Parameter(torch.ones(patches*heads*self.out_dim)).requires_grad_(True)
        self.scale_sp = nn.Parameter(torch.ones(1, heads, patches, head_dim)).requires_grad_(True)
        
        
        # self.lin_qk = nn.Linear(self.num, patches*heads*self.out_dim)
        # self.lin_q = nn.Linear(head_dim, self.out_dim)
        # self.lin_k = nn.Linear(head_dim, self.out_dim)
        self.lin_qk = nn.Linear(head_dim, self.out_dim)

        
        
    def forward(self, q, k, _):
        batch = q.shape[0]
        heads = q.shape[1]
        patches = q.shape[2]
        dim = q.shape[3]

        
        qk = torch.cat((q, k), dim = 0)
        
        base_qk = F.linear(self.base_fun(qk), self.base_weight_qk)

        qk = qk.reshape(int(2*batch), patches*heads*dim)
        
        qk = torch.unsqueeze(qk, dim = 1)
        grid = torch.unsqueeze(self.grid, dim = 1)
        
        # print('grid/qk shapes: ', grid.shape, ' ', qk.shape)
        f_out_qk = torch.sin(grid*qk).permute(2, 1, 0)
        # print('f out qk shape: ', f_out_qk.shape)
        f_out_q = f_out_qk[:, :, 0:batch]
        f_out_k = f_out_qk[:, :, batch:batch*2]
        
        # print('shapes: ', self.coef_q.shape, f_out_q.shape)
        f_out_q = torch.einsum('ij,ijk->ijk', self.coef_q, f_out_q).permute(2, 1, 0)
        f_out_k = torch.einsum('ij,ijk->ijk', self.coef_k, f_out_k).permute(2, 1, 0)
        # print('out shape: ', f_out_q.shape)
        f_out_q = torch.sum(f_out_q, dim = 1)
        f_out_k = torch.sum(f_out_k, dim = 1)
        # print('after sum shape: ', f_out_q.shape)
        f_out_q = f_out_q.reshape(batch, heads, patches, dim)
        f_out_k = f_out_k.reshape(batch, heads, patches, dim)
        
        base_q = base_qk[0:batch, :]
        base_k = base_qk[batch:batch*2, :]
        
        # print('*-*-*-*fout qk shape: ', f_out_q.shape, self.scale_sp.shape)
        # print('*-*-*-*base qk shape: ', base_q.shape, self.scale_base.shape)
        f_out_q = f_out_q*self.scale_sp + base_q*self.scale_base
        f_out_k = f_out_k*self.scale_sp + base_k*self.scale_base
        
        y = self.lin_qk(f_out_q + f_out_k)
        
        # f_out_q = self.lin_q(f_out_q)
        # f_out_k = self.lin_k(f_out_k)
        # f_out_qk = torch.sum(f_out_qk, dim = 1)
        
        # base_q = base_qk[0:batch, :]
        # base_k = base_qk[batch:batch*2, :]
        
        # print('*-*-*-*fout qk shape: ', f_out_q.shape, self.scale_sp.shape)
        # print('*-*-*-*base qk shape: ', base_q.shape, self.scale_base.shape)
        # y_q = f_out_q*self.scale_sp + base_q*self.scale_base
        # y_k = f_out_k*self.scale_sp + base_k*self.scale_base
        
        # y = y_k + y_q
        y = y.reshape(batch, heads, patches, self.out_dim)
        attn = y.softmax(dim=-1)
        
        return attn


# class KA_attention_reduced(nn.Module):
#     def __init__(self, out_dim, patches, heads, head_dim, num_f = 8, base_fun = nn.SiLU(), device = 'cuda'):
#         super(KA_attention_reduced, self).__init__()
        
#         self.num_f = num_f
#         self.num = patches*heads*head_dim
#         self.base_fun = base_fun
#         self.device = device
#         self.out_dim = out_dim
        
#         self.grid = nn.Parameter((torch.arange(num_f).float() + 1), requires_grad = True)
        
#         # self.base_weight_qk = nn.Parameter(torch.rand((patches*heads*self.out_dim, self.num), requires_grad = True))
#         self.base_weight_qk = nn.Parameter(torch.rand((self.out_dim, head_dim), requires_grad = True))
        
#         self.coef_qk = nn.Parameter(torch.rand((self.num, self.num_f), requires_grad = True))

#         self.scale_base = nn.Parameter(torch.ones(patches*heads*self.out_dim)).requires_grad_(True)
#         # self.scale_sp = nn.Parameter(torch.ones(patches*heads*self.out_dim)).requires_grad_(True)
#         self.scale_sp = nn.Parameter(torch.ones(1, heads, patches, self.out_dim)).requires_grad_(True)
        
        
#         # self.lin_qk = nn.Linear(self.num, patches*heads*self.out_dim)
#         self.lin_qk = nn.Linear(head_dim, self.out_dim)

        
        
#     def forward(self, q, k, _):
#         batch = q.shape[0]
#         heads = q.shape[1]
#         patches = q.shape[2]
#         dim = q.shape[3]

        
#         qk = torch.cat((q, k), dim = 0)
        
#         base_qk = F.linear(self.base_fun(qk), self.base_weight_qk)
        

#         qk = qk.reshape(int(2*batch), patches*heads*dim)
        
#         qk = torch.unsqueeze(qk, dim = 1)
#         grid = torch.unsqueeze(self.grid, dim = 1)
        
#         # print('grid/qk shapes: ', grid.shape, ' ', qk.shape)
#         f_out_qk = torch.sin(grid*qk).permute(2, 1, 0)
        

#         f_out_qk = torch.einsum('ij,ijk->ijk', self.coef_qk, f_out_qk).permute(2, 1, 0)
        
#         f_out_qk = torch.sum(f_out_qk, dim = 1)
#         # print('after coef shape: ', f_out_qk.shape)
#         f_out_qk = f_out_qk.reshape(int(2*batch), heads, patches, dim)
        
#         f_out_qk = self.lin_qk(f_out_qk)
#         # f_out_qk = torch.sum(f_out_qk, dim = 1)
        
#         # print('*-*-*-*fout qk shape: ', f_out_qk.shape, self.scale_sp.shape)
#         y_qk = f_out_qk*self.scale_sp + base_qk
#         y_k = y_qk[0:batch, :]
#         y_q = y_qk[batch:batch*2, :]
        
#         y = y_k + y_q
#         y = y.reshape(batch, heads, patches, self.out_dim)
#         attn = y.softmax(dim=-1)
        
#         return attn

class KA_attention(nn.Module):
    def __init__(self, out_dim, patches, heads, head_dim, num_f = 8, base_fun = nn.SiLU(), device = 'cuda'):
        super(KA_attention, self).__init__()
        
        self.num_f = num_f
        self.num = patches*heads*head_dim
        self.base_fun = base_fun
        self.device = device
        self.out_dim = out_dim
        
        self.grid = nn.Parameter((torch.arange(num_f).float() + 1), requires_grad = True)
        
        self.base_weight_q = nn.Parameter(torch.rand((patches*heads*self.out_dim, self.num), requires_grad = True))
        self.base_weight_k = nn.Parameter(torch.rand((patches*heads*self.out_dim, self.num), requires_grad = True))
        
        self.coef_q = nn.Parameter(torch.rand((self.num, self.num_f), requires_grad = True))
        self.coef_k = nn.Parameter(torch.rand((self.num, self.num_f), requires_grad = True))

        
        self.scale_base = nn.Parameter(torch.ones(patches*heads*self.out_dim)).requires_grad_(True)
        self.scale_sp = nn.Parameter(torch.ones(patches*heads*self.out_dim)).requires_grad_(True)
        
        # self.conv_layer_q = nn.Conv1d(self.num*num_f, patches*heads*self.out_dim*num_f, kernel_size = 1, groups = num_f)
        # self.conv_layer_k = nn.Conv1d(self.num*num_f, patches*heads*self.out_dim*num_f, kernel_size = 1, groups = num_f)
        
        self.lin_q = nn.Linear(self.num, patches*heads*self.out_dim)
        self.lin_k = nn.Linear(self.num, patches*heads*self.out_dim)
        
        
    def forward(self, q, k):
        batch = q.shape[0]
        heads = q.shape[1]
        patches = q.shape[2]
        dim = q.shape[3]
        # print('input shape: ', q.shape, ' ', k.shape)
        q = q.reshape(batch, heads*patches*dim)
        k = k.reshape(batch, heads*patches*dim)
        
        base_q = F.linear(self.base_fun(q), self.base_weight_q)
        base_k = F.linear(self.base_fun(k), self.base_weight_k)
        
        q = torch.unsqueeze(q, dim = 1)
        k = torch.unsqueeze(k, dim = 1)
        grid = torch.unsqueeze(self.grid, dim = 1)
        
        f_out_q = torch.sin(grid*q).permute(2, 1, 0)
        f_out_k = torch.sin(grid*k).permute(2, 1, 0)
        
        
        f_out_q = torch.einsum('ij,ijk->ijk', self.coef_q, f_out_q).permute(2, 1, 0)
        f_out_k = torch.einsum('ij,ijk->ijk', self.coef_k, f_out_k).permute(2, 1, 0)
        
        # f_out_q = f_out_q.reshape(batch, -1).unsqueeze(dim = 2)
        # f_out_k = f_out_k.reshape(batch, -1).unsqueeze(dim = 2)
        
        # f_out_q = self.conv_layer_q(f_out_q)
        f_out_q = self.lin_q(f_out_q)
        # f_out_q = f_out_q.reshape(batch, self.num_f, patches*heads*self.out_dim)
        f_out_q = torch.sum(f_out_q, dim = 1)
        
        # f_out_k = self.conv_layer_k(f_out_k)
        f_out_k = self.lin_k(f_out_k)
        # f_out_k = f_out_k.reshape(batch, self.num_f, patches*heads*self.out_dim)
        f_out_k = torch.sum(f_out_k, dim = 1)
        
        y_k = f_out_q*self.scale_sp + base_q
        y_q = f_out_k*self.scale_sp + base_k
        
        y = y_k + y_q
        y = y.reshape(batch, heads, patches, self.out_dim)
        attn = y.softmax(dim=-1)
        
        return attn
        


class Grouped_KA_attention(nn.Module):
    def __init__(self, out_dim, patches, heads, head_dim, groups, num_f = 8, base_fun = nn.SiLU(), device = 'cuda'):
        super(Grouped_KA_attention, self).__init__()
        
        self.num_f = num_f
        self.num = patches*heads*head_dim
        self.base_fun = base_fun
        self.device = device
        self.out_dim = out_dim
        
        self.groups = groups
        self.group_size = int(self.num / self.groups)
        
        self.grid = nn.Parameter((torch.arange(num_f).float() + 1), requires_grad = True)
        
        self.base_weight_q = nn.Parameter(torch.rand((patches*heads*self.out_dim, self.num), requires_grad = True))
        self.base_weight_k = nn.Parameter(torch.rand((patches*heads*self.out_dim, self.num), requires_grad = True))
        
        self.coef_q = nn.Parameter(torch.rand((self.groups, 1, self.num_f), requires_grad = True))
        self.coef_k = nn.Parameter(torch.rand((self.groups, 1, self.num_f), requires_grad = True))
        
        self.bias = nn.Linear(self.out_dim*patches*heads, 1, bias = False)
        
        self.conv_layer_q = nn.Conv1d(self.num*num_f, patches*heads*self.out_dim*num_f, kernel_size = 1, groups = num_f)#, bias = False)
        self.conv_layer_k = nn.Conv1d(self.num*num_f, patches*heads*self.out_dim*num_f, kernel_size = 1, groups = num_f)#, bias = False)

        self.scale_sp = nn.Parameter(torch.ones(patches*heads*self.out_dim)).requires_grad_(True)
        
        self.act = nn.ReLU()
        
        
    def forward(self, q, k):
        batch = q.shape[0]
        heads = q.shape[1]
        patches = q.shape[2]
        dim = q.shape[3]
        
        q = q.reshape(batch, heads*patches*dim)
        k = k.reshape(batch, heads*patches*dim)
        
        base_q = F.linear(self.base_fun(q), self.base_weight_q)
        base_k = F.linear(self.base_fun(k), self.base_weight_k)
        
        q = torch.unsqueeze(q, dim = 1)
        k = torch.unsqueeze(k, dim = 1)
        grid = torch.unsqueeze(self.grid, dim = 1)
        
        f_out_q = torch.sin(grid*q).permute(2, 1, 0)
        f_out_k = torch.sin(grid*k).permute(2, 1, 0)
        
        f_out_q  = f_out_q.reshape(self.groups, self.group_size, self.num_f, batch)
        f_out_k  = f_out_k.reshape(self.groups, self.group_size, self.num_f, batch)
        
        
        f_out_q = torch.einsum('ijk,ijkl->ijkl', self.coef_q, f_out_q)
        f_out_q  = f_out_q.reshape(self.groups*self.group_size, self.num_f, batch).permute(2, 1, 0)
        
        f_out_k = torch.einsum('ijk,ijkl->ijkl', self.coef_k, f_out_k)
        f_out_k  = f_out_k.reshape(self.groups*self.group_size, self.num_f, batch).permute(2, 1, 0)
        
        f_out_q = f_out_q.reshape(batch, -1).unsqueeze(dim = 2)
        f_out_k = f_out_k.reshape(batch, -1).unsqueeze(dim = 2)
        
        f_out_q = self.conv_layer_q(f_out_q)
        f_out_q = f_out_q.reshape(batch, self.num_f, patches*heads*self.out_dim)
        f_out_q = torch.sum(f_out_q, dim = 1)*self.scale_sp
        
        f_out_k = self.conv_layer_q(f_out_k)
        f_out_k = f_out_k.reshape(batch, self.num_f, patches*heads*self.out_dim)
        f_out_k = torch.sum(f_out_k, dim = 1)*self.scale_sp
        
        y_k = f_out_q*self.scale_sp + base_q
        y_q = f_out_k*self.scale_sp + base_k
        
        y = y_k + y_q
        y = y.reshape(batch, heads, patches, self.out_dim)
        attn = y.softmax(dim=-1)
        
        return attn

class KAN_attention(nn.Module):
    def __init__(self, in_dim, out_dim, patches, heads, head_dim, poly = 3, noise_scale=0.1, scale_base=1.0, scale_sp=1.0, num_f = 5, base_fun = nn.SiLU(), grid_range=[-1, 1], sp_trainable=True, sb_trainable=True, device = 'cuda'):
        super(KAN_attention, self).__init__()
        
        self.num_f = num_f
        self.patches = patches
        self.heads = heads
        self.head_dim = head_dim
        self.num = patches*heads*head_dim
        self.base_fun = base_fun
        self.device = device
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.act_in_dim = self.in_dim*patches*heads
        self.act_out_dim = self.out_dim*patches*heads
        self.poly = poly
        
        self.grid = torch.einsum('i,j->ij', torch.ones(self.act_in_dim*self.act_out_dim, device=device), torch.linspace(grid_range[0], grid_range[1], steps = self.num_f + 1, device = device))
        self.grid = torch.nn.Parameter(self.grid).requires_grad_(False)
        
        
        noises = (torch.rand(self.act_in_dim*self.act_out_dim, self.grid.shape[1]) - 1 / 2) * noise_scale / self.num_f
        noises = noises.to(device)
        # shape: (size, coef)
        self.coef_q = torch.nn.Parameter(curve2coef(self.grid, noises, self.grid, poly, device))
        self.coef_k = torch.nn.Parameter(curve2coef(self.grid, noises, self.grid, poly, device))
        
        if isinstance(scale_base, float):
            self.scale_base = torch.nn.Parameter(torch.ones(self.act_in_dim*self.act_out_dim, device = device) * scale_base).requires_grad_(sb_trainable)  # make scale trainable
        else:
            self.scale_base = torch.nn.Parameter(torch.FloatTensor(scale_base).to(device)).requires_grad_(sb_trainable)
            
        self.scale_sp = torch.nn.Parameter(torch.ones(self.act_in_dim*self.act_out_dim, device=device) * scale_sp).requires_grad_(sp_trainable)  # make scale trainable
        
        self.mask_q = torch.nn.Parameter(torch.ones(self.act_in_dim*self.act_out_dim, device=device)).requires_grad_(False)
        self.mask_k = torch.nn.Parameter(torch.ones(self.act_in_dim*self.act_out_dim, device=device)).requires_grad_(False)
        
        self.bias = nn.Linear(self.out_dim*patches*heads, 1, bias = False)

        
        self.act = nn.ReLU()
        
        self.weight_sharing = torch.arange(self.act_in_dim*self.act_out_dim)
        
        
    def forward(self, q, k):
        batch = q.shape[0]
        heads = q.shape[1]
        patches = q.shape[2]
        dim = q.shape[3]
        
        print('input shape: ', q.shape, k.shape)
        print('in_dim: ', self.in_dim)
        print('patches: ', self.patches)
        print('heads: ', self.heads)
        print('head dim: ', self.head_dim)
        
        q = q.reshape(batch, heads*patches*dim)#.permute(1, 0)
        k = k.reshape(batch, heads*patches*dim)#.permute(1, 0)
        q = torch.einsum('ij,k->ikj', q, torch.ones(self.act_out_dim, device = self.device)).reshape(batch, self.act_in_dim*self.act_out_dim).permute(1, 0)
        k = torch.einsum('ij,k->ikj', k, torch.ones(self.act_out_dim, device = self.device)).reshape(batch, self.act_in_dim*self.act_out_dim).permute(1, 0)
        
        base_q = self.base_fun(q).permute(1, 0)
        base_k = self.base_fun(k).permute(1, 0)
        

        f_out_q = coef2curve(x_eval = q, grid = self.grid[self.weight_sharing], coef = self.coef_q[self.weight_sharing], k = self.poly, device = self.device)
        f_out_k = coef2curve(x_eval = k, grid = self.grid[self.weight_sharing], coef = self.coef_k[self.weight_sharing], k = self.poly, device = self.device)

        
        f_out_q = f_out_q.permute(1, 0)
        f_out_k = f_out_k.permute(1, 0)
        
        f_out_q = self.base_q.unsqueeze(dim=0) * base + self.scale_sp.unsqueeze(dim=0) * f_out_q
        f_out_k = self.base_k.unsqueeze(dim=0) * base + self.scale_sp.unsqueeze(dim=0) * f_out_k
        
        f_out_q = self.mask_q[None, :] * f_out_q
        f_out_k = self.mask_k[None, :] * f_out_k
        
        y_k = torch.sum(f_out_q.reshape(batch, heads, patches, dim, self.out_dim), dim=3)
        y_q = torch.sum(f_out_k.reshape(batch, heads, patches, dim, self.out_dim), dim=3)
        
        y = y_k + y_q + self.bias.weight.reshape(1, heads, patches, self.out_dim)
        attn = y.softmax(dim=-1)
        
        return attn
        
