"""
original code from rwightman:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn

class lin_func(nn.Module):
    def __init__(self, min_val, max_val):
        super(lin_func, self).__init__()
        
        self.min_val = min_val
        self.max_val = max_val
        

    def smooth_step(self, x):

      if x < self.min_val:
        return self.min_val
      elif x > self.max_val:
        return self.max_val
      else:
        return x#(x - x_min) / (x_max - x_min) - (x - x_min)**2 / ((x_max - x_min)**2)
    
    def piecewise_linear(self, x):
      slope = 1.0
    
      # intercept = 0.0
      return slope * self.smooth_step(x)# * (x - 4.0) + intercept * self.smooth_step(50.0, 100.0, x)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class KA_attention(nn.Module):
    def __init__(self, out_dim, patches, heads, head_dim, num_f = 8, base_fun = nn.SiLU(), device = 'cuda'):
        super(KA_attention, self).__init__()
        
        self.num_f = num_f
        self.num = patches*heads*head_dim
        self.base_fun = base_fun
        self.device = device
        self.out_dim = out_dim
        
        self.grid = nn.Parameter((torch.arange(num_f).float() + 1), requires_grad = True)
        self.coef_q = nn.Parameter(torch.rand((self.num*self.out_dim, self.num_f), requires_grad = True))
        self.coef_k = nn.Parameter(torch.rand((self.num*self.out_dim, self.num_f), requires_grad = True))
        self.bias = nn.Linear(self.out_dim*patches*heads, 1, bias = False)

        
        self.scale_base = nn.Parameter(torch.ones(self.num*self.out_dim)).requires_grad_(True)
        self.scale_sp = nn.Parameter(torch.ones(self.num*self.out_dim)).requires_grad_(True)
        
        self.act = nn.ReLU()
        
        
    def forward(self, q, k):
        # Q, K and V -> [32, 8, 17, 16] - [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # print('Q, K, V shapes: ', q.shape, k.shape)
        # print('In self num: ', self.num) # [2175]
        batch = q.shape[0]
        heads = q.shape[1]
        patches = q.shape[2]
        dim = q.shape[3]
        
        q = q.reshape(batch, heads*patches*dim)#.permute(1, 0)
        k = k.reshape(batch, heads*patches*dim)#.permute(1, 0)
        q = torch.einsum('ij,k->ikj', q, torch.ones(self.out_dim, device = self.device)).reshape(batch, heads*patches*dim*self.out_dim).permute(1, 0)
        k = torch.einsum('ij,k->ikj', k, torch.ones(self.out_dim, device = self.device)).reshape(batch, heads*patches*dim*self.out_dim).permute(1, 0)
        
        base_q = self.base_fun(q)
        base_k = self.base_fun(k)
        
        q = torch.unsqueeze(q, dim = 1)
        k = torch.unsqueeze(k, dim = 1)
        grid = torch.unsqueeze(self.grid, dim = 1)
        # print('--before grid: ', q.shape) # [36992, 1, 32]
        f_out_q = torch.sin(grid*q)
        f_out_k = torch.sin(grid*k)
        # print('--after sin: ', f_out_q.shape) # [36992, 8, 32]
        # print('--coef shape: ', self.coef_q.shape) # [36992, 8]
        f_out_q = torch.einsum('ij,ijk->ik', self.coef_q, f_out_q).permute(1, 0)
        f_out_k = torch.einsum('ij,ijk->ik', self.coef_k, f_out_k).permute(1, 0)
        # print('--after coef: ', f_out_q.shape) # [32, 36992]
        
        f_out_q = self.scale_sp*f_out_q + self.scale_base*base_q.permute(1, 0)
        f_out_k = self.scale_sp*f_out_k + self.scale_base*base_k.permute(1, 0)
        # print('--after base: ', f_out_q.shape) # [32, 36992]
        
        # y_k = f_out_q.reshape(batch, heads, patches, dim, self.out_dim)
        # y_q = f_out_k.reshape(batch, heads, patches, dim, self.out_dim)
        y_k = torch.sum(f_out_q.reshape(batch, heads, patches, dim, self.out_dim), dim=3)
        y_q = torch.sum(f_out_k.reshape(batch, heads, patches, dim, self.out_dim), dim=3) # [ 32, 8, 17, 16, 17]
        # print('output shape: ', y_q.shape, self.bias.weight.reshape(1, heads, patches, self.out_dim).shape) # [32, 8, 17, 17], [1, 8, 17, 17]
        y = y_k + y_q + self.bias.weight.reshape(1, heads, patches, self.out_dim)
        attn = y.softmax(dim=-1)
        
        return attn
        
class Save_vals():
    def __init__(self):
        self.saves_record = []
    
    def save_val(self, x):
        self.saves_record.append(x)
        
    def return_array(self):
        return self.saves_record
    
    def return_unique(self):
        return torch.unique(self.saves_record)
    
    def return_last_val(self):
        return self.saves_record[len(self.saves_record)-1]
    


class Grouped_KA_attention(nn.Module):
    def __init__(self, out_dim, patches, heads, head_dim, groups, num_f = 8, base_fun = nn.SiLU(), device = 'cuda'):
        super(Grouped_KA_attention, self).__init__()
        
        self.num_f = num_f
        self.num = patches*heads*head_dim
        self.base_fun = base_fun
        self.device = device
        self.out_dim = out_dim
        
        self.groups = nn.Parameter(torch.tensor(groups).float())
        self.linear_func = lin_func(4, self.num*self.out_dim)
        
        
        self.grid = nn.Parameter((torch.arange(num_f).float() + 1), requires_grad = True)
        self.coef_q = nn.Parameter(torch.rand((int(self.groups), 1, self.num_f), requires_grad = True))
        self.coef_k = nn.Parameter(torch.rand((int(self.groups), 1, self.num_f), requires_grad = True))
        self.bias = nn.Linear(self.out_dim*patches*heads, 1, bias = False)

        
        self.scale_base = nn.Parameter(torch.ones(self.num*self.out_dim)).requires_grad_(True)
        self.scale_sp = nn.Parameter(torch.ones(self.num*self.out_dim)).requires_grad_(True)
        
        self.act = nn.ReLU()
        
        self.group_record = Save_vals()
        
        
    def forward(self, q, k):
        # Q, K and V -> [32, 8, 17, 16] - [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # print('Q, K, V shapes: ', q.shape, k.shape)
        # print('In self num: ', self.num) # [2175]
        batch = q.shape[0]
        heads = q.shape[1]
        patches = q.shape[2]
        dim = q.shape[3]
        
        q = q.reshape(batch, heads*patches*dim)#.permute(1, 0)
        k = k.reshape(batch, heads*patches*dim)#.permute(1, 0)
        q = torch.einsum('ij,k->ikj', q, torch.ones(self.out_dim, device = self.device)).reshape(batch, heads*patches*dim*self.out_dim).permute(1, 0)
        k = torch.einsum('ij,k->ikj', k, torch.ones(self.out_dim, device = self.device)).reshape(batch, heads*patches*dim*self.out_dim).permute(1, 0)
        
        base_q = self.base_fun(q)
        base_k = self.base_fun(k)
        
        q = torch.unsqueeze(q, dim = 1)
        k = torch.unsqueeze(k, dim = 1)
        grid = torch.unsqueeze(self.grid, dim = 1)
        # print('--before grid: ', q.shape) # [36992, 1, 32]
        f_out_q = torch.sin(grid*q)
        f_out_k = torch.sin(grid*k)
        # print('--after sin: ', f_out_q.shape) # [36992, 8, 32]
        print('--coef shape: ', self.coef_q.shape) # [36992, 8]
        
        group_val = torch.floor(self.linear_func.piecewise_linear(self.groups))
        print('== group value: ', group_val)
        self.group_record.save_val(group_val)
        group_size = int(heads*patches*dim*self.out_dim / group_val)
        # print('== group size: ', group_size)
        
        # print('== all shaeps; ', group_val.detach().cpu().numpy(), group_size, self.num_f, batch)
        f_out_q  = f_out_q.reshape(int(group_val.detach().cpu().numpy()), group_size, self.num_f, batch)
        f_out_k  = f_out_k.reshape(int(group_val.detach().cpu().numpy()), group_size, self.num_f, batch)
        
        # print('-- In shapes: ', self.coef_q.shape, f_out_q.shape)
        f_out_q = torch.einsum('ijk,ijkl->ijl', self.coef_q, f_out_q)
        f_out_k = torch.einsum('ijk,ijkl->ijl', self.coef_k, f_out_k)
        # print('--after coef: ', f_out_q.shape) # [32, 36992]
        
        f_out_q  = f_out_q.reshape(int(group_val.detach().cpu().numpy())*group_size, batch).permute(1, 0)
        f_out_k  = f_out_k.reshape(int(group_val.detach().cpu().numpy())*group_size, batch).permute(1, 0)
        
        f_out_q = self.scale_sp*f_out_q + self.scale_base*base_q.permute(1, 0)
        f_out_k = self.scale_sp*f_out_k + self.scale_base*base_k.permute(1, 0)
        # print('--after base: ', f_out_q.shape) # [32, 36992]
        
        # y_k = f_out_q.reshape(batch, heads, patches, dim, self.out_dim)
        # y_q = f_out_k.reshape(batch, heads, patches, dim, self.out_dim)
        y_k = torch.sum(f_out_q.reshape(batch, heads, patches, dim, self.out_dim), dim=3)
        y_q = torch.sum(f_out_k.reshape(batch, heads, patches, dim, self.out_dim), dim=3) # [ 32, 8, 17, 16, 17]
        # print('output shape: ', y_q.shape, self.bias.weight.reshape(1, heads, patches, self.out_dim).shape) # [32, 8, 17, 17], [1, 8, 17, 17]
        y = y_k + y_q + self.bias.weight.reshape(1, heads, patches, self.out_dim)
        attn = y.softmax(dim=-1)
        
        return attn

class Attention(nn.Module):
    def __init__(self,
                 dim, num_patches,  # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.num_patches = num_patches
        head_dim = dim // num_heads
        
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        
        # self.attention = KA_attention(out_dim = self.num_patches+1, patches = (self.num_patches+1), heads = self.num_heads, head_dim = head_dim)
        self.attention = Grouped_KA_attention(out_dim = self.num_patches+1, patches = (self.num_patches+1), heads = self.num_heads, head_dim = head_dim, groups = 4624)
        

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape # [32, 17, 128]
        # print('input x/patches shape: ', x.shape, self.num_patches)
        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        
        qkv = self.qkv(x) # [32, 17, 384]
        
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        
        attn = self.attention(q, k)
        # print('==out attention shape: ', attn.shape)
        
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        # attn = (q @ k.transpose(-2, -1)) * self.scale
        # attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)
        # print('==original attention shape: ', attn.shape)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim, num_patches,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_patches, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_patches = num_patches, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 196, 768]
        # [1, 1, 768] -> [B, 1, 768]
        print('Patch embed shape: ', x.shape)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
            
        print('Class embed shape: ', x.shape)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)

def kit_base_patch16_224(num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1zqb08naP0RPqqfSXfkB2EA  密码: eu9f
    """
    model = VisionTransformer(img_size=32,
                              patch_size=8,
                              embed_dim=128,
                              depth=6,
                              num_heads=8,
                              representation_size=None,
                              num_classes=num_classes)
    return model

def vit_base_patch16_224(num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1zqb08naP0RPqqfSXfkB2EA  密码: eu9f
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_base_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_base_patch32_224(num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1hCv0U8pQomwAtHBYc4hmZg  密码: s5hl
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_base_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch32_224_in21k-8db57226.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_large_patch16_224(num_classes: int = 1000):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1cxBgZJJ6qUWPSBNcE4TdRQ  密码: qqt8
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_large_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_large_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_huge_patch14_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: converted weights not currently available, too large for github release hosting.
    """
    model = VisionTransformer(img_size=224,
                              patch_size=14,
                              embed_dim=1280,
                              depth=32,
                              num_heads=16,
                              representation_size=1280 if has_logits else None,
                              num_classes=num_classes)
    return model
