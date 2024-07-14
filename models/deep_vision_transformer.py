""" 
Code for DeepViT. The implementation has heavy reference to timm.
"""
import torch
import torch.nn as nn
from functools import partial
import pickle
from torch.nn.parameter import Parameter

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.resnet import resnet26d, resnet50d
from timm.models.registry import register_model

from .layers import *


from torch.nn import functional as F

import numpy as np


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # patch models
    'Deepvit_base_patch16_224_16B': _cfg(
        url='',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'Deepvit_base_patch16_224_24B': _cfg(
        url='',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'Deepvit_base_patch16_224_32B': _cfg(
        url='',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'Deepvit_L_384': _cfg(
        url='',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
}

def KL_divergence_loss(patched_tensor):
    t1 = patched_tensor.unsqueeze(2)
    t2 = patched_tensor.unsqueeze(1)
    t1 = F.log_softmax(t1)
    t2 = F.softmax(t2)

    kl_function = nn.KLDivLoss(reduction = 'none')
    kl_loss = kl_function(t1, t2).sum(-1)
    kl_loss -= torch.diag_embed(kl_loss.diagonal(dim1=1, dim2=2))
    
    return kl_loss.mean()

def kl_divergence(p, q):
    return (p * (p / q).log()).sum(-1)

def Jeffrey_distance(patched_tensor):
    t1 = patched_tensor.unsqueeze(2)
    t2 = patched_tensor.unsqueeze(1)

    kl_function = nn.KLDivLoss(reduction = 'none')

    kl_1 = kl_function(F.log_softmax(t1), F.softmax(t2)).sum(-1)
    kl_2 = kl_function(F.log_softmax(t2), F.softmax(t1)).sum(-1)
    kl_1 -= torch.diag_embed(kl_1.diagonal(dim1=1, dim2=2))
    kl_2 -= torch.diag_embed(kl_2.diagonal(dim1=1, dim2=2))

    return 0.5*kl_1.mean() + 0.5*kl_2.mean()

def Jensen_Shannon_loss(patched_tensor):
    p = patched_tensor.unsqueeze(1)
    q = patched_tensor.unsqueeze(2)
    kl_function = nn.KLDivLoss(reduction = 'none')

    m = 0.5 * (p + q)
    kl1 = kl_function(F.log_softmax(p), F.softmax(m)).sum(-1)
    kl2 = kl_function(F.log_softmax(q), F.softmax(m)).sum(-1)
    # kl1 = kl_divergence(F.softmax(p), F.softmax(m))
    # kl2 = kl_divergence(F.softmax(q), F.softmax(m))
    JS_loss = 0.5 * (kl1 + kl2)
    # print('JS Loss shape: ', JS_loss.shape)
    
    # print('All KLs: ', kl1.mean(), ' ', kl2.mean())
    JS_loss -= torch.diag_embed(JS_loss.diagonal(dim1=1, dim2=2))
    # print('JS loss shape: ', JS_loss.shape)
    return abs(JS_loss.mean())

def cosine_similarity(patched_tensor):
    p = patched_tensor.unsqueeze(1)
    q = patched_tensor.unsqueeze(2)
    cosine_loss = nn.CosineSimilarity()

    cs_loss = cosine_loss(p, q)
    return cs_loss.mean()

def similarity(patches, context_target, high_k=7):
        num_batch, num_dim, img_size = patches.shape[0], patches.shape[1], patches.shape[-1]
        low_order = F.avg_pool2d(patches, kernel_size=high_k, stride=1, padding=(high_k - 1) // 2) - patches / (high_k ** 2)
        low_order *= (high_k ** 2) / (high_k ** 2 - 1)
        # print('Patches / first shape: ', patches.shape, ' ', context_target.shape)
        high_order = (patches.mean(dim=(1, 2), keepdim=True) - patches / (img_size ** 2)) * (img_size ** 2) / (img_size ** 2 - 1)

        # print('Shapes: ', patches.reshape(num_batch, num_dim, -1).shape)
        # pos_l = (context_target * patches.reshape(num_batch, num_dim, -1).permute(0, 2, 1)).sum(-1).reshape(-1, 1)
        pos_l = (context_target * patches.reshape(num_batch, num_dim, -1)).sum(-1).reshape(-1, 1)
        neg_l = (context_target * high_order.reshape(num_batch, num_dim, -1)).sum(-1).reshape(-1, 1)
        neg2_l = (context_target * low_order.reshape(num_batch, num_dim, -1)).sum(-1).reshape(-1, 1)
        return - torch.log(torch.cat((pos_l, neg_l, neg2_l), dim=-1).softmax(dim=-1)[:, 0] + 1e-8).mean()

class DeepVisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, group = False, re_atten=False, cos_reg = False,
                 use_cnn_embed=False, apply_transform=None, transform_scale=False, scale_adjustment=1.):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        # use cosine similarity as a regularization term
        self.cos_reg = cos_reg

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            if use_cnn_embed:
                self.patch_embed = PatchEmbed_CNN(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
            else:
                self.patch_embed = PatchEmbed(
                    img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        d = depth if isinstance(depth, int) else len(depth)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, d)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, share=depth[i], num_heads=num_heads, num_patches = num_patches + 1, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, group = group, 
                re_atten=re_atten, apply_transform=apply_transform[i], transform_scale=transform_scale, scale_adjustment=scale_adjustment)
            for i in range(len(depth))])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        # print('--- original input shape: ', x.shape)
        # if self.cos_reg:
        atten_list = []
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        attn = None
        # print('--- blocks input: ', x.shape)
        first = None
        for blk in self.blocks:
            x, attn = blk(x, attn)
            if first == None:
                first = x
            
            # if self.cos_reg:
            atten_list.append(attn)

        # print('--- blocks output: ', x.shape)
        # kl_loss = KL_divergence_loss(x)
        # JS_loss = Jensen_Shannon_loss(x)
        # sim_loss = similarity(x, first)
        # sim_loss = cosine_similarity(x) ############################################ Patch diversification code ###############################################################
        x = self.norm(x)
        if self.cos_reg and self.training:
            return x[:, 0], atten_list#, sim_loss
        else:
            return x[:, 0], atten_list

    def forward(self, x):
        if self.cos_reg and self.training:
            x, atten = self.forward_features(x)
            x = self.head(x)
            return x
        else:
            x, atten_list = self.forward_features(x)
            # print('x shape: ', x.shape)
            x = self.head(x)
            # print('head shape: ', x.shape)
            return x


@register_model
def deepvit_patch16_224_re_attn_16b(pretrained=False, **kwargs):
    apply_transform = [False] * 0 + [True] * 16
    model = DeepVisionTransformer(
        patch_size=16, embed_dim=256, depth=[False] * 16, apply_transform=apply_transform, num_heads=12, mlp_ratio=3, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),  **kwargs)
    # We following the same settings for original ViT
    model.default_cfg = default_cfgs['Deepvit_base_patch16_224_16B']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
    return model

@register_model
def deepvit_patch16_224_re_attn_24b(pretrained=False, **kwargs):
    apply_transform = [False] * 0 + [True] * 24
    model = DeepVisionTransformer(
        patch_size=16, embed_dim=384, depth=[False] * 24, apply_transform=apply_transform, num_heads=12, mlp_ratio=3, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),  **kwargs)
    # We following the same settings for original ViT
    model.default_cfg = default_cfgs['Deepvit_base_patch16_224_24B']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
    return model
 
@register_model
def deepvit_patch16_224_re_attn_32b(pretrained=False, **kwargs):
    apply_transform = [False] * 0 + [True] * 32
    model = DeepVisionTransformer(
        patch_size=16, embed_dim=384, depth=[False] * 32, apply_transform=apply_transform, num_heads=12, mlp_ratio=3, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),  **kwargs)
    # We following the same settings for original ViT
    model.default_cfg = default_cfgs['Deepvit_base_patch16_224_32B']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
    return model

@register_model
def deepvit_S(pretrained=False, **kwargs):
    apply_transform = [False] * 11 + [True] * 5
    model = DeepVisionTransformer(
        embed_dim=128, depth=[False] * 16, apply_transform=apply_transform, num_heads=8, mlp_ratio=3, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),  transform_scale=True, use_cnn_embed = True, scale_adjustment=0.5, **kwargs)
    
    # We following the same settings for original ViT
    model.default_cfg = default_cfgs['Deepvit_base_patch16_224_32B']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
    return model

# @register_model
# def deepvit_S(pretrained=False, **kwargs):
#     apply_transform = [False] * 11 + [True] * 5
#     model = DeepVisionTransformer(
#         patch_size=16, embed_dim=396, depth=[False] * 16, apply_transform=apply_transform, num_heads=12, mlp_ratio=3, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),  transform_scale=True, use_cnn_embed = True, scale_adjustment=0.5, **kwargs)
    
#     # We following the same settings for original ViT
#     model.default_cfg = default_cfgs['Deepvit_base_patch16_224_32B']
#     if pretrained:
#         load_pretrained(
#             model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
#     return model

@register_model
def deepvit_L(pretrained=False, **kwargs):
    apply_transform = [False] * 20 + [True] * 12
    model = DeepVisionTransformer(
        patch_size=16, embed_dim=256, depth=[False] * 32, apply_transform=apply_transform, num_heads=12, mlp_ratio=3, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), use_cnn_embed = True, scale_adjustment=0.5, **kwargs)
    # We following the same settings for original ViT
    model.default_cfg = default_cfgs['Deepvit_base_patch16_224_32B']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
    return model

@register_model
def deepvit_L_384(pretrained=False, **kwargs):
    apply_transform = [False] * 20 + [True] * 12
    model = DeepVisionTransformer(
        img_size=384, patch_size=16, embed_dim=420, depth=[False] * 32, apply_transform=apply_transform, num_heads=12, mlp_ratio=3, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), use_cnn_embed = True, scale_adjustment=0.5, **kwargs)
    # We following the same settings for original ViT
    model.default_cfg = default_cfgs['Deepvit_L_384']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
    return model
