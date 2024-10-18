import torch
import torch.nn as nn
from collections import OrderedDict

# 假设你有几个预定义的模型类
class SimpleModelA(nn.Module):
    def __init__(self):
        super(SimpleModelA, self).__init__()
        self.model = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 16, kernel_size=3)),
            ('relu', nn.ReLU()),
            ('pool', nn.MaxPool2d(2))
        ]))

    def forward(self, x):
        return self.model(x)

class SimpleModelB(nn.Module):
    def __init__(self):
        super(SimpleModelB, self).__init__()
        self.model = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(32*32*3, 100)),
            ('relu', nn.ReLU()),
            ('fc2', nn.Linear(100, 10)),
        ]))

    def forward(self, x):
        return self.model(x)

# 模型注册映射
MODEL_REGISTRY = {
    'simple_model_a': SimpleModelA,
    'simple_model_b': SimpleModelB,
}




register_model_deprecations(__name__, {
    'vit_tiny_patch16_224_in21k': 'vit_tiny_patch16_224.augreg_in21k',
    'vit_small_patch32_224_in21k': 'vit_small_patch32_224.augreg_in21k',
    'vit_small_patch16_224_in21k': 'vit_small_patch16_224.augreg_in21k',
    'vit_base_patch32_224_in21k': 'vit_base_patch32_224.augreg_in21k',
    'vit_base_patch16_224_in21k': 'vit_base_patch16_224.augreg_in21k',
    'vit_base_patch8_224_in21k': 'vit_base_patch8_224.augreg_in21k',
    'vit_large_patch32_224_in21k': 'vit_large_patch32_224.orig_in21k',
    'vit_large_patch16_224_in21k': 'vit_large_patch16_224.augreg_in21k',
    'vit_huge_patch14_224_in21k': 'vit_huge_patch14_224.orig_in21k',
    'vit_base_patch32_224_sam': 'vit_base_patch32_224.sam',
    'vit_base_patch16_224_sam': 'vit_base_patch16_224.sam',
    'vit_small_patch16_224_dino': 'vit_small_patch16_224.dino',
    'vit_small_patch8_224_dino': 'vit_small_patch8_224.dino',
    'vit_base_patch16_224_dino': 'vit_base_patch16_224.dino',
    'vit_base_patch8_224_dino': 'vit_base_patch8_224.dino',
    'vit_base_patch16_224_miil_in21k': 'vit_base_patch16_224_miil.in21k',
    'vit_base_patch32_224_clip_laion2b': 'vit_base_patch32_clip_224.laion2b',
    'vit_large_patch14_224_clip_laion2b': 'vit_large_patch14_clip_224.laion2b',
    'vit_huge_patch14_224_clip_laion2b': 'vit_huge_patch14_clip_224.laion2b',
    'vit_giant_patch14_224_clip_laion2b': 'vit_giant_patch14_clip_224.laion2b',
})




def create_model(
    model_name: str, 
    pretrained: bool = False, 
    checkpoint_path: str = '', 
    num_classes: int = 1000,
    **kwargs
):
    # 检查模型名称是否在注册的模型中
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} is not registered.")
    
    # 从注册的模型中创建模型实例
    model_class = MODEL_REGISTRY[model_name]
    model = model_class(**kwargs)
    
    # 如果需要预训练模型，可以加载权重
    if pretrained:
        print(f"Loading pretrained weights for {model_name}...")
        # 在这里实现加载预训练权重的逻辑
        # 比如通过加载保存的 checkpoint 文件来加载权重
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # 如果没有指定路径，可以默认加载一些权重（这一步需要根据具体实现自定义）
            raise NotImplementedError("Pretrained weights not available for this model.")
    
    return model
