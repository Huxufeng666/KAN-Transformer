import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import ParameterGrid

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import optuna

# 数据集的转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载 CIFAR-10 数据集
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=2)

# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=2)

# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
def get_cifar10_loaders(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader

class SingleVariableFunction(nn.Module):
    """单变量函数逼近器"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.SiLU()  # 可根据需要选择其他激活函数

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x

class GroupedKAAttention(nn.Module):
    """
    分组 Kolmogorov-Arnold 注意力机制
    """
    def __init__(self, total_dim, patches, heads, head_dim, hidden_dim, groups):
        super().__init__()
        self.total_dim = total_dim
        self.groups = groups
        self.group_size = total_dim // groups

        self.svfs_q = nn.ModuleList([
            SingleVariableFunction(self.group_size, hidden_dim, patches)
            for _ in range(groups)
        ])
        self.svfs_k = nn.ModuleList([
            SingleVariableFunction(self.group_size, hidden_dim, patches)
            for _ in range(groups)
        ])

        self.global_function = SingleVariableFunction(groups, hidden_dim, heads)

    def forward(self, q, k):
        batch_size = q.shape[0]
        print(q.shape)

        q_groups = q.view(batch_size, self.groups, self.group_size).transpose(1, 2)
        k_groups = k.view(batch_size, self.groups, self.group_size).transpose(1, 2)

        q_features = [svf(q_groups[:, :, i]) for i, svf in enumerate(self.svfs_q)]
        k_features = [svf(k_groups[:, :, i]) for i, svf in enumerate(self.svfs_k)]

        q_features = torch.stack(q_features, dim=2).view(batch_size, -1)
        k_features = torch.stack(k_features, dim=2).view(batch_size, -1)

        q_out = self.global_function(q_features).view(batch_size, -1)
        k_out = self.global_function(k_features).view(batch_size, -1)

        attn = (q_out * k_out).sum(dim=-1)
        attn = attn.softmax(dim=-1)

        return attn

# # 定义超参数搜索空间
# param_grid = {
#     'groups': [128,256,512,1024]  # 根据需求调整
# }

# # 初始化参数网格
# grid = ParameterGrid(param_grid)

# # 定义模型训练函数
# def train_model(groups):
#     # 实例化模型
#     model = GroupedKAAttention(groups=groups).cuda()
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#     # 训练模型
#     for epoch in range(2):  # 这里示范训练 2 个 epoch
#         running_loss = 0.0
#         for i, data in enumerate(trainloader, 0):
#             inputs, labels = data[0].cuda(), data[1].cuda()

#             optimizer.zero_grad()

#             outputs = model(inputs, inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()
#             if i % 1000 == 999:  # 每 1000 mini-batches 打印一次
#                 print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 1000:.3f}")
#                 running_loss = 0.0

#     print('Finished Training')
#     return model

# # 网格搜索
# best_acc = 0.0
# best_groups = None
# for params in grid:
#     groups = params['groups']
#     print(f"Training with groups = {groups}")
#     model = train_model(groups)

#     # 评估模型
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for data in testloader:
#             images, labels = data[0].cuda(), data[1].cuda()
#             outputs = model(images, images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#     acc = correct / total
#     print(f"Accuracy for groups = {groups}: {acc:.3f}")

#     if acc > best_acc:
#         best_acc = acc
#         best_groups = groups

# print(f"Best groups: {best_groups} with accuracy: {best_acc:.3f}")
    


def objective(trial):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 超参数
    groups = trial.suggest_int('groups', 128, 256)
    
    # 初始化模型
    model = GroupedKAAttention(
        total_dim=3*224*224,
        patches=7,
        heads=8,
        head_dim=16,
        hidden_dim=128,
        groups=groups
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    train_loader, test_loader = get_cifar10_loaders(batch_size=128)
    
    # 训练模型
    model.train()
    for epoch in range(10):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data, data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    
    # 测试模型
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data, data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    accuracy = correct / len(test_loader.dataset)
    return accuracy


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

print("最佳超参数: ", study.best_params)
print("最佳准确率: ", study.best_value)
