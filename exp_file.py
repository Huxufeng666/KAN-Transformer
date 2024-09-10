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
import tqdm
# 数据集的转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载 CIFAR-10 数据集

def get_cifar10_loaders(batch_size=16):
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
    def __init__(self, input_dim, hidden_dim, hidden_dim1, hidden_dim2=512, hidden_dim3=1024, hidden_dim4=2048, output_dim=2048):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        # self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.fc4 = nn.Linear(hidden_dim3, hidden_dim4)
        self.fc5 = nn.Linear(hidden_dim4, output_dim)
        self.act = nn.SiLU()  # 可根据需要选择其他激活函数

    def forward(self, x):
        x = self.act(self.fc1(x))
        # x = self.act(self.fc2(x))
        # x = self.act(self.fc3(x))
        x = self.act(self.fc4(x))
        print('x:', x.shape)
        return x

# class SingleVariableFunction(nn.Module):

#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super().__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, output_dim)
#         self.act = nn.SiLU()  # 可根据需要选择其他激活函数

#     def forward(self, x):
#         x = self.act(self.fc1(x))
#         x = self.fc2(x)
#         # print('x:', x.shape)
#         return x



class GroupedKAAttention(nn.Module):
    
    def __init__(self, total_dim, patches, heads, head_dim, hidden_dim, groups):
        super().__init__()
        self.total_dim = total_dim
        self.groups = groups
        # self.group_size = total_dim // groups
        self.group_size = 588
        

        self.svfs_q = nn.ModuleList([
            SingleVariableFunction(self.group_size, hidden_dim, patches)
            for _ in range(groups)
        ])
        self.svfs_k = nn.ModuleList([
            SingleVariableFunction(self.group_size, hidden_dim, patches)
            for _ in range(groups)
        ])

        self.global_function = SingleVariableFunction(groups* patches, hidden_dim, heads)

    def forward(self, q, k):
        batch_size = q.shape[0]
        q = q.reshape(batch_size, -1)  
        k = k.reshape(batch_size, -1)
        # product = q.size(0)*q.size(1)
        # group_size = product / self.groups
       

        q_groups = q.view(batch_size, self.groups,  self.group_size).transpose(1, 2)
        k_groups = k.view(batch_size, self.groups,  self.group_size).transpose(1, 2)
        
        q_features = [svf(q_groups[:, :, i]) for i, svf in enumerate(self.svfs_q)]
        k_features = [svf(k_groups[:, :, i]) for i, svf in enumerate(self.svfs_k)]

        q_features = torch.stack(q_features, dim=2).view(batch_size, -1)
        k_features = torch.stack(k_features, dim=2).view(batch_size, -1)

        q_out = self.global_function(q_features).view(batch_size, -1)
        k_out = self.global_function(k_features).view(batch_size, -1)

        attn = (q_out * k_out).sum(dim=-1)
        attn = attn.softmax(dim=-1)
        # print(attn.shape)
        return attn

    

def objective(trial):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 初始化模型
    model = GroupedKAAttention(
        total_dim=3*224*224,
        patches=7,
        heads=8,
        head_dim=16,
        hidden_dim=128,
        groups=256
    ).to(device)
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCELoss()
    
    train_loader, test_loader = get_cifar10_loaders(batch_size=16)
    

    # 训练模型
    model.train()
    for epoch in range(3):
        for data, target in tqdm.tqdm(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data, data)
            output = output.float()
            target = target.long() 
            loss = criterion(output, target.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # 测试模型
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data, data)
            pred = output.argmax(dim=0)
            correct += (pred == target).sum().item()
            # correct += pred.eq(target.view_as(pred)).sum().item()
    
    accuracy = correct / len(test_loader.dataset)
    return accuracy

# 运行超参数优化
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

print("最佳超参数: ", study.best_params)
print("最佳准确率: ", study.best_value)


