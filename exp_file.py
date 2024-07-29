import torch
import torch.nn as nn
import torch.nn.functional as F

from KANLayer import KANLayer
from models.Symbolic_KANLayer import *
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
    
    
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader



class SingleVariableFunction(nn.Module):
    """单变量函数逼近器"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        # self.fc3 = nn.Linear(output_dim,10)
        
        self.act = nn.ReLU()  # 可根据需要选择其他激活函数

    def forward(self, x):
        print("Before fc1:", x.shape)
        # x = (x.shape[0], 128)
        # x = torch.tensor(x).cuda()
           
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        print("After fc2:", x.shape)

        # x = self.fc3(x)
        # print("After fc3:", x.shape)
        return x

class GroupedKAAttention(nn.Module):
    """
    分组 Kolmogorov-Arnold 注意力机制
    """
    def __init__(self, out_dim=128, patches=4, heads=8, head_dim=16, groups=4, hidden_dim=128):
        super().__init__()
        self.out_dim = out_dim
        self.patches = patches
        self.heads = heads
        self.head_dim = head_dim
        self.groups = groups
        self.group_size = (patches * heads * head_dim) // groups

        # 使用单变量函数逼近器
        self.svfs_q = nn.ModuleList([
            SingleVariableFunction(self.group_size, hidden_dim, out_dim)
            for _ in range(groups)
        ])
        self.svfs_k = nn.ModuleList([
            SingleVariableFunction(self.group_size, hidden_dim, out_dim)
            for _ in range(groups)
        ])

        # 全局函数逼近器
        self.global_function = SingleVariableFunction(out_dim * groups, hidden_dim, out_dim)

    def forward(self, q, k):
        # q, k: (batch, heads, patches, head_dim)
        batch_size = q.shape[0]
        print("q.shape:",q.shape)

        # 分组并应用单变量函数
        q_groups = q.view(batch_size, 128, 1176).transpose(1, 2)
        k_groups = k.view(batch_size, 128, 1176).transpose(1, 2)
        
        q_features = [
            svf(q_groups[:, :, i]) for i, svf in enumerate(self.svfs_q)
        ]
        k_features = [
            svf(k_groups[:, :, i]) for i, svf in enumerate(self.svfs_k)
        ]

        # 特征融合
        q_features = torch.stack(q_features, dim=2).view(batch_size, -1)
        k_features = torch.stack(k_features, dim=2).view(batch_size, -1)

        # 应用全局函数
        q_out = self.global_function(q_features).view(batch_size, self.heads, self.patches, self.out_dim)
        k_out = self.global_function(k_features).view(batch_size, self.heads, self.patches, self.out_dim)

        # 计算注意力权重
        attn = (q_out * k_out).sum(dim=-1)
        attn = attn.softmax(dim=-1)

        return attn
    

# class GroupedKAAttention(nn.Module):
#     """
#     分组 Kolmogorov-Arnold 注意力机制
#     """
#     def __init__(self, out_dim=128, patches=4, heads=8, head_dim=16, hidden_dim=128):
#         super().__init__()
#         self.out_dim = out_dim
#         self.patches = patches
#         self.heads = heads
#         self.head_dim = head_dim

#         # 使 groups 和 group_size 成为可学习的参数
#         self.groups = nn.Parameter(torch.tensor(4.0), requires_grad=True)
#         self.group_size = nn.Parameter(torch.tensor((patches * heads * head_dim) / 4), requires_grad=True)

#         # 确保 group_size 为整数
#         self.group_size_int = nn.Parameter(torch.tensor((patches * heads * head_dim) // 4), requires_grad=False)

#         # 使用单变量函数逼近器
#         self.svfs_q = nn.ModuleList([
#             SingleVariableFunction(self.group_size_int, hidden_dim, out_dim)
#             for _ in range(int(self.groups.item()))
#         ])
#         self.svfs_k = nn.ModuleList([
#             SingleVariableFunction(self.group_size_int, hidden_dim, out_dim)
#             for _ in range(int(self.groups.item()))
#         ])

#         # 全局函数逼近器
#         self.global_function = SingleVariableFunction(out_dim * int(self.groups.item()), hidden_dim, out_dim)

#     def forward(self, q, k):
#         # 更新 group_size_int
#         self.group_size_int = nn.Parameter(self.group_size.to(torch.int), requires_grad=False)
#         groups = int(self.groups.item())
#         group_size = int(self.group_size_int.item())

#         # q, k: (batch, heads, patches, head_dim)
#         batch_size = q.shape[0]

#         # 分组并应用单变量函数
#         q_groups = q.view(batch_size, 128, 1176).transpose(1, 2)
#         k_groups = k.view(batch_size, 128, 1176).transpose(1, 2)
        
#         q_features = [
#             svf(q_groups[:, :, i]) for i, svf in enumerate(self.svfs_q)
#         ]
#         k_features = [
#             svf(k_groups[:, :, i]) for i, svf in enumerate(self.svfs_k)
#         ]

#         # 特征融合
#         q_features = torch.stack(q_features, dim=2).view(batch_size, -1)
#         k_features = torch.stack(k_features, dim=2).view(batch_size, -1)

#         # 应用全局函数
#         q_out = self.global_function(q_features).view(batch_size, self.heads, self.patches, self.out_dim)
#         k_out = self.global_function(k_features).view(batch_size, self.heads, self.patches, self.out_dim)

#         # 计算注意力权重
#         attn = (q_out * k_out).sum(dim=-1)
#         attn = attn.softmax(dim=-1)

#         return attn

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 下载和加载 CIFAR-10 数据集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=16, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=16, shuffle=False, num_workers=2)




def train_model(model, trainloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs, inputs)  # For simplicity, use the same input for q and k
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    print(f"Training loss: {running_loss / len(trainloader)}")

def evaluate_model(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs, inputs)  # For simplicity, use the same input for q and k
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy: {100 * correct / total:.2f}%")

# 初始化模型、损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GroupedKAAttention(out_dim=10, patches=4, heads=8, head_dim=16, hidden_dim=128).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练和评估模型
train_model(model, trainloader, criterion, optimizer, device)
evaluate_model(model, testloader, device)