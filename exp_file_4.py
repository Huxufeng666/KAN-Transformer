import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import torch.nn as nn
import torchvision.models as models

import torch.optim as optim

# 数据增强
train_transform = transforms.Compose([
    # transforms.RandomResizedCrop(size=32),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.4)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# 加载 CIFAR-10 数据集
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)




# 使用 ResNet-18 作为 backbone
class SimCLR(nn.Module):
    def __init__(self, base_model, out_dim):
        super(SimCLR, self).__init__()
        self.backbone = base_model(pretrained=False)
        dim_mlp = self.backbone.fc.in_features

        # 修改 ResNet 的最后一层为 MLP
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, out_dim))

    def forward(self, x):
        return self.backbone(x)

model = SimCLR(base_model=models.resnet18, out_dim=128).cuda()


class NTXentLoss(nn.Module):
    def __init__(self, batch_size, temperature):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        # No need to initialize mask here; it's calculated in the forward method

    def forward(self, z_i, z_j):
        z_i = nn.functional.normalize(z_i, dim=1)
        z_j = nn.functional.normalize(z_j, dim=1)

        # Combine the embeddings
        representations = torch.cat([z_i ,  z_j], dim=0)
        similarity_matrix = torch.matmul(representations, representations.T) / self.temperature

        # Create mask for positive pairs
        labels = torch.cat([torch.arange(self.batch_size) for _ in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        mask = torch.eye(2 * self.batch_size, dtype=torch.bool, device=z_i.device).cuda()
        mask = mask | mask.flip(0)

        
        # Use mask to select positive pairs
        positives = similarity_matrix[mask].view(2 * self.batch_size, -1)
        negatives = similarity_matrix[~mask].view(2 * self.batch_size, -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(z_i.device)

        loss = nn.CrossEntropyLoss(reduction="sum")(logits, labels)
        loss /= 2 * self.batch_size
        return loss




# 初始化损失函数和优化器
criterion = NTXentLoss(batch_size=128, temperature=0.5).cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 训练循环
for epoch in range(100):
    running_loss = 0.0
    for data in train_loader:
        images, _ = data
        images = images.cuda()

        # 生成两个不同的视角
        images_aug1, images_aug2 = images[:64], images[64:]
        
        # 前向传播
        z_i = model(images_aug1)
        z_j = model(images_aug2)

        # 计算对比损失
        loss = criterion(z_i, z_j)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch [{epoch + 1}/100], Loss: {running_loss / len(train_loader):.4f}')
