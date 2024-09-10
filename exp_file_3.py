import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # (B, 16, 16, 16)
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # (B, 32, 8, 8)
            nn.ReLU(True),
            nn.Conv2d(32, 64, 7)                      # (B, 64, 1, 1)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),            # (B, 32, 8, 8)
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # (B, 16, 16, 16)
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),   # (B, 3, 32, 32)
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



model = Autoencoder().cuda()  # 使用 GPU（如果可用）
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


num_epochs = 200

for epoch in range(num_epochs):
    running_loss = 0.0
    for data in trainloader:
        inputs, _ = data  # CIFAR-10是有标签的数据集，但我们在无监督学习中不使用标签
        inputs = inputs.cuda()

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, inputs)

        # 反向传播及优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(trainloader):.4f}')



torch.save(model.state_dict(), 'cifar10_autoencoder.pth')




# 加载测试数据
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=10, shuffle=False, num_workers=2)

# 展示原始图像和重建图像
dataiter = iter(testloader)
images, _ = dataiter.__next__()

# 原始图像
with torch.no_grad():
    outputs = model(images.cuda()).cpu()

# 展示前10张图片的原始图像和重建图像
fig, axes = plt.subplots(2, 10, figsize=(12, 2.5))
for i in range(10):
    axes[0, i].imshow((images[i].numpy().transpose(1, 2, 0) * 0.5) + 0.5)
    axes[1, i].imshow((outputs[i].numpy().transpose(1, 2, 0) * 0.5) + 0.5)
    axes[0, i].axis('off')
    axes[1, i].axis('off')

# plt.show()
