import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torch.optim as optim
import torchvision.transforms as transforms

from tqdm import tqdm
import numpy as np
import os
import time

from models_file import MLP_Net, KAN_Net, Sin_Net, Efficient_KAN_Net, Efficient_SKAN_Net
from SLayer import Slayer
from LBFGS import LBFGS

def calculate_accuracy(output, gt):
    output = torch.argmax(output, dim=1)
    acc = torch.sum(output == gt)
    return acc / gt.shape[0]

# 设置设备为GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device :",device)
# 加载数据集
cifar_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
cifar_testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=cifar_trainset, batch_size=64, shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=cifar_testset, batch_size=64, shuffle=False)

# 实例化并移动模型到GPU
model = Efficient_SKAN_Net().to(device)

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)

epochs = 50
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), 0.001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)

record_save_path = 'tt.txt'
weight_save_path = 'tt'

for epoch in range(epochs):
    train_accuracy = []
    train_loss = []
    val_accuracy = []
    val_loss = []

    if epoch > 15:
        optimizer = optim.Adam(model.parameters(), 0.0001)

    if epoch > 30:
        optimizer = optim.Adam(model.parameters(), 0.00005)

    tst = time.time()

    for sample in tqdm(train_loader):
        image, label = sample
        image, label = image.to(device), label.to(device)

        output = model(image)
        loss = loss_function(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accuracy = calculate_accuracy(output, label)
        train_accuracy.append(accuracy.detach().cpu().numpy())
        train_loss.append(loss.item())

    train_accuracy = np.array(train_accuracy)
    train_loss = np.array(train_loss)
    tt = time.time() - tst

    vst = time.time()
    for sample in tqdm(validation_loader):
        image, label = sample
        image, label = image.to(device), label.to(device)

        with torch.no_grad():
            output = model(image)

        loss = loss_function(output, label)

        accuracy = calculate_accuracy(output, label)
        val_accuracy.append(accuracy.detach().cpu().numpy())
        val_loss.append(loss.item())

    val_accuracy = np.array(val_accuracy)
    val_loss = np.array(val_loss)
    vt = time.time() - vst

    initial_name = 'Model_' + str(epoch) + '.pth'
    torch.save(model.state_dict(), os.path.join(weight_save_path, initial_name))

    print('Epoch: ', epoch)
    print('Training Loss: ', np.mean(train_loss))
    print('Training Accuracy: ', np.mean(train_accuracy))
    print('Training time: ', tt)
    print('Validation Loss: ', np.mean(val_loss))
    print('Validation Accuracy: ', np.mean(val_accuracy))
    print('Validation time: ', vt)
    print('\n')

    with open(record_save_path, 'a') as f:
        f.write(f'Epoch: {epoch}\n')
        f.write(f'Training Loss: {np.mean(train_loss)}\n')
        f.write(f'Training Accuracy: {np.mean(train_accuracy)}\n')
        f.write(f'Training Time: {tt}\n')
        f.write(f'Validation Loss: {np.mean(val_loss)}\n')
        f.write(f'Validation Accuracy: {np.mean(val_accuracy)}\n')
        f.write(f'Validation Time: {vt}\n')
        f.write('\n')
