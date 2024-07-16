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
from models.vision_kansformer import *
from models.vision_transformer import *
from models.deep_vision_transformer import *

# # 设置每个进程使用的显存分数
# torch.cuda.set_per_process_memory_fraction(0.1)  # 第一张GPU使用25%
# torch.cuda.set_per_process_memory_fraction(0.9)  # 第二张GPU使用75

def calculate_accuracy(output, gt):
    # print('before argmax: ', output.shape)
    output = torch.argmax(output, dim = 1)
    # print('after argmax: ', output.shape, output)
    acc = torch.sum(output == gt)
    return acc/gt.shape[0]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device :",device)
device = torch.device('cuda:0')
# 加载数据集
# mnist_trainset = datasets.MNIST(root = './data', train = True, download = True, transform = transforms.ToTensor())
# mnist_testset = datasets.MNIST(root = './data', train = False, download = True, transform = transforms.ToTensor())


transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化

])

cifar_trainset = torchvision.datasets.CIFAR10(root = './data', train = True, download = True, transform = transform)
cifar_testset = torchvision.datasets.CIFAR10(root = './data', train = False, download = True, transform = transform)

train_loader = torch.utils.data.DataLoader(dataset = cifar_trainset, batch_size = 32, shuffle = True) # MNIST = 1000, cifar orig = 4
validation_loader = torch.utils.data.DataLoader(dataset = cifar_testset, batch_size = 32, shuffle = False) # MNIST = 2000

# model = MLP_Net().cuda()
# model = KAN_Net().cuda()
# model = Sin_Net().cuda()
# model  = vit_large_patch16_224().to(device)#.cuda()
model  = deepvit_S(num_classes= 10).to(device)

# model = Efficient_KAN_Net().cuda()
# model = Efficient_SKAN_Net().cuda()

# if torch.cuda.device_count() > 1:
#     print(f"Let's use {torch.cuda.device_count()} GPUs!")
#     # model = nn.DataParallel(model)
#     model = nn.DataParallel(model, device_ids=[1])



for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)

epochs = 50
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), 0.001)
# optimizer = optim.SGD(model.parameters(), 0.01)

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.1)


record_save_path = 'tt/tt.txt'
weight_save_path = 'tt'

# model.load_state_dict(torch.load('D:\\mlp_modification\\Efficient_KAN_saves\\weight_saves\\tt\\Model_30.pth'))
# model.fc1.show_act()

for epoch in range(0, epochs):
    train_accuracy = []
    train_loss = []
    val_accuracy = []
    val_loss = []
    
    if epoch > 15:
        optimizer = optim.Adam(model.parameters(), 0.0001)
    
    if epoch > 30:
        optimizer = optim.Adam(model.parameters(), 0.00005)
        
        
    
    tst = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    for sample in tqdm(train_loader):
        image, label = sample
        label = label.to(device)#.cuda()
        # print('image shape: ', image.shape)
        # print('label: ', label)
        output = model(image.to(device))#cuda())
        # print('output/label shape: ', output.shape, label.shape)
        
        loss = loss_function(output, label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        accuracy = calculate_accuracy(output, label)
        train_accuracy.append(accuracy.detach().cpu().numpy())
        train_loss.append(loss.item())
        
    # model.fc1.show_act() ############################################################
    
    train_accuracy = np.array(train_accuracy)
    train_loss = np.array(train_loss)
    tt = time.time() - tst
    
    vst = time.time()
    for sample in tqdm(validation_loader):
        image, label = sample
        label = label.to(device)#cuda()
        with torch.no_grad():
            output = model(image.to(device))#cuda())
        
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
        f.write(f'Epoch: {epoch}')
        f.write('\n')
        f.write(f'Training Loss: {np.mean(train_loss)}')
        f.write('\n')
        f.write(f'Training Accuracy: {np.mean(train_accuracy)}')
        f.write('\n')
        f.write(f'Training Time: {tt}')
        f.write('\n')
        f.write(f'Validation Loss: {np.mean(val_loss)}')
        f.write('\n')
        f.write(f'Validation Accuracy: {np.mean(val_accuracy)}')
        f.write('\n')
        f.write(f'Validation Time: {vt}')
        f.write('\n')
        f.write('\n')
        

