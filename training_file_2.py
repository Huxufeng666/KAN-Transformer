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

from models_file import MLP_Net, KAN_Net, Sin_Net
from SLayer import Slayer
from LBFGS import LBFGS

def calculate_accuracy(output, gt):
    # print('before argmax: ', output.shape)
    output = torch.argmax(output, dim = 1)
    # print('after argmax: ', output.shape, output)
    acc = torch.sum(output == gt)
    return acc/gt.shape[0]


# mnist_trainset = datasets.MNIST(root = './data', train = True, download = True, transform = transforms.ToTensor())
# mnist_testset = datasets.MNIST(root = './data', train = False, download = True, transform = transforms.ToTensor())

cifar_trainset = torchvision.datasets.CIFAR10(root = './data', train = True, download = True, transform = transforms.ToTensor())
cifar_testset = torchvision.datasets.CIFAR10(root = './data', train = False, download = True, transform = transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset = cifar_trainset, batch_size = 16, shuffle = True) # MNIST = 1000
validation_loader = torch.utils.data.DataLoader(dataset = cifar_testset, batch_size = 16, shuffle = False) # MNIST = 2000

# model = MLP_Net().cuda()
model = KAN_Net().cuda()
# model = Sin_Net().cuda()

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)

epochs = 50
loss_function = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), 0.01)
optimizer = LBFGS(model.parameters(), lr = 0.01, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32, tolerance_ys=1e-32)


record_save_path = 'D:\\mlp_modification\\record_saves\\2_CIFAR_KAN_training.txt'
weight_save_path = 'D:\\mlp_modification\\weight_saves\\2_CIFAR_KAN_training'

# model.load_state_dict(torch.load('D:\\mlp_modification\\weight_saves\\2_Grouped_KAN_training\\Model_49.py'))
# model.fc1.show_act()

for epoch in range(0, epochs):
    train_accuracy = []
    train_loss = []
    val_accuracy = []
    val_loss = []
    
    # if epoch > 5:
    #     optimizer = optim.Adam(model.parameters(), 0.001)
    
    tst = time.time()
    for sample in tqdm(train_loader):
        image, label = sample
        
        def closure():
            global loss, reg_, accuracy
            optimizer.zero_grad()
            
            output = model(image.cuda())
            
            loss = loss_function(output, label.cuda())
            accuracy = calculate_accuracy(output, label.cuda())
            
            # reg_ = reg(self.acts_scale)
            # objective = train_loss + lamb * reg_
            loss.backward()
            return loss
        

        optimizer.step(closure)
        
        train_accuracy.append(accuracy.detach().cpu().numpy())
        # print('train loss shape: ', loss)
        train_loss.append(loss.item())
        
    # model.fc1.show_act() ############################################################
    
    train_accuracy = np.array(train_accuracy)
    train_loss = np.array(train_loss)
    tt = time.time() - tst
    
    
    vst = time.time()
    for sample in tqdm(validation_loader):
        image, label = sample
        
        with torch.no_grad():
            output = model(image.cuda())
        
        loss = loss_function(output, label.cuda())
        
        accuracy = calculate_accuracy(output, label.cuda())
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
        






















# def train(model, train_loader, loss_function, optimizer):
#     accuracy_record = []
#     loss_record = []
    
#     for sample in tqdm(train_loader):
#         image, label = sample
#         print('image shape: ', image.shape)
#         print('label: ', label)
#         output = model(image)
#         print('output/label shape: ', output.shape, label.shape)
        
#         loss = loss_function(output, label)
        
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         accuracy = calculate_accuracy(output, label)
#         accuracy_record.append(accuracy.numpy())
#         loss_record.append(loss.item())
    
#     accuracy_record = np.array(accuracy_record)
#     loss_record = np.array(loss_record)
#     return np.mean(accuracy_record), np.mean(loss_record)
    
# def evaluate(model, val_loader, loss_function, optimizer):
#     accuracy_record = []
#     loss_record = []
    
#     for sample in tqdm(val_loader):
#         image, label = sample
#         print('image shape: ', image.shape)
#         print('label: ', label)
#         with torch.no_grad():
#             output = model(image)
#         print('output/label shape: ', output.shape, label.shape)
        
#         loss = loss_function(output, label)
        

#         accuracy = calculate_accuracy(output, label)
#         accuracy_record.append(accuracy.numpy())
#         loss_record.append(loss.item())
        
#     accuracy_record = np.array(accuracy_record)
#     loss_record = np.array(loss_record)
#     return np.mean(accuracy_record), np.mean(loss_record)