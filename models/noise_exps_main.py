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

from scipy.stats import entropy

from models_file import MLP_Net, KAN_Net, Sin_Net, Efficient_SKAN_Net

from KAN_entropy_experiments.noise_loader import Noise_dataset

def calculate_accuracy(output, gt):
    output = torch.argmax(output, dim = 1)
    acc = torch.sum(output == gt)
    return acc/gt.shape[0]

def class_accuracy(output, gt, classes):
    output = torch.argmax(output, dim = 1)
    record = torch.zeros(classes)
    unique_vals = torch.unique(output)
    
    for cval in unique_vals:
        acc = torch.sum(output == cval)
        acc = acc/gt.shape[0]
        record[cval] = acc
        
    return record



data_transform = {
        "train": transforms.Compose([
            # transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ]),
        "val": transforms.Compose([
            # transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ])
    }


train_data_path = 'D:\\mlp_modification\\KAN_entropy_experiments\\Train_noise_dataset.npy'
validation_data_path = 'D:\\mlp_modification\\KAN_entropy_experiments\\Validation_noise_dataset.npy'
train_data = Noise_dataset(train_data_path)
validation_data = Noise_dataset(validation_data_path)

# cifar_trainset = torchvision.datasets.CIFAR10(root = './data', train = True, download = True, transform = data_transform["train"])
# cifar_testset = torchvision.datasets.CIFAR10(root = './data', train = False, download = True, transform = data_transform["val"])

train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size = 64, shuffle = True) # MNIST = 1000, cifar orig = 4
validation_loader = torch.utils.data.DataLoader(dataset = validation_data, batch_size = 64, shuffle = False) # MNIST = 2000

# model = MLP_Net().cuda()
model = KAN_Net().cuda()
# model = Sin_Net().cuda()
# model = Efficient_SKAN_Net().cuda()


# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name)

epochs = 50
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), 0.001)
# optimizer = optim.SGD(model.parameters(), 0.01)

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.1)


record_save_path = 'D:\\mlp_modification\\KAN_entropy_experiments\\record_saves\\tt.txt'
weight_save_path = 'D:\\mlp_modification\\KAN_entropy_experiments\\weight_saves\\tt'

# model.load_state_dict(torch.load('D:\\mlp_modification\\KAN_entropy_experiments\\weight_saves\\tt\\Model_30.pth'))
# model.fc1.show_act()

def generate_noise(shape, target_entropy):
  noise = np.random.rand(*shape)
  current_entropy = entropy(noise.flatten())

  while abs(current_entropy - target_entropy) > 0.001:
      noise = noise * np.random.rand(*shape)
      current_entropy = entropy(noise.flatten())

  return noise


train_dataset_array = []
validation_dataset_array = []

# 10-15, 20-25, 30-35, 40-45, 50-55
# base_entropy = [10, 30, 40, 50]

# for base_val in base_entropy:
    
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
    
    for sample in tqdm(train_loader):
        image, label = sample
        # print('image shape: ', image.shape)
        # print('label: ', label)
        output = model(image.float().cuda())
        # print('output/label shape: ', output.shape, label.shape)
        
        loss = loss_function(output, label.cuda())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        accuracy = calculate_accuracy(output, label.cuda())
        
        train_accuracy.append(accuracy.detach().cpu().numpy())
        train_loss.append(loss.item())
        
    # model.fc1.show_act() ############################################################
    
    train_accuracy = np.array(train_accuracy)
    train_loss = np.array(train_loss)
    tt = time.time() - tst
    
    
    class_records = []
    vst = time.time()
    for sample in tqdm(validation_loader):
        image, label = sample
        
        with torch.no_grad():
            output = model(image.float().cuda())
        
        loss = loss_function(output, label.cuda())
        
        accuracy = calculate_accuracy(output, label.cuda())
        val_accuracy.append(accuracy.detach().cpu().numpy())
        val_loss.append(loss.item())
        
    val_accuracy = np.array(val_accuracy)
    val_loss = np.array(val_loss)
    vt = time.time() - vst
    
    class_record = class_accuracy(output, label.cuda(), 5)
    class_records.append(class_record.detach().cpu().numpy())
    
    
    initial_name = 'Model_' + str(epoch) + '.pth'
    torch.save(model.state_dict(), os.path.join(weight_save_path, initial_name))
            
    print('Epoch: ', epoch)
    print('Training Loss: ', np.mean(train_loss))
    print('Training Accuracy: ', np.mean(train_accuracy))
    print('Training time: ', tt)
    print('Validation Loss: ', np.mean(val_loss))
    print('Validation Accuracy: ', np.mean(val_accuracy))
    print('Validation class accuracy: ', np.mean(class_records, axis = 0))
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
        f.write(f'Validation class Accuracy: {np.mean(class_records, axis = 0)}')
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