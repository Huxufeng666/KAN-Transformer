import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms

from tqdm import tqdm
import numpy as np
import os
import time

from torch.nn.parallel import DataParallel


from models.KA_SA_transformer import kit_base_patch16_224
from models.KA_SA_transformer_trainable_groups import kit_base_patch16_224
from models.vision_transformer import vit_base_patch16_224
from models.Efficient_SA_transformer import Eit_base_patch16_224
from models.Full_Efficient_transformer import Full_Eit_base_patch16_224

from models.deep_vision_transformer import deepvit_S
# from models.DEIT import deit_tiny_patch16_224, deit_small_patch16_224, deit_base_patch16_224

from data_loader import Noise_dataset, build_transform

def warmup(optimizer, warm_up_iters, warm_up_factor):
    def f(x):
        """根据step数返回一个学习率倍率因子, x代表step"""
        if x >= warm_up_iters:
            return 1
        
        alpha = float(x) / warm_up_iters
        return warm_up_factor * (1 - alpha) + alpha
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)

def calculate_accuracy(output, gt):
    output = torch.argmax(output, dim = 1)
    acc = torch.sum(output == gt)
    return acc/gt.shape[0]

data_transform = {
        "train": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ]),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ])
    }

data_transform_train = build_transform(is_train = True, input_size = 224)
data_transform_eval = build_transform(is_train = False, input_size = 224)

# mnist_trainset = datasets.MNIST(root = './data', train = True, download = True, transform = transforms.ToTensor())
# mnist_testset = datasets.MNIST(root = './data', train = False, download = True, transform = transforms.ToTensor())

# cifar_trainset = torchvision.datasets.CIFAR10(root = './data', train = True, download = True, transform = data_transform["train"])
# cifar_testset = torchvision.datasets.CIFAR10(root = './data', train = False, download = True, transform = data_transform["val"])

cifar_trainset = torchvision.datasets.CIFAR100(root = './data', train = True, download = True, transform = data_transform_train)#data_transform["train"])
cifar_testset = torchvision.datasets.CIFAR100(root = './data', train = False, download = True, transform = data_transform_eval)#data_transform["val"])


# train_path = 'D:\\mlp_modification\\data\\caltech_101_training_dataset.npy'
# validation_path = 'D:\\mlp_modification\\data\\caltech_101_validation_dataset.npy'

# train_dataset = Noise_dataset(train_path)
# validation_dataset = Noise_dataset(validation_path)

train_loader = torch.utils.data.DataLoader(dataset = cifar_trainset, batch_size = 42, shuffle = True, pin_memory = True, num_workers=2, prefetch_factor=2) # MNIST = 1000, cifar orig = 4
validation_loader = torch.utils.data.DataLoader(dataset = cifar_testset, batch_size = 42, shuffle = False, pin_memory = True, num_workers=2, prefetch_factor=2) # MNIST = 2000



# model = kit_base_patch16_224(num_classes = 10).cuda()
# model = vit_base_patch16_224(num_classes = 10).cuda()
# model = Eit_base_patch16_224(num_classes = 10).cuda()
# model = Full_Eit_base_patch16_224(num_classes = 10).cuda()


# classes in Oxford-IIIT = 37
# classes in caltech-101 = 33
model = deepvit_S(img_size = 224, patch_size = 16, num_classes = 100, pretrained = False).cuda() # patch = 24
# model = deit_tiny_patch16_224(pretrained = True).cuda()
# model = deit_base_patch16_224(pretrained = True).cuda()

if torch.cuda.device_count() > 1:
  device_ids = [0, 1]
  model = DataParallel(model, device_ids = device_ids)

# print(model)

# attn_params = []
# other_params = []
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         if 'attn' in name:
#             # print(name)
#             attn_params.append(param)
#         else:
#             other_params.append(param)

epochs = 50
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), 0.001)#, weight_decay = 0.01)
# optimizer = optim.SGD(model.parameters(), 0.001)
# optimizer = optim.AdamW(model.parameters(), 0.0001)


# optimizer = optim.Adam([
#     {'params':list(attn_params), 'lr': 0.01},
#     {'params':list(other_params), 'lr':0.001}
#     ])

# if epoch == 0: 
#     warmup_factor = 1.0/1000
#     warmup_iters = min(1000, len(data_loader) -1)

# lr_scheduler = warmup(optimizer, warmup_iters, warmup_factor)

record_save_path = 'D:\\mlp_modification\\transformer_saves\\\DeepVIT_saves\\record_saves\\tt2.txt'
weight_save_path = 'D:\\mlp_modification\\transformer_saves\\DeepVIT_saves\\weight_saves\\tt2'

# model.load_state_dict(torch.load('D:\\mlp_modification\\transformer_saves\\DeepVIT_saves\\weight_saves\\tt\\Model_10.pth'))
# model.fc1.show_act()

if __name__ == '__main__':
    
    for epoch in range(0, epochs):
        train_accuracy = []
        train_loss = []
        val_accuracy = []
        val_loss = []
        
        if epoch > 10:
            optimizer = optim.Adam(model.parameters(), 0.0001)
            
        if epoch > 29:
            optimizer = optim.Adam(model.parameters(), 0.00001)
            
        if epoch > 39:
            optimizer = optim.Adam(model.parameters(), 0.00001)
        
        tst = time.time()
        model.train()
        for sample in tqdm(train_loader):
            image, label = sample
            # print('image shape: ', image.shape)
            # print('label: ', label)
            output, attn_list = model(image.float().cuda())
            # print('output/label shape: ', output.shape, label)
            attn_list = attn_list[0:15]
            torch.save(attn_list, 'attention_list.pt')
            
            loss = loss_function(output, label.cuda())
            # print('--- Loss: ', loss)
            
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
        
        vst = time.time()
        model.eval()
        for sample in tqdm(validation_loader):
            image, label = sample
            
            with torch.no_grad():
                output, attn_list = model(image.float().cuda())
            
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