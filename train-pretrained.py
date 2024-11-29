import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.optim as optim
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
import time
import torch
from torch.utils.data import DataLoader
from util.file import save_training_info_csv ,save_top_k_weights


import timm



# 使用预训练的ViT模型


def calculate_accuracy(output, gt):
    # print('before argmax: ', output.shape)
    output = torch.argmax(output, dim = 1)
    # print('after argmax: ', output.shape, output)
    acc = torch.sum(output == gt)
    return acc/gt.shape[0]


transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])



cifar_trainset = datasets.CIFAR100(root = './data', train = True, download = True, transform = transform)
cifar_testset = datasets.CIFAR100(root = './data', train = False, download = True, transform = transform)

train_loader = torch.utils.data.DataLoader(dataset = cifar_trainset, batch_size = 16, shuffle = True , num_workers=2) # MNIST = 1000, cifar orig = 4
validation_loader = torch.utils.data.DataLoader(dataset = cifar_testset, batch_size = 16, shuffle = False , num_workers=2) # MNIST = 2000




model = timm.create_model('vit_base_patch16_224', pretrained=True).cuda()


    
    
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)


for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)

epochs = 5
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), 0.0001)

# optimizer = LBFGS(model.parameters(), lr = 0.01, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32, tolerance_ys=1e-32)


weight_save_path = 'output/kan_attention_patch16_224/'
                    


for epoch in range(0, epochs):
    train_accuracy = []
    train_loss = []
    val_accuracy = []
    val_loss = []
    
    tst = time.time()
    for sample in tqdm(train_loader):
        image, label = sample
        
        def closure():
            global loss, reg_, accuracy
            optimizer.zero_grad()
            
            output = model(image.cuda())
            
            loss = loss_function(output, label.cuda())
            accuracy = calculate_accuracy(output, label.cuda())
            

            loss.backward()
            return loss
        

        optimizer.step(closure)
        
        train_accuracy.append(accuracy.detach().cpu().numpy())
        # print('train loss shape: ', loss)
        train_loss.append(loss.item())
        
    
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
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型的参数总数: {total_params}")

    save_top_k_weights(

        weight_save_path,
        model,
        epoch,
        train_loss,
        train_accuracy,
        tt,
        val_loss,
        val_accuracy,
        vt,
        top_k=5
    )
    
            

    save_training_info_csv(
        weight_save_path,
        epoch,
        train_loss,
        train_accuracy,
        tt,
        val_loss,
        val_accuracy,
        vt,
        total_params 
    )

