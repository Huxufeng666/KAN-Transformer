import torch
from tqdm import tqdm
import numpy as np
import time
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from util.file import save_training_info_csv ,save_top_k_weights
from models.deep_vision_transformer import deepvit_S,deepvit_L
from models.Efficient_SA_transformer import vit_base_patch16_224



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


# train_dataset = datasets.ImageFolder(root='/home/user/PRPD-dataset/aaaa_dataset/png_dir/train', transform=transform)
# val_dataset = datasets.ImageFolder(root='/home/user/PRPD-dataset/aaaa_dataset/png_dir/val', transform=transform)

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

cifar_trainset = datasets.CIFAR10(root = './data', train = True, download = True, transform = transform)
cifar_testset = datasets.CIFAR10(root = './data', train = False, download = True, transform = transform)

train_loader = DataLoader(dataset = cifar_trainset, batch_size = 16, shuffle = True , num_workers=4) # MNIST = 1000, cifar orig = 4
validation_loader = DataLoader(dataset = cifar_testset, batch_size = 16, shuffle = False , num_workers=4) # MNIST = 2000



# model = MLP_Net().cuda()
# model = KAN_Net().cuda()
# model = vit_base_patch16_224().cuda()
model = deepvit_S().cuda()


for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)

epochs = 200
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), 0.0001)
# optimizer = LBFGS(model.parameters(), lr = 0.01, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32, tolerance_ys=1e-32)


weight_save_path = 'output/KA_attention_crossinf_multi_head/'
                    

# model.load_state_dict(torch.load('D:\\mlp_modification\\weight_saves\\2_Grouped_KAN_training\\Model_4
# model.fc1.show_act()
# if torch.cuda.device_count() > 1:
#     print(f"Using {torch.cuda.device_count()} GPUs!")
#     model = nn.DataParallel(model)
tst = time.time()
    
for epoch in range(0, epochs):
    train_accuracy = []
    train_loss = []
    val_accuracy = []
    val_loss = []
    
    # if epoch > 5:
    #     optimizer = optim.Adam(model.parameters(), 0.001)
   
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
        vt
    )

#     