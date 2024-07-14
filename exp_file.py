# import torch
# import torch.nn as nn


# device = 'cpu'

# class lin_func(nn.Module):
#     def __init__(self, min_val, max_val):
#         super(lin_func, self).__init__()
        
#         self.min_val = min_val
#         self.max_val = max_val
        

#     def smooth_step(self, x):

#       if x < self.min_val:
#         return self.min_val
#       elif x > self.max_val:
#         return self.max_val
#       else:
#         return x#(x - x_min) / (x_max - x_min) - (x - x_min)**2 / ((x_max - x_min)**2)
    
#     def piecewise_linear(self, x):
#       slope = 1.0
    
#       # intercept = 0.0
#       return slope * self.smooth_step(x)# * (x - 4.0) + intercept * self.smooth_step(50.0, 100.0, x)



# func = lin_func(4, 50)

# r = func.piecewise_linear(100)
# print('r: ', r)

''' --------------------------------------------------------------------------------------------------------------------------------------------- '''

import os
import cv2
import numpy as np

# train_folder = 'E:\\Oxford-IIIT\\train'
# validation_folder = 'E:\\Oxford-IIIT\\test'

# dataset = []
# vals = []

# for class_val in os.listdir(train_folder):
#     # print(class_val)
#     class_folder = os.path.join(train_folder, class_val)
    
#     for file in os.listdir(class_folder):
#         file_name = os.path.join(class_folder, file)
#         image = cv2.imread(file_name)
#         image = cv2.resize(image, (96, 96))
#         dataset.append([image, class_val])
#         vals.append(class_val)
        

# dataset = np.array(dataset)
# print('train dataset shape: ', dataset.shape)
# # np.save('Oxford_IIIT_train_dataset_96.npy', dataset)
# print('vals unique: ', np.unique(vals))



# dataset = []

# for class_val in os.listdir(validation_folder):
#     class_folder = os.path.join(train_folder, class_val)
    
#     for file in os.listdir(class_folder):
#         file_name = os.path.join(class_folder, file)
#         image = cv2.imread(file_name)
#         image = cv2.resize(image, (96, 96))
#         dataset.append([image, class_val])
        

# dataset = np.array(dataset)
# print('validation dataset shape: ', dataset.shape)
# # np.save('Oxford_IIIT_validation_dataset_96.npy', dataset)



train_folder = 'E:\\caltech-101\\caltech-101\\101_ObjectCategories'

training_dataset = []
validation_dataset = []
temporary = []
value = []

class_vals = 0

for i, class_val in enumerate(os.listdir(train_folder)):
    print(class_val)
    
    class_folder = os.path.join(train_folder, class_val)
    
    num_samples = len([name for name in os.listdir(class_folder)])
    print('number of samples: ', num_samples)
    
    if num_samples >= 70:
        
        value.append(class_vals)
        for j, file in enumerate(os.listdir(class_folder)):
            if j < 50:
                file_name = os.path.join(class_folder, file)
                image = cv2.imread(file_name)
                # print('image shape: ', image.shape)
                # image = cv2.resize(image, (96, 96))
                training_dataset.append([image, class_vals])
            
            if j >= 50 and j <= 69:
                file_name = os.path.join(class_folder, file)
                image = cv2.imread(file_name)
                # print('image shape: ', image.shape)
                # image = cv2.resize(image, (96, 96))
                validation_dataset.append([image, class_vals])
                
        class_vals += 1


training_dataset = np.array(training_dataset)
validation_dataset = np.array(validation_dataset)

print('Training dataset shape: ', training_dataset.shape)
print('Validation dataset shape: ', validation_dataset.shape)
np.save('caltech_101_training_dataset.npy', training_dataset)
np.save('caltech_101_validation_dataset.npy', validation_dataset)


print('class uniques: ', np.unique(value))







