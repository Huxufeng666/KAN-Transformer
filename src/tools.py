import pandas as pd
import matplotlib.pyplot as plt
import os


# 读取 CSV 文件
file_path = '/home/ami-1/HUXUFENG/KAN-Transformer/output/TF_patch_kan_cifar10_100'  # 替换为你的 CSV 文件路径
fil_path = os.path.join(file_path,'train.csv')
data = pd.read_csv(fil_path)

# 提取 Epoch 和 Validation_Accuracy 列
epochs = data['Epoch']
validation_accuracy = data['Validation_Accuracy']

# 绘制图表
plt.figure(figsize=(10, 6))
plt.plot(epochs, validation_accuracy, marker='o', linestyle='-', color='b')

output_dir = file_path  # 替换为你想要保存的文件夹路径
if not os.path.exists(output_dir):
    os.makedirs(output_dir)  # 如果文件夹不存在，创建文件夹

output_file = os.path.join(output_dir, 'validation_accuracy_vs_epoch.png')
plt.savefig(output_file)  # 保存图表