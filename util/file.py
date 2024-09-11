import os
import csv
import numpy as np
import torch


def save_training_info_csv(weight_save_path, epoch, train_loss, train_accuracy, tt, val_loss, val_accuracy, vt, total_params):
    # 确保目录存在
    if not os.path.exists(weight_save_path):
        os.makedirs(weight_save_path)

    # 定义csv文件路径
    csv_file_path = os.path.join(weight_save_path, "train.csv")

    # 检查文件是否存在，决定是否写入header
    file_exists = os.path.exists(csv_file_path)
     

    print('Epoch: ', epoch)

    column_format = "{:<20}"  # 每列设置为宽度20，左对齐

    with open(csv_file_path, 'a') as f:
        if not file_exists:
        # 写入header
            f.write(column_format.format('Epoch') + 
                    column_format.format('Training_Loss') +
                    column_format.format('Training_Accuracy') +
                    column_format.format('Training_Time') +
                    column_format.format('Validation_Loss') +
                    column_format.format('Validation_Accuracy') +
                    column_format.format('Validation_Time') +
                    column_format.format('Total_Params') + '\n')

        # 写入训练信息
        f.write(column_format.format(epoch) +
                column_format.format(np.mean(train_loss)) +
                column_format.format(np.mean(train_accuracy)) +
                column_format.format(tt) +
                column_format.format(np.mean(val_loss)) +
                column_format.format(np.mean(val_accuracy)) +
                column_format.format(vt) +
                column_format.format(total_params) + '\n')


    # with open(csv_file_path, 'a', newline='') as f:
    #     writer = csv.writer(f)

    #     # 如果文件不存在，写入header
    #     if not file_exists:
    #         writer.writerow(['Epoch', 'Training_Loss', 'Training_Accuracy', 'Training_Time',
    #                          'Validation_Loss', 'Validation_Accuracy', 'Validation_Time', 'Total_Params'])

    #     # 写入训练信息
    #     writer.writerow([epoch, np.mean(train_loss), np.mean(train_accuracy), tt,
    #                      np.mean(val_loss), np.mean(val_accuracy), vt,total_params])
        


def save_top_k_weights(weight_save_path, model, epoch, train_loss, train_accuracy, tt, val_loss, val_accuracy, vt, top_k=5):
    # 确保目录存在
    if not os.path.exists(weight_save_path):
        os.makedirs(weight_save_path)

    # 初始化一个用于保存模型权重信息的列表

    weights_info_path = os.path.join(weight_save_path, 'weights_info.npy')

    if os.path.exists(weights_info_path):
        weights_info = np.load(weights_info_path, allow_pickle=True).tolist()
    else:
        weights_info = []

    # 当前epoch的Validation Accuracy
    current_val_accuracy = np.mean(val_accuracy)
    
    # 检查当前的Validation Accuracy是否进入了前top_k名
    if len(weights_info) < top_k or current_val_accuracy > min(weights_info, key=lambda x: x['val_accuracy'])['val_accuracy']:
        # 保存当前的模型权重
        weight_file = os.path.join(weight_save_path, f'epoch_{epoch}_val_acc_{current_val_accuracy:.4f}.pth')
        torch.save(model.state_dict(), weight_file)
        
        # 更新权重信息
        weights_info.append({
            'epoch': epoch,
            'val_accuracy': current_val_accuracy,
            'weight_file': weight_file
        })

        # 如果超过top_k个权重，删除最低的那个
        if len(weights_info) > top_k:
            weights_info.sort(key=lambda x: x['val_accuracy'], reverse=True)
            to_remove = weights_info.pop(-1)
            os.remove(to_remove['weight_file'])
        
        # 保存更新后的权重信息
        np.save(weights_info_path, weights_info)