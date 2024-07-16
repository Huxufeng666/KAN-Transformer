import argparse
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from PIL import Image, ImageDraw, ImageFont
from models.deep_vision_transformer import deepvit_L  # 修改为你实际的模型导入路径

def get_args_parser():
    parser = argparse.ArgumentParser('Model Inference Script', add_help=False)
    parser.add_argument('--model', default='deepvit_L', type=str, help='Name of the model')
    parser.add_argument('--weight_path', default='output/deepvit_S/epoch_1_val_acc_0.3476.pth', type=str, help='Path to the model weights')
    parser.add_argument('--input_image', default='/home/user/PRPD-dataset/original_dataset/train/0/164.png', type=str, help='Path to the input image')
    parser.add_argument('--output_image', default='/home/user/train/output/predict/', type=str, help='Path to save the output image')
    parser.add_argument('--class_names', default='/home/user/PRPD-dataset/original_dataset/class.txt', type=str, help='Path to the class names file')
    parser.add_argument('--device', default="cuda", type=str, help='Device to run the inference on')
    parser.add_argument('--num_classes', default=5, type=int, help='Number of classes in the model')
    return parser

def load_model(weight_path, device):
    model = deepvit_L()  # 修改为你实际的模型初始化
    checkpoint = torch.load(weight_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model = model.to(device)
    model.eval()
    return model

def predict_image(model, image_path, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  # 增加批次维度
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        _, preds = torch.max(outputs, 1)

    return preds.item()

def postprocess_and_save(output, output_path):
    # 假设输出是单通道的segmentation mask
    output = output.squeeze(0).cpu().numpy()
    output_image = Image.fromarray((output * 255).astype('uint8'))
    output_image.save(output_path)

def load_labels(labels_path):
    class_names = {}
    with open(labels_path, 'r') as f:
        for line in f:
            idx, name = line.strip().split(' ', 1)
            class_names[int(idx)] = name
    return class_names


def save_result_image(image_path, output_path,label, class_names):
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    f = class_names[label]
    text = f"{f}"
    
    text_bbox = font.getbbox(text)
    text_size = (text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1])
    text_position = (10, 10)
    
    draw.rectangle([text_position, (text_position[0] + text_size[0], text_position[1] + text_size[1])], fill="black")
    draw.text(text_position, text, fill="red", font=font)
    output_path = os.path.join(output_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_path = os.path.join(output_path,f"{f}.png")
    image.save(output_path)

def main(args):
    device = torch.device(args.device)
    class_names = load_labels(args.class_names)

    model = load_model(args.weight_path, device)

    label = predict_image(model, args.input_image, args.device)

    save_result_image(args.input_image, args.output_image,label, class_names)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Model Inference Script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
