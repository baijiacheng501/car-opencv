import cv2
import torch
import os
import json
import numpy as np
import pickle
from tqdm import tqdm
from torchvision.models import resnet34
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# 1. 配置参数
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'device: {device}')

# 模型路径
model_path = r'D:\data\opencv\learn\car-opencv\resnet34_pretrain.pth'
# 车位字典路径
spot_dict_path = r'D:\data\opencv\learn\car-opencv\spot_dict.pickle'
# 测试图片路径
test_image_path = r'D:\data\opencv\learn\car-opencv\frames\frame_1049.jpg'

# 2. 数据预处理
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


# 3. 创建并加载模型
def create_model():
    net = resnet34(pretrained=False)
    in_channels = net.fc.in_features
    net.fc = nn.Linear(in_channels, 2)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device)
    net.eval()
    return net


# 4. 模型预测函数
def model_infer(img_pil, model):
    # 预处理
    img = data_transforms(img_pil)
    img = torch.unsqueeze(img, dim=0).to(device)

    with torch.no_grad():
        output = model(img)
        predict = torch.softmax(output, dim=1)
        predict_cla = torch.argmax(predict, dim=1).cpu().numpy()[0]
        predict_prob = predict.cpu().numpy()[0][predict_cla]

    return predict_cla, predict_prob


# 5. 在图片上预测所有车位
def predict_on_img(img, spot_dict, model):
    new_image = np.copy(img)
    overlay = np.copy(img)

    cnt_empty = 0
    all_spots = len(spot_dict)

    for spot in tqdm(spot_dict.keys(), desc="预测中"):
        x1, y1, x2, y2 = spot
        x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)

        # 裁剪车位图片
        spot_img = img[y1:y2, x1:x2]
        spot_img_pil = Image.fromarray(cv2.cvtColor(spot_img, cv2.COLOR_BGR2RGB))

        # 模型预测
        predict_cla, predict_prob = model_infer(spot_img_pil, model)

        # 如果是空车位（类别0），画绿色框
        if predict_cla == 0:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
            cnt_empty += 1

    # 混合叠加
    cv2.addWeighted(overlay, 0.5, new_image, 1 - 0.5, 0, new_image)

    # 添加文字说明
    cv2.putText(new_image, f"Available: {cnt_empty} spots", (30, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(new_image, f"Total: {all_spots} spots", (30, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return new_image, cnt_empty


def predict_on_video(video_path, spot_dict, model):
    """在视频上实时预测"""
    cap = cv2.VideoCapture(video_path)

    count = 0
    while True:
        ret, image = cap.read()
        if not ret:
            break

        count += 1
        if count % 5 == 0:  # 每5帧预测一次
            count = 0
            result_img, cnt_empty = predict_on_img(image, spot_dict, model)

            cv2.imshow('Parking Prediction', result_img)

            if cv2.waitKey(10) & 0xFF == ord('q'):  # 按q退出
                break

    cap.release()
    cv2.destroyAllWindows()

# 6. 主函数
def main():
    print("正在加载模型...")
    model = create_model()
    print("模型加载完成")

    print("正在加载车位字典...")
    with open(spot_dict_path, 'rb') as handle:
        spot_dict = pickle.load(handle)
    print(f"车位字典加载完成，共 {len(spot_dict)} 个车位")

    print("正在读取测试图片...")
    video_path =r"D:\data\opencv\learn\car-opencv\video\parking_video.mp4"
    print(f"正在预测视频：{video_path}")
    predict_on_video(video_path, spot_dict, model)
    print("预测完成")
    img = cv2.imread(test_image_path)
    if img is None:
        print(f"无法读取图片: {test_image_path}")
        return
    print("图片读取完成")

    print("正在预测...")
    result_img, cnt_empty = predict_on_img(img, spot_dict, model)  # 调用正确的函数名
    print(f"预测完成！空车位: {cnt_empty} 个")

    # 显示结果
    cv2.imshow('Parking Prediction', result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 保存结果
    cv2.imwrite('prediction_result.jpg', result_img)
    print("结果已保存到 prediction_result.jpg")


if __name__ == '__main__':
    main()