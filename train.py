import os
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torchvision.models import resnet34
import torch.optim as optim  #优化器，如adam
from transformers.models.esm.openfold_utils import data_transforms

#step1：数据预处理配置
data_transform = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        #RandomResizedCrop（224）：随机裁剪成224x224
        #RandomHorizontalFlip()随机水平翻转
        #ToTensor()转换为张量
        #Normalize：标准化：：
#(0.485, 0.456, 0.406) ：ImageNet数据集的RGB均值
#(0.229, 0.224, 0.225) ：ImageNet数据集的RGB方差
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),

    ])
}

#step2：创建数据加载器
def get_dataloader(train_root, val_root, transform ,batch_size = 8):
    """
    创建数据加载器
    参数：
        train_root:训练集路径
        val_root:验证集路径
        transform:数据预处理配置
        batch_size:每批图片数量
    返回：
        train_loader:训练数据加载器
        val_loader:验证数据加载器
        train_num:训练集图片数
        val_num:验证集图片数
    :param train_root:
    :param val_root:
    :param transform:
    :param batch_size:
    :return:
    """
    train_dataset = datasets.ImageFolder(root=train_root, transform=transform['train'])
    val_dataset = datasets.ImageFolder(root=val_root, transform=transform['val'])

    train_num = len(train_dataset)
    val_num = len(val_dataset)

    print(f'训练集{train_num}张图片')
    print(f'验证集{val_num}张图片')
    print(f'类别映射{train_dataset.class_to_idx}')

    #创建数据加载器
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,    #打乱顺序，防止过拟合
                                               num_workers=0)#windows下必须设为0
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=val_num,
                                             shuffle=False,
                                             num_workers=0)
    return train_loader, val_loader,train_num,val_num

# step3:创建模型
def create_model():
    """
    创建并配置模型（加载本地预训练权重）

    :return:
        net:配置好的模型
    """
    # 创建net34模型
    net = resnet34(pretrained=False)

    # 加载本地预训练权重
    model_weight_path = r'D:\data\opencv\learn\car-opencv\resnet34-b627a593.pth'
    net.load_state_dict(torch.load(model_weight_path,map_location='cpu'))

    # 修改最后一层，输出为2类
    in_channels = net.fc.in_features#获取最后一层的维度（512）
    net.fc = nn.Linear(in_channels, 2)
    return net

#step5:训练配置
def train_model():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备:{device}')

    # 2、获取数据加载器
    data_root = r"D:\data\opencv\learn\car-opencv\train_data"
    train_root = os.path.join(data_root, 'train')
    val_root = os.path.join(data_root, 'test')

    train_loader,val_loader,train_num,val_num = get_dataloader(train_root, val_root, data_transform, batch_size = 8)

    #3、创建模型
    net = create_model()
    net.to(device)#把模型放到设备上

    #4配置训练参数
    loss_function = nn.CrossEntropyLoss()#交叉熵损失函数
    optimizer = optim.Adam(net.parameters(), lr=0.001)#Adam优化器
    epochs = 100
    save_path = "resnet34_pretrain.pth"#模型保存路径
    best_acc = 0#最佳准确率

    #开始训练
    for epoch in range(epochs):
        print(f"\n{'='*50}")
        print(f"第{epoch+1}/{epochs}轮训练")
        print('='*50)

        #训练模式
        net.train()
        running_loss = 0.0

        for step,data in enumerate(train_loader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            #梯度清零，防止积累
            optimizer.zero_grad()

            #前向传播（输入图片，得到预测）
            outputs = net(images)

            #计算损失（预测和真实标签的差距）
            loss = loss_function(outputs, labels)

            #反向传播（计算梯度）
            loss.backward()

            #更新参数（根据梯度调整权重）
            optimizer.step()

            running_loss += loss.item()

            #每二十步打印一次
            if step % 20 == 19:
                print(f"第{step+1}步，损失：{loss.item():.4f}")

        # step7：验证阶段
        # 验证模式（关闭 dropout 等）
        net.eval()
        acc = 0.0

        # 不需要计算梯度（节省内存）
        with torch.no_grad():
            for val_data in val_loader:
                val_images, val_labels = val_data
                val_images = val_images.to(device)
                val_labels = val_labels.to(device)

                outputs = net(val_images)
                # 取概率最大的类别
                predict_y = torch.max(outputs, dim=1)[1]
                # 统计正确预测的数量
                acc += torch.eq(predict_y, val_labels).sum().item()

        val_acc = acc / val_num
        print(f"\n训练损失: {running_loss / len(train_loader):.4f}")
        print(f"验证准确率: {val_acc:.4f}")

        # 保存最优模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), save_path)
            print(f"保存模型（准确率: {best_acc:.4f}）")

    print(f"\n 训练完成！最佳准确率: {best_acc:.4f}")


if __name__ == '__main__':
    train_model()















