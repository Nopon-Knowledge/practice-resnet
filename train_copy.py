# 自己练习用的训练脚本
# 导包
import os
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.data
from torchvision import transforms,datasets
from tqdm import tqdm

from model import resnet34

def main():
    # 选择GPU或CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # 数据集的预处理
    data_transform = {
        # 训练集
        "train": transforms.Compose([
            # 将每张图片的尺寸调整为224x224
            transforms.RandomResizedCrop(224),
            # 水平翻转，0.5的概率
            transforms.RandomHorizontalFlip(),
            # 将图片转换为计算机可以识别的tensor格式
            transforms.ToTensor(),
            # 将像素值归一化到[-1,1]之间，加速模型收敛
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
        ]),
        # 验证集
        "val": transforms.Compose([
            # 调整图片尺寸
            transforms.Resize(224),
            # 中心裁剪
            transforms.CenterCrop(224),

            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # 获取当前脚本的目录
    data_root = os.path.abspath(os.getcwd())

    # 获取数据集路径
    image_path = os.path.join(data_root, "flower_data")  # flower_data数据集的路径
    # image_path = '/root/ResNet/flower_data'
    
    # 拿到数据集，并指定预处理的方式
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])

    # 获取训练集的长度
    train_num = len(train_dataset)
    print("训练集样本数量: {}".format(train_num))

    # 获取所有类别的索引
    flower_list = train_dataset.class_to_index

    # 交换key和value
    cla_dict = dict((val, key) for key, val in flower_list.items())

    # 将类别索引字典转换为json格式
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    # 每批次送入多少张图片 320/16=20 1epoch
    batch_size = 16

    # 返回cpu的数量
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    # 加载训练集
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    # 验证集准备
    val_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                       transform=data_transform["val"])

    val_num = len(val_dataset)
    # print("验证集样本数量: {}".format(val_num))

    # 使用多少个cpu去生成val数据加载器
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=nw)

    # 训练集和验证集数量的可视化
    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    # 拿到模型，送入gpu或cpu中
    net = resnet34()
    net.to(device)

    # 加载预训练权重
    # model_weight_path = "./resnet34-pre.pth"

    # 定义损失函数为交叉熵损失
    loss_function = nn.CrossEntropyLoss()

    # 打印模型参数
    params = [p for p in net.parameters() if p.requires_grad]
    print("可训练的参数数量为：", len(params))
    print(params)

    # 定义优化器
    optimizer = optim.Adam(params=net.parameters(), lr=0.0001)

    # epoch代表训练多少轮
    epochs = 10

    # 记录最佳准确率
    best_acc = 0.0

    # 模型权重的保存路径
    save_path = './resNet34.pth'

    # 在终端显示训练进度
    train_steps = len(train_loader)


    for epoch in range(epochs):
        # 训练模型
        net.train()

        # 监测训练损失、判断模型是否收敛
        running_loss = 0.0

        # 训练可视化，一个条形的进度条
        train_bar = tqdm(train_loader, file=sys.stdout)

        
        for step, data in enumerate(train_bar):
            # 拿到每批次的图像和标签
            images, labels = data

            # 计算训练值
            logits = net(images.to(device))

            # 计算损失
            loss = loss_function(logits, labels.to(device))

            # 优化器过往梯度清零
            optimizer.zero_grad()

            # 反向传播，计算当前梯度
            loss.backward()
            optimizer.step()

            # 累加每个batch的损失，梯度累加
            running_loss += loss.item()
            
            # 每训练完一个batch，更新一次训练进度条
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # 验证模式
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch

        with torch.no_grad():
            # 验证集可视化
            val_bar = tqdm(val_loader, file=sys.stdout)

            # 拿到每批次的验证图片和标签
            for val_data in val_bar:
                # 拿到图片和标签
                val_images, val_labels = val_data

                # 返回一个tensor，形状为[batch_size, num_classes]
                outputs = net(val_images.to(device))
                # 返回两个tensor，一个是概率最大的值，一个是对应的索引，取第二个tensor
                predict_y = torch.max(outputs, dim=1)[1]

                # 计算预测正确的数量
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                #验证集的可视化
                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                          epochs)

        # 计算整个验证集的准确率
        val_accurate = acc / val_num

        # 打印每一轮的训练损失和验证准确率
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        # 保存最佳模型
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

        # torch.save(net.state_dict(), "last.pth")

    print('Finished Training')


        
if __name__ == '__main__':
    main()