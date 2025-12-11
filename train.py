import os
import sys
import json
 
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
# 训练resnet34
from model import resnet34
 
 
def main():
    # 如果有NVIDA显卡，转到GPU训练，否则用CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
 
    data_transform = {
        # 训练
        # Compose()：将多个transforms的操作整合在一起
        "train": transforms.Compose([
            # RandomResizedCrop(224)：将给定图像随机裁剪为不同的大小和宽高比，然后缩放所裁剪得到的图像为给定大小
            transforms.RandomResizedCrop(224),
            # RandomVerticalFlip()：以0.5的概率竖直翻转给定的PIL图像
            transforms.RandomHorizontalFlip(),
            # ToTensor()：数据转化为Tensor格式
            transforms.ToTensor(),
            # Normalize()：将图像的像素值归一化到[-1,1]之间，使模型更容易收敛
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        # 验证
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    # abspath()：获取文件当前目录的绝对路径
    # join()：用于拼接文件路径，可以传入多个路径
    # getcwd()：该函数不需要传递参数，获得当前所运行脚本的路径
    data_root = os.path.abspath(os.getcwd())
    # 得到数据集的路径
    image_path = os.path.join(data_root, "flower_data")
    # exists()：判断括号里的文件是否存在，可以是文件路径
    # 如果image_path不存在，序会抛出AssertionError错误，报错为参数内容“ ”
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    # 训练集长度
    train_num = len(train_dataset)
 
    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    # class_to_idx：获取分类名称对应索引
    flower_list = train_dataset.class_to_idx
    # dict()：创建一个新的字典
    # 循环遍历数组索引并交换val和key的值重新赋值给数组，这样模型预测的直接就是value类别值
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # 把字典编码成json格式
    json_str = json.dumps(cla_dict, indent=4)
    # 把字典类别索引写入json文件
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)
 
    # 一次训练载入16张图像
    batch_size = 16
    # 确定进程数
    # min()：返回给定参数的最小值，参数可以为序列
    # cpu_count()：返回一个整数值，表示系统中的CPU数量，如果不确定CPU的数量，则不返回任何内容
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))
    # DataLoader：将读取的数据按照batch size大小封装给训练集
    # dataset (Dataset)：输入的数据集
    # batch_size (int, optional)：每个batch加载多少个样本，默认: 1
    # shuffle (bool, optional)：设置为True时会在每个epoch重新打乱数据，默认: False
    # num_workers(int, optional): 决定了有几个进程来处理，默认为0意味着所有的数据都会被load进主进程
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)
    # 加载测试数据集
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    # 测试集长度
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)
 
    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
 
    # 模型实例化
    net = resnet34()
    net.to(device)

    # # 加载预训练模型权重
    # model_weight_path = "./resnet34-pre.pth"
    # # exists()：判断括号里的文件是否存在，可以是文件路径
    # assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    # net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
    # # 输入通道数
    # in_channel = net.fc.in_features
    # # 全连接层
    # net.fc = nn.Linear(in_channel, 5)
 
    # 定义损失函数（交叉熵损失）
    loss_function = nn.CrossEntropyLoss()
 
    # 抽取模型参数
    params = [p for p in net.parameters() if p.requires_grad]
    # 定义adam优化器
    # params(iterable)：要训练的参数，一般传入的是model.parameters()
    # lr(float)：learning_rate学习率，也就是步长，默认：1e-3
    optimizer = optim.Adam(params, lr=0.0001)
 
    # 迭代次数（训练次数）
    epochs = 20
    # 用于判断最佳模型
    best_acc = 0.0
    # 最佳模型保存地址
    save_path = './resNet34.pth'
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # 训练
        net.train()
        running_loss = 0.0
        # tqdm：进度条显示
        train_bar = tqdm(train_loader, file=sys.stdout)
        # train_bar: 传入数据（数据包括：训练数据和标签）
        # enumerate()：将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在for循环当中
        # enumerate返回值有两个：一个是序号，一个是数据（包含训练数据和标签）
        # x：训练数据（inputs）(tensor类型的），y：标签（labels）(tensor类型）
        for step, data in enumerate(train_bar):
            # 前向传播
            images, labels = data
            # 计算训练值
            logits = net(images.to(device))
            # 计算损失
            loss = loss_function(logits, labels.to(device))
            # 反向传播
            # 清空过往梯度
            optimizer.zero_grad()
            # 反向传播，计算当前梯度
            loss.backward()
            optimizer.step()
 
            # item()：得到元素张量的元素值
            running_loss += loss.item()
 
            # 进度条的前缀
            # .3f：表示浮点数的精度为3（小数位保留3位）
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
 
        # 测试
        # eval()：如果模型中有Batch Normalization和Dropout，则不启用，以防改变权值
        net.eval()
        acc = 0.0
        # 清空历史梯度，与训练最大的区别是测试过程中取消了反向传播
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # torch.max(input, dim)函数
                # input是具体的tensor，dim是max函数索引的维度，0是每列的最大值，1是每行的最大值输出
                # 函数会返回两个tensor，第一个tensor是每行的最大值；第二个tensor是每行最大值的索引
                predict_y = torch.max(outputs, dim=1)[1]
                # 对两个张量Tensor进行逐元素的比较，若相同位置的两个元素相同，则返回True；若不同，返回False
                # .sum()对输入的tensor数据的某一维度求和
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
 
                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)
 
        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))
 
        # 保存最好的模型权重
        if val_accurate > best_acc:
            best_acc = val_accurate
            # torch.save(state, dir)保存模型等相关参数，dir表示保存文件的路径+保存文件名
            # model.state_dict()：返回的是一个OrderedDict，存储了网络结构的名字和对应的参数
            torch.save(net.state_dict(), save_path)
 
    print('Finished Training')
 

    
 
if __name__ == '__main__':
    main()