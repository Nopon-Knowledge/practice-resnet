import torch.nn as nn
import torch
 
# 定义残差网络18/34的残差结构，为2个3*3的卷积
class BasicBlock(nn.Module):
    # 判断残差结构中，主分支的卷积核个数是否发生变化，不变则为1
    expansion = 1
    def __init__(self,in_channel,out_channel,stride=1,downsample=None,**kwargs):
        super(BasicBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel,out_channels=out_channel,
                               kernel_size=3,stride=stride,padding=1,bias=False
        )
        # 使用批量归一化
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel,out_channels=out_channel,
                               kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample =downsample
    def forward(self,x):
 
        identity = x
        if self.downsample is not None:
            identity=self.downsample(x)  # self当前对象实例，用于访问对象的属性和方法
        out = self.conv1(x)
        out =self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
        out +=identity
        out = self.relu(out)
        return out
class Bottleneck(nn.Module):
    # expansion是指在每个小残差块内，减少尺度增加维度的倍数
    # 该类输出的通道是输入的四倍
    expansion = 4
    def __init__(self,in_channel,out_channel,strdie=1,downsample=None,
                 groups=1,width_per_group=64):
        super(Bottleneck,self).__init__()
 
        width = int(out_channel*(width_per_group/64.))*groups
 
        self.conv1 = nn.Conv2d(in_channels=in_channel,out_channels=width,kernel_size=1,stride=1,bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2= nn.Conv2d(in_channels=width,out_channels=width,groups=groups
                              ,kernel_size=3,stride=strdie,bias=False,padding=1)
        self.bn2 = nn.BatchNorm2d(width)
 
        self.conv3 = nn.Conv2d(in_channels=width,out_channels=width,groups=groups,
                               kernel_size=3,stride=strdie,bias=False,padding=1)
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansio)
 
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        
    def forward(self,x):
        identify = x
        if self.downsample is not None:
            identify = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
 
        out = self.conv3(out)
        out = self.bn3(out)
 
        out+=identify
        out = self.relu(out)
        return out
#定义ResNet类
class ResNet(nn.Module):
    def __init__(self,block,blocks_num,num_classes=1000,include_top=True,groups=1,width_per_group=64):
        super(ResNet,self).__init__()
        self.include_top = include_top
        # maxpool的输出通道数为64，残差结构的输入通道为64
        self.in_channel = 64
        self.groups = groups
        self.width_per_group = width_per_group
        self.conv1 = nn.Conv2d(3,self.in_channel,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        # 浅层步长为1，深层的步长为2
        # block：定义了两种残差模块
        # block_num:定义了残差快的个数
        self.layer1 = self._make_layer(block,64,blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1],stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2],stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3],stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1,1)) # 自适应平均池化，指定输出（H，w），通道数不变
            self.fc = nn.Linear(512*block.expansion,num_classes)
        # 遍历网络中的每一层
        # 继承nn.Moudle类中的一个方法：self.Moudle(),它会返回该网络中所有的moudles
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                # kaiming正态分布初始化，使得卷积层反向传播的输出的方差都为1
                # fan_in权重是通过线性层隐形确定
                # fan_out：通过创建随机矩阵显式创建权重
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
    def _make_layer(self,block,channel,block_num,stride=1):
        downsample = None
        if stride !=1 or self.in_channel !=channel*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel,channel*block.expansion,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(channel*block.expansion)
            )
        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group = self.width_per_group))
        self.in_channel = channel*block.expansion
 
        for _ in range(1,block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group = self.width_per_group))
        # Sequential:自定义顺序连接成模型，生成网络结构
        return nn.Sequential(*layers)
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # 上述都是静态层，下边是动态层
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
 
        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x,1)
            x = self.fc(x)
        return x
 
 
# ResNet()中block参数对应的位置是BasicBlock或Bottleneck
# ResNet()中blocks_num[0-3]对应[3, 4, 6, 3]，表示残差模块中的残差数
# 34层的resnet
def resnet34(num_classes=5, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)
 
 
# 50层的resnet
def resnet50(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)
 
 
# 101层的resnet
def resnet101(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)
 