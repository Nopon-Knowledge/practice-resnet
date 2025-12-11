import os
from shutil import copy, rmtree
import random
 
 
def mk_file(file_path: str):
    if os.path.exists(file_path):
        # 如果文件夹存在，则先删除原文件夹再重新创建
        rmtree(file_path)
    os.makedirs(file_path)
 
 
def main():
    # 保证随机可复现
    random.seed(0)
 
    # 将数据集中10%的数据划分到验证集中
    split_rate = 0.1
 
    # 指向解压后的flower_photos文件夹
    # getcwd()：该函数不需要传递参数，获得当前所运行脚本的路径
    cwd = os.getcwd()
    # join()：用于拼接文件路径，可以传入多个路径
    data_root = os.path.join(cwd, "flower_data")
    origin_flower_path = os.path.join(data_root, "flower_photos")
    # 确定路径存在，否则反馈错误
    assert os.path.exists(origin_flower_path), "path '{}' does not exist.".format(origin_flower_path)
    # isdir()：判断某一路径是否为目录
    # listdir()：返回指定的文件夹包含的文件或文件夹的名字的列表
    flower_class = [cla for cla in os.listdir(origin_flower_path)
                    if os.path.isdir(os.path.join(origin_flower_path, cla))]
 
    # 创建训练集train文件夹，并由类名在其目录下创建子目录
    train_root = os.path.join(data_root, "train")
    mk_file(train_root)
    for cla in flower_class:
        # 建立每个类别对应的文件夹
        mk_file(os.path.join(train_root, cla))
 
    # 创建验证集val文件夹，并由类名在其目录下创建子目录
    val_root = os.path.join(data_root, "val")
    mk_file(val_root)
    for cla in flower_class:
        # 建立每个类别对应的文件夹
        mk_file(os.path.join(val_root, cla))
 
    # 遍历所有类别的图像并按比例分成训练集和验证集
    for cla in flower_class:
        cla_path = os.path.join(origin_flower_path, cla)
        # iamges列表存储了该目录下所有图像的名称
        images = os.listdir(cla_path)
        num = len(images)
        # 随机采样验证集的索引
        # 从images列表中随机抽取k个图像名称
        # random.sample：用于截取列表的指定长度的随机数，返回列表
        # eval_index保存验证集val的图像名称
        eval_index = random.sample(images, k=int(num * split_rate))
        for index, image in enumerate(images):
            if image in eval_index:
                # 将分配至验证集中的文件复制到相应目录
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(val_root, cla)
                copy(image_path, new_path)
            else:
                # 将分配至训练集中的文件复制到相应目录
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(train_root, cla)
                copy(image_path, new_path)
                # '\r'回车，回到当前行的行首，而不会换到下一行，如果接着输出，本行以前的内容会被逐一覆盖
                # end=""：将print自带的换行用end中指定的str代替
            print("\r[{}] processing [{}/{}]".format(cla, index + 1, num), end="")
        print()
 
    print("processing done!")
 
 
if __name__ == '__main__':
    main()