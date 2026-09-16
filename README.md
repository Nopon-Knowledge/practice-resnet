# 运行方法

以下命令均在项目根目录执行。

## 1. 安装依赖

创建并激活虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell 使用：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

安装依赖：

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2. 使用现有权重预测图片

项目已包含 `resNet34.pth` 和 `class_indices.json`，可以直接用于预测。

先将 `predict.py` 中的 `img_path` 和 `weights_path` 分别改为：

```python
img_path = "./flower_data/val/roses/15011625580_7974c44bce.jpg"
weights_path = "./resNet34.pth"
```

`img_path` 也可以替换为自己的 RGB 图片路径。然后运行：

```bash
python predict.py
```

终端会输出各类别的预测概率，并弹出带有预测结果的图片窗口。无图形界面的服务器可使用以下命令，仅查看终端输出：

```bash
python -c "import matplotlib; matplotlib.use('Agg'); import predict; predict.main()"
```

## 3. 重新训练（可选）

项目已包含划分好的 `flower_data/train` 和 `flower_data/val`，运行：

```bash
python train.py
```

默认训练 20 轮，每批 16 张图片，可在 `train.py` 中修改 `epochs` 和 `batch_size`。脚本在 CUDA 可用时使用 GPU，否则使用 CPU；当前代码在 Mac 上也使用 CPU。

训练会写入 `class_indices.json`，并将验证准确率最高的权重保存到 `resNet34.pth`，覆盖同名文件。需要保留仓库自带权重时，请先备份该文件。训练完成后按第 2 步运行预测。

如需重新划分数据集，先把按类别分文件夹存放的原始图片放在 `flower_data/flower_photos/` 下，再运行：

```bash
python split_data.py
python train.py
```

划分脚本会删除并重建 `flower_data/train` 和 `flower_data/val`，按约 9:1 分配训练集和验证集。
