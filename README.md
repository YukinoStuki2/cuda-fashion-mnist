# GPUtask1（仅说明 `提交/` 目录）

本仓库中与作业/评测直接相关的内容集中在 `提交/` 目录。其它文件属于实验/中间产物/个人记录，这里不做说明。

## 目录：`提交/`

`提交/` 下包含 3 个文件：

- `inference.cu`：CUDA 推理程序（读取导出的权重 `.txt` + FashionMNIST 测试集，输出推理耗时与准确率/结果）。
- `train.py`：PyTorch + SpikingJelly 训练脚本（训练 SCNN，并将权重导出为一组 `.txt` 文件，供 `inference.cu` 读取）。
- `weight.zip`：一份导出的权重参数打包（解压后得到若干 `*.txt` 权重文件）。

> 注意：`inference.cu` 会读取 `conv1.bias.txt`，而 `weight.zip` 的内容是否包含该文件以压缩包内实际文件为准；如缺失，可通过运行 `提交/train.py` 重新导出完整权重（训练脚本会导出 `conv1.bias.txt`）。

---

## 1) `提交/train.py`（训练 + 导出权重）

### 功能
- 使用 FashionMNIST 数据集训练一个时序步长为 **T=2** 的 SCNN（结构与 `inference.cu` 中的宏定义对应）。
- 训练过程中维护 EMA/SWA，验证集提升时会将当前最优模型的参数导出为文本文件（`*.txt`）。

### 运行
在 `提交/` 目录执行（或从任意目录执行但保证脚本能写入同目录）：

- 安装依赖（示例）：
  - `pip install torch torchvision spikingjelly`
- 开始训练：
  - `python 提交/train.py`

### 输出（权重导出）
当精度刷新时，会在脚本所在目录导出权重文本文件（文件名形如）：
- `conv1.weight.txt` / `conv1.bias.txt`
- `conv2.weight.txt` / `conv2.bias.txt`
- `fc1.weight.txt` / `fc1.bias.txt`
- `fc2.weight.txt` / `fc2.bias.txt`
- `fc3.weight.txt` / `fc3.bias.txt`

---

## 2) `提交/weight.zip`（权重包）

### 功能
- 提供一份已导出的权重文件集合（若你不想重新训练，可直接使用）。

### 使用
- 将其解压到一个“权重目录”中（比如 `提交/model/weights/best/`）：
  - `tar -xf 提交/weight.zip -C 提交/model/weights/best`

解压后，该目录内应当出现若干 `*.txt` 权重文件，供 `inference.cu` 加载。

---

## 3) `提交/inference.cu`（CUDA 推理）

### 功能
- 从命令行参数传入一个目录 `dir`，程序会：
  1) 从 `dir/` 下读取权重文件（`conv1.weight.txt`、`conv1.bias.txt`、…、`fc3.bias.txt`）
  2) 从 `dir/../../..` 下读取 FashionMNIST 测试集：
     - `data/FashionMNIST/raw/t10k-images-idx3-ubyte`
     - `data/FashionMNIST/raw/t10k-labels-idx1-ubyte`
  3) 运行推理并统计结果

### 目录约定（很重要）
因为数据集路径是按 `dir/../../..` 拼出来的，你需要让目录结构满足：

- `dir` 指向“权重所在目录”（里面放一堆 `*.txt`）
- `dir` 的上三级目录（`dir/../../..`）下面存在 `data/FashionMNIST/raw/`

一个推荐的示例结构（在仓库根目录下）：

- `提交/data/FashionMNIST/raw/...(数据文件)`
- `提交/model/weights/best/...(权重 txt)`

此时运行时令 `dir = 提交/model/weights/best`，则 `dir/../../..` 正好回到 `提交/`，可以找到 `提交/data/...`。

### 编译与运行（示例）
在仓库根目录执行（示例命令，可按你的 CUDA/显卡架构调整）：

- 编译：
  - `nvcc -O3 -std=c++17 -arch=sm_70 提交/inference.cu -o inference`
- 运行：
  - `./inference 提交/model/weights/best`

> 说明：代码中默认 `T_STEPS=2`，需要与训练脚本的 `T_TIMESTEPS=2` 保持一致。

---

## 常见问题

### VS Code 仍然显示被忽略文件？
- 如果只是“显示”，但不会被 `git add` 加入，一般是 VS Code 的 Git 视图设置导致（比如显示 ignored/untracked）。
- 以 Git 为准：只要 `git status` 不出现这些文件，它们就不会被提交。
