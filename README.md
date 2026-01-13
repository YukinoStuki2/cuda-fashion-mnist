# Fashion-mnist的cuda推理与权重训练

**2025年秋季中国科学院大学《GPU架构与编程》项目一**

---

本仓库中最终用来提交评测的在 `提交/` 目录。其它文件属于个人记录，这里不做说明。

## 目录：`提交/`

`提交/` 下包含 3 个文件：

- `inference.cu`：CUDA 推理程序（读取导出的权重 `.txt` + FashionMNIST 测试集，输出推理耗时与准确率/结果）。

- `train.py`：PyTorch + SpikingJelly 训练脚本（训练 SCNN，并将权重导出为一组 `.txt` 文件，供 `inference.cu` 读取）。

- `weight.zip`：一份导出的权重参数打包（解压后得到若干 `*.txt` 权重文件）。

> 注意：`inference.cu` 会读取 `conv1.bias.txt`，有特殊目录要求权重位于../../../weight/才能正常读取）。

---

## 1) `提交/train.py`（训练 + 导出权重）

### 功能

- 使用 FashionMNIST 数据集训练一个时序步长为 **T=2** 的 SCNN（结构与 `inference.cu` 中的宏定义对应）。

- 训练过程中维护 EMA/SWA，验证集提升时会将当前最优模型的参数导出为文本文件（`*.txt`）。

### 运行

在 `提交/` 目录执行（或从任意目录执行但保证脚本能写入同目录）：

- 安装依赖：

  - conda create -n cuda-cnn-tester-2025 python=3.12 

  - pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url <https://download.pytorch.org/whl/cu118> 

  - pip install spikingjelly

- 开始训练：

  - `python 提交/train.py`

### 输出（权重导出）

当精度刷新时，会在脚本所在目录导出权重文本文件：

- `conv1.weight.txt` / `conv1.bias.txt`

- `conv2.weight.txt` / `conv2.bias.txt`

- `fc1.weight.txt` / `fc1.bias.txt`

- `fc2.weight.txt` / `fc2.bias.txt`

- `fc3.weight.txt` / `fc3.bias.txt`

---

## 2) `提交/weight.zip`（权重包）

### 功能

- 提供一份已导出的权重文件集合，用于上传评测，精度满足要求。

### 使用

- 解压后，该目录内应当出现若干 `*.txt` 权重文件，供 `inference.cu` 加载。

---

## 3) `提交/inference.cu`（CUDA 推理）

### 功能

- 从命令行参数传入一个目录 `dir`，程序会：

  1. 读取权重文件（`conv1.weight.txt`、`conv1.bias.txt`、…、`fc3.bias.txt`）

  2. 读取 FashionMNIST 测试集：

     - `data/FashionMNIST/raw/t10k-images-idx3-ubyte`

     - `data/FashionMNIST/raw/t10k-labels-idx1-ubyte`

  3. 运行推理并统计结果

### 编译与运行

评测服务器CUDA版本为11.8，

**编译命令为：**

 nvcc [inference.cu](http://inference.cu) -o inference_prog -Xcompiler "-O3 -std=c++14" -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -rdc=true 

评测将在单块Tesla V100S-PCIE-32GB上进行。

---

## 4) `提交/inference.cu` 详细代码解释

### 4.1 依赖库

| 头文件 / 库 | 用途 |
|-------------|------|
| `<cuda_runtime.h>` | CUDA 运行时 API（内存分配、流、同步等） |
| `<cuda_fp16.h>` | 半精度浮点类型 `__half` 及其内建转换函数 |
| `<mma.h>` | NVIDIA Tensor Core WMMA（Warp Matrix Multiply-Accumulate）API |
| `<intrin.h>` (MSVC) / `__builtin_bswap32` (GCC) | 字节序转换，用于读取 big-endian 的 MNIST 文件头 |
| 标准 C++ 头（`<iostream>`, `<fstream>`, `<vector>`, `<chrono>` 等） | 文件读写、计时、容器 |

---

### 4.2 网络结构宏定义

代码使用宏描述 SCNN 的结构，与 `train.py` 保持一致：

| 宏 | 含义 |
|----|------|
| `T_STEPS` | SNN 时间步数（默认 2） |
| `C1_IN=1, C1_OUT=8, K1=5` | conv1：输入 1 通道 → 输出 8 通道，卷积核 5×5 |
| `C2_IN=8, C2_OUT=24, K2=5` | conv2：8 → 24 通道，5×5 |
| `FC1_IN=384, FC1_OUT=160` | 全连接层 1（24×4×4 → 160） |
| `FC2_IN=160, FC2_OUT=96` | 全连接层 2（160 → 96） |
| `FC3_IN=96, FC3_OUT=10` | 输出层（96 → 10 类） |
| `V_TH=0.7f` | IF 神经元阈值电压 |

---

### 4.3 内核（Kernel）列表及作用

#### 卷积 & 池化阶段

| 内核 | 作用 |
|------|------|
| `conv2d_conv1_const_kernel` | conv1 前向：从 constant memory 读取权重，执行 5×5 卷积（28×28 → 24×24） |
| `if1_pool1_fused_kernel` | IF1 神经元 + 2×2 MaxPool 融合：对 conv1 输出做膜电位累加、阈值发放、池化（24×24 → 12×12），脉冲以 half 存储 |
| `conv2_if_pool2_fused_const_smem_flat_oc2` | conv2 + IF2 + pool2 三合一：利用 shared memory 缓存输入 tile，一次内核完成 5×5 卷积→膜电位→脉冲→2×2 池化→输出 half 脉冲（12×12 → 4×4） |

#### 全连接阶段

| 内核 | 作用 |
|------|------|
| `flatten_kernel` | 将 [N, C, H, W] 转为 [N, F] 一维排布（辅助，非主路径） |
| `fc1_transpose_pad_colmajor_v2` | 将 FC1 权重转置并 pad 为 16 对齐的列主序，供 WMMA 使用 |
| `fc2_transpose_pad_colmajor` | FC2 权重转置 + pad |
| `fc1_wmma_if_kernel` | FC1 使用 Tensor Core WMMA 做矩阵乘 + IF 神经元（sm_70+ 路径） |
| `fc2_wmma_if_kernel` | FC2 WMMA + IF |
| `linear_if_fp16W_cached_kernel_fc1` | FC1 fallback 路径（非 Tensor Core）：使用 shared memory 缓存权重，FMA 向量化 |
| `linear_if_fp16W_cached_kernel_fc2` | FC2 fallback |
| `fc3_accum_kernel_floatW` | 输出层 FC3：纯 float 权重，PTX `float4` 向量读取 + FMA，结果累加到 logits |

#### 辅助内核

| 内核 | 作用 |
|------|------|
| `float_to_half_kernel` | float → half 批量转换 |
| `half_to_float_kernel` | half → float 批量转换（fallback 路径用） |
| `argmax_kernel` | 对 logits 取 argmax，得到预测类别 |

---

### 4.4 性能优化技术

1. **Constant Memory 存放权重/偏置**  
   conv1/conv2 权重、所有层的 bias 放入 `__constant__`，编译器可进行广播缓存，减少 global memory 带宽。

2. **PTX 内联汇编**  
   - `ptx_ld_const_f32`：显式 constant load，提示缓存行为。  
   - `ptx_ld_global_f32x4` / `ptx_fma`：向量化读取 + FMA 流水，减少指令数。

3. **半精度（FP16）存储**  
   - 膜电位 `mem_t` 可配置为 `__half`，节省显存带宽。  
   - conv2 权重可存为 half（`USE_HALF_CONV2_W`）。

4. **Shared Memory Tile**  
   - `conv2_if_pool2_fused_const_smem_flat_oc2` 将输入 tile（12×12）加载到 shared memory，多输出通道复用，减少 global load 次数。

5. **Tensor Core WMMA（sm_70+）**  
   - FC1/FC2 在 V100 等 sm_70+ GPU 上使用 `nvcuda::wmma` 做 16×16×16 矩阵乘，吞吐远超纯 FMA。  
   - 权重预先转置 + pad 为 16 倍数，满足 WMMA 对齐要求。

6. **Kernel Fusion**  
   - IF 神经元、池化、展平等操作与卷积/全连接融合，减少 kernel launch 开销和中间数据读写。

7. **双缓冲 + 异步流**  
   - 使用两块输入缓冲 `d_in[0/1]` 与 `cudaMemcpyAsync`，实现 Host→Device 传输与计算的 overlap。

8. **`__launch_bounds__`**  
   - 大部分内核声明 `__launch_bounds__(256, 2)`，提示编译器寄存器分配，保障 occupancy。

9. **向量化读写**  
   - 使用 `float4` / `__half2` 读写，提高内存事务效率。

---

### 4.5 推理主流程（`scnn_inference` 函数）

```text
对于每个 batch：
  1. cudaMemcpyAsync 将图像拷入 d_in（双缓冲）
  2. conv1 (constant weight) → half 输出
  3. IF1 + pool1 融合 → half 脉冲
  4. conv2 + IF2 + pool2 融合 → half 脉冲 (flatten)
  5. 若 sm_70+：
       FC1 WMMA + IF → FC2 WMMA + IF
     否则：
       half→float 转换 → FC1 fallback → FC2 fallback
  6. FC3 累加 logits
  7. 重复 T 步，logits 累加
  8. argmax → 预测结果
```

---

### 4.6 编译开关

| 宏 | 默认 | 说明 |
|----|------|------|
| `T_STEPS` | 2 | 时间步数 |
| `USE_FP16_STATE` | 1 | 膜电位用 half 存储 |
| `USE_HALF_CONV2_W` | 1 | conv2 权重用 half |
| `USE_PTX` | 1 | 启用 PTX 内联优化 |
| `USE_WMMA` | 1 | 启用 Tensor Core 路径（运行时检测 arch） |
| `IF_SUBTRACT_RESET` | 0 | 0 = 硬复位；1 = 减阈值软复位 |