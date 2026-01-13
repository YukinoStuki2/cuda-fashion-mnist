cmake_minimum_required(VERSION 3.28)
project(cuda LANGUAGES CUDA CXX)

# 评测机使用 c++14，这里也对齐
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 编译期宏（按需）
add_compile_definitions(
#        USE_PTX=1
#        USE_WMMA=1
#        USE_WMMA_FUSED=1
#        T_STEPS=2
)

add_executable(cuda
        inference.cu
)

# 等价于 nvcc -rdc=true
set_target_properties(cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

# 架构对齐评测机（仅 60/61/70 的 SASS）
# 本机 4090 将从 compute_70 的 PTX JIT 运行，从而路径更接近评测机
set_target_properties(cuda PROPERTIES CUDA_ARCHITECTURES "60;61;70")

# NVCC 选项：
# - ptxas O3 + 全局缓存策略
# - 额外补一条 PTX 兜底：compute_70,code=compute_70（评测机无需，但本机 4090 需要）
# - 仅在 Release/RelWithDebInfo 下传递 /O2 /DNDEBUG，避免与 Debug 的 /RTC1 冲突
target_compile_options(cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
        -Xptxas=-O3,-dlcm=ca
        -gencode;arch=compute_70,code=compute_70
        $<$<CONFIG:Release>:-Xcompiler=/O2>
        $<$<CONFIG:Release>:-Xcompiler=/DNDEBUG>
        $<$<CONFIG:RelWithDebInfo>:-Xcompiler=/O2>
        $<$<CONFIG:RelWithDebInfo>:-Xcompiler=/DNDEBUG>
        >
)

# 如需查看 nvlink 选用了哪些 cubin，可临时打开
# target_link_options(cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xnvlink=-v>)