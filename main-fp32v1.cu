// Windows 兼容：把 GCC 的 __builtin_bswap32 映射到 MSVC 的 _byteswap_ulong
#ifdef _MSC_VER
    #include <intrin.h>
    #ifndef __builtin_bswap32
        #define __builtin_bswap32 _byteswap_ulong
    #endif
#endif
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <cuda_runtime.h>


// ========== 优化开关 ==========
// 你当前用“硬重置”达标，保持硬重置
#define IF_SUBTRACT_RESET 0

// 卷积形状常量（固定结构）
#define C1_IN   1
#define C1_OUT  6
#define K1      5
#define C2_IN   6
#define C2_OUT  16
#define K2      5

#ifndef USE_PTX
    #define USE_PTX 1    // 1 开启 PTX，0 关闭（回退到原始 C++ 实现）
#endif

// 以防万一：在 Windows/MSVC 下 __builtin_bswap32 已做过兼容，保持不动

// PTX 辅助：从常量内存加载 float
static __device__ __forceinline__ float ptx_ld_const_f32(const float* p) {
    float v;
    asm volatile(
            "{\n\t"
            ".reg .u64 a;\n\t"
            "cvta.to.const.u64 a, %1;\n\t"
            "ld.const.f32 %0, [a];\n\t"
            "}\n"
            : "=f"(v) : "l"(p));
    return v;
}

// PTX 辅助：从全局内存加载 float（cache at all levels）
static __device__ __forceinline__ float ptx_ld_global_f32(const float* p) {
    float v;
    asm volatile(
            "{\n\t"
            ".reg .u64 a;\n\t"
            "cvta.to.global.u64 a, %1;\n\t"
            "ld.global.ca.f32 %0, [a];\n\t"
            "}\n"
            : "=f"(v) : "l"(p));
    return v;
}

// PTX 辅助：FMA acc = xv * wv + acc
static __device__ __forceinline__ void ptx_fma(float& acc, float xv, float wv) {
    asm volatile("fma.rn.f32 %0, %1, %2, %0;" : "+f"(acc) : "f"(xv), "f"(wv));
}

// 将卷积权重/偏置放到常量内存（Step 2）
__constant__ float c_conv1_w[C1_OUT * C1_IN * K1 * K1];
__constant__ float c_conv1_b[C1_OUT];
__constant__ float c_conv2_w[C2_OUT * C2_IN * K2 * K2];
__constant__ float c_conv2_b[C2_OUT];

inline int div_up(int a, int b) { return (a + b - 1) / b; }
__host__ __device__ inline int idx4(int n,int c,int h,int w,int C,int H,int W){ return ((n*C+c)*H+h)*W+w; }
__host__ __device__ inline int idx2(int o,int i,int I){ return o*I+i; }

// ---------------- CUDA KERNELS ----------------

// conv1：常量内存（NCHW，stride=1，无padding）
__global__ void conv2d_conv1_const_kernel(
        const float* __restrict__ x,      // [N, C1_IN, H, W]
        float* __restrict__ y,            // [N, C1_OUT, H-4, W-4]
        int N, int H, int W
){
    const int Cout = C1_OUT, Cin = C1_IN, KH = K1, KW = K1;
    const int Wout = W - KW + 1;
    const int Hout = H - KH + 1;

    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    int z  = blockIdx.z; // 0..N*Cout-1
    if (ox >= Wout || oy >= Hout || z >= N * Cout) return;

    int n  = z / Cout;
    int oc = z % Cout;

    float acc = c_conv1_b[oc];
#pragma unroll
    for (int ic = 0; ic < Cin; ++ic) {
#pragma unroll
        for (int ky = 0; ky < KH; ++ky) {
            int iy = oy + ky;
#pragma unroll
            for (int kx = 0; kx < KW; ++kx) {
                int ix = ox + kx;
                float xv = x[idx4(n, ic, iy, ix, Cin, H, W)];
                int widx = (((oc * Cin) + ic) * KH + ky) * KW + kx;
                float wv = c_conv1_w[widx];
                acc += xv * wv;
            }
        }
    }
    y[idx4(n, oc, oy, ox, Cout, Hout, Wout)] = acc;
}

// conv2：常量内存（NCHW，stride=1，无padding）
__global__ void conv2d_conv2_const_kernel(
        const float* __restrict__ x,      // [N, C2_IN, H, W]
        float* __restrict__ y,            // [N, C2_OUT, H-4, W-4]
        int N, int H, int W
){
    const int Cout = C2_OUT, Cin = C2_IN, KH = K2, KW = K2;
    const int Wout = W - KW + 1;
    const int Hout = H - KH + 1;

    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    int z  = blockIdx.z; // 0..N*Cout-1
    if (ox >= Wout || oy >= Hout || z >= N * Cout) return;

    int n  = z / Cout;
    int oc = z % Cout;

    float acc = c_conv2_b[oc];
#pragma unroll
    for (int ic = 0; ic < Cin; ++ic) {
#pragma unroll
        for (int ky = 0; ky < KH; ++ky) {
            int iy = oy + ky;
#pragma unroll
            for (int kx = 0; kx < KW; ++kx) {
                int ix = ox + kx;
                float xv = x[idx4(n, ic, iy, ix, Cin, H, W)];
                int widx = (((oc * Cin) + ic) * KH + ky) * KW + kx;
                float wv = c_conv2_w[widx];
                acc += xv * wv;
            }
        }
    }
    y[idx4(n, oc, oy, ox, Cout, Hout, Wout)] = acc;
}

// 2x2 最大池化，stride=2
__global__ void maxpool2x2_kernel(
        const float* __restrict__ x, // [N, C, H, W]
        float* __restrict__ y,       // [N, C, H/2, W/2]
        int N, int C, int H, int W
){
    int Hout = H / 2, Wout = W / 2;
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    int z  = blockIdx.z;
    if (ox >= Wout || oy >= Hout || z >= N * C) return;

    int n = z / C;
    int c = z % C;
    int ih = oy * 2, iw = ox * 2;

    float m = -1e30f;
    m = fmaxf(m, x[idx4(n, c, ih + 0, iw + 0, C, H, W)]);
    m = fmaxf(m, x[idx4(n, c, ih + 0, iw + 1, C, H, W)]);
    m = fmaxf(m, x[idx4(n, c, ih + 1, iw + 0, C, H, W)]);
    m = fmaxf(m, x[idx4(n, c, ih + 1, iw + 1, C, H, W)]);
    y[idx4(n, c, oy, ox, C, Hout, Wout)] = m;
}

// IF 脉冲层
__global__ void ifnode_kernel(
        const float* __restrict__ in, // [N, C, H, W] 或 H=W=1 的向量
        float* __restrict__ v,        // 同形状
        float* __restrict__ out,      // spike 0/1
        int N, int C, int H, int W,
        float v_th
){
    int x = blockIdx.x * blockDim.x + threadIdx.x; // 0..W-1
    int y = blockIdx.y * blockDim.y + threadIdx.y; // 0..H-1
    int z = blockIdx.z;                             // 0..N*C-1
    if (x >= W || y >= H || z >= N * C) return;

    int n = z / C;
    int c = z % C;
    int id = idx4(n, c, y, x, C, H, W);

    float vv = v[id] + in[id];
    float s  = vv >= v_th ? 1.0f : 0.0f;
#if IF_SUBTRACT_RESET
    vv = vv - s * v_th;
#else
    vv = (s > 0.0f) ? 0.0f : vv;
#endif
    v[id]   = vv;
    out[id] = s;
}

// 展平 [N,C,H,W] → [N, C*H*W]
__global__ void flatten_kernel(
        const float* __restrict__ x, // [N,C,H,W]
        float* __restrict__ y,       // [N, F]
        int N, int C, int H, int W
){
    int F = C * H * W;
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // 0..N*F-1
    if (tid >= N * F) return;
    int n = tid / F;
    int f = tid % F;
    int c = f / (H * W);
    int rem = f % (H * W);
    int h = rem / W;
    int w = rem % W;
    y[n * F + f] = x[idx4(n, c, h, w, C, H, W)];
}

//// 全连接 y = x * W^T + b，x:[N,I] W:[O,I] y:[N,O]
//__global__ void linear_kernel(
//        const float* __restrict__ x, // [N, I]
//        const float* __restrict__ W, // [O, I]
//        const float* __restrict__ b, // [O]
//        float* __restrict__ y,       // [N, O]
//        int N, int I, int O
//){
//    int ox = blockIdx.x * blockDim.x + threadIdx.x; // out feature
//    int n  = blockIdx.y * blockDim.y + threadIdx.y; // batch
//    if (ox >= O || n >= N) return;
//
//    float acc = b[ox];
//    const float* xrow = x + n * I;
//    const float* wrow = W + ox * I;
//    for (int i = 0; i < I; ++i) acc += xrow[i] * wrow[i];
//    y[n * O + ox] = acc;
//}
// 全连接 y = x * W^T + b，x:[N,I] W:[O,I] y:[N,O] —— 带 PTX 的累加分支
__global__ void linear_kernel(
        const float* __restrict__ x, // [N, I]
        const float* __restrict__ W, // [O, I]
        const float* __restrict__ b, // [O]
        float* __restrict__ y,       // [N, O]
        int N, int I, int O
){
    int ox = blockIdx.x * blockDim.x + threadIdx.x; // out feature
    int n  = blockIdx.y * blockDim.y + threadIdx.y; // batch
    if (ox >= O || n >= N) return;

    float acc = b[ox];
    const float* xrow = x + n * I;
    const float* wrow = W + ox * I;

#if USE_PTX
    // 纯标量 PTX 版（可在此处做向量化展开：以4为步长+1个尾部）
    int i = 0;
    for (; i + 4 <= I; i += 4) {
        // 向量化加载（可选）：这里先用 4 次标量，保证兼容性稳定
        float x0 = ptx_ld_global_f32(xrow + i + 0);
        float w0 = ptx_ld_global_f32(wrow + i + 0);
        ptx_fma(acc, x0, w0);

        float x1 = ptx_ld_global_f32(xrow + i + 1);
        float w1 = ptx_ld_global_f32(wrow + i + 1);
        ptx_fma(acc, x1, w1);

        float x2 = ptx_ld_global_f32(xrow + i + 2);
        float w2 = ptx_ld_global_f32(wrow + i + 2);
        ptx_fma(acc, x2, w2);

        float x3 = ptx_ld_global_f32(xrow + i + 3);
        float w3 = ptx_ld_global_f32(wrow + i + 3);
        ptx_fma(acc, x3, w3);
    }
    for (; i < I; ++i) {
        float xv = ptx_ld_global_f32(xrow + i);
        float wv = ptx_ld_global_f32(wrow + i);
        ptx_fma(acc, xv, wv);
    }
#else
    for (int i = 0; i < I; ++i) {
        acc += xrow[i] * wrow[i];
    }
#endif

    y[n * O + ox] = acc;
}

__global__ void add_inplace_kernel(float* __restrict__ acc, const float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) acc[i] += y[i];
}
__global__ void scale_inplace_kernel(float* __restrict__ a, float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] *= s;
}
__global__ void argmax_kernel(const float* __restrict__ a, int* __restrict__ out, int N, int C) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const float* row = a + n * C;
    int best = 0; float bv = row[0];
    for (int i = 1; i < C; ++i) if (row[i] > bv) { bv = row[i]; best = i; }
    out[n] = best;
}

/// conv2 + IF2 + pool2 融合核 —— 内环使用 PTX（ld.const/ld.global + fma）
__global__ void conv2_if_pool2_fused_const(
        const float* __restrict__ x_p1, // [N, C2_IN, 12, 12]
        float* __restrict__ v2,         // [N, C2_OUT, 8, 8]
        float* __restrict__ y_p2,       // [N, C2_OUT, 4, 4]
        int N, int H_in, int W_in,      // 12, 12
        float v_th                      // 1.0
){
    const int Hout = H_in - K2 + 1;  // 8
    const int Wout = W_in - K2 + 1;  // 8
    const int Hp   = Hout / 2;       // 4
    const int Wp   = Wout / 2;       // 4

    int oxp = blockIdx.x * blockDim.x + threadIdx.x; // 0..Wp-1
    int oyp = blockIdx.y * blockDim.y + threadIdx.y; // 0..Hp-1
    int z   = blockIdx.z;                             // 0..N*C2_OUT-1
    if (oxp >= Wp || oyp >= Hp || z >= N * C2_OUT) return;

    int n  = z / C2_OUT;
    int oc = z % C2_OUT;

    int py0 = oyp * 2;
    int px0 = oxp * 2;

    float s_max = 0.0f;

#pragma unroll
    for (int dy = 0; dy < 2; ++dy) {
#pragma unroll
        for (int dx = 0; dx < 2; ++dx) {
            int py = py0 + dy;
            int px = px0 + dx;

            float acc;
#if USE_PTX
            // 从常量内存读偏置
            acc = ptx_ld_const_f32(c_conv2_b + oc);
#else
            acc = c_conv2_b[oc];
#endif

            // 卷积核 5x5 × C2_IN 累加
#pragma unroll
            for (int ic = 0; ic < C2_IN; ++ic) {
#pragma unroll
                for (int ky = 0; ky < K2; ++ky) {
                    int iy = py + ky;
#pragma unroll
                    for (int kx = 0; kx < K2; ++kx) {
                        int ix = px + kx;
                        int widx = (((oc * C2_IN) + ic) * K2 + ky) * K2 + kx;

#if USE_PTX
                        // PTX 方式加载 x / w 并 FMA
                        float xv = ptx_ld_global_f32(x_p1 + idx4(n, ic, iy, ix, C2_IN, H_in, W_in));
                        float wv = ptx_ld_const_f32(c_conv2_w + widx);
                        ptx_fma(acc, xv, wv);
#else
                        float xv = x_p1[idx4(n, ic, iy, ix, C2_IN, H_in, W_in)];
                        float wv = c_conv2_w[widx];
                        acc += xv * wv;
#endif
                    }
                }
            }

            // IF2 on 8x8 grid
            int vidx = idx4(n, oc, py, px, C2_OUT, Hout, Wout);
            float vv  = v2[vidx] + acc;
            float s   = (vv >= v_th) ? 1.0f : 0.0f;
#if IF_SUBTRACT_RESET
            vv = vv - s * v_th;
#else
            vv = (s > 0.0f) ? 0.0f : vv;
#endif
            v2[vidx] = vv;

            s_max = fmaxf(s_max, s);
        }
    }

    // 写 pool2（4x4）
    y_p2[idx4(n, oc, oyp, oxp, C2_OUT, Hp, Wp)] = s_max;
}

// 融合 IF1 + pool1：从 conv1(x) 的 24x24 直接得到 12x12 的脉冲输出，期间更新 IF1 的 24x24 膜电位
__global__ void if1_pool1_fused_kernel(
        const float* __restrict__ x_c1, // [N, C1_OUT, 24, 24]  conv1(x) 的输出（T 外预计算）
        float* __restrict__ v1,         // [N, C1_OUT, 24, 24]  IF1 的膜电位（会被更新）
        float* __restrict__ y_p1,       // [N, C1_OUT, 12, 12]  pool1 的脉冲输出（0/1）
        int N,                          // batch
        int H_in, int W_in,             // 24, 24
        float v_th                      // 阈值 1.0
){
    const int Hp = H_in / 2; // 12
    const int Wp = W_in / 2; // 12

    int oxp = blockIdx.x * blockDim.x + threadIdx.x; // 0..Wp-1
    int oyp = blockIdx.y * blockDim.y + threadIdx.y; // 0..Hp-1
    int z   = blockIdx.z;                             // 0..N*C1_OUT-1
    if (oxp >= Wp || oyp >= Hp || z >= N * C1_OUT) return;

    int n  = z / C1_OUT;
    int oc = z % C1_OUT;

    int py0 = oyp * 2;
    int px0 = oxp * 2;

    float s_max = 0.0f;

#pragma unroll
    for (int dy = 0; dy < 2; ++dy) {
#pragma unroll
        for (int dx = 0; dx < 2; ++dx) {
            int py = py0 + dy;
            int px = px0 + dx;

            int id = idx4(n, oc, py, px, C1_OUT, H_in, W_in);
            float vv = v1[id] + x_c1[id];
            float s  = (vv >= v_th) ? 1.0f : 0.0f;
#if IF_SUBTRACT_RESET
            vv = vv - s * v_th;
#else
            vv = (s > 0.0f) ? 0.0f : vv;
#endif
            v1[id] = vv;
            s_max = fmaxf(s_max, s);
        }
    }

    y_p1[idx4(n, oc, oyp, oxp, C1_OUT, Hp, Wp)] = s_max;
}

// 向量版 IF：输入/膜电位/输出形状都是 [N, F]，用二维网格避免 grid.z 溢出
__global__ void ifnode_vec_kernel(
        const float* __restrict__ in,  // [N, F]
        float* __restrict__ v,         // [N, F]
        float* __restrict__ out,       // [N, F]
        int N, int F, float v_th
){
    int f = blockIdx.x * blockDim.x + threadIdx.x; // 特征维
    int n = blockIdx.y * blockDim.y + threadIdx.y; // batch 维
    if (f >= F || n >= N) return;
    int id = n * F + f;

    float vv = v[id] + in[id];
    float s  = (vv >= v_th) ? 1.0f : 0.0f;
#if IF_SUBTRACT_RESET
    vv = vv - s * v_th;      // 软重置
#else
    vv = (s > 0.0f) ? 0.0f : vv;  // 硬重置
#endif
    v[id]   = vv;
    out[id] = s;
}
// conv2 + IF2 + pool2 融合核（shared-memory tiling + PTX FMA）
// 每个 block 处理一个 (n, oc) 的 4x4 池化网格；对每个输入通道，先把 12x12 平面搬进 shared memory
__global__ void conv2_if_pool2_fused_const_smem(
        const float* __restrict__ x_p1, // [N, C2_IN, 12, 12]
        float* __restrict__ v2,         // [N, C2_OUT, 8, 8]
        float* __restrict__ y_p2,       // [N, C2_OUT, 4, 4]
        int N, int H_in, int W_in,      // 12, 12
        float v_th                      // 1.0
){
    extern __shared__ float s_x[];  // 大小 = H_in*W_in floats（每次 ic 覆盖）
    const int Hout = H_in - K2 + 1; // 8
    const int Wout = W_in - K2 + 1; // 8
    const int Hp   = Hout / 2;      // 4
    const int Wp   = Wout / 2;      // 4

    int oxp = blockIdx.x * blockDim.x + threadIdx.x; // 0..Wp-1
    int oyp = blockIdx.y * blockDim.y + threadIdx.y; // 0..Hp-1
    int z   = blockIdx.z;                             // 0..N*C2_OUT-1
    if (oxp >= Wp || oyp >= Hp || z >= N * C2_OUT) return;

    int n  = z / C2_OUT;
    int oc = z % C2_OUT;

    // 该池化位置对应 2x2 conv 输出左上角
    int py0 = oyp * 2;
    int px0 = oxp * 2;

    // 四个卷积位置的累加寄存器
    float acc00, acc01, acc10, acc11;
#if USE_PTX
    acc00 = ptx_ld_const_f32(c_conv2_b + oc);
    acc01 = acc00;
    acc10 = acc00;
    acc11 = acc00;
#else
    acc00 = c_conv2_b[oc];
    acc01 = acc00;
    acc10 = acc00;
    acc11 = acc00;
#endif

    const int tid   = threadIdx.y * blockDim.x + threadIdx.x;     // 0..15（建议 block 4x4）
    const int tcount= blockDim.x * blockDim.y;                     // 16
    const int plane = H_in * W_in;                                 // 144

    // 对每个输入通道：把 12x12 平面搬到 shared，然后做 4 个位置的 5x5 卷积
    for (int ic = 0; ic < C2_IN; ++ic) {

        // coop load: 每线程搬运 ceil(plane/tcount) 个元素
        for (int idx = tid; idx < plane; idx += tcount) {
            int iy = idx / W_in;
            int ix = idx % W_in;
#if USE_PTX
            s_x[idx] = ptx_ld_global_f32(x_p1 + idx4(n, ic, iy, ix, C2_IN, H_in, W_in));
#else
            s_x[idx] = x_p1[idx4(n, ic, iy, ix, C2_IN, H_in, W_in)];
#endif
        }
        __syncthreads();

        // 5x5 卷积贡献（四个相邻位置：00, 01, 10, 11）
#pragma unroll
        for (int ky = 0; ky < K2; ++ky) {
#pragma unroll
            for (int kx = 0; kx < K2; ++kx) {
                // 权重
                int widx = (((oc * C2_IN) + ic) * K2 + ky) * K2 + kx;
#if USE_PTX
                float wv = ptx_ld_const_f32(c_conv2_w + widx);
#else
                float wv = c_conv2_w[widx];
#endif
                // 输入（来自 shared memory）
                int p00 = (py0 + ky) * W_in + (px0 + kx);
                int p01 = p00 + 1;
                int p10 = p00 + W_in;
                int p11 = p10 + 1;

                float x00 = s_x[p00];
                float x01 = s_x[p01];
                float x10 = s_x[p10];
                float x11 = s_x[p11];

#if USE_PTX
                ptx_fma(acc00, x00, wv);
                ptx_fma(acc01, x01, wv);
                ptx_fma(acc10, x10, wv);
                ptx_fma(acc11, x11, wv);
#else
                acc00 += x00 * wv;
                acc01 += x01 * wv;
                acc10 += x10 * wv;
                acc11 += x11 * wv;
#endif
            }
        }
        __syncthreads(); // 下一通道前同步
    }

    // IF2 + 2x2 池化
    float s_max = 0.0f;

    int vidx00 = idx4(n, oc, py0 + 0, px0 + 0, C2_OUT, Hout, Wout);
    float vv00 = v2[vidx00] + acc00;
    float s00  = (vv00 >= v_th) ? 1.0f : 0.0f;
#if IF_SUBTRACT_RESET
    vv00 = vv00 - s00 * v_th;
#else
    vv00 = (s00 > 0.0f) ? 0.0f : vv00;
#endif
    v2[vidx00] = vv00;
    s_max = fmaxf(s_max, s00);

    int vidx01 = idx4(n, oc, py0 + 0, px0 + 1, C2_OUT, Hout, Wout);
    float vv01 = v2[vidx01] + acc01;
    float s01  = (vv01 >= v_th) ? 1.0f : 0.0f;
#if IF_SUBTRACT_RESET
    vv01 = vv01 - s01 * v_th;
#else
    vv01 = (s01 > 0.0f) ? 0.0f : vv01;
#endif
    v2[vidx01] = vv01;
    s_max = fmaxf(s_max, s01);

    int vidx10 = idx4(n, oc, py0 + 1, px0 + 0, C2_OUT, Hout, Wout);
    float vv10 = v2[vidx10] + acc10;
    float s10  = (vv10 >= v_th) ? 1.0f : 0.0f;
#if IF_SUBTRACT_RESET
    vv10 = vv10 - s10 * v_th;
#else
    vv10 = (s10 > 0.0f) ? 0.0f : vv10;
#endif
    v2[vidx10] = vv10;
    s_max = fmaxf(s_max, s10);

    int vidx11 = idx4(n, oc, py0 + 1, px0 + 1, C2_OUT, Hout, Wout);
    float vv11 = v2[vidx11] + acc11;
    float s11  = (vv11 >= v_th) ? 1.0f : 0.0f;
#if IF_SUBTRACT_RESET
    vv11 = vv11 - s11 * v_th;
#else
    vv11 = (s11 > 0.0f) ? 0.0f : vv11;
#endif
    v2[vidx11] = vv11;
    s_max = fmaxf(s_max, s11);

    y_p2[idx4(n, oc, oyp, oxp, C2_OUT, Hp, Wp)] = s_max;
}
// ===================================================================================
// Helper for CUDA Error Handling - DO NOT MODIFY BEGIN
// ===================================================================================
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
}
// ===================================================================================
// Helper for CUDA Error Handling - DO NOT MODIFY END
// ===================================================================================

// ===================================================================================
// Data and Parameter Loading Functions - DO NOT MODIFY BEGIN
// ===================================================================================
std::vector<std::vector<float>> read_mnist_images(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { std::cerr << "Cannot open file: " << path << std::endl; return {}; }
    int magic_number = 0, num_images = 0, num_rows = 0, num_cols = 0;
    file.read((char*)&magic_number, 4); magic_number = __builtin_bswap32(magic_number);
    file.read((char*)&num_images, 4); num_images = __builtin_bswap32(num_images);
    file.read((char*)&num_rows, 4); num_rows = __builtin_bswap32(num_rows);
    file.read((char*)&num_cols, 4); num_cols = __builtin_bswap32(num_cols);
    std::vector<std::vector<float>> images(num_images, std::vector<float>(num_rows * num_cols));
    std::vector<unsigned char> buffer(num_rows * num_cols);
    for (int i = 0; i < num_images; ++i) {
        file.read((char*)buffer.data(), buffer.size());
        for (size_t j = 0; j < buffer.size(); ++j) {
            images[i][j] = (static_cast<float>(buffer[j]) / 255.0f - 0.5f) / 0.5f; // Normalization
        }
    }
    return images;
}

std::vector<int> read_mnist_labels(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { std::cerr << "Cannot open file: " << path << std::endl; return {}; }
    int magic_number = 0, num_items = 0;
    file.read((char*)&magic_number, 4); magic_number = __builtin_bswap32(magic_number);
    file.read((char*)&num_items, 4); num_items = __builtin_bswap32(num_items);
    std::vector<int> labels(num_items);
    std::vector<unsigned char> buffer(num_items);
    file.read((char*)buffer.data(), num_items);
    for(int i = 0; i < num_items; ++i) { labels[i] = static_cast<int>(buffer[i]); }
    return labels;
}

std::vector<float> read_param(const std::string& path) {
    std::ifstream file(path);
    if (!file) { std::cerr << "Cannot open parameter file: " << path << std::endl; return {}; }
    std::vector<float> params; float param;
    while (file >> param) { params.push_back(param); }
    return params;
}

// ===================================================================================
// Data and Parameter Loading Functions - DO NOT MODIFY END
// ===================================================================================


//std::vector<int> scnn_inference(
//        const std::vector<std::vector<float>>& images,
//        // Device pointers for parameters
//        float* d_conv1_w, float* d_conv1_b, float* d_conv2_w, float* d_conv2_b,
//        float* d_fc1_w,   float* d_fc1_b,   float* d_fc2_w,   float* d_fc2_b,
//        float* d_fc3_w,   float* d_fc3_b
//        // YOU CAN ADD MORE PARAMETERS HERE!!!
//)
//{
//    std::vector<int> predictions;
//    const int num_images = images.size();
//    predictions.reserve(num_images);
//
//    // SNN-specific parameter, must match training
//    const int T = 8;
//
//    // --- Loop over each image ---
//    for (int i = 0; i < num_images; ++i) {
//
//        predictions.push_back(1);
//    }
//
//    // Memory is freed in main.
//
//    return predictions;
//}
//
std::vector<int> scnn_inference(
        const std::vector<std::vector<float>>& images,
        float* d_conv1_w, float* d_conv1_b, float* d_conv2_w, float* d_conv2_b,
        float* d_fc1_w,   float* d_fc1_b,   float* d_fc2_w,   float* d_fc2_b,
        float* d_fc3_w,   float* d_fc3_b
){
    const int T = 8;
    const int N_all = static_cast<int>(images.size());
    const int IMG_C = 1, IMG_H = 28, IMG_W = 28;

    // 尺寸
    const int C1_H=IMG_H-K1+1, C1_W=IMG_W-K1+1; // 24x24
    const int P1_H=C1_H/2,     P1_W=C1_W/2;     // 12x12
    const int C2_H=P1_H-K2+1,  C2_W=P1_W-K2+1;  // 8x8
    const int P2_H=C2_H/2,     P2_W=C2_W/2;     // 4x4
    const int FLAT=16*4*4;
    const int FC1_IN=FLAT, FC1_OUT=120;
    const int FC2_IN=FC1_OUT, FC2_OUT=84;
    const int FC3_IN=FC2_OUT, FC3_OUT=10;

    // 批大小：你可设为 2048/4096/8192……
    const int BATCH = 8192;

    std::vector<int> predictions;
    predictions.reserve(N_all);

    // 常量内存填充（设备到设备）
    checkCudaErrors(cudaMemcpyToSymbol(c_conv1_w, d_conv1_w, sizeof(float)*C1_OUT*C1_IN*K1*K1, 0, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(c_conv1_b, d_conv1_b, sizeof(float)*C1_OUT,              0, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(c_conv2_w, d_conv2_w, sizeof(float)*C2_OUT*C2_IN*K2*K2, 0, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(c_conv2_b, d_conv2_b, sizeof(float)*C2_OUT,              0, cudaMemcpyDeviceToDevice));

    // 设备缓冲与尺寸
    float *d_in[2] = {nullptr, nullptr};
    float *d_c1_static=nullptr;
    float *d_if1_v=nullptr, *d_if1_s=nullptr, *d_p1=nullptr;
    float *d_if2_v=nullptr, *d_p2=nullptr;
    float *d_flat=nullptr;
    float *d_fc1=nullptr, *d_if3_v=nullptr, *d_if3_s=nullptr;
    float *d_fc2=nullptr, *d_if4_v=nullptr, *d_if4_s=nullptr;
    float *d_fc3=nullptr, *d_acc=nullptr;
    int   *d_pred=nullptr;

    size_t sz_in   = (size_t)BATCH*IMG_C*IMG_H*IMG_W*sizeof(float);
    size_t sz_c1   = (size_t)BATCH*C1_OUT*C1_H*C1_W*sizeof(float);
    size_t sz_if1  = sz_c1;
    size_t sz_p1   = (size_t)BATCH*C1_OUT*P1_H*P1_W*sizeof(float);
    size_t sz_if2v = (size_t)BATCH*C2_OUT*C2_H*C2_W*sizeof(float);
    size_t sz_p2   = (size_t)BATCH*C2_OUT*P2_H*P2_W*sizeof(float);
    size_t sz_flat = (size_t)BATCH*FLAT*sizeof(float);
    size_t sz_fc1  = (size_t)BATCH*FC1_OUT*sizeof(float);
    size_t sz_if3  = sz_fc1;
    size_t sz_fc2  = (size_t)BATCH*FC2_OUT*sizeof(float);
    size_t sz_if4  = sz_fc2;
    size_t sz_fc3  = (size_t)BATCH*FC3_OUT*sizeof(float);

    checkCudaErrors(cudaMalloc(&d_in[0], sz_in));
    checkCudaErrors(cudaMalloc(&d_in[1], sz_in));
    checkCudaErrors(cudaMalloc(&d_c1_static, sz_c1));
    checkCudaErrors(cudaMalloc(&d_if1_v, sz_if1));
    checkCudaErrors(cudaMalloc(&d_if1_s, sz_if1));
    checkCudaErrors(cudaMalloc(&d_p1,     sz_p1));
    checkCudaErrors(cudaMalloc(&d_if2_v,  sz_if2v));
    checkCudaErrors(cudaMalloc(&d_p2,     sz_p2));
    checkCudaErrors(cudaMalloc(&d_flat,   sz_flat));
    checkCudaErrors(cudaMalloc(&d_fc1,    sz_fc1));
    checkCudaErrors(cudaMalloc(&d_if3_v,  sz_if3));
    checkCudaErrors(cudaMalloc(&d_if3_s,  sz_if3));
    checkCudaErrors(cudaMalloc(&d_fc2,    sz_fc2));
    checkCudaErrors(cudaMalloc(&d_if4_v,  sz_if4));
    checkCudaErrors(cudaMalloc(&d_if4_s,  sz_if4));
    checkCudaErrors(cudaMalloc(&d_fc3,    sz_fc3));
    checkCudaErrors(cudaMalloc(&d_acc,    sz_fc3));
    checkCudaErrors(cudaMalloc(&d_pred,   BATCH*sizeof(int)));

    // pinned host 双缓冲 + 两条流 + 事件同步
    float* h_in_pinned[2] = {nullptr, nullptr};
    checkCudaErrors(cudaHostAlloc(&h_in_pinned[0], sz_in, cudaHostAllocDefault));
    checkCudaErrors(cudaHostAlloc(&h_in_pinned[1], sz_in, cudaHostAllocDefault));
    std::vector<int> h_pred_vec(BATCH);
    cudaStream_t stream_copy, stream_comp;
    checkCudaErrors(cudaStreamCreate(&stream_copy));
    checkCudaErrors(cudaStreamCreate(&stream_comp));
    cudaEvent_t h2d_done[2];
    checkCudaErrors(cudaEventCreateWithFlags(&h2d_done[0], cudaEventDisableTiming));
    checkCudaErrors(cudaEventCreateWithFlags(&h2d_done[1], cudaEventDisableTiming));

    dim3 blk2d(16, 16, 1);
    dim3 blkF(std::min(8, P2_W), std::min(8, P2_H), 1); // (4,4,1) for fused conv2
    const int total_batches = (N_all + BATCH - 1) / BATCH;

    // 预取第0批
    {
        int curB0 = std::min(BATCH, N_all);
        for (int n = 0; n < curB0; ++n) {
            const auto& v = images[n];
            std::copy(v.begin(), v.end(), h_in_pinned[0] + n * IMG_H * IMG_W);
        }
        checkCudaErrors(cudaMemcpyAsync(d_in[0], h_in_pinned[0],
                                        (size_t)curB0*IMG_H*IMG_W*sizeof(float),
                                        cudaMemcpyHostToDevice, stream_copy));
        checkCudaErrors(cudaEventRecord(h2d_done[0], stream_copy));
        checkCudaErrors(cudaStreamWaitEvent(stream_comp, h2d_done[0], 0));
    }

    // z 维最大块数
    const int Z_MAX = 65535;
    auto sliceB = [&](int curB, int Cout)->int {
        return std::max(1, std::min(curB, Z_MAX / Cout));
    };

    for (int bi = 0; bi < total_batches; ++bi) {
        int cur = bi & 1, next = cur ^ 1;
        int start_idx = bi * BATCH;
        int curB = std::min(BATCH, N_all - start_idx);

        // 等待本批 H2D 完成
        checkCudaErrors(cudaStreamWaitEvent(stream_comp, h2d_done[cur], 0));

        // conv1 预计算（按 z 分片：每片 z = thisB*C1_OUT <= 65535）
        {
            int step = sliceB(curB, C1_OUT);
            for (int b0 = 0; b0 < curB; b0 += step) {
                int thisB = std::min(step, curB - b0);
                const float* x_ptr = d_in[cur] + (size_t)b0 * IMG_C * IMG_H * IMG_W;
                float* y_ptr = d_c1_static + (size_t)b0 * C1_OUT * C1_H * C1_W;
                dim3 grd(div_up(C1_W, blk2d.x), div_up(C1_H, blk2d.y), thisB * C1_OUT);
                conv2d_conv1_const_kernel<<<grd, blk2d, 0, stream_comp>>>(x_ptr, y_ptr, thisB, IMG_H, IMG_W);
                checkCudaErrors(cudaGetLastError());
            }
        }

        // 清零膜电位与 logits 累加器
        checkCudaErrors(cudaMemsetAsync(d_if1_v, 0, (size_t)curB*C1_OUT*C1_H*C1_W*sizeof(float), stream_comp));
        checkCudaErrors(cudaMemsetAsync(d_if2_v, 0, (size_t)curB*C2_OUT*C2_H*C2_W*sizeof(float), stream_comp));
        checkCudaErrors(cudaMemsetAsync(d_if3_v, 0, (size_t)curB*FC1_OUT*sizeof(float), stream_comp));
        checkCudaErrors(cudaMemsetAsync(d_if4_v, 0, (size_t)curB*FC2_OUT*sizeof(float), stream_comp));
        checkCudaErrors(cudaMemsetAsync(d_acc,   0, (size_t)curB*FC3_OUT*sizeof(float), stream_comp));

        for (int t = 0; t < T; ++t) {
            // IF1 + pool1 融合（按 z 分片；block 建议 (8,8,1)，grid (ceil(12/8), ceil(12/8), thisB*6)）
            {
                dim3 blkP1(8, 8, 1);
                int step = sliceB(curB, C1_OUT);
                for (int b0 = 0; b0 < curB; b0 += step) {
                    int thisB = std::min(step, curB - b0);
                    const float* in_ptr = d_c1_static + (size_t)b0 * C1_OUT * C1_H * C1_W;
                    float*       v_ptr  = d_if1_v     + (size_t)b0 * C1_OUT * C1_H * C1_W;
                    float*       out_ptr= d_p1        + (size_t)b0 * C1_OUT * P1_H * P1_W;

                    dim3 grdP1(div_up(P1_W, blkP1.x), div_up(P1_H, blkP1.y), thisB * C1_OUT);
                    if1_pool1_fused_kernel<<<grdP1, blkP1, 0, stream_comp>>>(
                            in_ptr, v_ptr, out_ptr, thisB, C1_H, C1_W, 1.0f
                    );
                    checkCudaErrors(cudaGetLastError());
                }
            }
            // conv2 + IF2 + pool2（按 z 分片；注意用 blkF 和 shared memory 大小）
            {
                int step = sliceB(curB, C2_OUT);
                for (int b0 = 0; b0 < curB; b0 += step) {
                    int thisB = std::min(step, curB - b0);
                    const float* x_ptr = d_p1    + (size_t)b0 * C1_OUT * P1_H * P1_W;
                    float*       v2_ptr= d_if2_v + (size_t)b0 * C2_OUT * C2_H * C2_W;
                    float*       y_ptr = d_p2    + (size_t)b0 * C2_OUT * P2_H * P2_W;

                    dim3 grdF(div_up(P2_W, blkF.x), div_up(P2_H, blkF.y), thisB * C2_OUT);
                    size_t smem_bytes = (size_t)P1_H * P1_W * sizeof(float); // 12*12*4=576
                    conv2_if_pool2_fused_const_smem<<<grdF, blkF, smem_bytes, stream_comp>>>(
                            x_ptr, v2_ptr, y_ptr, thisB, P1_H, P1_W, 1.0f
                    );
                    checkCudaErrors(cudaGetLastError());
                }
            }
            // flatten（不需要分片，按 N 维展开）
            {
                int tot = curB * FLAT;
                flatten_kernel<<<div_up(tot, 256), 256, 0, stream_comp>>>(d_p2, d_flat, curB, 16, 4, 4);
                checkCudaErrors(cudaGetLastError());
            }
            // fc1 + IF3
            {
                dim3 blk(32, 8);
                dim3 grd(div_up(FC1_OUT, blk.x), div_up(curB, blk.y));
                linear_kernel<<<grd, blk, 0, stream_comp>>>(d_flat, d_fc1_w, d_fc1_b, d_fc1, curB, FC1_IN, FC1_OUT);
                checkCudaErrors(cudaGetLastError());
                // FC1 后的 IF（二维网格）
                {
                    dim3 blk_if(128, 4);
                    dim3 grd_if(div_up(FC1_OUT, blk_if.x), div_up(curB, blk_if.y));
                    ifnode_vec_kernel<<<grd_if, blk_if, 0, stream_comp>>>(d_fc1, d_if3_v, d_if3_s, curB, FC1_OUT, 1.0f);
                    checkCudaErrors(cudaGetLastError());
                }
                checkCudaErrors(cudaGetLastError());
            }
            // fc2 + IF4
            {
                dim3 blk(32, 8);
                dim3 grd(div_up(FC2_OUT, blk.x), div_up(curB, blk.y));
                linear_kernel<<<grd, blk, 0, stream_comp>>>(d_if3_s, d_fc2_w, d_fc2_b, d_fc2, curB, FC2_IN, FC2_OUT);
                checkCudaErrors(cudaGetLastError());
                // FC2 后的 IF（二维网格）
                {
                    dim3 blk_if(128, 4);
                    dim3 grd_if(div_up(FC2_OUT, blk_if.x), div_up(curB, blk_if.y));
                    ifnode_vec_kernel<<<grd_if, blk_if, 0, stream_comp>>>(d_fc2, d_if4_v, d_if4_s, curB, FC2_OUT, 1.0f);
                    checkCudaErrors(cudaGetLastError());
                }
                checkCudaErrors(cudaGetLastError());
            }
            // fc3 累加
            {
                dim3 blk(32, 8);
                dim3 grd(div_up(FC3_OUT, blk.x), div_up(curB, blk.y));
                linear_kernel<<<grd, blk, 0, stream_comp>>>(d_if4_s, d_fc3_w, d_fc3_b, d_fc3, curB, FC3_IN, FC3_OUT);
                checkCudaErrors(cudaGetLastError());
                int tot = curB * FC3_OUT;
                add_inplace_kernel<<<div_up(tot, 256), 256, 0, stream_comp>>>(d_acc, d_fc3, tot);
                checkCudaErrors(cudaGetLastError());
            }
        } // end T

        // 平均 + argmax
        {
            int tot = curB * FC3_OUT;
            scale_inplace_kernel<<<div_up(tot, 256), 256, 0, stream_comp>>>(d_acc, 1.0f / T, tot);
            checkCudaErrors(cudaGetLastError());
            argmax_kernel<<<div_up(curB, 256), 256, 0, stream_comp>>>(d_acc, d_pred, curB, FC3_OUT);
            checkCudaErrors(cudaGetLastError());
        }

        // 异步预取下一批
        if (bi + 1 < total_batches) {
            int nextStart = (bi + 1) * BATCH;
            int nextB = std::min(BATCH, N_all - nextStart);
            for (int n = 0; n < nextB; ++n) {
                const auto& v = images[nextStart + n];
                std::copy(v.begin(), v.end(), h_in_pinned[next] + n * IMG_H * IMG_W);
            }
            checkCudaErrors(cudaMemcpyAsync(d_in[next], h_in_pinned[next],
                                            (size_t)nextB*IMG_H*IMG_W*sizeof(float),
                                            cudaMemcpyHostToDevice, stream_copy));
            checkCudaErrors(cudaEventRecord(h2d_done[next], stream_copy));
        }

        // 收集预测
        checkCudaErrors(cudaStreamSynchronize(stream_comp));
        checkCudaErrors(cudaMemcpy(h_pred_vec.data(), d_pred, (size_t)curB*sizeof(int), cudaMemcpyDeviceToHost));
        predictions.insert(predictions.end(), h_pred_vec.begin(), h_pred_vec.begin() + curB);
    }

    // 清理
    checkCudaErrors(cudaEventDestroy(h2d_done[0]));
    checkCudaErrors(cudaEventDestroy(h2d_done[1]));
    checkCudaErrors(cudaStreamDestroy(stream_copy));
    checkCudaErrors(cudaStreamDestroy(stream_comp));
    checkCudaErrors(cudaFreeHost(h_in_pinned[0]));
    checkCudaErrors(cudaFreeHost(h_in_pinned[1]));

    checkCudaErrors(cudaFree(d_in[0]));
    checkCudaErrors(cudaFree(d_in[1]));
    checkCudaErrors(cudaFree(d_c1_static));
    checkCudaErrors(cudaFree(d_if1_v));
    checkCudaErrors(cudaFree(d_if1_s));
    checkCudaErrors(cudaFree(d_p1));
    checkCudaErrors(cudaFree(d_if2_v));
    checkCudaErrors(cudaFree(d_p2));
    checkCudaErrors(cudaFree(d_flat));
    checkCudaErrors(cudaFree(d_fc1));
    checkCudaErrors(cudaFree(d_if3_v));
    checkCudaErrors(cudaFree(d_if3_s));
    checkCudaErrors(cudaFree(d_fc2));
    checkCudaErrors(cudaFree(d_if4_v));
    checkCudaErrors(cudaFree(d_if4_s));
    checkCudaErrors(cudaFree(d_fc3));
    checkCudaErrors(cudaFree(d_acc));
    checkCudaErrors(cudaFree(d_pred));

    return predictions;
}


// ===================================================================================
// Main Function -  DO NOT MODIFY BEGIN
// ===================================================================================
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_model_and_data_dir>" << std::endl;
        return 1;
    }
    std::string dir = argv[1];

    // Load test data
    auto images = read_mnist_images(dir + "/../../.." + "/data/FashionMNIST/raw/t10k-images-idx3-ubyte");
    auto labels = read_mnist_labels(dir + "/../../.." + "/data/FashionMNIST/raw/t10k-labels-idx1-ubyte");
    if (images.empty() || labels.empty()) return 1;

    // Load model parameters to host memory
    auto conv1_w = read_param(dir + "/conv1.weight.txt");
    auto conv1_b = read_param(dir + "/conv1.bias.txt");
    auto conv2_w = read_param(dir + "/conv2.weight.txt");
    auto conv2_b = read_param(dir + "/conv2.bias.txt");
    auto fc1_w = read_param(dir + "/fc1.weight.txt");
    auto fc1_b = read_param(dir + "/fc1.bias.txt");
    auto fc2_w = read_param(dir + "/fc2.weight.txt");
    auto fc2_b = read_param(dir + "/fc2.bias.txt");
    auto fc3_w = read_param(dir + "/fc3.weight.txt");
    auto fc3_b = read_param(dir + "/fc3.bias.txt");

    // --- 1. Allocate all necessary GPU memory ---
    // Device pointers for parameters
    float *d_conv1_w, *d_conv1_b, *d_conv2_w, *d_conv2_b;
    float *d_fc1_w, *d_fc1_b, *d_fc2_w, *d_fc2_b, *d_fc3_w, *d_fc3_b;

    // Allocate parameters
    checkCudaErrors(cudaMalloc(&d_conv1_w, conv1_w.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_conv1_b, conv1_b.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_conv2_w, conv2_w.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_conv2_b, conv2_b.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc1_w,   fc1_w.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc1_b,   fc1_b.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc2_w,   fc2_w.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc2_b,   fc2_b.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc3_w,   fc3_w.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc3_b,   fc3_b.size() * sizeof(float)));

    // --- 2. Copy constant parameters from host to device ---
    checkCudaErrors(cudaMemcpy(d_conv1_w, conv1_w.data(), conv1_w.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_conv1_b, conv1_b.data(), conv1_b.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_conv2_w, conv2_w.data(), conv2_w.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_conv2_b, conv2_b.data(), conv2_b.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc1_w, fc1_w.data(), fc1_w.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc1_b, fc1_b.data(), fc1_b.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc2_w, fc2_w.data(), fc2_w.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc2_b, fc2_b.data(), fc2_b.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc3_w, fc3_w.data(), fc3_w.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc3_b, fc3_b.data(), fc3_b.size() * sizeof(float), cudaMemcpyHostToDevice));

    // Start timer
    auto start = std::chrono::high_resolution_clock::now();

// ===================================================================================
// Main Function -  DO NOT MODIFY END
// ===================================================================================

    // --- 3. Perform inference ---
    // Pass device pointers to the inference function
    std::vector<int> predictions = scnn_inference(images,
                                                  d_conv1_w, d_conv1_b, d_conv2_w, d_conv2_b,
                                                  d_fc1_w, d_fc1_b, d_fc2_w, d_fc2_b, d_fc3_w, d_fc3_b
            // YOU CAN ADD MORE PARAMETERS HERE!!!
    );

// ===================================================================================
// Main Function -  DO NOT MODIFY BEGIN
// ===================================================================================

    // Synchronize to ensure all GPU work is done before stopping the timer
    checkCudaErrors(cudaDeviceSynchronize());

    // Stop timer
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    // --- 4. Free all allocated GPU memory ---
    checkCudaErrors(cudaFree(d_conv1_w));
    checkCudaErrors(cudaFree(d_conv1_b));
    checkCudaErrors(cudaFree(d_conv2_w));
    checkCudaErrors(cudaFree(d_conv2_b));
    checkCudaErrors(cudaFree(d_fc1_w));
    checkCudaErrors(cudaFree(d_fc1_b));
    checkCudaErrors(cudaFree(d_fc2_w));
    checkCudaErrors(cudaFree(d_fc2_b));
    checkCudaErrors(cudaFree(d_fc3_w));
    checkCudaErrors(cudaFree(d_fc3_b));

    // Calculate accuracy
    int correct_predictions = 0;
    for (size_t i = 0; i < labels.size(); ++i) {
        if (predictions[i] == labels[i]) {
            correct_predictions++;
        }
    }
    double accuracy = static_cast<double>(correct_predictions) / labels.size();

    // Output result in the required format
    std::cout << std::fixed << std::setprecision(4) << diff.count() << ":" << accuracy << std::endl;

    return 0;
}
// ===================================================================================
// Main Function -  DO NOT MODIFY END
// ===================================================================================