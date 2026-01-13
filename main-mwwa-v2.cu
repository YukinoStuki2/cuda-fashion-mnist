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
#include <cuda_fp16.h>

// ========== 可选项 ==========
#define IF_SUBTRACT_RESET 0
#ifndef USE_PTX
#define USE_PTX 1
#endif
#ifndef USE_WMMA
#define USE_WMMA 1   // 运行时仍会判断是否在 >=sm_70 上
#endif

// LeNet 结构（固定）
#define C1_IN   1
#define C1_OUT  6
#define K1      5
#define C2_IN   6
#define C2_OUT  16
#define K2      5

// 全连接维度
#define FC1_IN 256  // 16*4*4
#define FC1_OUT 120
#define FC2_IN 120
#define FC2_OUT 84
#define FC3_IN 84
#define FC3_OUT 10

// WMMA tiles（V100 Tensor Core 使用 16x16x16 tile）
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// FC1 WMMA 维度（K/N 都需 16 对齐；Npad 向上到 128）
#define FC1_K        FC1_IN        // 256
#define FC1_N        FC1_OUT       // 120
#define FC1_NPAD     128

// FC2 WMMA 维度（K=120 pad 到 128；N=84 pad 到 96）
#define FC2_K        FC2_IN        // 120
#define FC2_KPAD     128
#define FC2_N        FC2_OUT       // 84
#define FC2_NPAD     96

// conv2 每 block 计算的输出通道瓦片
#ifndef OC_TILE
#define OC_TILE 4
#endif

#if USE_WMMA
// 统一 WMMA 使用的半精度类型
using half = __half;
#include <mma.h>
#endif

static_assert((C2_OUT % OC_TILE) == 0, "C2_OUT must be divisible by OC_TILE.");

inline int div_up(int a, int b) { return (a + b - 1) / b; }
__host__ __device__ inline int idx4(int n,int c,int h,int w,int C,int H,int W){ return ((n*C+c)*H+h)*W+w; }
__host__ __device__ inline int idx2(int o,int i,int I){ return o*I+i; }

// ---------------- PTX helpers ----------------
static __device__ __forceinline__ float ptx_ld_const_f32(const float* p) {
    float v;
    asm volatile("{.reg .u64 a; cvta.to.const.u64 a, %1; ld.const.f32 %0, [a];}"
            : "=f"(v) : "l"(p));
    return v;
}
static __device__ __forceinline__ float ptx_ld_global_f32(const float* p) {
    float v;
    asm volatile("{.reg .u64 a; cvta.to.global.u64 a, %1; ld.global.ca.f32 %0, [a];}"
            : "=f"(v) : "l"(p));
    return v;
}
static __device__ __forceinline__ float4 ptx_ld_global_f32x4(const float* p) {
    float4 v;
    asm volatile("{.reg .u64 a; cvta.to.global.u64 a, %4; ld.global.ca.v4.f32 {%0,%1,%2,%3}, [a];}"
            : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p));
    return v;
}
static __device__ __forceinline__ void ptx_fma(float& acc, float xv, float wv) {
    asm volatile("fma.rn.f32 %0, %1, %2, %0;" : "+f"(acc) : "f"(xv), "f"(wv));
}
static __device__ __forceinline__ float ptx_max4(float a, float b, float c, float d) {
    float max_ab, max_cd, result;
    asm("max.f32 %0, %1, %2;" : "=f"(max_ab) : "f"(a), "f"(b));
    asm("max.f32 %0, %1, %2;" : "=f"(max_cd) : "f"(c), "f"(d));
    asm("max.f32 %0, %1, %2;" : "=f"(result) : "f"(max_ab), "f"(max_cd));
    return result;
}

// 常量内存（conv 权重/偏置 + fc1/fc2/fc3 偏置）
__constant__ float c_conv1_w[C1_OUT * C1_IN * K1 * K1];
__constant__ float c_conv1_b[C1_OUT];
__constant__ float c_conv2_w[C2_OUT * C2_IN * K2 * K2];
__constant__ float c_conv2_b[C2_OUT];
__constant__ float c_fc1_b[FC1_OUT];
__constant__ float c_fc2_b[FC2_OUT];
__constant__ float c_fc3_b[FC3_OUT];

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

// ---------------- CUDA KERNELS ----------------

// conv1（stride=1，无 padding）
__launch_bounds__(256, 2)
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

// IF1 + pool1 融合：输入/输出 half，膜电位 v1 用 float
__launch_bounds__(256, 2)
__global__ void if1_pool1_fused_kernel(
        const __half* __restrict__ x_c1_h, // [N, C1_OUT, 24, 24] (half)
        float* __restrict__ v1,           // [N, C1_OUT, 24, 24] (float)
        __half* __restrict__ y_p1_h,      // [N, C1_OUT, 12, 12] (half)
        int N,
        int H_in, int W_in,               // 24, 24
        float v_th
){
    const int Hp = H_in / 2; // 12
    const int Wp = W_in / 2; // 12

    int oxp = blockIdx.x * blockDim.x + threadIdx.x;
    int oyp = blockIdx.y * blockDim.y + threadIdx.y;
    int z   = blockIdx.z; // 0..N*C1_OUT-1
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
            float xv = __half2float(x_c1_h[id]); // half -> float
            float vv = v1[id] + xv;
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

    y_p1_h[idx4(n, oc, oyp, oxp, C1_OUT, Hp, Wp)] = __float2half(s_max);
}

// conv2 + IF2 + pool2 融合（half2 搬运 x 到 shared），一次算 OC_TILE 个输出通道，并直接写扁平化输出（half）
__global__ void conv2_if_pool2_fused_const_smem_flat_oc2(
        const __half* __restrict__ x_p1_h, // [N, C2_IN, 12,12] half
        float* __restrict__ v2,            // [N, C2_OUT, 8,8] float
        __half* __restrict__ y_flat,       // [N, 16*4*4] half
        int N, int H_in, int W_in, float v_th
){
    extern __shared__ float s_x[];             // 12*12 floats
    const int Hout = H_in - K2 + 1;            // 8
    const int Wout = W_in - K2 + 1;            // 8
    const int Hp   = Hout / 2;                 // 4
    const int Wp   = Wout / 2;                 // 4

    int oxp = blockIdx.x * blockDim.x + threadIdx.x; // 0..Wp-1
    int oyp = blockIdx.y * blockDim.y + threadIdx.y; // 0..Hp-1
    int z   = blockIdx.z;                             // 0..N*(C2_OUT/OC_TILE)-1
    if (oxp >= Wp || oyp >= Hp || z >= N * (C2_OUT/OC_TILE)) return;

    int n        = z / (C2_OUT/OC_TILE);
    int oc_base  = (z % (C2_OUT/OC_TILE)) * OC_TILE;

    int py0 = oyp * 2;
    int px0 = oxp * 2;

    float acc00[OC_TILE], acc01[OC_TILE], acc10[OC_TILE], acc11[OC_TILE];
#pragma unroll
    for (int t = 0; t < OC_TILE; ++t) {
#if USE_PTX
        float b = ptx_ld_const_f32(c_conv2_b + (oc_base + t));
#else
        float b = c_conv2_b[oc_base + t];
#endif
        acc00[t] = b; acc01[t] = b; acc10[t] = b; acc11[t] = b;
    }

    const int tid    = threadIdx.y * blockDim.x + threadIdx.x;
    const int tcount = blockDim.x * blockDim.y;
    const int plane  = H_in * W_in;     // 144
    const int plane2 = plane >> 1;      // 72

    for (int ic = 0; ic < C2_IN; ++ic) {
        int base = ((n * C2_IN + ic) * H_in * W_in);
        const __half*  gptr  = x_p1_h + base;
        const __half2* gptr2 = reinterpret_cast<const __half2*>(gptr);
        for (int idx2 = tid; idx2 < plane2; idx2 += tcount) {
            __half2 h2 = gptr2[idx2];
            float2 f2 = __half22float2(h2);
            int i0 = idx2 << 1;
            s_x[i0 + 0] = f2.x;
            s_x[i0 + 1] = f2.y;
        }
        if ((plane & 1) && tid == 0) s_x[plane - 1] = __half2float(gptr[plane - 1]);
        __syncthreads();

#pragma unroll
        for (int ky = 0; ky < K2; ++ky) {
#pragma unroll
            for (int kx = 0; kx < K2; ++kx) {
                int p00 = (py0 + ky) * W_in + (px0 + kx);
                int p01 = p00 + 1;
                int p10 = p00 + W_in;
                int p11 = p10 + 1;

                float x00 = s_x[p00];
                float x01 = s_x[p01];
                float x10 = s_x[p10];
                float x11 = s_x[p11];

#pragma unroll
                for (int t = 0; t < OC_TILE; ++t) {
                    int oc = oc_base + t;
                    int widx = (((oc * C2_IN) + ic) * K2 + ky) * K2 + kx;
#if USE_PTX
                    float wv = ptx_ld_const_f32(c_conv2_w + widx);
#else
                    float wv = c_conv2_w[widx];
#endif
#if USE_PTX
                    ptx_fma(acc00[t], x00, wv);
                    ptx_fma(acc01[t], x01, wv);
                    ptx_fma(acc10[t], x10, wv);
                    ptx_fma(acc11[t], x11, wv);
#else
                    acc00[t] += x00 * wv;
                    acc01[t] += x01 * wv;
                    acc10[t] += x10 * wv;
                    acc11[t] += x11 * wv;
#endif
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int t = 0; t < OC_TILE; ++t) {
        int oc = oc_base + t;
        float s_max = 0.0f;

        int vidx00 = idx4(n, oc, py0 + 0, px0 + 0, C2_OUT, Hout, Wout);
        float vv00 = v2[vidx00] + acc00[t];
        float s00  = (vv00 >= v_th) ? 1.0f : 0.0f;
#if IF_SUBTRACT_RESET
        vv00 -= s00 * v_th;
#else
        vv00 = (s00 > 0.0f) ? 0.0f : vv00;
#endif
        v2[vidx00] = vv00; s_max = fmaxf(s_max, s00);

        int vidx01 = idx4(n, oc, py0 + 0, px0 + 1, C2_OUT, Hout, Wout);
        float vv01 = v2[vidx01] + acc01[t];
        float s01  = (vv01 >= v_th) ? 1.0f : 0.0f;
#if IF_SUBTRACT_RESET
        vv01 -= s01 * v_th;
#else
        vv01 = (s01 > 0.0f) ? 0.0f : vv01;
#endif
        v2[vidx01] = vv01; s_max = fmaxf(s_max, s01);

        int vidx10 = idx4(n, oc, py0 + 1, px0 + 0, C2_OUT, Hout, Wout);
        float vv10 = v2[vidx10] + acc10[t];
        float s10  = (vv10 >= v_th) ? 1.0f : 0.0f;
#if IF_SUBTRACT_RESET
        vv10 -= s10 * v_th;
#else
        vv10 = (s10 > 0.0f) ? 0.0f : vv10;
#endif
        v2[vidx10] = vv10; s_max = fmaxf(s_max, s10);

        int vidx11 = idx4(n, oc, py0 + 1, px0 + 1, C2_OUT, Hout, Wout);
        float vv11 = v2[vidx11] + acc11[t];
        float s11  = (vv11 >= v_th) ? 1.0f : 0.0f;
#if IF_SUBTRACT_RESET
        vv11 -= s11 * v_th;
#else
        vv11 = (s11 > 0.0f) ? 0.0f : vv11;
#endif
        v2[vidx11] = vv11; s_max = fmaxf(s_max, s11);

        int Hp   = Hout / 2;
        int Wp   = Wout / 2;
        int flat_idx = n * (C2_OUT * Hp * Wp) + oc * (Hp * Wp) + oyp * Wp + oxp;
        y_flat[flat_idx] = __float2half(s_max); // half 输出
    }
}

// flatten（保留备用）
__launch_bounds__(256, 2)
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

// 线性 + IF（float 权重；备用）
__launch_bounds__(256, 2)
__global__ void linear_if_fused_kernel(
        const float* __restrict__ x,    // [N, I]
        const float* __restrict__ W,    // [O, I]
        const float* __restrict__ b,    // [O]
        float* __restrict__ v,          // [N, O]
        float* __restrict__ out,        // [N, O]
        int N, int I, int O, float v_th
) {
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= O || n >= N) return;

    float acc = b[ox];
    const float* xrow = x + n * I;
    const float* wrow = W + ox * I;

#if USE_PTX
    int i = 0;
    for (; i + 4 <= I; i += 4) {
        float4 xv4 = ptx_ld_global_f32x4(xrow + i);
        float4 wv4 = ptx_ld_global_f32x4(wrow + i);
        ptx_fma(acc, xv4.x, wv4.x);
        ptx_fma(acc, xv4.y, wv4.y);
        ptx_fma(acc, xv4.z, wv4.z);
        ptx_fma(acc, xv4.w, wv4.w);
    }
    for (; i < I; ++i) {
        float xv = ptx_ld_global_f32(xrow + i);
        float wv = ptx_ld_global_f32(wrow + i);
        ptx_fma(acc, xv, wv);
    }
#else
    for (int i = 0; i < I; ++i) acc += xrow[i] * wrow[i];
#endif

    int id = n * O + ox;
    float vv = v[id] + acc;
    float s = vv >= v_th ? 1.0f : 0.0f;
#if IF_SUBTRACT_RESET
    vv = vv - s * v_th;
#else
    vv = (s > 0.0f) ? 0.0f : vv;
#endif
    v[id] = vv;
    out[id] = s;
}

// FC3 累加（float 权重，bias 来自常量内存）— 融合 scale
__launch_bounds__(256, 2)
__global__ void fc3_accum_kernel_floatW(
        const float* __restrict__ spikes,  // [N, FC3_IN]
        const float* __restrict__ weights, // [FC3_OUT, FC3_IN]
        float* __restrict__ output,        // [N, FC3_OUT]
        int N, float scale
){
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int n  = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= FC3_OUT || n >= N) return;

#if USE_PTX
    float acc = ptx_ld_const_f32(c_fc3_b + ox);
#else
    float acc = c_fc3_b[ox];
#endif
    const float* xrow = spikes + n * FC3_IN;
    const float* wrow = weights + ox * FC3_IN;

#if USE_PTX
    int i = 0;
    for (; i + 4 <= FC3_IN; i += 4) {
        float4 xv4 = ptx_ld_global_f32x4(xrow + i);
        float4 wv4 = ptx_ld_global_f32x4(wrow + i);
        ptx_fma(acc, xv4.x, wv4.x);
        ptx_fma(acc, xv4.y, wv4.y);
        ptx_fma(acc, xv4.z, wv4.z);
        ptx_fma(acc, xv4.w, wv4.w);
    }
    for (; i < FC3_IN; ++i) {
        float xv = ptx_ld_global_f32(xrow + i);
        float wv = ptx_ld_global_f32(wrow + i);
        ptx_fma(acc, xv, wv);
    }
#else
    for (int i = 0; i < FC3_IN; ++i) acc = fmaf(xrow[i], wrow[i], acc);
#endif

    output[n * FC3_OUT + ox] += acc * scale;
}

__launch_bounds__(256, 2)
__global__ void scale_inplace_kernel(float* __restrict__ a, float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] *= s;
}
__launch_bounds__(256, 2)
__global__ void argmax_kernel(const float* __restrict__ a, int* __restrict__ out, int N, int C) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const float* row = a + n * C;
    int best = 0; float bv = row[0];
    for (int i = 1; i < C; ++i) if (row[i] > bv) { bv = row[i]; best = i; }
    out[n] = best;
}

// ======= FC 半精度权重的 fallback（FC1/FC2） =======

// float -> half 一维转换
__launch_bounds__(256, 2)
__global__ void float_to_half_kernel(const float* __restrict__ in, __half* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(in[i]);
}

// half -> float 一维转换（fallback 时用于 d_flat_h -> d_flat）
__launch_bounds__(256, 2)
__global__ void half_to_float_kernel(const __half* __restrict__ in, float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __half2float(in[i]);
}

// 每个 block 固定一个输出通道 ox；block 内多个样本并行；使用 shared 缓存该输出通道的权重（半精度转 float）
template<int I_MAX>
__launch_bounds__(256, 2)
__global__ void linear_if_fp16W_cached_kernel(
        const float* __restrict__ x,    // [N, I]
        const __half* __restrict__ W_h, // [O, I] (half)
        const float* __restrict__ b,    // [O]
        float* __restrict__ v,          // [N, O]
        float* __restrict__ out,        // [N, O]
        int N, int I, int O, float v_th // I <= I_MAX
) {
    extern __shared__ float shW[]; // I floats
    const int ox = blockIdx.x;
    int n  = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= O || n >= N) return;

    const __half* wrow_h = W_h + ox * I;
    for (int i = threadIdx.y; i < I; i += blockDim.y) {
        shW[i] = __half2float(wrow_h[i]);
    }
    __syncthreads();

    const float* xrow = x + n * I;
    float acc = b[ox];

    int i = 0;
#pragma unroll
    for (; i + 4 <= I; i += 4) {
        float4 xv4 = reinterpret_cast<const float4*>(xrow)[i >> 2];
        acc = fmaf(xv4.x, shW[i+0], acc);
        acc = fmaf(xv4.y, shW[i+1], acc);
        acc = fmaf(xv4.z, shW[i+2], acc);
        acc = fmaf(xv4.w, shW[i+3], acc);
    }
    for (; i < I; ++i) acc = fmaf(xrow[i], shW[i], acc);

    int id = n * O + ox;
    float vv = v[id] + acc;
    float s  = (vv >= v_th) ? 1.0f : 0.0f;
#if IF_SUBTRACT_RESET
    vv -= s * v_th;
#else
    vv = (s > 0.0f) ? 0.0f : vv;
#endif
    v[id]   = vv;
    out[id] = s;
}

// 去掉 b 指针，bias 从常量内存取（FC1/FC2 专用）
template<int I_MAX>
__launch_bounds__(256, 2)
__global__ void linear_if_fp16W_cached_kernel_fc1(
        const float* __restrict__ x,    // [N, I]
        const __half* __restrict__ W_h, // [O, I] (half)
        float* __restrict__ v,          // [N, O]
        float* __restrict__ out,        // [N, O]
        int N, int I, int O, float v_th // I <= I_MAX
) {
    extern __shared__ float shW[]; // I floats
    const int ox = blockIdx.x;
    int n  = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= O || n >= N) return;

    const __half* wrow_h = W_h + ox * I;
    for (int i = threadIdx.y; i < I; i += blockDim.y) {
        shW[i] = __half2float(wrow_h[i]);
    }
    __syncthreads();

    const float* xrow = x + n * I;
    float acc = c_fc1_b[ox];

    int i = 0;
#pragma unroll
    for (; i + 4 <= I; i += 4) {
        float4 xv4 = reinterpret_cast<const float4*>(xrow)[i >> 2];
        acc = fmaf(xv4.x, shW[i+0], acc);
        acc = fmaf(xv4.y, shW[i+1], acc);
        acc = fmaf(xv4.z, shW[i+2], acc);
        acc = fmaf(xv4.w, shW[i+3], acc);
    }
    for (; i < I; ++i) acc = fmaf(xrow[i], shW[i], acc);

    int id = n * O + ox;
    float vv = v[id] + acc;
    float s  = (vv >= v_th) ? 1.0f : 0.0f;
#if IF_SUBTRACT_RESET
    vv -= s * v_th;
#else
    vv = (s > 0.0f) ? 0.0f : vv;
#endif
    v[id]   = vv;
    out[id] = s;
}

template<int I_MAX>
__launch_bounds__(256, 2)
__global__ void linear_if_fp16W_cached_kernel_fc2(
        const float* __restrict__ x,    // [N, I]
        const __half* __restrict__ W_h, // [O, I] (half)
        float* __restrict__ v,          // [N, O]
        float* __restrict__ out,        // [N, O]
        int N, int I, int O, float v_th // I <= I_MAX
) {
    extern __shared__ float shW[]; // I floats
    const int ox = blockIdx.x;
    int n  = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= O || n >= N) return;

    const __half* wrow_h = W_h + ox * I;
    for (int i = threadIdx.y; i < I; i += blockDim.y) {
        shW[i] = __half2float(wrow_h[i]);
    }
    __syncthreads();

    const float* xrow = x + n * I;
    float acc = c_fc2_b[ox];

    int i = 0;
#pragma unroll
    for (; i + 4 <= I; i += 4) {
        float4 xv4 = reinterpret_cast<const float4*>(xrow)[i >> 2];
        acc = fmaf(xv4.x, shW[i+0], acc);
        acc = fmaf(xv4.y, shW[i+1], acc);
        acc = fmaf(xv4.z, shW[i+2], acc);
        acc = fmaf(xv4.w, shW[i+3], acc);
    }
    for (; i < I; ++i) acc = fmaf(xrow[i], shW[i], acc);

    int id = n * O + ox;
    float vv = v[id] + acc;
    float s  = (vv >= v_th) ? 1.0f : 0.0f;
#if IF_SUBTRACT_RESET
    vv -= s * v_th;
#else
    vv = (s > 0.0f) ? 0.0f : vv;
#endif
    v[id]   = vv;
    out[id] = s;
}

// 权重转置+pad 到列主（col_major）存储的 W^T：
// FC1: in W_h[O=120, I=256] -> out Wt[I=256, Npad=128]，列主 ld=I
__launch_bounds__(256, 2)
__global__ void fc1_transpose_pad_colmajor_v2(
        const __half* __restrict__ W_h, // [O, I] row-major
        __half* __restrict__ Wt_col,    // [I, Npad] col-major (ld = I)
        int O, int I, int Npad
){
    int i = blockIdx.y * blockDim.y + threadIdx.y; // row in I
    int n = blockIdx.x * blockDim.x + threadIdx.x; // col in Npad
    if (i >= I || n >= Npad) return;
    __half v = __float2half(0.f);
    if (n < O) {
        v = W_h[n * I + i]; // 取 W[o=n, i]
    }
    Wt_col[n * I + i] = v;  // 列主：地址 = n*ld + i，ld = I
}

// FC2: in W_h[O=84, I=120] -> out Wt[Kpad=128, Npad=96]，列主 ld=Kpad
__launch_bounds__(256, 2)
__global__ void fc2_transpose_pad_colmajor(
        const __half* __restrict__ W_h, // [O, I] row-major
        __half* __restrict__ Wt_col,    // [Kpad, Npad] col-major (ld = Kpad)
        int O, int I, int Kpad, int Npad
){
    int k = blockIdx.y * blockDim.y + threadIdx.y; // 0..Kpad-1
    int n = blockIdx.x * blockDim.x + threadIdx.x; // 0..Npad-1
    if (k >= Kpad || n >= Npad) return;
    __half v = __float2half(0.f);
    if (n < O && k < I) {
        // 原权重 W[o, i]，转置后位置 (k=i, n=o)
        v = W_h[n * I + k];
    }
    Wt_col[n * Kpad + k] = v;
}

// FC1 WMMA 内核：sm_70+ 用 WMMA；否则用纯 FMA 回退，且在写回时同时产出 N×128 的 half（供 FC2 直接使用）
__global__ void fc1_wmma_if_kernel(
        const half* __restrict__ x_h,        // [N, 256] row-major, ldA = 256
        const half* __restrict__ wT_h,       // [256, 128] col-major, ldB = 256
        float* __restrict__ v,               // [N, 120]
        float* __restrict__ s_out,           // [N, 120]
        half*  __restrict__ s_out_pad_h,     // [N, 128] half（pad 区域 0）
        int N
){
    const int M = N, K = FC1_K, Npad = FC1_NPAD;
    const int tile_m = blockIdx.y;
    const int tile_n = blockIdx.x;
    const int m0 = tile_m * WMMA_M;
    const int n0 = tile_n * WMMA_N;
    if (m0 >= M || n0 >= Npad) return;

#if __CUDA_ARCH__ >= 700
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
    nvcuda::wmma::fill_fragment(acc, 0.0f);
    for (int k0 = 0; k0 < K; k0 += WMMA_K) {
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> a;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major> b;
        const half* Aptr = x_h  + m0 * K + k0;  // ldA = K
        const half* Bptr = wT_h + n0 * K + k0;  // ldB = K
        nvcuda::wmma::load_matrix_sync(a, Aptr, K);
        nvcuda::wmma::load_matrix_sync(b, Bptr, K);
        nvcuda::wmma::mma_sync(acc, a, b, acc);
    }
    __shared__ float Csh[WMMA_M * WMMA_N]; // 256 floats
    nvcuda::wmma::store_matrix_sync(Csh, acc, WMMA_N, nvcuda::wmma::mem_row_major);
    __syncthreads();
#else
    __shared__ float Csh[WMMA_M * WMMA_N];
    const int lane = threadIdx.x & 31;
    for (int i = lane; i < WMMA_M * WMMA_N; i += 32) {
        int r = i / WMMA_N, c = i % WMMA_N;
        int m = m0 + r, n = n0 + c;
        float acc = 0.0f;
        if (m < M && n < FC1_NPAD) {
            int aoff = m * K, boff = n * K;
            for (int k = 0; k < K; ++k) {
                acc = fmaf(__half2float(x_h[aoff + k]),
                           __half2float(wT_h[boff + k]), acc);
            }
            Csh[i] = acc;
        }
    }
    __syncthreads();
#endif

    // 写回：float spikes + half(128) padded
    const int lane2 = threadIdx.x & 31;
    for (int i = lane2; i < WMMA_M * WMMA_N; i += 32) {
        int r = i / WMMA_N, c = i % WMMA_N;
        int m = m0 + r, n = n0 + c;
        if (m < M) {
            if (n < FC1_OUT) {
                float val = Csh[i] + c_fc1_b[n];
                int id = m * FC1_OUT + n;
                float vv = v[id] + val;
                float sp = (vv >= 1.0f) ? 1.0f : 0.0f;
#if IF_SUBTRACT_RESET
                vv -= sp * 1.0f;
#else
                vv = (sp > 0.0f) ? 0.0f : vv;
#endif
                v[id]                 = vv;
                s_out[id]             = sp;
                s_out_pad_h[m*FC1_NPAD + n] = __float2half(sp);
            } else if (n < FC1_NPAD) {
                s_out_pad_h[m*FC1_NPAD + n] = __float2half(0.0f);
            }
        }
    }
}

// FC2 WMMA 内核：sm_70+ 用 WMMA；否则用纯 FMA 回退
__global__ void fc2_wmma_if_kernel(
        const half* __restrict__ x_h_pad, // [N, 128] row-major, ldA = 128
        const half* __restrict__ wT_h,    // [128, 96]  col-major, ldB = 128
        float* __restrict__ v,            // [N, 84]
        float* __restrict__ s_out,        // [N, 84]
        int N
){
    const int M = N, K = FC2_KPAD, Npad = FC2_NPAD;
    const int tile_m = blockIdx.y;
    const int tile_n = blockIdx.x;
    const int m0 = tile_m * WMMA_M;
    const int n0 = tile_n * WMMA_N;
    if (m0 >= M || n0 >= Npad) return;

#if __CUDA_ARCH__ >= 700
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
    nvcuda::wmma::fill_fragment(acc, 0.0f);
    for (int k0 = 0; k0 < K; k0 += WMMA_K) {
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> a;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major> b;
        const half* Aptr = x_h_pad + m0 * K + k0; // ldA = K
        const half* Bptr = wT_h    + n0 * K + k0; // ldB = K
        nvcuda::wmma::load_matrix_sync(a, Aptr, K);
        nvcuda::wmma::load_matrix_sync(b, Bptr, K);
        nvcuda::wmma::mma_sync(acc, a, b, acc);
    }
    __shared__ float Csh[WMMA_M * WMMA_N];
    nvcuda::wmma::store_matrix_sync(Csh, acc, WMMA_N, nvcuda::wmma::mem_row_major);
    __syncthreads();

    const int lane = threadIdx.x & 31;
    for (int i = lane; i < WMMA_M * WMMA_N; i += 32) {
        const int r = i / WMMA_N;
        const int c = i % WMMA_N;
        const int m = m0 + r;
        const int n = n0 + c;
        if (m < M && n < FC2_OUT) {
            float val = Csh[i] + c_fc2_b[n];
            const int id = m * FC2_OUT + n;
            float vv = v[id] + val;
            const float sp = (vv >= 1.0f) ? 1.0f : 0.0f;
#if IF_SUBTRACT_RESET
            vv -= sp * 1.0f;
#else
            vv = (sp > 0.0f) ? 0.0f : vv;
#endif
            v[id]     = vv;
            s_out[id] = sp;
        }
    }
#else
    // 纯 FMA 回退
    const int lane = threadIdx.x & 31;
    for (int i = lane; i < WMMA_M * WMMA_N; i += 32) {
        const int r = i / WMMA_N;
        const int c = i % WMMA_N;
        const int m = m0 + r;
        const int n = n0 + c;
        if (m < M && n < FC2_OUT) {
            float acc = 0.0f;
            const int a_row_off = m * K;
            const int b_col_off = n * K; // 列主：列 n 的起始偏移
#pragma unroll 1
            for (int k = 0; k < K; ++k) {
                float a = __half2float(x_h_pad[a_row_off + k]);
                float b = __half2float(wT_h   [b_col_off + k]);
                acc = fmaf(a, b, acc);
            }
            acc += c_fc2_b[n];
            const int id = m * FC2_OUT + n;
            float vv = v[id] + acc;
            const float sp = (vv >= 1.0f) ? 1.0f : 0.0f;
#if IF_SUBTRACT_RESET
            vv -= sp * 1.0f;
#else
            vv = (sp > 0.0f) ? 0.0f : vv;
#endif
            v[id]     = vv;
            s_out[id] = sp;
        }
    }
#endif
}

// —— FC1 WMMA 1x2 CTA：64 线程（2 warp）/块，两个相邻 N 瓦片共享 A ——
// 说明：共享内存 Ash 装载 A 瓦片（16x16），两个 warp 分别计算 n0 与 n0+16
//       C 瓦片分别落到 Csh[0]/Csh[1]，再加 bias + IF 写回
__global__ void fc1_wmma_if_kernel_1x2(
        const half* __restrict__ x_h,        // [N, 256] row-major, ldA=256
        const half* __restrict__ wT_h,       // [256, 128] col-major, ldB=256
        float* __restrict__ v,               // [N, 120]
        float* __restrict__ s_out,           // [N, 120]
        half*  __restrict__ s_out_pad_h,     // [N, 128]（可传 nullptr，表示不输出 half 版本）
        int N
){
    const int M = N, K = FC1_K, Npad = FC1_NPAD;
    const int warpId = threadIdx.x >> 5;     // 0 or 1
    const int lane   = threadIdx.x & 31;

    const int tile_m = blockIdx.y;           // 沿 M（batch）
    const int tile_n_pair = blockIdx.x;      // 每个 block 覆盖两个 N 瓦片
    const int m0 = tile_m * WMMA_M;
    const int n0_base = tile_n_pair * (WMMA_N * 2);
    const int n0 = n0_base + warpId * WMMA_N;  // 当前 warp 的 N 起点

    if (m0 >= M || n0 >= Npad) return;

    // 共享 A（16x16），以及每个 warp 各自的 C 缓冲
    __shared__ half  Ash[WMMA_M * WMMA_K];     // 16*16
    __shared__ float Csh[2][WMMA_M * WMMA_N];  // 每个 warp 一个 16*16

#if __CUDA_ARCH__ >= 700
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
    nvcuda::wmma::fill_fragment(acc, 0.0f);

    // k 循环
    for (int k0 = 0; k0 < K; k0 += WMMA_K) {
        // 由64线程协同把 A(m0..m0+15, k0..k0+15) 拷到 Ash（row-major）
        for (int t = threadIdx.x; t < WMMA_M * WMMA_K; t += blockDim.x) {
            int r = t / WMMA_K, c = t % WMMA_K;
            Ash[t] = x_h[(m0 + r) * K + (k0 + c)];
        }
        __syncthreads();

        // 两个 warp 分别加载 A from shared、B from global，做一次 mma
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> a;
        nvcuda::wmma::load_matrix_sync(a, Ash, WMMA_K);

        const half* Bptr = wT_h + n0 * K + k0; // 列主，ldB=K
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major> b;
        nvcuda::wmma::load_matrix_sync(b, Bptr, K);

        nvcuda::wmma::mma_sync(acc, a, b, acc);

        __syncthreads(); // 保护 Ash 重用
    }

    // 把每个 warp 的 C 瓦片落到各自 Csh[warpId]
    nvcuda::wmma::store_matrix_sync(Csh[warpId], acc, WMMA_N, nvcuda::wmma::mem_row_major);
    __syncthreads();

    // 写回：加 bias + IF；可选输出 half(128) 供 FC2 直接使用
    for (int i = lane; i < WMMA_M * WMMA_N; i += 32) {
        int r = i / WMMA_N, c = i % WMMA_N;
        int m = m0 + r, n = n0 + c;
        if (m < M && n < FC1_NPAD) {
            float val = Csh[warpId][i];
            if (n < FC1_OUT) {
                val += c_fc1_b[n];
                int id = m * FC1_OUT + n;
                float vv = v[id] + val;
                float sp = (vv >= 1.0f) ? 1.0f : 0.0f;
#if IF_SUBTRACT_RESET
                vv -= sp * 1.0f;
#else
                vv = (sp > 0.0f) ? 0.0f : vv;
#endif
                v[id]     = vv;
                s_out[id] = sp;
                if (s_out_pad_h) s_out_pad_h[m * FC1_NPAD + n] = __float2half(sp);
            } else {
                if (s_out_pad_h) s_out_pad_h[m * FC1_NPAD + n] = __float2half(0.0f);
            }
        }
    }
#else
    // <sm_70 纯 FMA 回退（两个 warp 各算一块）
    for (int i = lane; i < WMMA_M * WMMA_N; i += 32) {
        int r = i / WMMA_N, c = i % WMMA_N;
        int m = m0 + r, n = n0 + c;
        if (m < M && n < FC1_NPAD) {
            float acc = 0.f;
            const int aoff = m * K;
            const int boff = n * K;
#pragma unroll 1
            for (int k = 0; k < K; ++k) {
                float a = __half2float(x_h[aoff + k]);
                float b = __half2float(wT_h[boff + k]);
                acc = fmaf(a, b, acc);
            }
            if (n < FC1_OUT) {
                acc += c_fc1_b[n];
                int id = m * FC1_OUT + n;
                float vv = v[id] + acc;
                float sp = (vv >= 1.0f) ? 1.0f : 0.0f;
#if IF_SUBTRACT_RESET
                vv -= sp * 1.0f;
#else
                vv = (sp > 0.0f) ? 0.0f : vv;
#endif
                v[id]     = vv;
                s_out[id] = sp;
                if (s_out_pad_h) s_out_pad_h[m * FC1_NPAD + n] = __float2half(sp);
            } else {
                if (s_out_pad_h) s_out_pad_h[m * FC1_NPAD + n] = __float2half(0.0f);
            }
        }
    }
#endif
}

// —— FC2 WMMA 1x2 CTA（与 FC1 同理） ——
__global__ void fc2_wmma_if_kernel_1x2(
        const half* __restrict__ x_h_pad, // [N, 128] row-major, ldA=128
        const half* __restrict__ wT_h,    // [128, 96]  col-major, ldB=128
        float* __restrict__ v,            // [N, 84]
        float* __restrict__ s_out,        // [N, 84]
        int N
){
    const int M = N, K = FC2_KPAD, Npad = FC2_NPAD;
    const int warpId = threadIdx.x >> 5;
    const int lane   = threadIdx.x & 31;

    const int tile_m = blockIdx.y;
    const int tile_n_pair = blockIdx.x;
    const int m0 = tile_m * WMMA_M;
    const int n0_base = tile_n_pair * (WMMA_N * 2);
    const int n0 = n0_base + warpId * WMMA_N;

    if (m0 >= M || n0 >= Npad) return;

    __shared__ half  Ash[WMMA_M * WMMA_K];
    __shared__ float Csh[2][WMMA_M * WMMA_N];

#if __CUDA_ARCH__ >= 700
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
    nvcuda::wmma::fill_fragment(acc, 0.0f);

    for (int k0 = 0; k0 < K; k0 += WMMA_K) {
        // stage A(m0.., k0..) -> Ash
        for (int t = threadIdx.x; t < WMMA_M * WMMA_K; t += blockDim.x) {
            int r = t / WMMA_K, c = t % WMMA_K;
            Ash[t] = x_h_pad[(m0 + r) * K + (k0 + c)];
        }
        __syncthreads();

        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> a;
        nvcuda::wmma::load_matrix_sync(a, Ash, WMMA_K);

        const half* Bptr = wT_h + n0 * K + k0;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major> b;
        nvcuda::wmma::load_matrix_sync(b, Bptr, K);

        nvcuda::wmma::mma_sync(acc, a, b, acc);

        __syncthreads();
    }

    nvcuda::wmma::store_matrix_sync(Csh[warpId], acc, WMMA_N, nvcuda::wmma::mem_row_major);
    __syncthreads();

    for (int i = lane; i < WMMA_M * WMMA_N; i += 32) {
        int r = i / WMMA_N, c = i % WMMA_N;
        int m = m0 + r, n = n0 + c;
        if (m < M && n < FC2_OUT) {
            float val = Csh[warpId][i] + c_fc2_b[n];
            int id = m * FC2_OUT + n;
            float vv = v[id] + val;
            float sp = (vv >= 1.0f) ? 1.0f : 0.0f;
#if IF_SUBTRACT_RESET
            vv -= sp * 1.0f;
#else
            vv = (sp > 0.0f) ? 0.0f : vv;
#endif
            v[id]     = vv;
            s_out[id] = sp;
        }
    }
#else
    for (int i = lane; i < WMMA_M * WMMA_N; i += 32) {
        int r = i / WMMA_N, c = i % WMMA_N;
        int m = m0 + r, n = n0 + c;
        if (m < M && n < FC2_OUT) {
            float acc = 0.0f;
            const int aoff = m * K;
            const int boff = n * K;
#pragma unroll 1
            for (int k = 0; k < K; ++k) {
                float a = __half2float(x_h_pad[aoff + k]);
                float b = __half2float(wT_h   [boff + k]);
                acc = fmaf(a, b, acc);
            }
            acc += c_fc2_b[n];
            int id = m * FC2_OUT + n;
            float vv = v[id] + acc;
            float sp = (vv >= 1.0f) ? 1.0f : 0.0f;
#if IF_SUBTRACT_RESET
            vv -= sp * 1.0f;
#else
            vv = (sp > 0.0f) ? 0.0f : vv;
#endif
            v[id]     = vv;
            s_out[id] = sp;
        }
    }
#endif
}
// ================== 推理主流程 ==================
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

    const int BATCH = 4096;

    std::vector<int> predictions;
    predictions.reserve(N_all);

    // 常量内存拷贝（device->device）
    checkCudaErrors(cudaMemcpyToSymbol(c_conv1_w, d_conv1_w, sizeof(float)*C1_OUT*C1_IN*K1*K1, 0, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(c_conv1_b, d_conv1_b, sizeof(float)*C1_OUT,              0, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(c_conv2_w, d_conv2_w, sizeof(float)*C2_OUT*C2_IN*K2*K2, 0, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(c_conv2_b, d_conv2_b, sizeof(float)*C2_OUT,              0, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(c_fc3_b,   d_fc3_b,   sizeof(float)*FC3_OUT,             0, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(c_fc1_b,   d_fc1_b,   sizeof(float)*FC1_OUT,             0, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(c_fc2_b,   d_fc2_b,   sizeof(float)*FC2_OUT,             0, cudaMemcpyDeviceToDevice));

    // 设备缓冲
    float *d_in[2] = {nullptr, nullptr};
    float *d_c1_static=nullptr;
    __half *d_c1_static_h=nullptr;
    float *d_if1_v=nullptr; __half *d_p1_h=nullptr;
    float *d_if2_v=nullptr;
    float *d_flat=nullptr;   // 回退路径 FC1 输入（float）
    float *d_if3_v=nullptr, *d_if3_s=nullptr;
    float *d_if4_v=nullptr, *d_if4_s=nullptr;
    float *d_acc=nullptr;
    int   *d_pred=nullptr;

    // WMMA 相关
    __half *d_flat_h=nullptr;              // [BATCH, 256] conv2 直接写的 half flatten
    __half *d_if3_s_pad_h=nullptr;         // [BATCH, 128] FC1 输出 half（pad）
    __half *d_fc1_w_h=nullptr, *d_fc2_w_h=nullptr; // 原始半精度权重 [O,I]
    __half *d_fc1_wT_h=nullptr;            // [256, 128] col-major（ld=256）
    __half *d_fc2_wT_h=nullptr;            // [128, 96]  col-major（ld=128）

    size_t sz_in   = (size_t)BATCH*IMG_C*IMG_H*IMG_W*sizeof(float);
    size_t sz_c1   = (size_t)BATCH*C1_OUT*C1_H*C1_W*sizeof(float);
    size_t sz_c1_h = (size_t)BATCH*C1_OUT*C1_H*C1_W*sizeof(__half);
    size_t sz_if1v = sz_c1;
    size_t sz_p1_h = (size_t)BATCH*C1_OUT*P1_H*P1_W*sizeof(__half);
    size_t sz_if2v = (size_t)BATCH*C2_OUT*C2_H*C2_W*sizeof(float);
    size_t sz_flat = (size_t)BATCH*FLAT*sizeof(float);
    size_t sz_if3  = (size_t)BATCH*FC1_OUT*sizeof(float);
    size_t sz_if4  = (size_t)BATCH*FC2_OUT*sizeof(float);
    size_t sz_fc3  = (size_t)BATCH*FC3_OUT*sizeof(float);

    checkCudaErrors(cudaMalloc(&d_in[0], sz_in));
    checkCudaErrors(cudaMalloc(&d_in[1], sz_in));
    checkCudaErrors(cudaMalloc(&d_c1_static, sz_c1));
    checkCudaErrors(cudaMalloc(&d_c1_static_h, sz_c1_h));
    checkCudaErrors(cudaMalloc(&d_if1_v, sz_if1v));
    checkCudaErrors(cudaMalloc(&d_p1_h,  sz_p1_h));
    checkCudaErrors(cudaMalloc(&d_if2_v, sz_if2v));
    checkCudaErrors(cudaMalloc(&d_flat,  sz_flat));     // 回退路径需要
    checkCudaErrors(cudaMalloc(&d_if3_v, sz_if3));
    checkCudaErrors(cudaMalloc(&d_if3_s, sz_if3));
    checkCudaErrors(cudaMalloc(&d_if4_v, sz_if4));
    checkCudaErrors(cudaMalloc(&d_if4_s, sz_if4));
    checkCudaErrors(cudaMalloc(&d_acc,   sz_fc3));
    checkCudaErrors(cudaMalloc(&d_pred,  BATCH*sizeof(int)));

    // FC 半精度权重（原始 OxI）
    checkCudaErrors(cudaMalloc(&d_fc1_w_h, FC1_OUT * FC1_IN * sizeof(__half)));
    checkCudaErrors(cudaMalloc(&d_fc2_w_h, FC2_OUT * FC2_IN * sizeof(__half)));
    {
        int n1 = FC1_OUT * FC1_IN;
        int n2 = FC2_OUT * FC2_IN;
        float_to_half_kernel<<<div_up(n1,256), 256>>>(d_fc1_w, d_fc1_w_h, n1);
        float_to_half_kernel<<<div_up(n2,256), 256>>>(d_fc2_w, d_fc2_w_h, n2);
        checkCudaErrors(cudaGetLastError());
    }

    // WMMA 额外缓冲分配
    checkCudaErrors(cudaMalloc(&d_flat_h,       (size_t)BATCH*FC1_IN*sizeof(__half)));     // N x 256
    checkCudaErrors(cudaMalloc(&d_if3_s_pad_h,  (size_t)BATCH*FC2_KPAD*sizeof(__half)));   // N x 128
    checkCudaErrors(cudaMalloc(&d_fc1_wT_h,     (size_t)FC1_IN*FC1_NPAD*sizeof(__half)));  // 256 x 128
    checkCudaErrors(cudaMalloc(&d_fc2_wT_h,     (size_t)FC2_KPAD*FC2_NPAD*sizeof(__half))); // 128 x 96

    // 转置+pad 权重到列主
    {
        dim3 blkW(16, 16);
        // FC1
        dim3 grd1(div_up(FC1_NPAD, blkW.x), div_up(FC1_IN, blkW.y));
        fc1_transpose_pad_colmajor_v2<<<grd1, blkW>>>(d_fc1_w_h, d_fc1_wT_h, FC1_OUT, FC1_IN, FC1_NPAD);
        // FC2
        dim3 grd2(div_up(FC2_NPAD, blkW.x), div_up(FC2_KPAD, blkW.y));
        fc2_transpose_pad_colmajor<<<grd2, blkW>>>(d_fc2_w_h, d_fc2_wT_h, FC2_OUT, FC2_IN, FC2_KPAD, FC2_NPAD);
        checkCudaErrors(cudaGetLastError());
    }

    // L1 偏好
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(if1_pool1_fused_kernel, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(conv2_if_pool2_fused_const_smem_flat_oc2, cudaFuncCachePreferL1);

    // 运行期判断是否使用 WMMA（>=sm_70）
    cudaDeviceProp prop{};
    int dev = 0;
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&prop, dev);
    bool wmma_available = (prop.major >= 7);
    bool use_wmma = (wmma_available && (USE_WMMA != 0));

    // pinned host 双缓冲 + streams
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

    // launch 配置
    dim3 blk2d(16, 16, 1);
    dim3 blkP1(8, 8, 1);
    dim3 blkF(std::min(8, P2_W), std::min(8, P2_H), 1); // (4,4,1)
    const int tileN = 128; // FC fallback 的一次样本 tile
    dim3 blkLinear(1, tileN);

    const int total_batches = (N_all + BATCH - 1) / BATCH;

    // 预取 batch 0
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

    // z 轴切片，避免 grid.z>65535
    const int Z_MAX = 65535;
    auto sliceB = [&](int curB, int Cout)->int {
        return std::max(1, std::min(curB, Z_MAX / Cout));
    };

    for (int bi = 0; bi < total_batches; ++bi) {
        int cur = bi & 1, next = cur ^ 1;
        int start_idx = bi * BATCH;
        int curB = std::min(BATCH, N_all - start_idx);

        checkCudaErrors(cudaStreamWaitEvent(stream_comp, h2d_done[cur], 0));

        // conv1 静态前向
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

        // conv1 输出一次性转半精度（供 T 次复用）
        {
            int n_conv1 = curB * C1_OUT * C1_H * C1_W;
            float_to_half_kernel<<<div_up(n_conv1,256), 256, 0, stream_comp>>>(d_c1_static, d_c1_static_h, n_conv1);
            checkCudaErrors(cudaGetLastError());
        }

        // 清零膜电位/累加器
        checkCudaErrors(cudaMemsetAsync(d_if1_v, 0, (size_t)curB*C1_OUT*C1_H*C1_W*sizeof(float), stream_comp));
        checkCudaErrors(cudaMemsetAsync(d_if2_v, 0, (size_t)curB*C2_OUT*C2_H*C2_W*sizeof(float), stream_comp));
        checkCudaErrors(cudaMemsetAsync(d_if3_v, 0, (size_t)curB*FC1_OUT*sizeof(float), stream_comp));
        checkCudaErrors(cudaMemsetAsync(d_if4_v, 0, (size_t)curB*FC2_OUT*sizeof(float), stream_comp));
        checkCudaErrors(cudaMemsetAsync(d_acc,   0, (size_t)curB*FC3_OUT*sizeof(float), stream_comp));

        for (int t = 0; t < T; ++t) {
            // IF1 + pool1（half in/out）
            {
                int step = sliceB(curB, C1_OUT);
                for (int b0 = 0; b0 < curB; b0 += step) {
                    int thisB = std::min(step, curB - b0);
                    const __half* in_ptr = d_c1_static_h + (size_t)b0 * C1_OUT * C1_H * C1_W;
                    float*        v_ptr  = d_if1_v        + (size_t)b0 * C1_OUT * C1_H * C1_W;
                    __half*       out_ptr= d_p1_h         + (size_t)b0 * C1_OUT * P1_H * P1_W;

                    dim3 grdP1(div_up(P1_W, blkP1.x), div_up(P1_H, blkP1.y), thisB * C1_OUT);
                    if1_pool1_fused_kernel<<<grdP1, blkP1, 0, stream_comp>>>(
                            in_ptr, v_ptr, out_ptr, thisB, C1_H, C1_W, 1.0f
                    );
                    checkCudaErrors(cudaGetLastError());
                }
            }

            // conv2 + IF2 + pool2（half 输入，直接写扁平化 half）
            {
                int step = sliceB(curB, C2_OUT/OC_TILE);
                for (int b0 = 0; b0 < curB; b0 += step) {
                    int thisB = std::min(step, curB - b0);
                    const __half* x_ptr = d_p1_h   + (size_t)b0 * C1_OUT * P1_H * P1_W;
                    float*        v2_ptr= d_if2_v  + (size_t)b0 * C2_OUT * C2_H * C2_W;
                    __half*       yflat = d_flat_h + (size_t)b0 * FLAT; // half flatten

                    dim3 grdF(div_up(P2_W, blkF.x), div_up(P2_H, blkF.y), thisB * (C2_OUT/OC_TILE));
                    size_t smem_bytes = (size_t)P1_H * P1_W * sizeof(float); // 576B
                    conv2_if_pool2_fused_const_smem_flat_oc2<<<grdF, blkF, smem_bytes, stream_comp>>>(
                            x_ptr, v2_ptr, yflat, thisB, P1_H, P1_W, 1.0f
                    );
                    checkCudaErrors(cudaGetLastError());
                }
            }

            if (use_wmma) {
#if USE_WMMA
                // FC1 WMMA：直接消费 d_flat_h（half），并产出 d_if3_s_pad_h（N×128 half）
                {
                    dim3 grid(FC1_NPAD / WMMA_N, div_up(curB, WMMA_M));
                    dim3 block(32); // 单 warp
                    fc1_wmma_if_kernel<<<grid, block, 0, stream_comp>>>(
                            d_flat_h, d_fc1_wT_h, d_if3_v, d_if3_s, d_if3_s_pad_h, curB);
                    checkCudaErrors(cudaGetLastError());
                }
                // FC2 WMMA：直接消费 d_if3_s_pad_h（half, N×128）
                {
                    dim3 grid(FC2_NPAD / WMMA_N, div_up(curB, WMMA_M));
                    dim3 block(32); // 单 warp
                    fc2_wmma_if_kernel<<<grid, block, 0, stream_comp>>>(
                            d_if3_s_pad_h, d_fc2_wT_h, d_if4_v, d_if4_s, curB);
                    checkCudaErrors(cudaGetLastError());
                }
#endif
            } else {
                // 回退：把 d_flat_h 转 float 再跑半精度权重的 GEMV
                {
                    int tot = curB * FC1_IN;
                    half_to_float_kernel<<<div_up(tot,256), 256, 0, stream_comp>>>(d_flat_h, d_flat, tot);
                    checkCudaErrors(cudaGetLastError());
                }
                {
                    dim3 grd(FC1_OUT, div_up(curB, tileN));
                    size_t shmem = FC1_IN * sizeof(float);
                    linear_if_fp16W_cached_kernel_fc1<FC1_IN><<<grd, blkLinear, shmem, stream_comp>>>(
                            d_flat, d_fc1_w_h, d_if3_v, d_if3_s, curB, FC1_IN, FC1_OUT, 1.0f);
                    checkCudaErrors(cudaGetLastError());
                }
                {
                    dim3 grd(FC2_OUT, div_up(curB, tileN));
                    size_t shmem = FC2_IN * sizeof(float);
                    linear_if_fp16W_cached_kernel_fc2<FC2_IN><<<grd, blkLinear, shmem, stream_comp>>>(
                            d_if3_s, d_fc2_w_h, d_if4_v, d_if4_s, curB, FC2_IN, FC2_OUT, 1.0f);
                    checkCudaErrors(cudaGetLastError());
                }
            }

            // FC3 累加（融合 scale = 1/T）
            {
                dim3 blk(32, 8);
                dim3 grd(div_up(FC3_OUT, blk.x), div_up(curB, blk.y));
                fc3_accum_kernel_floatW<<<grd, blk, 0, stream_comp>>>(d_if4_s, d_fc3_w, d_acc, curB, 1.0f / T);
                checkCudaErrors(cudaGetLastError());
            }
        } // end T

        // 平均已融合；只做 argmax
        {
            argmax_kernel<<<div_up(curB, 256), 256, 0, stream_comp>>>(d_acc, d_pred, curB, FC3_OUT);
            checkCudaErrors(cudaGetLastError());
        }

        // 预取下一批
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

        // 收集结果
        checkCudaErrors(cudaStreamSynchronize(stream_comp));
        checkCudaErrors(cudaMemcpy(h_pred_vec.data(), d_pred, (size_t)curB*sizeof(int), cudaMemcpyDeviceToHost));
        predictions.insert(predictions.end(), h_pred_vec.begin(), h_pred_vec.begin() + curB);
    }

    // 释放
    checkCudaErrors(cudaEventDestroy(h2d_done[0]));
    checkCudaErrors(cudaEventDestroy(h2d_done[1]));
    checkCudaErrors(cudaStreamDestroy(stream_copy));
    checkCudaErrors(cudaStreamDestroy(stream_comp));
    checkCudaErrors(cudaFreeHost(h_in_pinned[0]));
    checkCudaErrors(cudaFreeHost(h_in_pinned[1]));

    checkCudaErrors(cudaFree(d_in[0]));
    checkCudaErrors(cudaFree(d_in[1]));
    checkCudaErrors(cudaFree(d_c1_static));
    checkCudaErrors(cudaFree(d_c1_static_h));
    checkCudaErrors(cudaFree(d_if1_v));
    checkCudaErrors(cudaFree(d_p1_h));
    checkCudaErrors(cudaFree(d_if2_v));
    checkCudaErrors(cudaFree(d_flat));
    checkCudaErrors(cudaFree(d_if3_v));
    checkCudaErrors(cudaFree(d_if3_s));
    checkCudaErrors(cudaFree(d_if4_v));
    checkCudaErrors(cudaFree(d_if4_s));
    checkCudaErrors(cudaFree(d_acc));
    checkCudaErrors(cudaFree(d_pred));

    // WMMA 相关
    checkCudaErrors(cudaFree(d_fc1_w_h));
    checkCudaErrors(cudaFree(d_fc2_w_h));
    checkCudaErrors(cudaFree(d_flat_h));
    checkCudaErrors(cudaFree(d_if3_s_pad_h));
    checkCudaErrors(cudaFree(d_fc1_wT_h));
    checkCudaErrors(cudaFree(d_fc2_wT_h));

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
    std::vector<int> predictions = scnn_inference(images,
                                                  d_conv1_w, d_conv1_b, d_conv2_w, d_conv2_b,
                                                  d_fc1_w, d_fc1_b, d_fc2_w, d_fc2_b, d_fc3_w, d_fc3_b
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

    std::cout << std::fixed << std::setprecision(4) << diff.count() << ":" << accuracy << std::endl;
    return 0;
}
// ===================================================================================
// Main Function -  DO NOT MODIFY END
// ===================================================================================