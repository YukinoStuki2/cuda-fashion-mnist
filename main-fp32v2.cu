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

// ========== 可调宏 ==========
#define IF_SUBTRACT_RESET 0

// LeNet 结构（固定）
#define C1_IN   1
#define C1_OUT  6
#define K1      5
#define C2_IN   6
#define C2_OUT  16
#define K2      5

#ifndef USE_PTX
#define USE_PTX 1
#endif

// 全连接维度
#define FC1_IN 256  // 16*4*4
#define FC1_OUT 120
#define FC2_IN 120
#define FC2_OUT 84
#define FC3_IN 84
#define FC3_OUT 10

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

// 常量内存（conv 权重/偏置 + fc3 偏置）
__constant__ float c_conv1_w[C1_OUT * C1_IN * K1 * K1];
__constant__ float c_conv1_b[C1_OUT];
__constant__ float c_conv2_w[C2_OUT * C2_IN * K2 * K2];
__constant__ float c_conv2_b[C2_OUT];
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
// Data and Parameter Loading Functions - DO MODIFY END
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

// IF1 + pool1 fused，写出 uint8 spike
__launch_bounds__(256, 2)
__global__ void if1_pool1_fused_u8_kernel(
        const float* __restrict__ x_c1,   // [N, C1_OUT, 24, 24]
        float* __restrict__ v1,           // [N, C1_OUT, 24, 24]
        unsigned char* __restrict__ y_p1, // [N, C1_OUT, 12, 12] uint8
        int N, int H_in, int W_in, float v_th
){
    const int Hp = H_in / 2; // 12
    const int Wp = W_in / 2; // 12

    int oxp = blockIdx.x * blockDim.x + threadIdx.x;
    int oyp = blockIdx.y * blockDim.y + threadIdx.y;
    int z   = blockIdx.z;
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
            s_max  = fmaxf(s_max, s);
        }
    }
    y_p1[idx4(n, oc, oyp, oxp, C1_OUT, Hp, Wp)] = (unsigned char)(s_max > 0.0f);
}

// conv2 + IF2 + pool2 融合（shared-memory tiling），读取 uint8 p1
__launch_bounds__(16, 16)
__global__ void conv2_if_pool2_fused_const_smem_u8(
        const unsigned char* __restrict__ x_p1, // [N, C2_IN, 12, 12] uint8
        float* __restrict__ v2,                 // [N, C2_OUT, 8, 8]
        unsigned char* __restrict__ y_p2,       // [N, C2_OUT, 4, 4] uint8
        int N, int H_in, int W_in, float v_th
){
    extern __shared__ float s_x[];  // 12*12 floats
    const int Hout = H_in - K2 + 1; // 8
    const int Wout = W_in - K2 + 1; // 8
    const int Hp   = Hout / 2;      // 4
    const int Wp   = Wout / 2;      // 4

    int oxp = blockIdx.x * blockDim.x + threadIdx.x;
    int oyp = blockIdx.y * blockDim.y + threadIdx.y;
    int z   = blockIdx.z;
    if (oxp >= Wp || oyp >= Hp || z >= N * C2_OUT) return;

    int n  = z / C2_OUT;
    int oc = z % C2_OUT;

    int py0 = oyp * 2;
    int px0 = oxp * 2;

#if USE_PTX
    float acc00 = ptx_ld_const_f32(c_conv2_b + oc);
#else
    float acc00 = c_conv2_b[oc];
#endif
    float acc01 = acc00, acc10 = acc00, acc11 = acc00;

    const int tid   = threadIdx.y * blockDim.x + threadIdx.x; // 0..15
    const int tcount= blockDim.x * blockDim.y;                 // 16
    const int plane = H_in * W_in;                             // 144

    for (int ic = 0; ic < C2_IN; ++ic) {
        // coop load uint8 -> float
        for (int idx = tid; idx < plane; idx += tcount) {
            int iy = idx / W_in;
            int ix = idx % W_in;
            unsigned char v = x_p1[idx4(n, ic, iy, ix, C2_IN, H_in, W_in)];
            s_x[idx] = (float)v;
        }
        __syncthreads();

#pragma unroll
        for (int ky = 0; ky < K2; ++ky) {
#pragma unroll
            for (int kx = 0; kx < K2; ++kx) {
                int widx = (((oc * C2_IN) + ic) * K2 + ky) * K2 + kx;
#if USE_PTX
                float wv = ptx_ld_const_f32(c_conv2_w + widx);
#else
                float wv = c_conv2_w[widx];
#endif
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
        __syncthreads();
    }

    float s_max = 0.0f;

    int vidx00 = idx4(n, oc, py0 + 0, px0 + 0, C2_OUT, Hout, Wout);
    float vv00 = v2[vidx00] + acc00;
    float s00  = (vv00 >= v_th) ? 1.0f : 0.0f;
#if IF_SUBTRACT_RESET
    vv00 = vv00 - s00 * v_th;
#else
    vv00 = (s00 > 0.0f) ? 0.0f : vv00;
#endif
    v2[vidx00] = vv00; s_max = fmaxf(s_max, s00);

    int vidx01 = idx4(n, oc, py0 + 0, px0 + 1, C2_OUT, Hout, Wout);
    float vv01 = v2[vidx01] + acc01;
    float s01  = (vv01 >= v_th) ? 1.0f : 0.0f;
#if IF_SUBTRACT_RESET
    vv01 = vv01 - s01 * v_th;
#else
    vv01 = (s01 > 0.0f) ? 0.0f : vv01;
#endif
    v2[vidx01] = vv01; s_max = fmaxf(s_max, s01);

    int vidx10 = idx4(n, oc, py0 + 1, px0 + 0, C2_OUT, Hout, Wout);
    float vv10 = v2[vidx10] + acc10;
    float s10  = (vv10 >= v_th) ? 1.0f : 0.0f;
#if IF_SUBTRACT_RESET
    vv10 = vv10 - s10 * v_th;
#else
    vv10 = (s10 > 0.0f) ? 0.0f : vv10;
#endif
    v2[vidx10] = vv10; s_max = fmaxf(s_max, s10);

    int vidx11 = idx4(n, oc, py0 + 1, px0 + 1, C2_OUT, Hout, Wout);
    float vv11 = v2[vidx11] + acc11;
    float s11  = (vv11 >= v_th) ? 1.0f : 0.0f;
#if IF_SUBTRACT_RESET
    vv11 = vv11 - s11 * v_th;
#else
    vv11 = (s11 > 0.0f) ? 0.0f : vv11;
#endif
    v2[vidx11] = vv11; s_max = fmaxf(s_max, s11);

    y_p2[idx4(n, oc, oyp, oxp, C2_OUT, Hp, Wp)] = (unsigned char)(s_max > 0.0f);
}

// FC1：直接从 p2（uint8）读取并做 IF（免 flatten）
__launch_bounds__(256, 2)
__global__ void fc1_from_p2_u8_if_kernel(
        const unsigned char* __restrict__ p2_u8, // [N,16,4,4] uint8
        const float* __restrict__ W,             // [FC1_OUT, 256] float
        const float* __restrict__ b,             // [FC1_OUT] float
        float* __restrict__ v,                   // [N, FC1_OUT]
        float* __restrict__ out,                 // [N, FC1_OUT]
        int N
){
    int ox = blockIdx.x * blockDim.x + threadIdx.x; // 0..FC1_OUT-1
    int n  = blockIdx.y * blockDim.y + threadIdx.y; // 0..N-1
    if (ox >= FC1_OUT || n >= N) return;

    float acc = b[ox];

#pragma unroll
    for (int c = 0; c < 16; ++c) {
#pragma unroll
        for (int h = 0; h < 4; ++h) {
            const unsigned char* xptr = p2_u8 + idx4(n, c, h, 0, 16, 4, 4);
            const float* wptr = W + ox * FC1_IN + (c * 16 + h * 4);
            // 标量读取四个字节，权重 vector 化
            unsigned char u0 = xptr[0], u1 = xptr[1], u2 = xptr[2], u3 = xptr[3];
            float4 wv4 = reinterpret_cast<const float4*>(wptr)[0];
            acc = fmaf((float)u0, wv4.x, acc);
            acc = fmaf((float)u1, wv4.y, acc);
            acc = fmaf((float)u2, wv4.z, acc);
            acc = fmaf((float)u3, wv4.w, acc);
        }
    }

    int id = n * FC1_OUT + ox;
    float vv = v[id] + acc;
    float s  = (vv >= 1.0f) ? 1.0f : 0.0f;
#if IF_SUBTRACT_RESET
    vv = vv - s * 1.0f;
#else
    vv = (s > 0.0f) ? 0.0f : vv;
#endif
    v[id]   = vv;
    out[id] = s;
}

// 线性 + IF（FC2） float 权重，向量化
__launch_bounds__(256, 2)
__global__ void linear_if_fused_kernel(
        const float* __restrict__ x, // [N, I]
        const float* __restrict__ W, // [O, I]
        const float* __restrict__ b, // [O]
        float* __restrict__ v,       // [N, O]
        float* __restrict__ y,       // [N, O]
        int N, int I, int O, float v_th
){
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int n  = blockIdx.y * blockDim.y + threadIdx.y;
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
    float s  = (vv >= v_th) ? 1.0f : 0.0f;
#if IF_SUBTRACT_RESET
    vv = vv - s * v_th;
#else
    vv = (s > 0.0f) ? 0.0f : vv;
#endif
    v[id] = vv;
    y[id] = s;
}

// FC3 累加 float（偏置从常量内存），向量化
__launch_bounds__(256, 2)
__global__ void fc3_accum_kernel(
        const float* __restrict__ spikes, // [N, FC3_IN]
        const float* __restrict__ weights,// [FC3_OUT, FC3_IN]
        float* __restrict__ output,       // [N, FC3_OUT]
        int N
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
    for (int i = 0; i < FC3_IN; ++i) acc += xrow[i] * wrow[i];
#endif

    output[n * FC3_OUT + ox] += acc;
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
    // const int FLAT=16*4*4;  // 不再需要中间 flatten

    // 批大小：V100S-32GB 建议 8192（也可 4096 视实测）
    const int BATCH = 8192;

    std::vector<int> predictions;
    predictions.reserve(N_all);

    // 常量内存拷贝（device->device）
    checkCudaErrors(cudaMemcpyToSymbol(c_conv1_w, d_conv1_w, sizeof(float)*C1_OUT*C1_IN*K1*K1, 0, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(c_conv1_b, d_conv1_b, sizeof(float)*C1_OUT,              0, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(c_conv2_w, d_conv2_w, sizeof(float)*C2_OUT*C2_IN*K2*K2, 0, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(c_conv2_b, d_conv2_b, sizeof(float)*C2_OUT,              0, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(c_fc3_b,   d_fc3_b,   sizeof(float)*FC3_OUT,             0, cudaMemcpyDeviceToDevice));

    // 设备中间缓冲
    float *d_in[2] = {nullptr, nullptr};
    float *d_c1_static=nullptr;
    float *d_if1_v=nullptr;
    unsigned char *d_p1_u8=nullptr; // uint8
    float *d_if2_v=nullptr;
    unsigned char *d_p2_u8=nullptr; // uint8
    float *d_if3_v=nullptr, *d_if3_s=nullptr;
    float *d_if4_v=nullptr, *d_if4_s=nullptr;
    float *d_acc=nullptr;
    int   *d_pred=nullptr;

    size_t sz_in    = (size_t)BATCH*IMG_C*IMG_H*IMG_W*sizeof(float);
    size_t sz_c1    = (size_t)BATCH*C1_OUT*C1_H*C1_W*sizeof(float);
    size_t sz_if1v  = sz_c1;
    size_t sz_p1u8  = (size_t)BATCH*C1_OUT*P1_H*P1_W;                 // uint8
    size_t sz_if2v  = (size_t)BATCH*C2_OUT*C2_H*C2_W*sizeof(float);
    size_t sz_p2u8  = (size_t)BATCH*C2_OUT*P2_H*P2_W;                 // uint8
    size_t sz_if3   = (size_t)BATCH*FC1_OUT*sizeof(float);
    size_t sz_if4   = (size_t)BATCH*FC2_OUT*sizeof(float);
    size_t sz_fc3   = (size_t)BATCH*FC3_OUT*sizeof(float);

    checkCudaErrors(cudaMalloc(&d_in[0], sz_in));
    checkCudaErrors(cudaMalloc(&d_in[1], sz_in));
    checkCudaErrors(cudaMalloc(&d_c1_static, sz_c1));
    checkCudaErrors(cudaMalloc(&d_if1_v, sz_if1v));
    checkCudaErrors(cudaMalloc(&d_p1_u8, sz_p1u8));
    checkCudaErrors(cudaMalloc(&d_if2_v,  sz_if2v));
    checkCudaErrors(cudaMalloc(&d_p2_u8,  sz_p2u8));
    checkCudaErrors(cudaMalloc(&d_if3_v,  sz_if3));
    checkCudaErrors(cudaMalloc(&d_if3_s,  sz_if3));
    checkCudaErrors(cudaMalloc(&d_if4_v,  sz_if4));
    checkCudaErrors(cudaMalloc(&d_if4_s,  sz_if4));
    checkCudaErrors(cudaMalloc(&d_acc,    sz_fc3));
    checkCudaErrors(cudaMalloc(&d_pred,   BATCH*sizeof(int)));

    // pinned host 双缓冲 + stream
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
    dim3 blkLinear(128, 2);  // FC 层
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

    // z 维切片（避免 grid.z>65535）
    const int Z_MAX = 65535;
    auto sliceB = [&](int curB, int Cout)->int {
        return std::max(1, std::min(curB, Z_MAX / Cout));
    };

    for (int bi = 0; bi < total_batches; ++bi) {
        int cur = bi & 1, next = cur ^ 1;
        int start_idx = bi * BATCH;
        int curB = std::min(BATCH, N_all - start_idx);

        checkCudaErrors(cudaStreamWaitEvent(stream_comp, h2d_done[cur], 0));

        // conv1 预计算（z 切片）
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

        // 清零膜电位 / logits 累加器
        checkCudaErrors(cudaMemsetAsync(d_if1_v, 0, (size_t)curB*C1_OUT*C1_H*C1_W*sizeof(float), stream_comp));
        checkCudaErrors(cudaMemsetAsync(d_if2_v, 0, (size_t)curB*C2_OUT*C2_H*C2_W*sizeof(float), stream_comp));
        checkCudaErrors(cudaMemsetAsync(d_if3_v, 0, (size_t)curB*FC1_OUT*sizeof(float), stream_comp));
        checkCudaErrors(cudaMemsetAsync(d_if4_v, 0, (size_t)curB*FC2_OUT*sizeof(float), stream_comp));
        checkCudaErrors(cudaMemsetAsync(d_acc,   0, (size_t)curB*FC3_OUT*sizeof(float), stream_comp));

        for (int t = 0; t < T; ++t) {
            // IF1 + pool1（写出 uint8）
            {
                int step = sliceB(curB, C1_OUT);
                for (int b0 = 0; b0 < curB; b0 += step) {
                    int thisB = std::min(step, curB - b0);
                    const float* in_ptr = d_c1_static + (size_t)b0 * C1_OUT * C1_H * C1_W;
                    float*       v_ptr  = d_if1_v     + (size_t)b0 * C1_OUT * C1_H * C1_W;
                    unsigned char* out_ptr = d_p1_u8  + (size_t)b0 * C1_OUT * P1_H * P1_W;

                    dim3 grdP1(div_up(P1_W, blkP1.x), div_up(P1_H, blkP1.y), thisB * C1_OUT);
                    if1_pool1_fused_u8_kernel<<<grdP1, blkP1, 0, stream_comp>>>(
                            in_ptr, v_ptr, out_ptr, thisB, C1_H, C1_W, 1.0f
                    );
                    checkCudaErrors(cudaGetLastError());
                }
            }

            // conv2 + IF2 + pool2（读 uint8 p1，写 uint8 p2）
            {
                int step = sliceB(curB, C2_OUT);
                for (int b0 = 0; b0 < curB; b0 += step) {
                    int thisB = std::min(step, curB - b0);
                    const unsigned char* x_ptr = d_p1_u8 + (size_t)b0 * C1_OUT * P1_H * P1_W;
                    float*       v2_ptr= d_if2_v + (size_t)b0 * C2_OUT * C2_H * C2_W;
                    unsigned char* y_ptr = d_p2_u8 + (size_t)b0 * C2_OUT * P2_H * P2_W;

                    dim3 grdF(div_up(P2_W, blkF.x), div_up(P2_H, blkF.y), thisB * C2_OUT);
                    size_t smem_bytes = (size_t)P1_H * P1_W * sizeof(float);
                    conv2_if_pool2_fused_const_smem_u8<<<grdF, blkF, smem_bytes, stream_comp>>>(
                            x_ptr, v2_ptr, y_ptr, thisB, P1_H, P1_W, 1.0f
                    );
                    checkCudaErrors(cudaGetLastError());
                }
            }

            // FC1：直接从 p2_u8 做 IF（免 flatten）
            {
                dim3 grd(div_up(FC1_OUT, blkLinear.x), div_up(curB, blkLinear.y));
                fc1_from_p2_u8_if_kernel<<<grd, blkLinear, 0, stream_comp>>>(
                        d_p2_u8, d_fc1_w, d_fc1_b, d_if3_v, d_if3_s, curB
                );
                checkCudaErrors(cudaGetLastError());
            }

            // FC2 + IF
            {
                dim3 grd(div_up(FC2_OUT, blkLinear.x), div_up(curB, blkLinear.y));
                linear_if_fused_kernel<<<grd, blkLinear, 0, stream_comp>>>(
                        d_if3_s, d_fc2_w, d_fc2_b, d_if4_v, d_if4_s,
                        curB, FC2_IN, FC2_OUT, 1.0f);
                checkCudaErrors(cudaGetLastError());
            }

            // FC3 累加
            {
                dim3 blk(32, 8);
                dim3 grd(div_up(FC3_OUT, blk.x), div_up(curB, blk.y));
                fc3_accum_kernel<<<grd, blk, 0, stream_comp>>>(d_if4_s, d_fc3_w, d_acc, curB);
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

        // 回收预测
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
    checkCudaErrors(cudaFree(d_if1_v));
    checkCudaErrors(cudaFree(d_p1_u8));
    checkCudaErrors(cudaFree(d_if2_v));
    checkCudaErrors(cudaFree(d_p2_u8));
    checkCudaErrors(cudaFree(d_if3_v));
    checkCudaErrors(cudaFree(d_if3_s));
    checkCudaErrors(cudaFree(d_if4_v));
    checkCudaErrors(cudaFree(d_if4_s));
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