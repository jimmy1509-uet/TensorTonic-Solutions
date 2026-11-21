#include <cuda_runtime.h>
#include <cfloat>

#define THREADS    256
#define MAX_BLOCKS 1024

__device__ float        g_pmax[MAX_BLOCKS];
__device__ float        g_psum[MAX_BLOCKS];
__device__ unsigned int g_count = 0;
__device__ float        g_M;
__device__ float        g_invS;

// Thêm 1 phần tử vào trạng thái online (m, s)
__device__ __forceinline__ void online_add(float& m, float& s, float x) {
    if (x > m) { s = s * expf(m - x) + 1.0f; m = x; }
    else       { s += expf(x - m); }
}

// Gộp 2 trạng thái (m, s) và (m2, s2)
__device__ __forceinline__ void combine(float& m, float& s, float m2, float s2) {
    float nm = fmaxf(m, m2);
    s = s * expf(m - nm) + s2 * expf(m2 - nm);
    m = nm;
}

__device__ __forceinline__ void warp_reduce(float& m, float& s) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float m2 = __shfl_xor_sync(0xffffffff, m, off);
        float s2 = __shfl_xor_sync(0xffffffff, s, off);
        combine(m, s, m2, s2);
    }
}

// Sau hàm này, mọi thread trong block đều có (m, s) của cả block
__device__ __forceinline__ void block_reduce(float& m, float& s) {
    __shared__ float sm[32], ss[32];
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    int nwarps = blockDim.x >> 5;

    warp_reduce(m, s);
    if (lane == 0) { sm[wid] = m; ss[wid] = s; }
    __syncthreads();

    if (wid == 0) {
        m = (lane < nwarps) ? sm[lane] : -FLT_MAX;
        s = (lane < nwarps) ? ss[lane] : 0.0f;
        warp_reduce(m, s);
        if (lane == 0) { sm[0] = m; ss[0] = s; }
    }
    __syncthreads();
    m = sm[0]; s = ss[0];
}

// Kernel 1: tính M = max(x), S = sum exp(x - M) trong 1 lượt đọc
__global__ void reduce_kernel(const float* __restrict__ in, int N) {
    float m = -FLT_MAX, s = 0.0f;
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    int n4 = N >> 2;
    const float4* in4 = reinterpret_cast<const float4*>(in);
    for (int i = tid; i < n4; i += stride) {
        float4 v = in4[i];
        online_add(m, s, v.x); online_add(m, s, v.y);
        online_add(m, s, v.z); online_add(m, s, v.w);
    }
    for (int i = (n4 << 2) + tid; i < N; i += stride)   // phần dư khi N % 4 != 0
        online_add(m, s, in[i]);

    block_reduce(m, s);

    // Ghi partial của block, block xong cuối cùng sẽ gộp tất cả
    __shared__ bool is_last;
    if (threadIdx.x == 0) {
        g_pmax[blockIdx.x] = m;
        g_psum[blockIdx.x] = s;
        __threadfence();                                  // đảm bảo partial đã hiển thị toàn cục
        unsigned int t = atomicAdd(&g_count, 1u);
        is_last = (t == gridDim.x - 1);
    }
    __syncthreads();

    if (is_last) {
        m = -FLT_MAX; s = 0.0f;
        for (int i = threadIdx.x; i < gridDim.x; i += blockDim.x)
            combine(m, s, __ldcg(&g_pmax[i]), __ldcg(&g_psum[i]));  // bỏ qua L1, tránh đọc dữ liệu cũ
        block_reduce(m, s);
        if (threadIdx.x == 0) {
            g_M    = m;
            g_invS = 1.0f / s;
            g_count = 0;                                  // reset cho lần gọi sau
        }
    }
}

// Kernel 2: out[i] = exp(x[i] - M) / S
__global__ void normalize_kernel(const float* __restrict__ in, float* __restrict__ out, int N) {
    const float M = g_M, invS = g_invS;
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    int n4 = N >> 2;
    const float4* in4  = reinterpret_cast<const float4*>(in);
    float4*       out4 = reinterpret_cast<float4*>(out);
    for (int i = tid; i < n4; i += stride) {
        float4 v = in4[i], o;
        o.x = expf(v.x - M) * invS;
        o.y = expf(v.y - M) * invS;
        o.z = expf(v.z - M) * invS;
        o.w = expf(v.w - M) * invS;
        out4[i] = o;
    }
    for (int i = (n4 << 2) + tid; i < N; i += stride)
        out[i] = expf(in[i] - M) * invS;
}

extern "C" void solve(const float* input, float* output, int N) {
    if (N <= 0) return;

    // Kernel 1: mỗi thread xử lý khoảng >= 8 phần tử, tối đa MAX_BLOCKS block
    int b1 = (N + THREADS * 8 - 1) / (THREADS * 8);
    if (b1 > MAX_BLOCKS) b1 = MAX_BLOCKS;
    if (b1 < 1) b1 = 1;
    reduce_kernel<<<b1, THREADS>>>(input, N);

    // Kernel 2: mỗi thread khoảng 1 float4
    int b2 = ((N + 3) / 4 + THREADS - 1) / THREADS;
    if (b2 > 65535) b2 = 65535;
    normalize_kernel<<<b2, THREADS>>>(input, output, N);

    cudaDeviceSynchronize();
}