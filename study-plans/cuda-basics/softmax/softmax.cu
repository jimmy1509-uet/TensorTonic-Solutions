#include <cuda_runtime.h>

__global__ void softmax_kernel(const float* input, float* output, int N) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >=N) return;
    float maxv = input[0];
    for (int j = 1; j < N; j++) maxv = fmaxf(maxv, input[j]);
    float sum = 0.0f;
    for(int j = 0;j < N;j++){
        sum += expf(input[j] - maxv);
    }
    output[i] = expf(input[i]-maxv)/sum;
    
    
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    softmax_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}