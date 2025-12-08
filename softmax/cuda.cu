#include <cuda_runtime.h>
#include <float.h>

__global__ void reduce_max_kernel(const float* input, float* max_buf, int N) {
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float v = -FLT_MAX;
    if (i < N) v = input[i];

    sdata[tid] = v;
    __syncthreads();

    // block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) max_buf[blockIdx.x] = sdata[0];
}

__global__ void reduce_sum_kernel(const float* input, float* output,
                                  const float* max_val, float* sum_buf, int N) {

    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float maxv = max_val[0];

    float v = 0.0f;
    if (i < N) {
        float ex = expf(input[i] - maxv);
        output[i] = ex;   // temporarily store exp(x - max)
        v = ex;
    }

    sdata[tid] = v;
    __syncthreads();

    // block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) sum_buf[blockIdx.x] = sdata[0];
}

__global__ void normalize_kernel(float* output, const float* sum_val, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        output[i] = output[i] / sum_val[0];
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    float* d_maxbuf;
    float* d_sumbuf;
    float* d_maxval;
    float* d_sumval;

    cudaMalloc(&d_maxbuf, blocksPerGrid * sizeof(float));
    cudaMalloc(&d_sumbuf, blocksPerGrid * sizeof(float));
    cudaMalloc(&d_maxval, sizeof(float));
    cudaMalloc(&d_sumval, sizeof(float));

    // 1. Reduce max
    reduce_max_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, d_maxbuf, N);
    cudaDeviceSynchronize();

    // Second-stage reduction on host (small array)
    float* hbuf = new float[blocksPerGrid];
    cudaMemcpy(hbuf, d_maxbuf, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);
    float maxv = -FLT_MAX;
    for (int i = 0; i < blocksPerGrid; i++) maxv = fmaxf(maxv, hbuf[i]);
    cudaMemcpy(d_maxval, &maxv, sizeof(float), cudaMemcpyHostToDevice);

    // 2. Compute exp(x - max) and reduce sum
    reduce_sum_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, d_maxval, d_sumbuf, N);
    cudaDeviceSynchronize();

    // Reduce sums on host (again small)
    cudaMemcpy(hbuf, d_sumbuf, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);
    float sumv = 0.0f;
    for (int i = 0; i < blocksPerGrid; i++) sumv += hbuf[i];
    cudaMemcpy(d_sumval, &sumv, sizeof(float), cudaMemcpyHostToDevice);

    delete[] hbuf;

    // 3. Normalize
    normalize_kernel<<<blocksPerGrid, threadsPerBlock>>>(output, d_sumval, N);
    cudaDeviceSynchronize();

    cudaFree(d_maxbuf);
    cudaFree(d_sumbuf);
    cudaFree(d_maxval);
    cudaFree(d_sumval);
}
