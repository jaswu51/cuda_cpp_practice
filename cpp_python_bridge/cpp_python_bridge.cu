#include <stdio.h>

__global__ void negative_number_kernel(float* input, float* output, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        output[index] = -input[index];
    }
}

extern "C" {
    void negative_number_main(float* input, float* output, int n) {
        float *d_input, *d_output;

        cudaMalloc(&d_input, n * sizeof(float));
        cudaMalloc(&d_output, n * sizeof(float));
        cudaMemcpy(d_input, input, n * sizeof(float), cudaMemcpyHostToDevice);

        int threads_per_block = 256;
        int blocks = (n + threads_per_block - 1) / threads_per_block;
        negative_number_kernel<<<blocks, threads_per_block>>>(d_input, d_output, n);
        cudaMemcpy(output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_input);
        cudaFree(d_output);
    }

}
