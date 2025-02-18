#include <stdio.h>

__global__ void matrixMultiplicationKernel(float* A, float* B, float* C, 
                                         int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(row < M && col < N) {
        float sum = 0.0f;  // 初始化为0
        for(int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;  // 存储最终结果
    }   
}
/*
如果我C的维度超过8怎么办
矩阵C (12 x 12)
|------8------|----4----|
|             |         |
|    块0      |   块1   |  8行
|             |         |
|-------------|---------|
|             |         |
|    块2      |   块3   |  4行
|             |         |
|-------------|---------|
这个矩阵乘法是在线程级别（thread level）并行的
在当前的实现中，每个线程都需要从全局内存（DRAM）加载数据，这是非常低效的。
*/

int main() {
    int M = 3, N = 4, K = 2;
    
    // 创建CUDA事件用于计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    float* h_A = (float*)malloc(M * K * sizeof(float));
    float* h_B = (float*)malloc(K * N * sizeof(float));
    float* h_C = (float*)malloc(M * N * sizeof(float));

    // 初始化矩阵A和B   
    for(int i = 0; i < M * K; i++) {
        h_A[i] = i + 1;
    }
    for(int i = 0; i < K * N; i++) {
        h_B[i] = i + 1;
    }

    // 分配GPU内存
    float* d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    // 将数据从CPU复制到GPU
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // 定义线程块和网格
    dim3 block(8, 8);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    // 开始计时
    cudaEventRecord(start);

    // 执行矩阵乘法
    matrixMultiplicationKernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);    

    // 停止计时
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // 计算执行时间
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU执行时间: %f 毫秒\n", milliseconds);

    // 将结果从GPU复制到CPU
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印结果
    printf("矩阵A:\n"); 
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < K; j++) {
            printf("%f ", h_A[i * K + j]);
        }
        printf("\n");
    }   
    printf("矩阵B:\n");
    for(int i = 0; i < K; i++) {
        for(int j = 0; j < N; j++) {
            printf("%f ", h_B[i * N + j]);
        }
        printf("\n");
    }   
    printf("矩阵C:\n");     
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            printf("%f ", h_C[i * N + j]);
        }
        printf("\n");
    }      

    // 释放内存
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    // 销毁CUDA事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}