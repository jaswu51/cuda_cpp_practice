#include <cuda_runtime.h>
#include <stdio.h>

// 使用extern "C"让C++函数能被Python的ctypes调用
extern "C" {
    void get_gpu_memory_info(float* free_mem, float* total_mem) {
        size_t free_memory, total_memory;
        cudaError_t error = cudaMemGetInfo(&free_memory, &total_memory);
        
        if(error != cudaSuccess) {
            printf("CUDA错误: %s\n", cudaGetErrorString(error));
            return;
        }
        
        *free_mem = free_memory/1024.0/1024.0;  // 转换为MB
        *total_mem = total_memory/1024.0/1024.0; // 转换为MB
    }
}
