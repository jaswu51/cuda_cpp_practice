import ctypes
import os

# 加载编译好的库
lib_path = os.path.join(os.path.dirname(__file__), "libnegate.so")
cuda_lib = ctypes.CDLL(lib_path)

# 设置函数参数类型
cuda_lib.get_gpu_memory_info.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float)
]

def get_gpu_memory():
    free_mem = ctypes.c_float()
    total_mem = ctypes.c_float()
    
    cuda_lib.get_gpu_memory_info(ctypes.byref(free_mem), ctypes.byref(total_mem))
    
    return {
        "可用内存(MB)": free_mem.value,
        "总内存(MB)": total_mem.value,
        "已使用内存(MB)": total_mem.value - free_mem.value
    }

if __name__ == "__main__":
    memory_info = get_gpu_memory()
    print("\nGPU内存信息:")
    for key, value in memory_info.items():
        print(f"{key}: {value:.2f}") 