import numpy as np
import ctypes
from numpy.ctypeslib import ndpointer

# 加载编译好的共享库
lib = ctypes.CDLL('cpp_python_bridge/libnegate.so')

# 设置函数参数类型
lib.negative_number_main.argtypes = [
    ndpointer(dtype=np.float32),  # input array
    ndpointer(dtype=np.float32),  # output array
    ctypes.c_int                  # array length
]
lib.negative_number_main.restype = None  # 函数返回值类型为void

def negate_array(input_array):
    """
    对输入数组中的每个元素取负值
    
    参数:
        input_array: numpy数组，元素类型为float32
        
    返回:
        numpy数组，包含取负后的结果
    """
    # 确保输入是float32类型
    input_array = np.asarray(input_array, dtype=np.float32)
    
    # 创建输出数组
    output_array = np.empty_like(input_array)
    
    # 调用CUDA函数
    lib.negative_number_main(
        input_array,
        output_array,
        input_array.size
    )
    
    return output_array

# 测试代码
if __name__ == '__main__':
    # 创建测试数组
    test_array = np.array([1.0, -2.0, 3.0, -4.0, 5.0], dtype=np.float32)
    print("输入数组:", test_array)
    
    # 调用函数
    result = negate_array(test_array)
    print("输出数组:", result)
    
    # 验证结果
    print("验证结果:", np.allclose(-test_array, result))
