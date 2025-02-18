# cuda_cpp_practice

## cpp_python_bridge

### 1. 编译

```bash
nvcc -Xcompiler -fPIC -shared cpp_python_bridge/test.cu -o cpp_python_bridge/libnegate.so
```

### 2. 运行

```bash
python cpp_python_bridge/cpp_python_bridge.py
```

## matrix_multiplication

CUDA实现的矩阵乘法。

### 1. 编译

```bash
nvcc -o matrix_mult matrix_multiplication/matrix_multiplication.cu
```

### 2. 运行

```bash
./matrix_mult
```

### 3. 性能分析

使用 Nsight Systems 进行性能分析：
```bash
nsys profile --stats=true ./matrix_mult
```

