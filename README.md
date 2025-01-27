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