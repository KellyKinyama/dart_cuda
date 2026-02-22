nvcc --shared -o libmatmul.so engine.cu -Xcompiler -fPIC

nvcc --shared -o dart_cuda.so dart_cuda.cu -Xcompiler -fPIC