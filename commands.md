nvcc --shared -o libmat_mul.so engine.cu -Xcompiler -fPIC

nvcc --shared -o dart_cuda.so dart_cuda.cu -Xcompiler -fPIC



nvcc --shared -o libmat_mul.so engine_v2.cu -Xcompiler -fPIC