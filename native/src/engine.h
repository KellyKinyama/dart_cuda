#ifndef ENGINE_H
#define ENGINE_H

typedef void* TensorHandle;

extern "C" {
    TensorHandle create_tensor(int rows, int cols, float* initial_data);
    void destroy_tensor(TensorHandle handle);
    void get_tensor_data(TensorHandle handle, float* cpu_buffer);
    void get_tensor_grad(TensorHandle handle, float* cpu_buffer);
    void backward(TensorHandle handle);
    TensorHandle add_tensors(TensorHandle a, TensorHandle b);
    TensorHandle sub_tensors(TensorHandle a, TensorHandle b);
    TensorHandle mul_tensors(TensorHandle a, TensorHandle b);
    TensorHandle div_tensors(TensorHandle a, TensorHandle b);
    TensorHandle pow_tensor(TensorHandle a, float exponent);
    TensorHandle relu_tensor(TensorHandle a);
    TensorHandle tanh_tensor(TensorHandle a);
    TensorHandle sigmoid_tensor(TensorHandle a); // If you add it to C++
}

#endif