#pragma once

// An "opaque pointer" to our internal C++ Tensor object.
typedef void* TensorHandle;

extern "C" {
    // --- Memory Management ---
    TensorHandle create_tensor(int rows, int cols, float* initial_data);
    void destroy_tensor(TensorHandle handle);

    // --- Data Retrieval ---
    void get_tensor_data(TensorHandle handle, float* cpu_buffer);
    void get_tensor_grad(TensorHandle handle, float* cpu_buffer);

    // --- Operations ---
    TensorHandle add_tensors(TensorHandle a, TensorHandle b);
    TensorHandle sub_tensors(TensorHandle a, TensorHandle b);
    TensorHandle mul_tensors(TensorHandle a, TensorHandle b);
    TensorHandle div_tensors(TensorHandle a, TensorHandle b);
    TensorHandle neg_tensor(TensorHandle a);
    TensorHandle pow_tensor(TensorHandle a, float exponent);
    TensorHandle relu_tensor(TensorHandle a);
    TensorHandle tanh_tensor(TensorHandle a);
    TensorHandle exp_tensor(TensorHandle a);
    TensorHandle log_tensor(TensorHandle a);

    // --- Autograd ---
    void backward(TensorHandle handle);
}
