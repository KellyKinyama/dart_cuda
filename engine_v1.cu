#include "engine.h"
#include <cstdio>
#include <vector>
#include <unordered_set>
#include <functional>
#include <cmath>

#define DLLEXPORT __attribute__((visibility("default")))

// --- CUDA Kernels (No changes here) ---

__global__ void add_forward_kernel(float* a, float* b, float* out, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) out[i] = a[i] + b[i]; }
__global__ void add_backward_kernel(float* grad_out, float* grad_a, float* grad_b, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) { atomicAdd(&grad_a[i], grad_out[i]); atomicAdd(&grad_b[i], grad_out[i]); } }
__global__ void sub_forward_kernel(float* a, float* b, float* out, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) out[i] = a[i] - b[i]; }
__global__ void sub_backward_kernel(float* grad_out, float* grad_a, float* grad_b, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) { atomicAdd(&grad_a[i], grad_out[i]); atomicAdd(&grad_b[i], -grad_out[i]); } }
__global__ void mul_forward_kernel(float* a, float* b, float* out, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) out[i] = a[i] * b[i]; }
__global__ void mul_backward_kernel(float* data_a, float* data_b, float* grad_out, float* grad_a, float* grad_b, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) { atomicAdd(&grad_a[i], data_b[i] * grad_out[i]); atomicAdd(&grad_b[i], data_a[i] * grad_out[i]); } }
__global__ void div_forward_kernel(float* a, float* b, float* out, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) out[i] = a[i] / b[i]; }
__global__ void div_backward_kernel(float* data_a, float* data_b, float* grad_out, float* grad_a, float* grad_b, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) { float b_val = data_b[i]; atomicAdd(&grad_a[i], (1.0f / b_val) * grad_out[i]); atomicAdd(&grad_b[i], (-data_a[i] / (b_val * b_val)) * grad_out[i]); } }
__global__ void neg_forward_kernel(float* a, float* out, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) out[i] = -a[i]; }
__global__ void neg_backward_kernel(float* grad_out, float* grad_a, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) atomicAdd(&grad_a[i], -grad_out[i]); }
__global__ void pow_forward_kernel(float* a, float exponent, float* out, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) out[i] = powf(a[i], exponent); }
__global__ void pow_backward_kernel(float* data_a, float exponent, float* grad_out, float* grad_a, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) atomicAdd(&grad_a[i], (exponent * powf(data_a[i], exponent - 1.0f)) * grad_out[i]); }
__global__ void relu_forward_kernel(float* a, float* out, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) out[i] = a[i] > 0.0f ? a[i] : 0.0f; }
__global__ void relu_backward_kernel(float* data_a, float* grad_out, float* grad_a, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) if (data_a[i] > 0.0f) atomicAdd(&grad_a[i], grad_out[i]); }
__global__ void tanh_forward_kernel(float* a, float* out, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) out[i] = tanhf(a[i]); }
__global__ void tanh_backward_kernel(float* data_out, float* grad_out, float* grad_a, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) { float out_val = data_out[i]; atomicAdd(&grad_a[i], (1.0f - out_val * out_val) * grad_out[i]); } }
__global__ void exp_forward_kernel(float* a, float* out, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) out[i] = expf(a[i]); }
__global__ void exp_backward_kernel(float* data_out, float* grad_out, float* grad_a, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) atomicAdd(&grad_a[i], data_out[i] * grad_out[i]); }
__global__ void log_forward_kernel(float* a, float* out, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) out[i] = logf(a[i]); }
__global__ void log_backward_kernel(float* data_a, float* grad_out, float* grad_a, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) atomicAdd(&grad_a[i], (1.0f / data_a[i]) * grad_out[i]); }

// --- Tensor Struct ---
struct Tensor {
    float* data_gpu;
    float* grad_gpu;
    int rows, cols, size;
    std::vector<Tensor*> _children;
    std::function<void()> _backward = [](){};
};

// --- C API Implementation ---
extern "C" DLLEXPORT TensorHandle create_tensor(int rows, int cols, float* initial_data) {
    Tensor* t = new Tensor();
    t->rows = rows; t->cols = cols; t->size = rows * cols;
    size_t byte_size = t->size * sizeof(float);
    cudaMalloc(&t->data_gpu, byte_size);
    cudaMalloc(&t->grad_gpu, byte_size);
    cudaMemset(t->grad_gpu, 0, byte_size);
    if (initial_data != nullptr) cudaMemcpy(t->data_gpu, initial_data, byte_size, cudaMemcpyHostToDevice);
    return static_cast<TensorHandle>(t);
}
extern "C" DLLEXPORT void destroy_tensor(TensorHandle handle) {
    Tensor* t = static_cast<Tensor*>(handle);
    cudaFree(t->data_gpu); cudaFree(t->grad_gpu); delete t;
}
extern "C" DLLEXPORT void get_tensor_data(TensorHandle handle, float* cpu_buffer) {
    Tensor* t = static_cast<Tensor*>(handle);
    cudaMemcpy(cpu_buffer, t->data_gpu, t->size * sizeof(float), cudaMemcpyDeviceToHost);
}
extern "C" DLLEXPORT void get_tensor_grad(TensorHandle handle, float* cpu_buffer) {
    Tensor* t = static_cast<Tensor*>(handle);
    cudaMemcpy(cpu_buffer, t->grad_gpu, t->size * sizeof(float), cudaMemcpyDeviceToHost);
}
extern "C" DLLEXPORT void backward(TensorHandle handle) {
    Tensor* t = static_cast<Tensor*>(handle);
    std::vector<Tensor*> topo;
    std::unordered_set<Tensor*> visited;
    std::function<void(Tensor*)> build_topo = [&](Tensor* v) {
        if (visited.find(v) == visited.end()) {
            visited.insert(v);
            for (Tensor* child : v->_children) build_topo(child);
            topo.push_back(v);
        }
    };
    build_topo(t);
    cudaMemset(t->grad_gpu, 0, t->size * sizeof(float));
    float* one_data = new float[t->size];
    for(int i = 0; i < t->size; ++i) one_data[i] = 1.0;
    cudaMemcpy(t->grad_gpu, one_data, t->size * sizeof(float), cudaMemcpyHostToDevice);
    delete[] one_data;
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) (*it)->_backward();
}

// --- Operator Implementations (Written out explicitly) ---

extern "C" DLLEXPORT TensorHandle add_tensors(TensorHandle a_handle, TensorHandle b_handle) {
    Tensor* a = static_cast<Tensor*>(a_handle);
    Tensor* b = static_cast<Tensor*>(b_handle);
    Tensor* out = static_cast<Tensor*>(create_tensor(a->rows, a->cols, nullptr));
    out->_children = {a, b};
    int threads = 256; int blocks = (a->size + threads - 1) / threads;
    add_forward_kernel<<<blocks, threads>>>(a->data_gpu, b->data_gpu, out->data_gpu, a->size);
    out->_backward = [out, a, b, blocks, threads]() {
        add_backward_kernel<<<blocks, threads>>>(out->grad_gpu, a->grad_gpu, b->grad_gpu, a->size);
    };
    return static_cast<TensorHandle>(out);
}

extern "C" DLLEXPORT TensorHandle sub_tensors(TensorHandle a_handle, TensorHandle b_handle) {
    Tensor* a = static_cast<Tensor*>(a_handle);
    Tensor* b = static_cast<Tensor*>(b_handle);
    Tensor* out = static_cast<Tensor*>(create_tensor(a->rows, a->cols, nullptr));
    out->_children = {a, b};
    int threads = 256; int blocks = (a->size + threads - 1) / threads;
    sub_forward_kernel<<<blocks, threads>>>(a->data_gpu, b->data_gpu, out->data_gpu, a->size);
    out->_backward = [out, a, b, blocks, threads]() {
        sub_backward_kernel<<<blocks, threads>>>(out->grad_gpu, a->grad_gpu, b->grad_gpu, a->size);
    };
    return static_cast<TensorHandle>(out);
}

extern "C" DLLEXPORT TensorHandle mul_tensors(TensorHandle a_handle, TensorHandle b_handle) {
    Tensor* a = static_cast<Tensor*>(a_handle);
    Tensor* b = static_cast<Tensor*>(b_handle);
    Tensor* out = static_cast<Tensor*>(create_tensor(a->rows, a->cols, nullptr));
    out->_children = {a, b};
    int threads = 256; int blocks = (a->size + threads - 1) / threads;
    mul_forward_kernel<<<blocks, threads>>>(a->data_gpu, b->data_gpu, out->data_gpu, a->size);
    out->_backward = [out, a, b, blocks, threads]() {
        mul_backward_kernel<<<blocks, threads>>>(a->data_gpu, b->data_gpu, out->grad_gpu, a->grad_gpu, b->grad_gpu, a->size);
    };
    return static_cast<TensorHandle>(out);
}

extern "C" DLLEXPORT TensorHandle div_tensors(TensorHandle a_handle, TensorHandle b_handle) {
    Tensor* a = static_cast<Tensor*>(a_handle);
    Tensor* b = static_cast<Tensor*>(b_handle);
    Tensor* out = static_cast<Tensor*>(create_tensor(a->rows, a->cols, nullptr));
    out->_children = {a, b};
    int threads = 256; int blocks = (a->size + threads - 1) / threads;
    div_forward_kernel<<<blocks, threads>>>(a->data_gpu, b->data_gpu, out->data_gpu, a->size);
    out->_backward = [out, a, b, blocks, threads]() {
        div_backward_kernel<<<blocks, threads>>>(a->data_gpu, b->data_gpu, out->grad_gpu, a->grad_gpu, b->grad_gpu, a->size);
    };
    return static_cast<TensorHandle>(out);
}

extern "C" DLLEXPORT TensorHandle pow_tensor(TensorHandle a_handle, float exponent) {
    Tensor* a = static_cast<Tensor*>(a_handle);
    Tensor* out = static_cast<Tensor*>(create_tensor(a->rows, a->cols, nullptr));
    out->_children = {a};
    int threads = 256; int blocks = (a->size + threads - 1) / threads;
    pow_forward_kernel<<<blocks, threads>>>(a->data_gpu, exponent, out->data_gpu, a->size);
    out->_backward = [out, a, exponent, blocks, threads]() {
        pow_backward_kernel<<<blocks, threads>>>(a->data_gpu, exponent, out->grad_gpu, a->grad_gpu, a->size);
    };
    return static_cast<TensorHandle>(out);
}

extern "C" DLLEXPORT TensorHandle relu_tensor(TensorHandle a_handle) {
    Tensor* a = static_cast<Tensor*>(a_handle);
    Tensor* out = static_cast<Tensor*>(create_tensor(a->rows, a->cols, nullptr));
    out->_children = {a};
    int threads = 256; int blocks = (a->size + threads - 1) / threads;
    relu_forward_kernel<<<blocks, threads>>>(a->data_gpu, out->data_gpu, a->size);
    out->_backward = [out, a, blocks, threads]() {
        relu_backward_kernel<<<blocks, threads>>>(a->data_gpu, out->grad_gpu, a->grad_gpu, a->size);
    };
    return static_cast<TensorHandle>(out);
}

extern "C" DLLEXPORT TensorHandle tanh_tensor(TensorHandle a_handle) {
    Tensor* a = static_cast<Tensor*>(a_handle);
    Tensor* out = static_cast<Tensor*>(create_tensor(a->rows, a->cols, nullptr));
    out->_children = {a};
    int threads = 256; int blocks = (a->size + threads - 1) / threads;
    tanh_forward_kernel<<<blocks, threads>>>(a->data_gpu, out->data_gpu, a->size);
    out->_backward = [out, a, blocks, threads]() {
        tanh_backward_kernel<<<blocks, threads>>>(out->data_gpu, out->grad_gpu, a->grad_gpu, a->size);
    };
    return static_cast<TensorHandle>(out);
}

extern "C" DLLEXPORT TensorHandle exp_tensor(TensorHandle a_handle) {
    Tensor* a = static_cast<Tensor*>(a_handle);
    Tensor* out = static_cast<Tensor*>(create_tensor(a->rows, a->cols, nullptr));
    out->_children = {a};
    int threads = 256; int blocks = (a->size + threads - 1) / threads;
    exp_forward_kernel<<<blocks, threads>>>(a->data_gpu, out->data_gpu, a->size);
    out->_backward = [out, a, blocks, threads]() {
        exp_backward_kernel<<<blocks, threads>>>(out->data_gpu, out->grad_gpu, a->grad_gpu, a->size);
    };
    return static_cast<TensorHandle>(out);
}

extern "C" DLLEXPORT TensorHandle log_tensor(TensorHandle a_handle) {
    Tensor* a = static_cast<Tensor*>(a_handle);
    Tensor* out = static_cast<Tensor*>(create_tensor(a->rows, a->cols, nullptr));
    out->_children = {a};
    int threads = 256; int blocks = (a->size + threads - 1) / threads;
    log_forward_kernel<<<blocks, threads>>>(a->data_gpu, out->data_gpu, a->size);
    out->_backward = [out, a, blocks, threads]() {
        log_backward_kernel<<<blocks, threads>>>(a->data_gpu, out->grad_gpu, a->grad_gpu, a->size);
    };
    return static_cast<TensorHandle>(out);
}

extern "C" DLLEXPORT TensorHandle neg_tensor(TensorHandle a_handle) {
    Tensor* a = static_cast<Tensor*>(a_handle);
    Tensor* out = static_cast<Tensor*>(create_tensor(a->rows, a->cols, nullptr));
    out->_children = {a};
    int threads = 256; int blocks = (a->size + threads - 1) / threads;
    neg_forward_kernel<<<blocks, threads>>>(a->data_gpu, out->data_gpu, a->size);
    out->_backward = [out, a, blocks, threads]() {
        neg_backward_kernel<<<blocks, threads>>>(out->grad_gpu, a->grad_gpu, a->size);
    };
    return static_cast<TensorHandle>(out);
}
