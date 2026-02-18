#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <unordered_set>
#include <functional>
#include <cmath>

#define DLLEXPORT __attribute__((visibility("default")))

// --- Global Kernels (Must be outside functions) ---
__global__ void add_fwd(float *a, float *b, float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = a[i] + b[i];
}
__global__ void add_bwd(float *go, float *ga, float *gb, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        atomicAdd(&ga[i], go[i]);
        atomicAdd(&gb[i], go[i]);
    }
}
__global__ void sub_fwd(float *a, float *b, float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = a[i] - b[i];
}
__global__ void sub_bwd(float *go, float *ga, float *gb, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        atomicAdd(&ga[i], go[i]);
        atomicAdd(&gb[i], -go[i]);
    }
}
__global__ void mul_fwd(float *a, float *b, float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = a[i] * b[i];
}
__global__ void mul_bwd(float *da, float *db, float *go, float *ga, float *gb, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        atomicAdd(&ga[i], db[i] * go[i]);
        atomicAdd(&gb[i], da[i] * go[i]);
    }
}
__global__ void div_fwd(float *a, float *b, float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = a[i] / b[i];
}
__global__ void div_bwd(float *da, float *db, float *go, float *ga, float *gb, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        atomicAdd(&ga[i], go[i] / db[i]);
        atomicAdd(&gb[i], -go[i] * da[i] / (db[i] * db[i]));
    }
}
__global__ void pow_fwd(float *a, float exp, float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = powf(a[i], exp);
}
__global__ void pow_bwd(float *da, float exp, float *go, float *ga, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        atomicAdd(&ga[i], exp * powf(da[i], exp - 1.0f) * go[i]);
}
__global__ void relu_fwd(float *a, float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = fmaxf(0.0f, a[i]);
}
__global__ void relu_bwd(float *da, float *go, float *ga, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        atomicAdd(&ga[i], (da[i] > 0 ? 1.0f : 0.0f) * go[i]);
}
__global__ void sigmoid_fwd(float *a, float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = 1.0f / (1.0f + expf(-a[i]));
}
__global__ void sigmoid_bwd(float *out, float *go, float *ga, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        float s = out[i];
        atomicAdd(&ga[i], s * (1.0f - s) * go[i]);
    }
}
__global__ void tanh_fwd(float *a, float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = tanhf(a[i]);
}
__global__ void tanh_bwd(float *out, float *go, float *ga, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        float t = out[i];
        atomicAdd(&ga[i], (1.0f - t * t) * go[i]);
    }
}
__global__ void step_kernel(float *data, float *grad, float lr, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        data[i] -= lr * grad[i];
        grad[i] = 0;
    }
}
__global__ void matmul_fwd(float *A, float *B, float *C, int M, int K, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N)
    {
        float sum = 0;
        for (int i = 0; i < K; i++)
            sum += A[row * K + i] * B[i * N + col];
        C[row * N + col] = sum;
    }
}
__global__ void matmul_bwd(float *A, float *B, float *dO, float *dA, float *dB, int M, int K, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N)
    {
        float g = dO[row * N + col];
        for (int i = 0; i < K; i++)
        {
            atomicAdd(&dA[row * K + i], B[i * N + col] * g);
            atomicAdd(&dB[i * N + col], A[row * K + i] * g);
        }
    }
}

__global__ void log_fwd(float *a, float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        // Epsilon prevents log(0)
        out[i] = logf(a[i] + 1e-7f);
    }
}

__global__ void log_bwd(float *a, float *go, float *ga, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        // Derivative of log(x) is 1/x
        atomicAdd(&ga[i], (1.0f / (a[i] + 1e-7f)) * go[i]);
    }
}

__global__ void aft_full_fwd(float *Q, float *K, float *V, float *WB, float *out,
                             int T, int D, bool masked)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x; // Current timestep
    if (t < T)
    {
        for (int d = 0; d < D; d++)
        {
            float numerator = 0.0f;
            float denominator = 0.0f;
            int limit = masked ? (t + 1) : T;

            for (int tp = 0; tp < limit; tp++)
            {
                // Bias index: w[t, tp]
                float w_ttp = WB[t * T + tp];
                // exp(K[tp, d] + w_ttp)
                float exp_weight = expf(K[tp * D + d] + w_ttp);

                numerator += exp_weight * V[tp * D + d];
                denominator += exp_weight;
            }
            // Y[t, d] = sigmoid(Q[t, d]) * (num / (den + epsilon))
            float sigQ = 1.0f / (1.0f + expf(-Q[t * D + d]));
            out[t * D + d] = sigQ * (numerator / (denominator + 1e-9f));
        }
    }
}

__global__ void aft_cross_fwd(float *Q, float *K, float *V, float *WB, float *out,
                              int TDec, int TEnc, int D)
{
    // Each thread handles one row (timestep) of the Decoder output
    int t = blockIdx.x * blockDim.x + threadIdx.x;

    if (t < TDec)
    {
        for (int d = 0; d < D; d++)
        {
            float numerator = 0.0f;
            float denominator = 0.0f;

            // Decoder position 't' attends to all Encoder positions 'tp'
            for (int tp = 0; tp < TEnc; tp++)
            {
                // posBias is [TDec, TEnc]
                float w_ttp = WB[t * TEnc + tp];

                // exp(K[tp, d] + w_ttp)
                float exp_weight = expf(K[tp * D + d] + w_ttp);

                numerator += exp_weight * V[tp * D + d];
                denominator += exp_weight;
            }

            // Sigmoid(Q) is applied here for efficiency
            float sigQ = 1.0f / (1.0f + expf(-Q[t * D + d]));
            out[t * D + d] = sigQ * (numerator / (denominator + 1e-9f));
        }
    }
}

__global__ void aft_cross_bwd(
    float* Q, float* K, float* V, float* WB, 
    float* grad_out, 
    float* grad_Q, float* grad_K, float* grad_V, float* grad_WB,
    int TDec, int TEnc, int D) 
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= TDec) return;

    for (int d = 0; d < D; d++) {
        // --- Re-calculate forward components for this specific d ---
        float num = 0.0f;
        float den = 0.0f;
        for (int tp = 0; tp < TEnc; tp++) {
            float weight = expf(K[tp * D + d] + WB[t * TEnc + tp]);
            num += weight * V[tp * D + d];
            den += weight;
        }
        
        float den_inv = 1.0f / (den + 1e-9f);
        float ratio = num * den_inv;
        float sigQ = 1.0f / (1.0f + expf(-Q[t * D + d]));
        float gO = grad_out[t * D + d];

        // 1. Gradient for Q: dL/dQ = gO * ratio * sigQ * (1 - sigQ)
        atomicAdd(&grad_Q[t * D + d], gO * ratio * sigQ * (1.0f - sigQ));

        // 2. Gradients for K, V, and WB (Loop through encoder steps)
        for (int tp = 0; tp < TEnc; tp++) {
            float weight = expf(K[tp * D + d] + WB[t * TEnc + tp]);
            
            // dL/dV = gO * sigQ * (weight / den)
            float dV = gO * sigQ * (weight * den_inv);
            atomicAdd(&grad_V[tp * D + d], dV);

            // dL/dWeight (common for K and WB)
            // Derivative of (num/den) w.r.t weight_i is: (V_i * den - num) / den^2
            float dW = gO * sigQ * weight * (V[tp * D + d] - ratio) * den_inv;
            
            atomicAdd(&grad_K[tp * D + d], dW);
            atomicAdd(&grad_WB[t * TEnc + tp], dW); // Note: WB sum is across D
        }
    }
}

extern "C"
{

    struct Tensor
    {
        float *data_gpu, *grad_gpu;
        int rows, cols, size;
        std::vector<Tensor *> _children;
        std::function<void()> _backward = []() {};
        Tensor(int r, int c) : rows(r), cols(c), size(r * c)
        {
            cudaMalloc(&data_gpu, size * sizeof(float));
            cudaMalloc(&grad_gpu, size * sizeof(float));
            cudaMemset(grad_gpu, 0, size * sizeof(float));
        }
        ~Tensor()
        {
            cudaFree(data_gpu);
            cudaFree(grad_gpu);
        }
    };

    DLLEXPORT void *create_tensor(int r, int c, float *d)
    {
        Tensor *t = new Tensor(r, c);
        if (d)
            cudaMemcpy(t->data_gpu, d, t->size * sizeof(float), cudaMemcpyHostToDevice);
        return (void *)t;
    }
    DLLEXPORT void destroy_tensor(void *h) { delete (Tensor *)h; }
    DLLEXPORT void get_tensor_data(void *h, float *b)
    {
        Tensor *t = (Tensor *)h;
        cudaMemcpy(b, t->data_gpu, t->size * sizeof(float), cudaMemcpyDeviceToHost);
    }
    DLLEXPORT void get_tensor_grad(void *h, float *b)
    {
        Tensor *t = (Tensor *)h;
        cudaMemcpy(b, t->grad_gpu, t->size * sizeof(float), cudaMemcpyDeviceToHost);
    }
    DLLEXPORT void zero_grad(void *h)
    {
        Tensor *t = (Tensor *)h;
        cudaMemset(t->grad_gpu, 0, t->size * sizeof(float));
    }
    DLLEXPORT void tensor_step(void *h, float lr)
    {
        Tensor *t = (Tensor *)h;
        step_kernel<<<(t->size + 255) / 256, 256>>>(t->data_gpu, t->grad_gpu, lr, t->size);
        cudaDeviceSynchronize();
    }

    DLLEXPORT void backward(void *h)
    {
        Tensor *t = (Tensor *)h;
        std::vector<Tensor *> topo;
        std::unordered_set<Tensor *> visited;
        std::function<void(Tensor *)> build = [&](Tensor *v)
        {
            if (visited.find(v) == visited.end())
            {
                visited.insert(v);
                for (Tensor *c : v->_children)
                    build(c);
                topo.push_back(v);
            }
        };
        build(t);
        std::vector<float> ones(t->size, 1.0f);
        cudaMemset(t->grad_gpu, 0, t->size * sizeof(float)); // Reset current grad
        cudaMemcpy(t->grad_gpu, ones.data(), t->size * sizeof(float), cudaMemcpyHostToDevice);
        for (auto it = topo.rbegin(); it != topo.rend(); ++it)
            (*it)->_backward();
    }

    // Op wrappers...
    DLLEXPORT void *add_tensors(void *ah, void *bh)
    {
        Tensor *a = (Tensor *)ah, *b = (Tensor *)bh;
        Tensor *out = new Tensor(a->rows, a->cols);
        out->_children = {a, b};
        add_fwd<<<(a->size + 255) / 256, 256>>>(a->data_gpu, b->data_gpu, out->data_gpu, a->size);
        out->_backward = [out, a, b]()
        { add_bwd<<<(a->size + 255) / 256, 256>>>(out->grad_gpu, a->grad_gpu, b->grad_gpu, a->size); };
        return (void *)out;
    }
    DLLEXPORT void *sub_tensors(void *ah, void *bh)
    {
        Tensor *a = (Tensor *)ah, *b = (Tensor *)bh;
        Tensor *out = new Tensor(a->rows, a->cols);
        out->_children = {a, b};
        sub_fwd<<<(a->size + 255) / 256, 256>>>(a->data_gpu, b->data_gpu, out->data_gpu, a->size);
        out->_backward = [out, a, b]()
        { sub_bwd<<<(a->size + 255) / 256, 256>>>(out->grad_gpu, a->grad_gpu, b->grad_gpu, a->size); };
        return (void *)out;
    }
    DLLEXPORT void *mul_tensors(void *ah, void *bh)
    {
        Tensor *a = (Tensor *)ah, *b = (Tensor *)bh;
        Tensor *out = new Tensor(a->rows, a->cols);
        out->_children = {a, b};
        mul_fwd<<<(a->size + 255) / 256, 256>>>(a->data_gpu, b->data_gpu, out->data_gpu, a->size);
        out->_backward = [out, a, b]()
        { mul_bwd<<<(a->size + 255) / 256, 256>>>(a->data_gpu, b->data_gpu, out->grad_gpu, a->grad_gpu, b->grad_gpu, a->size); };
        return (void *)out;
    }
    DLLEXPORT void *div_tensors(void *ah, void *bh)
    {
        Tensor *a = (Tensor *)ah, *b = (Tensor *)bh;
        Tensor *out = new Tensor(a->rows, a->cols);
        out->_children = {a, b};
        div_fwd<<<(a->size + 255) / 256, 256>>>(a->data_gpu, b->data_gpu, out->data_gpu, a->size);
        out->_backward = [out, a, b]()
        { div_bwd<<<(a->size + 255) / 256, 256>>>(a->data_gpu, b->data_gpu, out->grad_gpu, a->grad_gpu, b->grad_gpu, a->size); };
        return (void *)out;
    }
    DLLEXPORT void *pow_tensor(void *ah, float exp)
    {
        Tensor *a = (Tensor *)ah;
        Tensor *out = new Tensor(a->rows, a->cols);
        out->_children = {a};
        pow_fwd<<<(a->size + 255) / 256, 256>>>(a->data_gpu, exp, out->data_gpu, a->size);
        out->_backward = [out, a, exp]()
        { pow_bwd<<<(a->size + 255) / 256, 256>>>(a->data_gpu, exp, out->grad_gpu, a->grad_gpu, a->size); };
        return (void *)out;
    }
    DLLEXPORT void *relu_tensor(void *ah)
    {
        Tensor *a = (Tensor *)ah;
        Tensor *out = new Tensor(a->rows, a->cols);
        out->_children = {a};
        relu_fwd<<<(a->size + 255) / 256, 256>>>(a->data_gpu, out->data_gpu, a->size);
        out->_backward = [out, a]()
        { relu_bwd<<<(a->size + 255) / 256, 256>>>(a->data_gpu, out->grad_gpu, a->grad_gpu, a->size); };
        return (void *)out;
    }
    DLLEXPORT void *sigmoid_tensor(void *ah)
    {
        Tensor *a = (Tensor *)ah;
        Tensor *out = new Tensor(a->rows, a->cols);
        out->_children = {a};
        sigmoid_fwd<<<(a->size + 255) / 256, 256>>>(a->data_gpu, out->data_gpu, a->size);
        out->_backward = [out, a]()
        { sigmoid_bwd<<<(a->size + 255) / 256, 256>>>(out->data_gpu, out->grad_gpu, a->grad_gpu, a->size); };
        return (void *)out;
    }
    DLLEXPORT void *tanh_tensor(void *ah)
    {
        Tensor *a = (Tensor *)ah;
        Tensor *out = new Tensor(a->rows, a->cols);
        out->_children = {a};
        tanh_fwd<<<(a->size + 255) / 256, 256>>>(a->data_gpu, out->data_gpu, a->size);
        out->_backward = [out, a]()
        { tanh_bwd<<<(a->size + 255) / 256, 256>>>(out->data_gpu, out->grad_gpu, a->grad_gpu, a->size); };
        return (void *)out;
    }
    DLLEXPORT void *matmul_tensors(void *ah, void *bh)
    {
        Tensor *a = (Tensor *)ah, *b = (Tensor *)bh;
        int M = a->rows, K = a->cols, N = b->cols;
        Tensor *out = new Tensor(M, N);
        out->_children = {a, b};
        dim3 th(16, 16);
        dim3 bl((N + 15) / 16, (M + 15) / 16);
        matmul_fwd<<<bl, th>>>(a->data_gpu, b->data_gpu, out->data_gpu, M, K, N);
        out->_backward = [out, a, b, M, K, N]()
        { matmul_bwd<<<dim3((N + 15) / 16, (M + 15) / 16), dim3(16, 16)>>>(a->data_gpu, b->data_gpu, out->grad_gpu, a->grad_gpu, b->grad_gpu, M, K, N); };
        return (void *)out;
    }

    // Wrapper
    DLLEXPORT void *log_tensor(void *ah)
    {
        Tensor *a = (Tensor *)ah;
        Tensor *out = new Tensor(a->rows, a->cols);
        out->_children = {a};
        log_fwd<<<(a->size + 255) / 256, 256>>>(a->data_gpu, out->data_gpu, a->size);
        out->_backward = [out, a]()
        {
            log_bwd<<<(a->size + 255) / 256, 256>>>(a->data_gpu, out->grad_gpu, a->grad_gpu, a->size);
        };
        return (void *)out;
    }

    DLLEXPORT void *aft_forward(void *qh, void *kh, void *vh, void *wbh, bool masked)
    {
        Tensor *Q = (Tensor *)qh, *K = (Tensor *)kh, *V = (Tensor *)vh, *WB = (Tensor *)wbh;
        Tensor *out = new Tensor(Q->rows, Q->cols);
        out->_children = {Q, K, V, WB};

        int T = Q->rows;
        int D = Q->cols;

        aft_full_fwd<<<(T + 255) / 256, 256>>>(Q->data_gpu, K->data_gpu, V->data_gpu, WB->data_gpu, out->data_gpu, T, D, masked);

        // Note: AFT Backward is complex and usually requires a custom kernel as well.
        // For now, we focus on getting the Forward pass running on GPU.
        return (void *)out;
    }

    DLLEXPORT void* aft_cross_forward(void* qh, void* kh, void* vh, void* wbh) {
        Tensor *Q = (Tensor*)qh, *K = (Tensor*)kh, *V = (Tensor*)vh, *WB = (Tensor*)wbh;
        Tensor* out = new Tensor(Q->rows, Q->cols);
        out->_children = {Q, K, V, WB};
    
        int TDec = Q->rows, TEnc = K->rows, D = Q->cols;
        aft_cross_fwd<<<(TDec + 255) / 256, 256>>>(Q->data_gpu, K->data_gpu, V->data_gpu, WB->data_gpu, out->data_gpu, TDec, TEnc, D);
        
        // Attach the autograd backward function
        out->_backward = [out, Q, K, V, WB, TDec, TEnc, D]() {
            aft_cross_bwd<<<(TDec + 255) / 256, 256>>>(
                Q->data_gpu, K->data_gpu, V->data_gpu, WB->data_gpu, 
                out->grad_gpu, 
                Q->grad_gpu, K->grad_gpu, V->grad_gpu, WB->grad_gpu, 
                TDec, TEnc, D
            );
        };
        return (void*)out;
    }
}