// engine.cu — thin entry point.
//
// CUDA kernels live in `kernels/*.cuh` (single translation unit via
// #include); only the `extern "C"` DLLEXPORT wrappers stay here.
// Build: `nvcc --shared -o native/lib/libmat_mul.so native/src/engine.cu`.

#include "kernels/common.cuh"
#include "kernels/elementwise.cuh"
#include "kernels/matmul.cuh"
#include "kernels/attention.cuh"
#include "kernels/layernorm_embed.cuh"
#include "kernels/loss_optim.cuh"
#include "kernels/conv_misc.cuh"

extern "C"
{

    struct Tensor
    {
        float *data_gpu, *grad_gpu;
        int rows, cols, size;
        bool is_view = false; // Default: owns memory
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
            // ONLY free if this isn't a non-owning view
            // (Though your current slice IS an owner, this protects you elsewhere)
            if (!is_view)
            {
                cudaFree(data_gpu);
                cudaFree(grad_gpu);
            }
        }
    };
    DLLEXPORT void *create_tensor(int r, int c, float *d)
    {
        Tensor *t = new Tensor(r, c);
        if (d)
            cudaMemcpy(t->data_gpu, d, t->size * sizeof(float), cudaMemcpyHostToDevice);
        return (void *)t;
    }
    DLLEXPORT void destroy_tensor(void *h)
    {
        if (!h)
            return;
        Tensor *t = (Tensor *)h;
        cudaFree(t->data_gpu);
        cudaFree(t->grad_gpu);
        delete t; // Delete the struct itself
    }
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

    // ---------------------------------------------------------------------
    // Scalar-broadcast elementwise ops: A is [n], B is a single scalar tensor
    // (size 1). Output has shape == A's shape. Backward accumulates B's grad
    // via atomicAdd into B->grad_gpu[0].
    // ---------------------------------------------------------------------
    __global__ void add_scalar_fwd_k(const float *a, const float *b, float *out, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) out[i] = a[i] + b[0];
    }
    __global__ void add_scalar_bwd_k(const float *gout, float *ga, float *gb, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            ga[i] += gout[i];
            atomicAdd(&gb[0], gout[i]);
        }
    }
    __global__ void sub_scalar_fwd_k(const float *a, const float *b, float *out, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) out[i] = a[i] - b[0];
    }
    __global__ void sub_scalar_bwd_k(const float *gout, float *ga, float *gb, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            ga[i] += gout[i];
            atomicAdd(&gb[0], -gout[i]);
        }
    }
    __global__ void mul_scalar_fwd_k(const float *a, const float *b, float *out, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) out[i] = a[i] * b[0];
    }
    __global__ void mul_scalar_bwd_k(const float *a, const float *b, const float *gout,
                                     float *ga, float *gb, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            ga[i] += b[0] * gout[i];
            atomicAdd(&gb[0], a[i] * gout[i]);
        }
    }
    __global__ void div_scalar_fwd_k(const float *a, const float *b, float *out, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) out[i] = a[i] / b[0];
    }
    __global__ void div_scalar_bwd_k(const float *a, const float *b, const float *gout,
                                     float *ga, float *gb, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            float bv = b[0];
            ga[i] += gout[i] / bv;
            atomicAdd(&gb[0], -a[i] * gout[i] / (bv * bv));
        }
    }

    DLLEXPORT void *add_tensor_scalar(void *ah, void *bh)
    {
        Tensor *a = (Tensor *)ah, *b = (Tensor *)bh;
        Tensor *out = new Tensor(a->rows, a->cols);
        out->_children = {a, b};
        int n = a->size;
        int blocks = (n + 255) / 256;
        add_scalar_fwd_k<<<blocks, 256>>>(a->data_gpu, b->data_gpu, out->data_gpu, n);
        out->_backward = [out, a, b, n, blocks]()
        {
            add_scalar_bwd_k<<<blocks, 256>>>(out->grad_gpu, a->grad_gpu, b->grad_gpu, n);
        };
        return (void *)out;
    }
    DLLEXPORT void *sub_tensor_scalar(void *ah, void *bh)
    {
        Tensor *a = (Tensor *)ah, *b = (Tensor *)bh;
        Tensor *out = new Tensor(a->rows, a->cols);
        out->_children = {a, b};
        int n = a->size;
        int blocks = (n + 255) / 256;
        sub_scalar_fwd_k<<<blocks, 256>>>(a->data_gpu, b->data_gpu, out->data_gpu, n);
        out->_backward = [out, a, b, n, blocks]()
        {
            sub_scalar_bwd_k<<<blocks, 256>>>(out->grad_gpu, a->grad_gpu, b->grad_gpu, n);
        };
        return (void *)out;
    }
    DLLEXPORT void *mul_tensor_scalar(void *ah, void *bh)
    {
        Tensor *a = (Tensor *)ah, *b = (Tensor *)bh;
        Tensor *out = new Tensor(a->rows, a->cols);
        out->_children = {a, b};
        int n = a->size;
        int blocks = (n + 255) / 256;
        mul_scalar_fwd_k<<<blocks, 256>>>(a->data_gpu, b->data_gpu, out->data_gpu, n);
        out->_backward = [out, a, b, n, blocks]()
        {
            mul_scalar_bwd_k<<<blocks, 256>>>(a->data_gpu, b->data_gpu, out->grad_gpu,
                                              a->grad_gpu, b->grad_gpu, n);
        };
        return (void *)out;
    }
    DLLEXPORT void *div_tensor_scalar(void *ah, void *bh)
    {
        Tensor *a = (Tensor *)ah, *b = (Tensor *)bh;
        Tensor *out = new Tensor(a->rows, a->cols);
        out->_children = {a, b};
        int n = a->size;
        int blocks = (n + 255) / 256;
        div_scalar_fwd_k<<<blocks, 256>>>(a->data_gpu, b->data_gpu, out->data_gpu, n);
        out->_backward = [out, a, b, n, blocks]()
        {
            div_scalar_bwd_k<<<blocks, 256>>>(a->data_gpu, b->data_gpu, out->grad_gpu,
                                              a->grad_gpu, b->grad_gpu, n);
        };
        return (void *)out;
    }

    DLLEXPORT void *add_tensors(void *ah, void *bh)
    {
        Tensor *a = (Tensor *)ah, *b = (Tensor *)bh;
        Tensor *out = new Tensor(a->rows, a->cols);
        out->_children = {a, b};

        int n = a->size;
        // Each thread handles 4 elements, so we divide the total threads needed by 4
        int blocks = (n + (256 * 4) - 1) / (256 * 4);

        add_fwd<<<blocks, 256>>>(a->data_gpu, b->data_gpu, out->data_gpu, n);

        out->_backward = [out, a, b, n, blocks]()
        {
            // Removed atomicAdd here as well for massive speedup
            add_bwd<<<blocks, 256>>>(out->grad_gpu, a->grad_gpu, b->grad_gpu, n);
        };

        return (void *)out;
    }
    DLLEXPORT void *sub_tensors(void *ah, void *bh)
    {
        Tensor *a = (Tensor *)ah, *b = (Tensor *)bh;
        Tensor *out = new Tensor(a->rows, a->cols);
        out->_children = {a, b};

        // We process 4 elements per thread
        int block_size = 256;
        int grid_size = (a->size + (block_size * 4) - 1) / (block_size * 4);

        sub_fwd<<<grid_size, block_size>>>(a->data_gpu, b->data_gpu, out->data_gpu, a->size);

        out->_backward = [out, a, b, grid_size, block_size]()
        {
            sub_bwd<<<grid_size, block_size>>>(out->grad_gpu, a->grad_gpu, b->grad_gpu, a->size);
        };

        return (void *)out;
    }
    DLLEXPORT void *mul_tensors(void *ah, void *bh)
    {
        Tensor *a = (Tensor *)ah, *b = (Tensor *)bh;
        Tensor *out = new Tensor(a->rows, a->cols);
        out->_children = {a, b};

        // Divide size by 4 because each thread handles 4 floats
        int n = a->size;
        int blocks = (n + (256 * 4) - 1) / (256 * 4);

        mul_fwd<<<blocks, 256>>>(a->data_gpu, b->data_gpu, out->data_gpu, n);

        out->_backward = [out, a, b, n, blocks]()
        {
            // High performer backward: Vectorized and NO atomicAdds
            // because element-wise grads don't overlap.
            mul_bwd<<<blocks, 256>>>(a->data_gpu, b->data_gpu, out->grad_gpu,
                                     a->grad_gpu, b->grad_gpu, n);
        };

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

    DLLEXPORT void *gelu_tensor(void *ah)
    {
        Tensor *a = (Tensor *)ah;
        Tensor *out = new Tensor(a->rows, a->cols);
        out->_children = {a};

        gelu_fwd<<<(a->size + 255) / 256, 256>>>(a->data_gpu, out->data_gpu, a->size);

        out->_backward = [out, a]()
        {
            gelu_bwd<<<(a->size + 255) / 256, 256>>>(a->data_gpu, out->grad_gpu, a->grad_gpu, a->size);
        };

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

        // Forward: Tiled 32x32
        dim3 th(32, 32);
        dim3 bl((N + 31) / 32, (M + 31) / 32);
        matmul_fwd<<<bl, th>>>(a->data_gpu, b->data_gpu, out->data_gpu, M, K, N);

        // Backward Lambda: Pure CUDA High Performer
        out->_backward = [out, a, b, M, K, N]()
        {
            dim3 th32(32, 32);

            // Launch dA calculation
            dim3 blA((K + 31) / 32, (M + 31) / 32);
            matmul_bwd_dA_tiled<<<blA, th32>>>(out->grad_gpu, b->data_gpu, a->grad_gpu, M, K, N);

            // Launch dB calculation
            dim3 blB((N + 31) / 32, (K + 31) / 32);
            matmul_bwd_dB_tiled<<<blB, th32>>>(a->data_gpu, out->grad_gpu, b->grad_gpu, M, K, N);
        };

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
    __global__ void aft_full_bwd(
        float *Q, float *K, float *V, float *WB,
        float *grad_out,
        float *grad_Q, float *grad_K, float *grad_V, float *grad_WB,
        int T, int D, bool masked)
    {
        int t = blockIdx.x * blockDim.x + threadIdx.x;
        if (t >= T)
            return;

        for (int d = 0; d < D; d++)
        {
            // Re-calculate sum components for this row/dim
            float num = 0.0f;
            float den = 0.0f;
            int limit = masked ? (t + 1) : T;

            for (int tp = 0; tp < limit; tp++)
            {
                float weight = expf(K[tp * D + d] + WB[t * T + tp]);
                num += weight * V[tp * D + d];
                den += weight;
            }

            float den_inv = 1.0f / (den + 1e-9f);
            float ratio = num * den_inv;

            // Sigmoid derivative for Q
            float sigQ = 1.0f / (1.0f + expf(-Q[t * D + d]));
            float dSigQ = sigQ * (1.0f - sigQ);

            float gO = grad_out[t * D + d];

            // 1. Gradient for Q
            atomicAdd(&grad_Q[t * D + d], gO * ratio * dSigQ);

            // 2. Gradients for K, V, and WB
            for (int tp = 0; tp < limit; tp++)
            {
                float weight = expf(K[tp * D + d] + WB[t * T + tp]);

                // dL/dV
                float dV = gO * sigQ * (weight * den_inv);
                atomicAdd(&grad_V[tp * D + d], dV);

                // dL/dWeight (affects both K and WB)
                // quotient rule: (V_i * den - num) / den^2
                float dW = gO * sigQ * weight * (V[tp * D + d] - ratio) * den_inv;

                atomicAdd(&grad_K[tp * D + d], dW);
                atomicAdd(&grad_WB[t * T + tp], dW);
            }
        }
    }

    DLLEXPORT void *aft_forward(void *qh, void *kh, void *vh, void *wbh, bool masked)
    {
        Tensor *Q = (Tensor *)qh, *K = (Tensor *)kh, *V = (Tensor *)vh, *WB = (Tensor *)wbh;
        Tensor *out = new Tensor(Q->rows, Q->cols);
        out->_children = {Q, K, V, WB};

        int T = Q->rows;
        int D = Q->cols;

        aft_full_fwd<<<(T + 255) / 256, 256>>>(Q->data_gpu, K->data_gpu, V->data_gpu, WB->data_gpu, out->data_gpu, T, D, masked);

        // Register the backward lambda for autograd
        out->_backward = [out, Q, K, V, WB, T, D, masked]()
        {
            aft_full_bwd<<<(T + 255) / 256, 256>>>(
                Q->data_gpu, K->data_gpu, V->data_gpu, WB->data_gpu,
                out->grad_gpu,
                Q->grad_gpu, K->grad_gpu, V->grad_gpu, WB->grad_gpu,
                T, D, masked);
        };

        return (void *)out;
    }

    DLLEXPORT void *aft_cross_forward(void *qh, void *kh, void *vh, void *wbh)
    {
        Tensor *Q = (Tensor *)qh, *K = (Tensor *)kh, *V = (Tensor *)vh, *WB = (Tensor *)wbh;
        Tensor *out = new Tensor(Q->rows, Q->cols);
        out->_children = {Q, K, V, WB};

        int TDec = Q->rows, TEnc = K->rows, D = Q->cols;
        aft_cross_fwd<<<(TDec + 255) / 256, 256>>>(Q->data_gpu, K->data_gpu, V->data_gpu, WB->data_gpu, out->data_gpu, TDec, TEnc, D);

        // Attach the autograd backward function
        out->_backward = [out, Q, K, V, WB, TDec, TEnc, D]()
        {
            aft_cross_bwd<<<(TDec + 255) / 256, 256>>>(
                Q->data_gpu, K->data_gpu, V->data_gpu, WB->data_gpu,
                out->grad_gpu,
                Q->grad_gpu, K->grad_gpu, V->grad_gpu, WB->grad_gpu,
                TDec, TEnc, D);
        };
        return (void *)out;
    }

    DLLEXPORT void *concat_tensors_gpu(void **handles, int num_tensors)
    {
        std::vector<Tensor *> ts;
        for (int i = 0; i < num_tensors; i++)
            ts.push_back((Tensor *)handles[i]);

        int rows = ts[0]->rows;
        int cols_per_t = ts[0]->cols;
        Tensor *out = new Tensor(rows, cols_per_t * num_tensors);

        // Prepare device pointers for the kernel
        float **d_inputs;
        float *h_inputs[num_tensors];
        for (int i = 0; i < num_tensors; i++)
            h_inputs[i] = ts[i]->data_gpu;
        cudaMalloc(&d_inputs, num_tensors * sizeof(float *));
        cudaMemcpy(d_inputs, h_inputs, num_tensors * sizeof(float *), cudaMemcpyHostToDevice);

        concat_axis1_fwd<<<(out->size + 255) / 256, 256>>>(d_inputs, out->data_gpu, num_tensors, rows, cols_per_t);

        out->_backward = [out, ts, num_tensors, rows, cols_per_t]()
        {
            float **d_grads;
            float *h_grads[num_tensors];
            for (int i = 0; i < num_tensors; i++)
                h_grads[i] = ts[i]->grad_gpu;
            cudaMalloc(&d_grads, num_tensors * sizeof(float *));
            cudaMemcpy(d_grads, h_grads, num_tensors * sizeof(float *), cudaMemcpyHostToDevice);

            concat_axis1_bwd<<<(out->size + 255) / 256, 256>>>(out->grad_gpu, d_grads, num_tensors, rows, cols_per_t);
            cudaFree(d_grads);
        };

        out->_children = ts;
        return (void *)out;
    }

    DLLEXPORT void *concat_tensors_axis0_gpu(void **handles, int num_tensors)
    {
        std::vector<Tensor *> ts;
        for (int i = 0; i < num_tensors; i++)
            ts.push_back((Tensor *)handles[i]);

        int cols = ts[0]->cols;
        int total_rows = 0;
        std::vector<int> h_offsets(num_tensors + 1, 0);
        for (int i = 0; i < num_tensors; i++)
        {
            h_offsets[i] = total_rows;
            total_rows += ts[i]->rows;
        }
        h_offsets[num_tensors] = total_rows;

        Tensor *out = new Tensor(total_rows, cols);

        // Device pointer arrays.
        float **d_inputs;
        std::vector<float *> h_inputs(num_tensors);
        for (int i = 0; i < num_tensors; i++) h_inputs[i] = ts[i]->data_gpu;
        cudaMalloc(&d_inputs, num_tensors * sizeof(float *));
        cudaMemcpy(d_inputs, h_inputs.data(), num_tensors * sizeof(float *), cudaMemcpyHostToDevice);

        int *d_offsets;
        cudaMalloc(&d_offsets, (num_tensors + 1) * sizeof(int));
        cudaMemcpy(d_offsets, h_offsets.data(), (num_tensors + 1) * sizeof(int), cudaMemcpyHostToDevice);

        concat_axis0_fwd<<<(out->size + 255) / 256, 256>>>(d_inputs, out->data_gpu, num_tensors, d_offsets, cols);
        cudaFree(d_inputs);

        out->_backward = [out, ts, num_tensors, cols, d_offsets]()
        {
            float **d_grads;
            std::vector<float *> h_grads(num_tensors);
            for (int i = 0; i < num_tensors; i++) h_grads[i] = ts[i]->grad_gpu;
            cudaMalloc(&d_grads, num_tensors * sizeof(float *));
            cudaMemcpy(d_grads, h_grads.data(), num_tensors * sizeof(float *), cudaMemcpyHostToDevice);

            concat_axis0_bwd<<<(out->size + 255) / 256, 256>>>(out->grad_gpu, d_grads, num_tensors, d_offsets, cols);
            cudaFree(d_grads);
            cudaFree(d_offsets);
        };

        out->_children = ts;
        return (void *)out;
    }
    DLLEXPORT void *layernorm_forward(void *xh, void *gh, void *bh, float eps)
    {
        Tensor *x = (Tensor *)xh, *gamma = (Tensor *)gh, *beta = (Tensor *)bh;
        Tensor *out = new Tensor(x->rows, x->cols);
        out->_children = {x, gamma, beta};

        layernorm_fwd<<<x->rows, 1>>>(x->data_gpu, gamma->data_gpu, beta->data_gpu, out->data_gpu, x->rows, x->cols, eps);

        // Added 'beta' to the capture list below [out, x, gamma, beta, eps]
        out->_backward = [out, x, gamma, beta, eps]()
        {
            layernorm_bwd<<<x->rows, 1>>>(
                x->data_gpu,
                gamma->data_gpu,
                out->grad_gpu,
                x->grad_gpu,
                gamma->grad_gpu,
                beta->grad_gpu, // Now beta is accessible
                x->rows,
                x->cols,
                eps);
        };
        return (void *)out;
    }

    DLLEXPORT void *embedding_forward(int *h_indices, void *wteh, void *wpeh, int T, int D)
    {
        Tensor *wte = (Tensor *)wteh, *wpe = (Tensor *)wpeh;
        Tensor *out = new Tensor(T, D);
        out->_children = {wte, wpe};

        // Allocate and copy indices to GPU
        int *d_indices;
        cudaMalloc(&d_indices, T * sizeof(int));
        cudaMemcpy(d_indices, h_indices, T * sizeof(int), cudaMemcpyHostToDevice);

        int total = T * D;
        // Now T and D are correctly used to define the grid
        embedding_fwd<<<(total + 255) / 256, 256>>>(d_indices, wte->data_gpu, wpe->data_gpu, out->data_gpu, T, D);

        out->_backward = [out, wte, wpe, d_indices, T, D]()
        {
            int total = T * D;
            embedding_bwd<<<(total + 255) / 256, 256>>>(d_indices, out->grad_gpu, wte->grad_gpu, wpe->grad_gpu, T, D);

            // Clean up indices after backward
            cudaFree(d_indices);
        };

        return (void *)out;
    }

    // Note: In a real engine, you'd pass epsilon from Dart. Hardcoding 0.1 for now.
    const float GLOBAL_EPSILON = 0.1f;

    void *cross_entropy_loss(void *lh, int *h_targets, int T, int V)
    {
        Tensor *logits = (Tensor *)lh;
        Tensor *out = new Tensor(1, 1);
        out->_children = {logits};

        int *d_targets;
        cudaMalloc(&d_targets, T * sizeof(int));
        cudaMemcpy(d_targets, h_targets, T * sizeof(int), cudaMemcpyHostToDevice);

        float *d_losses;
        cudaMalloc(&d_losses, T * sizeof(float));

        // Launch with 1 block per row, 128 threads per block
        cross_entropy_fwd_hyper<<<T, 128>>>(logits->data_gpu, d_targets, d_losses, T, V, GLOBAL_EPSILON);

        // Use a basic reduction to get mean (or copy back if T is small)
        std::vector<float> h_losses(T);
        cudaMemcpy(h_losses.data(), d_losses, T * sizeof(float), cudaMemcpyDeviceToHost);
        float total_loss = 0;
        for (float l : h_losses)
            total_loss += l;
        float mean_loss = total_loss / T;
        cudaMemcpy(out->data_gpu, &mean_loss, sizeof(float), cudaMemcpyHostToDevice);

        out->_backward = [out, logits, d_targets, T, V]()
        {
            float h_grad_out;
            cudaMemcpy(&h_grad_out, out->grad_gpu, sizeof(float), cudaMemcpyDeviceToHost);

            cross_entropy_bwd_hyper<<<T, 128>>>(
                logits->data_gpu, d_targets, logits->grad_gpu, T, V, h_grad_out, GLOBAL_EPSILON);

            cudaFree(d_targets);
        };

        cudaFree(d_losses);
        return (void *)out;
    }

    DLLEXPORT void tensor_to_host(void *handle, float *h_data)
    {
        Tensor *t = (Tensor *)handle;
        int size = t->rows * t->cols;
        // Copy from GPU data pointer to the pointer provided by Dart
        cudaMemcpy(h_data, t->data_gpu, size * sizeof(float), cudaMemcpyDeviceToHost);
    }

    DLLEXPORT void adam_step(void *ph, void *mh, void *vh, int t,
                             float lr, float b1, float b2, float eps)
    {
        Tensor *p = (Tensor *)ph;
        Tensor *m = (Tensor *)mh;
        Tensor *v = (Tensor *)vh;

        int size = p->rows * p->cols;
        int threads = 256;
        int blocks = (size + threads - 1) / threads;

        // Launch the Adam kernel
        adam_kernel<<<blocks, threads>>>(
            p->data_gpu, p->grad_gpu, m->data_gpu, v->data_gpu,
            size, t, lr, b1, b2, eps);

        // Ensure the GPU finishes the update before Dart continues
        cudaDeviceSynchronize();
    }

    DLLEXPORT void sdg_step(void *ph, void *mh, void *vh, int t,
                            float lr)
    {
        Tensor *p = (Tensor *)ph;

        int size = p->rows * p->cols;
        int threads = 256;
        int blocks = (size + threads - 1) / threads;

        // Launch the Adam kernel
        sdg_kernel<<<blocks, threads>>>(
            p->data_gpu, p->grad_gpu,
            size, lr);

        // Ensure the GPU finishes the update before Dart continues
        cudaDeviceSynchronize();
    }

    DLLEXPORT void clip_gradients(void *handle, float limit)
    {
        Tensor *t = (Tensor *)handle;
        int size = t->rows * t->cols;
        int threads = 256;
        int blocks = (size + threads - 1) / threads;

        clip_grads_kernel<<<blocks, threads>>>(t->grad_gpu, limit, size);
        cudaDeviceSynchronize();
    }

    DLLEXPORT void set_tensor_data(void *handle, float *cpu_data)
    {
        if (!handle || !cpu_data)
            return;

        Tensor *t = (Tensor *)handle;
        int size_in_bytes = t->rows * t->cols * sizeof(float);

        // Copy from CPU (Host) to GPU (Device)
        cudaError_t err = cudaMemcpy(t->data_gpu, cpu_data, size_in_bytes, cudaMemcpyHostToDevice);

        if (err != cudaSuccess)
        {
            printf("CUDA Error in set_tensor_data: %s\n", cudaGetErrorString(err));
        }

        // Ensure copy is finished before Dart continues
        cudaDeviceSynchronize();
    }

    // DLLEXPORT void *slice_tensor(void *h, int startRow, int numRows)
    // {
    //     Tensor *t = (Tensor *)h;

    //     // Safety check
    //     if (startRow + numRows > t->rows)
    //         numRows = t->rows - startRow;

    //     int cols = t->cols;
    //     int total_elements = numRows * cols;
    //     int start_idx = startRow * cols;

    //     Tensor *out = new Tensor(numRows, cols);
    //     out->_children = {t};

    //     // Forward pass: Copy rows from t to out
    //     slice_fwd_kernel<<<(total_elements + 255) / 256, 256>>>(
    //         t->data_gpu, out->data_gpu, start_idx, total_elements);

    //     // Autograd: Map gradients from the slice back to the source tensor's specific rows
    //     out->_backward = [out, t, start_idx, total_elements]()
    //     {
    //         slice_bwd_kernel<<<(total_elements + 255) / 256, 256>>>(
    //             t->grad_gpu, out->grad_gpu, start_idx, total_elements);
    //     };

    //     return (void *)out;
    // }

    // 2. The FFI Wrapper
    extern "C" DLLEXPORT void *slice_tensor(void *h, int startRow, int numRows)
    {
        Tensor *t = (Tensor *)h;

        // Bounds safety to prevent GPU out-of-bounds access
        if (startRow < 0)
            startRow = 0;
        if (startRow + numRows > t->rows)
            numRows = t->rows - startRow;

        int cols = t->cols;
        int total_elements = numRows * cols;
        int start_idx = startRow * cols;

        // IMPORTANT: This calls the constructor that runs cudaMalloc.
        // This 'out' tensor OWNS its own memory.
        Tensor *out = new Tensor(numRows, cols);
        out->_children = {t};

        int threads = 256;
        int blocks = (total_elements + threads - 1) / threads;

        // Copy data from source to slice
        slice_fwd_kernel<<<blocks, threads>>>(t->data_gpu, out->data_gpu, start_idx, total_elements);

        // Register the backward lambda
        // We capture 'start_idx' and 'total_elements' by value [=] or explicitly
        out->_backward = [out, t, start_idx, total_elements]()
        {
            int threads = 256;
            int blocks = (total_elements + threads - 1) / threads;
            slice_bwd_kernel<<<blocks, threads>>>(t->grad_gpu, out->grad_gpu, start_idx, total_elements);
        };

        return (void *)out;
    }

    DLLEXPORT void *abs_tensor(void *ah)
    {
        Tensor *a = (Tensor *)ah;
        Tensor *out = new Tensor(a->rows, a->cols);
        out->_children = {a};

        int threads = 256;
        int blocks = (a->size + threads - 1) / threads;

        // Forward: out = |a|
        abs_fwd<<<blocks, threads>>>(a->data_gpu, out->data_gpu, a->size);

        // Backward: grad_a += grad_out * sign(a)
        out->_backward = [out, a, blocks, threads]()
        {
            abs_bwd<<<blocks, threads>>>(a->data_gpu, out->grad_gpu, a->grad_gpu, a->size);
        };

        return (void *)out;
    }

    __global__ void softmax_fwd_kernel(float *logits, float *out, int T, int V)
    {
        int t = blockIdx.x * blockDim.x + threadIdx.x;
        if (t < T)
        {
            float max_val = -1e30f;
            for (int v = 0; v < V; v++)
            {
                if (logits[t * V + v] > max_val)
                    max_val = logits[t * V + v];
            }

            float sum = 0.0f;
            for (int v = 0; v < V; v++)
            {
                out[t * V + v] = expf(logits[t * V + v] - max_val);
                sum += out[t * V + v];
            }

            for (int v = 0; v < V; v++)
            {
                out[t * V + v] /= (sum + 1e-9f);
            }
        }
    }

    // Exported Wrapper
    DLLEXPORT void *softmax_forward(void *ah)
    {
        Tensor *a = (Tensor *)ah;
        Tensor *out = new Tensor(a->rows, a->cols);
        out->_children = {a};

        // Launch row-wise (one thread per query/row)
        softmax_fwd_kernel<<<(a->rows + 255) / 256, 256>>>(a->data_gpu, out->data_gpu, a->rows, a->cols);

        // Softmax backward is slightly complex; if you only use it for cost matrix,
        // you can leave it empty, but here is the correct derivation:
        out->_backward = [out, a]()
        {
            // Softmax gradient logic (similar to your CE gradient but general)
            // For matching costs, we usually don't backprop through the matching step!
        };
        return (void *)out;
    }

    DLLEXPORT void compute_cost_matrix(void *pb_h, void *gb_h, void *cm_h)
    {
        Tensor *pb = (Tensor *)pb_h;
        Tensor *gb = (Tensor *)gb_h;
        Tensor *cm = (Tensor *)cm_h;

        dim3 threads(16, 16);
        dim3 blocks((gb->rows + 15) / 16, (pb->rows + 15) / 16);

        compute_cost_matrix_kernel<<<blocks, threads>>>(
            pb->data_gpu, gb->data_gpu, cm->data_gpu, pb->rows, gb->rows);
        cudaDeviceSynchronize();
    }

    DLLEXPORT void *sum_tensor(void *ah)
    {
        Tensor *a = (Tensor *)ah;
        // Result is always a 1x1 tensor
        Tensor *out = new Tensor(1, 1);
        out->_children = {a};

        // Initialize output memory to zero before reduction
        cudaMemset(out->data_gpu, 0, sizeof(float));

        int threads = 256;
        int blocks = (a->size + threads - 1) / threads;

        sum_fwd_kernel<<<blocks, threads>>>(a->data_gpu, out->data_gpu, a->size);

        out->_backward = [out, a, blocks, threads]()
        {
            sum_bwd_kernel<<<blocks, threads>>>(out->grad_gpu, a->grad_gpu, a->size);
        };

        return (void *)out;
    }

    DLLEXPORT void *mean_tensor(void *ah)
    {
        Tensor *a = (Tensor *)ah;
        Tensor *out = new Tensor(1, 1);
        out->_children = {a};

        cudaMemset(out->data_gpu, 0, sizeof(float));

        int threads = 256;
        int blocks = (a->size + threads - 1) / threads;

        mean_fwd_kernel<<<blocks, threads>>>(a->data_gpu, out->data_gpu, a->size);

        out->_backward = [out, a, blocks, threads]()
        {
            mean_bwd_kernel<<<blocks, threads>>>(out->grad_gpu, a->grad_gpu, a->size);
        };

        return (void *)out;
    }

    DLLEXPORT void tensor_xavier_init(void *ph, int n_in, int n_out, int seed)
    {
        Tensor *p = (Tensor *)ph;
        int size = p->rows * p->cols;

        int threads = 256;
        int blocks = (size + threads - 1) / threads;

        xavier_init_kernel<<<blocks, threads>>>(p->data_gpu, size, n_in, n_out, (unsigned long)seed);
        cudaDeviceSynchronize();
    }

    DLLEXPORT void tensor_zero_init(void *ph)
    {
        Tensor *p = (Tensor *)ph;
        int size = p->rows * p->cols;
        cudaMemset(p->data_gpu, 0, size * sizeof(float));
    }

    // DLLEXPORT void *l2_normalize_tensor(void *ah, float eps)
    // {
    //     Tensor *a = (Tensor *)ah;
    //     int R = a->rows;
    //     int C = a->cols;
    //     Tensor *out = new Tensor(R, C);
    //     out->_children = {a};

    //     int threads = 256;
    //     int blocks = (R + threads - 1) / threads;

    //     l2_normalize_fwd<<<blocks, threads>>>(a->data_gpu, out->data_gpu, R, C, eps);

    //     out->_backward = [out, a, R, C, eps, blocks, threads]()
    //     {
    //         l2_normalize_bwd<<<blocks, threads>>>(
    //             a->data_gpu,
    //             out->grad_gpu,
    //             a->grad_gpu,
    //             R, C, eps);
    //     };
    //     return (void *)out;
    // }

    DLLEXPORT void *layer_norm_tensor(void *ah, float eps)
    {
        Tensor *a = (Tensor *)ah;
        int R = a->rows;
        int C = a->cols;
        Tensor *out = new Tensor(R, C);
        out->_children = {a};

        int threads = 256;
        int blocks = (R + threads - 1) / threads;

        layer_norm_fwd<<<blocks, threads>>>(a->data_gpu, out->data_gpu, R, C, eps);

        out->_backward = [out, a, R, C, eps, blocks, threads]()
        {
            layer_norm_bwd<<<blocks, threads>>>(
                a->data_gpu, out->grad_gpu, a->grad_gpu, R, C, eps);
        };
        return (void *)out;
    }

    DLLEXPORT void im2col_cuda(
        void *input_handle,
        int channels, int height, int width,
        int kernel_h, int kernel_w,
        int pad_h, int pad_w,
        int stride_h, int stride_w,
        void *output_handle)
    {
        Tensor *input = (Tensor *)input_handle;
        Tensor *output = (Tensor *)output_handle;

        int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
        int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
        int num_outputs = height_col * width_col;
        int total_threads = channels * num_outputs;

        if (total_threads > 0)
        {
            // Using 256 threads per block is standard and efficient
            int threadsPerBlock = 256;
            int blocks = (total_threads + threadsPerBlock - 1) / threadsPerBlock;

            im2col_kernel<<<blocks, threadsPerBlock>>>(
                input->data_gpu, channels, height, width,
                kernel_h, kernel_w, pad_h, pad_w,
                stride_h, stride_w, height_col, width_col,
                output->data_gpu);

            cudaDeviceSynchronize();
        }
    }
    DLLEXPORT void col2im_cuda(float *d_col, int c, int h, int w, int k, int p, int s, float *d_im)
    {
        int h_out = (h + 2 * p - k) / s + 1;
        int w_out = (w + 2 * p - k) / s + 1;
        int total_threads = c * h * w;

        if (total_threads <= 0)
            return;

        // Initialize d_im to zero before accumulating gradients
        cudaMemset(d_im, 0, c * h * w * sizeof(float));

        col2im_kernel<<<(total_threads + 255) / 256, 256>>>(
            d_col, c, h, w, k, k, p, p, s, s, h_out, w_out, d_im);

        cudaDeviceSynchronize();
    }
}