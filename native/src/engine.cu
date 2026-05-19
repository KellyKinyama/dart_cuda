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
#include "kernels/transpose.cuh"

extern "C"
{

    // Forward declarations for the registry / ChildRef machinery defined
    // below. The autograd graph used to store raw `Tensor *` pointers in
    // `_children`, which made it impossible for `destroy_tensor` to free
    // the C++ struct safely (a still-live parent could segfault on the
    // next `backward()` walk). We now wrap each child in a `ChildRef`
    // that holds a `shared_ptr<Tensor>`, so a parent keeps its inputs
    // alive even after the Dart wrapper has been disposed.
    struct Tensor;
    static std::shared_ptr<Tensor> shared_from(Tensor *t);

    struct ChildRef
    {
        std::shared_ptr<Tensor> ptr;
        ChildRef() = default;
        ChildRef(Tensor *t) : ptr(shared_from(t)) {}
        ChildRef(std::shared_ptr<Tensor> sp) : ptr(std::move(sp)) {}
    };

    struct Tensor
    {
        float *data_gpu, *grad_gpu;
        int rows, cols, size;
        bool is_view = false; // Default: owns memory
        std::vector<ChildRef> _children;
        std::function<void()> _backward = []() {};

        Tensor(int r, int c) : rows(r), cols(c), size(r * c)
        {
            cudaMalloc(&data_gpu, size * sizeof(float));
            cudaMalloc(&grad_gpu, size * sizeof(float));
            cudaMemset(grad_gpu, 0, size * sizeof(float));
        }

        ~Tensor()
        {
            if (!is_view)
            {
                if (data_gpu) cudaFree(data_gpu);
                if (grad_gpu) cudaFree(grad_gpu);
                data_gpu = nullptr;
                grad_gpu = nullptr;
            }
        }
    };

    // Process-wide registry: every handed-out handle (raw `Tensor *`) is
    // pinned by a shared_ptr in this map. `destroy_tensor` simply erases
    // the entry; the struct dies (running `~Tensor`) when no parent's
    // `_children` still holds a shared_ptr to it.
    static std::unordered_map<Tensor *, std::shared_ptr<Tensor>> g_tensor_registry;
    static std::mutex g_registry_mutex;

    static std::shared_ptr<Tensor> shared_from(Tensor *t)
    {
        if (!t) return nullptr;
        std::lock_guard<std::mutex> lock(g_registry_mutex);
        auto it = g_tensor_registry.find(t);
        if (it != g_tensor_registry.end()) return it->second;
        // Aliasing fallback (non-owning). Shouldn't normally happen for
        // tensors created through the wrappers below; views (which share
        // another tensor's handle) are the main reason this path exists.
        return std::shared_ptr<Tensor>(std::shared_ptr<Tensor>{}, t);
    }

    // Allocate a Tensor and pin it in the registry. Returns the raw
    // pointer (== handle). Use this everywhere instead of `make_tensor(...)`.
    static Tensor *make_tensor(int r, int c)
    {
        Tensor *t = new Tensor(r, c);
        std::lock_guard<std::mutex> lock(g_registry_mutex);
        g_tensor_registry[t] = std::shared_ptr<Tensor>(t);
        return t;
    }

    DLLEXPORT void *create_tensor(int r, int c, float *d)
    {
        Tensor *t = make_tensor(r, c);
        if (d)
            cudaMemcpy(t->data_gpu, d, t->size * sizeof(float), cudaMemcpyHostToDevice);
        return (void *)t;
    }
    DLLEXPORT void destroy_tensor(void *h)
    {
        if (!h)
            return;
        Tensor *t = (Tensor *)h;
        // Views share another tensor's handle/buffers and must never be
        // freed via this path. The Dart wrapper already short-circuits
        // `dispose()` on views, but be defensive.
        if (t->is_view) return;
        std::shared_ptr<Tensor> kept;
        {
            std::lock_guard<std::mutex> lock(g_registry_mutex);
            auto it = g_tensor_registry.find(t);
            if (it == g_tensor_registry.end()) return;
            // Move the shared_ptr out so it can be released *outside* the
            // lock — `~Tensor` may transitively destroy other tensors,
            // and we don't want to recursively re-enter the mutex.
            kept = std::move(it->second);
            g_tensor_registry.erase(it);
        }
        // `kept` goes out of scope here; the struct is freed iff no
        // autograd parent still holds a shared_ptr to it.
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
        // Drain any in-flight forward kernels before walking the graph.
        // Without this, an asynchronous forward kernel can still be
        // writing to a Tensor's data buffer while we read its children
        // pointer here, producing spurious corruption / segfaults that
        // surface several training steps later.
        cudaDeviceSynchronize();
        Tensor *root = (Tensor *)h;
        std::vector<Tensor *> topo;
        std::unordered_set<Tensor *> visited;

        // Iterative post-order DFS. The previous version used a recursive
        // std::function lambda, which would overflow the FFI thread stack
        // for deep autograd graphs (e.g. multi-layer ViT encoders) and
        // segfault inside backward(). Using an explicit stack keeps the
        // recursion depth bounded by heap memory instead.
        std::vector<std::pair<Tensor *, size_t>> stack;
        stack.reserve(64);
        topo.reserve(256);
        visited.insert(root);
        stack.push_back({root, 0});
        while (!stack.empty())
        {
            auto &top = stack.back();
            Tensor *v = top.first;
            if (top.second < v->_children.size())
            {
                Tensor *c = v->_children[top.second++].ptr.get();
                if (c && visited.insert(c).second)
                {
                    stack.push_back({c, 0});
                }
            }
            else
            {
                topo.push_back(v);
                stack.pop_back();
            }
        }

        std::vector<float> ones(root->size, 1.0f);
        cudaMemset(root->grad_gpu, 0, root->size * sizeof(float));
        cudaMemcpy(root->grad_gpu, ones.data(), root->size * sizeof(float), cudaMemcpyHostToDevice);
        for (auto it = topo.rbegin(); it != topo.rend(); ++it)
            (*it)->_backward();
        // Make sure all backward kernels have completed before returning,
        // so any subsequent disposal of intermediate tensors cannot race
        // with in-flight backward kernels still holding their pointers.
        cudaDeviceSynchronize();
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
            // Sign-preserving floor on |b| keeps 1/b and 1/b^2 finite if
            // the broadcast scalar transiently approaches zero. Without
            // this, a single small denominator can poison every grad with
            // +/-Inf and Adam will then turn them into NaN at the next
            // sqrtf(v_hat) step.
            float bv = b[0];
            const float floor_abs = 1e-20f;
            if (fabsf(bv) < floor_abs) bv = copysignf(floor_abs, bv == 0.0f ? 1.0f : bv);
            float inv_b = 1.0f / bv;
            ga[i] += gout[i] * inv_b;
            atomicAdd(&gb[0], -a[i] * gout[i] * inv_b * inv_b);
        }
    }

    DLLEXPORT void *add_tensor_scalar(void *ah, void *bh)
    {
        Tensor *a = (Tensor *)ah, *b = (Tensor *)bh;
        Tensor *out = make_tensor(a->rows, a->cols);
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
        Tensor *out = make_tensor(a->rows, a->cols);
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
        Tensor *out = make_tensor(a->rows, a->cols);
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
        Tensor *out = make_tensor(a->rows, a->cols);
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
        Tensor *out = make_tensor(a->rows, a->cols);
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

    // Row-broadcast add: out[M, N] = a[M, N] + b[1, N]. Used by Linear-style
    // layers to add a [1, out] bias to a [batch, out] matmul result. Plain
    // `add_tensors` does element-wise add over `a->size`, which silently
    // reads past the bias allocation for batch > 1.
    DLLEXPORT void *add_tensor_row_broadcast(void *ah, void *bh)
    {
        Tensor *a = (Tensor *)ah, *b = (Tensor *)bh;
        int M = a->rows, N = a->cols;
        Tensor *out = make_tensor(M, N);
        out->_children = {a, b};

        dim3 block(16, 16);
        dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

        add_row_broadcast_fwd<<<grid, block>>>(
            a->data_gpu, b->data_gpu, out->data_gpu, M, N);

        out->_backward = [out, a, b, M, N, block, grid]()
        {
            add_row_broadcast_bwd<<<grid, block>>>(
                out->grad_gpu, a->grad_gpu, b->grad_gpu, M, N);
        };

        return (void *)out;
    }
    DLLEXPORT void *sub_tensors(void *ah, void *bh)
    {
        Tensor *a = (Tensor *)ah, *b = (Tensor *)bh;
        Tensor *out = make_tensor(a->rows, a->cols);
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
        Tensor *out = make_tensor(a->rows, a->cols);
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
        Tensor *out = make_tensor(a->rows, a->cols);
        out->_children = {a, b};
        div_fwd<<<(a->size + 255) / 256, 256>>>(a->data_gpu, b->data_gpu, out->data_gpu, a->size);
        out->_backward = [out, a, b]()
        { div_bwd<<<(a->size + 255) / 256, 256>>>(a->data_gpu, b->data_gpu, out->grad_gpu, a->grad_gpu, b->grad_gpu, a->size); };
        return (void *)out;
    }
    DLLEXPORT void *pow_tensor(void *ah, float exp)
    {
        Tensor *a = (Tensor *)ah;
        Tensor *out = make_tensor(a->rows, a->cols);
        out->_children = {a};
        pow_fwd<<<(a->size + 255) / 256, 256>>>(a->data_gpu, exp, out->data_gpu, a->size);
        out->_backward = [out, a, exp]()
        { pow_bwd<<<(a->size + 255) / 256, 256>>>(a->data_gpu, exp, out->grad_gpu, a->grad_gpu, a->size); };
        return (void *)out;
    }
    DLLEXPORT void *relu_tensor(void *ah)
    {
        Tensor *a = (Tensor *)ah;
        Tensor *out = make_tensor(a->rows, a->cols);
        out->_children = {a};
        relu_fwd<<<(a->size + 255) / 256, 256>>>(a->data_gpu, out->data_gpu, a->size);
        out->_backward = [out, a]()
        { relu_bwd<<<(a->size + 255) / 256, 256>>>(a->data_gpu, out->grad_gpu, a->grad_gpu, a->size); };
        return (void *)out;
    }

    DLLEXPORT void *gelu_tensor(void *ah)
    {
        Tensor *a = (Tensor *)ah;
        Tensor *out = make_tensor(a->rows, a->cols);
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
        Tensor *out = make_tensor(a->rows, a->cols);
        out->_children = {a};
        sigmoid_fwd<<<(a->size + 255) / 256, 256>>>(a->data_gpu, out->data_gpu, a->size);
        out->_backward = [out, a]()
        { sigmoid_bwd<<<(a->size + 255) / 256, 256>>>(out->data_gpu, out->grad_gpu, a->grad_gpu, a->size); };
        return (void *)out;
    }
    DLLEXPORT void *tanh_tensor(void *ah)
    {
        Tensor *a = (Tensor *)ah;
        Tensor *out = make_tensor(a->rows, a->cols);
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
        Tensor *out = make_tensor(M, N);
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
        Tensor *out = make_tensor(a->rows, a->cols);
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
            int limit = masked ? (t + 1) : T;

            // --- Stable softmax: subtract per-(t,d) max before exp.
            // The forward pass (aft_full_fwd) does this; the original
            // backward did not, so any K[tp,d] + WB[t,tp] >> ~88 made
            // expf overflow to +inf, num/den became 0/inf or inf/inf,
            // and NaN propagated into every gradient. Mirror the forward
            // exactly so the recomputed weights match bit-for-bit.
            float max_val = -1e20f;
            for (int tp = 0; tp < limit; tp++)
            {
                float v = K[tp * D + d] + WB[t * T + tp];
                if (v > max_val) max_val = v;
            }

            float num = 0.0f;
            float den = 0.0f;
            for (int tp = 0; tp < limit; tp++)
            {
                float weight = expf(K[tp * D + d] + WB[t * T + tp] - max_val);
                num += weight * V[tp * D + d];
                den += weight;
            }

            // den >= 1 because the max term contributes exp(0) = 1, so
            // 1/den is bounded; the historical 1e-9 guard is no longer
            // needed but kept harmless.
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
                float weight = expf(K[tp * D + d] + WB[t * T + tp] - max_val);

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
        Tensor *out = make_tensor(Q->rows, Q->cols);
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
        Tensor *out = make_tensor(Q->rows, Q->cols);
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
        Tensor *out = make_tensor(rows, cols_per_t * num_tensors);

        // Prepare device pointers for the kernel
        float **d_inputs;
        float *h_inputs[num_tensors];
        for (int i = 0; i < num_tensors; i++)
            h_inputs[i] = ts[i]->data_gpu;
        cudaMalloc(&d_inputs, num_tensors * sizeof(float *));
        cudaMemcpy(d_inputs, h_inputs, num_tensors * sizeof(float *), cudaMemcpyHostToDevice);

        concat_axis1_fwd<<<(out->size + 255) / 256, 256>>>(d_inputs, out->data_gpu, num_tensors, rows, cols_per_t);
        // The forward kernel only reads from d_inputs; once it has been
        // dispatched (the device pointer table is consumed by the launch
        // before this returns from the host's perspective), we can free
        // the temporary device array. cudaDeviceSynchronize ensures the
        // kernel has actually consumed it.
        cudaDeviceSynchronize();
        cudaFree(d_inputs);

        out->_backward = [out, ts, num_tensors, rows, cols_per_t]()
        {
            float **d_grads;
            float *h_grads[num_tensors];
            for (int i = 0; i < num_tensors; i++)
                h_grads[i] = ts[i]->grad_gpu;
            cudaMalloc(&d_grads, num_tensors * sizeof(float *));
            cudaMemcpy(d_grads, h_grads, num_tensors * sizeof(float *), cudaMemcpyHostToDevice);

            concat_axis1_bwd<<<(out->size + 255) / 256, 256>>>(out->grad_gpu, d_grads, num_tensors, rows, cols_per_t);
            // Must sync before freeing the device pointer table -- the
            // kernel reads from d_grads asynchronously; freeing it while
            // the kernel still runs corrupts the CUDA context and causes
            // every subsequent op to fail with `illegal memory access`.
            cudaDeviceSynchronize();
            cudaFree(d_grads);
        };

        out->_children.clear(); out->_children.reserve(ts.size()); for (auto *__p : ts) out->_children.emplace_back(__p);
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

        Tensor *out = make_tensor(total_rows, cols);

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
        cudaDeviceSynchronize();
        cudaFree(d_inputs);

        // d_offsets must outlive the forward (we keep it for backward) but
        // we have to make sure it is freed even if backward never runs (NaN
        // early-exit in training loops). Capture it in a small refcounted
        // owner so the destructor will free it when both the forward path
        // and any pending backward closure have released their reference.
        //
        // IMPORTANT: this struct owns a CUDA allocation, so it MUST be
        // non-copyable / non-movable. The previous version used
        // `std::make_shared<OffsetsHolder>(OffsetsHolder{d_offsets})`,
        // which constructed a temporary, copied it into the shared_ptr's
        // managed object, then ran the temporary's destructor —
        // cudaFree'ing d_offsets while the shared_ptr still held a copy
        // of the (now dangling) pointer. The backward kernel then read
        // freed device memory and triggered cudaErrorIllegalAddress.
        struct OffsetsHolder {
            int *ptr = nullptr;
            OffsetsHolder() = default;
            explicit OffsetsHolder(int *p) : ptr(p) {}
            OffsetsHolder(const OffsetsHolder &) = delete;
            OffsetsHolder &operator=(const OffsetsHolder &) = delete;
            OffsetsHolder(OffsetsHolder &&) = delete;
            OffsetsHolder &operator=(OffsetsHolder &&) = delete;
            ~OffsetsHolder() { if (ptr) cudaFree(ptr); }
        };
        auto offsetsHolder = std::make_shared<OffsetsHolder>(d_offsets);

        out->_backward = [out, ts, num_tensors, cols, offsetsHolder]()
        {
            float **d_grads;
            std::vector<float *> h_grads(num_tensors);
            for (int i = 0; i < num_tensors; i++) h_grads[i] = ts[i]->grad_gpu;
            cudaMalloc(&d_grads, num_tensors * sizeof(float *));
            cudaMemcpy(d_grads, h_grads.data(), num_tensors * sizeof(float *), cudaMemcpyHostToDevice);

            concat_axis0_bwd<<<(out->size + 255) / 256, 256>>>(out->grad_gpu, d_grads, num_tensors, offsetsHolder->ptr, cols);
            // Must sync before freeing the device pointer table -- the
            // kernel reads from d_grads asynchronously; freeing it while
            // the kernel still runs corrupts the CUDA context.
            cudaDeviceSynchronize();
            cudaFree(d_grads);
            // offsetsHolder is freed automatically when this lambda is
            // destroyed (i.e. when `out` is destroyed).
        };

        out->_children.clear(); out->_children.reserve(ts.size()); for (auto *__p : ts) out->_children.emplace_back(__p);
        return (void *)out;
    }
    DLLEXPORT void *layernorm_forward(void *xh, void *gh, void *bh, float eps)
    {
        Tensor *x = (Tensor *)xh, *gamma = (Tensor *)gh, *beta = (Tensor *)bh;
        Tensor *out = make_tensor(x->rows, x->cols);
        out->_children = {x, gamma, beta};

        // Block-cooperative: one block per row, 256 threads cooperate on
        // the column-wise mean/variance reductions. Previously launched
        // <<<x->rows, 1>>>, which left ~99% of the GPU idle.
        layernorm_fwd<<<x->rows, 256>>>(x->data_gpu, gamma->data_gpu, beta->data_gpu, out->data_gpu, x->rows, x->cols, eps);

        // Added 'beta' to the capture list below [out, x, gamma, beta, eps]
        out->_backward = [out, x, gamma, beta, eps]()
        {
            layernorm_bwd<<<x->rows, 256>>>(
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
        Tensor *out = make_tensor(T, D);
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

            // Clean up indices after backward. Must sync first: the
            // kernel reads d_indices asynchronously, so freeing it
            // before the kernel completes corrupts CUDA state.
            cudaDeviceSynchronize();
            cudaFree(d_indices);
        };

        return (void *)out;
    }

    // Note: In a real engine, you'd pass epsilon from Dart. Hardcoding 0.1 for now.
    const float GLOBAL_EPSILON = 0.1f;

    // GPU-side mean reduction: avoids the previous D->H copy of T floats,
    // CPU sum, and H->D copy of the scalar mean.
    __global__ void ce_mean_reduce(const float *losses, float *out, int T)
    {
        __shared__ float smem[32];
        float local = 0.0f;
        for (int i = threadIdx.x; i < T; i += blockDim.x)
            local += losses[i];
        float total = block_reduce_sum_bcast(local, smem);
        if (threadIdx.x == 0) out[0] = total / (float)T;
    }

    void *cross_entropy_loss(void *lh, int *h_targets, int T, int V)
    {
        Tensor *logits = (Tensor *)lh;
        Tensor *out = make_tensor(1, 1);
        out->_children = {logits};

        int *d_targets;
        cudaMalloc(&d_targets, T * sizeof(int));
        cudaMemcpy(d_targets, h_targets, T * sizeof(int), cudaMemcpyHostToDevice);

        float *d_losses;
        cudaMalloc(&d_losses, T * sizeof(float));

        // Launch with 1 block per row, 128 threads per block
        cross_entropy_fwd_hyper<<<T, 128>>>(logits->data_gpu, d_targets, d_losses, T, V, GLOBAL_EPSILON);

        // Reduce on the device directly into out->data_gpu (no D->H/H->D
        // round-trip). d_losses is no longer needed by the backward pass,
        // so free it immediately -- matches the previous lifetime exactly.
        ce_mean_reduce<<<1, 256>>>(d_losses, out->data_gpu, T);
        cudaDeviceSynchronize();
        cudaFree(d_losses);

        out->_backward = [out, logits, d_targets, T, V]()
        {
            float h_grad_out;
            cudaMemcpy(&h_grad_out, out->grad_gpu, sizeof(float), cudaMemcpyDeviceToHost);

            cross_entropy_bwd_hyper<<<T, 128>>>(
                logits->data_gpu, d_targets, logits->grad_gpu, T, V, h_grad_out, GLOBAL_EPSILON);

            // Sync before freeing: the kernel reads d_targets
            // asynchronously.
            cudaDeviceSynchronize();
            cudaFree(d_targets);
        };

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

    //     Tensor *out = make_tensor(numRows, cols);
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
        Tensor *out = make_tensor(numRows, cols);
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
        Tensor *out = make_tensor(a->rows, a->cols);
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

    // Block-cooperative softmax: one block per row, 256 threads cooperate
    // on the row-wide max/sum reductions. The previous version ran one
    // *thread* per row, doing two/three sequential passes over V -- a
    // catastrophic bottleneck for vocab/logit-sized rows.
    __global__ void softmax_fwd_kernel(float *logits, float *out, int T, int V)
    {
        int t = blockIdx.x;
        if (t >= T) return;

        __shared__ float smem[32];
        __shared__ float s_max, s_sum;

        const float *row_in  = logits + t * V;
        float       *row_out = out    + t * V;

        // Pass 1: max
        float local_max = -1e30f;
        for (int v = threadIdx.x; v < V; v += blockDim.x)
            local_max = fmaxf(local_max, row_in[v]);
        float mx = block_reduce_max_bcast(local_max, smem);
        if (threadIdx.x == 0) s_max = mx;
        __syncthreads();
        float row_max = s_max;

        // Pass 2: write exp(x - max), accumulate sum
        float local_sum = 0.0f;
        for (int v = threadIdx.x; v < V; v += blockDim.x) {
            float e = expf(row_in[v] - row_max);
            row_out[v] = e;
            local_sum += e;
        }
        float sum = block_reduce_sum_bcast(local_sum, smem);
        if (threadIdx.x == 0) s_sum = sum + 1e-9f;
        __syncthreads();
        float inv_sum = 1.0f / s_sum;

        // Pass 3: normalize
        for (int v = threadIdx.x; v < V; v += blockDim.x)
            row_out[v] *= inv_sum;
    }

    // Exported Wrapper
    DLLEXPORT void *softmax_forward(void *ah)
    {
        Tensor *a = (Tensor *)ah;
        Tensor *out = make_tensor(a->rows, a->cols);
        out->_children = {a};

        // One block per row.
        softmax_fwd_kernel<<<a->rows, 256>>>(a->data_gpu, out->data_gpu, a->rows, a->cols);

        // Row-wise softmax backward (see kernels/transpose.cuh).
        // dL/dx_i = y_i * (gy_i - sum_j y_j * gy_j).
        out->_backward = [out, a]()
        {
            softmax_bwd_kernel<<<a->rows, 256>>>(out->data_gpu, out->grad_gpu,
                                                 a->grad_gpu, a->rows, a->cols);
        };
        return (void *)out;
    }

    // Transpose: out[j,i] = a[i,j]. Output shape is [a->cols, a->rows].
    DLLEXPORT void *transpose_tensor(void *ah)
    {
        Tensor *a = (Tensor *)ah;
        Tensor *out = make_tensor(a->cols, a->rows);
        out->_children = {a};
        dim3 block(DC_TRANSPOSE_TILE, DC_TRANSPOSE_TILE);
        dim3 grid((a->cols + DC_TRANSPOSE_TILE - 1) / DC_TRANSPOSE_TILE,
                  (a->rows + DC_TRANSPOSE_TILE - 1) / DC_TRANSPOSE_TILE);
        transpose_fwd_kernel<<<grid, block>>>(a->data_gpu, out->data_gpu,
                                              a->rows, a->cols);
        out->_backward = [out, a]()
        {
            dim3 b(DC_TRANSPOSE_TILE, DC_TRANSPOSE_TILE);
            dim3 g((a->rows + DC_TRANSPOSE_TILE - 1) / DC_TRANSPOSE_TILE,
                   (a->cols + DC_TRANSPOSE_TILE - 1) / DC_TRANSPOSE_TILE);
            transpose_bwd_kernel<<<g, b>>>(out->grad_gpu, a->grad_gpu,
                                           a->rows, a->cols);
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
        Tensor *out = make_tensor(1, 1);
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
        Tensor *out = make_tensor(1, 1);
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
    //     Tensor *out = make_tensor(R, C);
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
        Tensor *out = make_tensor(R, C);
        out->_children = {a};

        // Block-cooperative: one block per row, 256 threads/block.
        // Previously launched (R+255)/256 blocks of 256 threads with
        // each thread handling a whole row sequentially -- this got
        // a single warp's worth of parallelism per row. The new layout
        // gets 256x parallelism per row.
        layer_norm_fwd<<<R, 256>>>(a->data_gpu, out->data_gpu, R, C, eps);

        out->_backward = [out, a, R, C, eps]()
        {
            layer_norm_bwd<<<R, 256>>>(
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