#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <unordered_set>
#include <functional>
#include <cmath>
#include <curand_kernel.h>

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

__global__ void gelu_fwd(float *a, float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        float x = a[i];
        // Standard GELU approximation
        out[i] = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    }
}

__global__ void gelu_bwd(float *a, float *go, float *ga, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        float x = a[i];
        float x3 = x * x * x;
        // Derivative of the GELU approximation
        float inner = 0.7978845608f * (x + 0.044715f * x3);
        float t = tanhf(inner);
        float sech2 = 1.0f - t * t;
        float deriv = 0.5f * (1.0f + t) + 0.5f * x * sech2 * 0.7978845608f * (1.0f + 3.0f * 0.044715f * x * x);
        atomicAdd(&ga[i], deriv * go[i]);
    }
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
    float *Q, float *K, float *V, float *WB,
    float *grad_out,
    float *grad_Q, float *grad_K, float *grad_V, float *grad_WB,
    int TDec, int TEnc, int D)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= TDec)
        return;

    for (int d = 0; d < D; d++)
    {
        // --- Re-calculate forward components for this specific d ---
        float num = 0.0f;
        float den = 0.0f;
        for (int tp = 0; tp < TEnc; tp++)
        {
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
        for (int tp = 0; tp < TEnc; tp++)
        {
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

__global__ void concat_axis1_fwd(float **inputs, float *out, int num_tensors, int rows, int cols_per_tensor)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total_cols = num_tensors * cols_per_tensor;
    int size = rows * total_cols;

    if (i < size)
    {
        int r = i / total_cols;
        int c = i % total_cols;
        int tensor_idx = c / cols_per_tensor;
        int local_c = c % cols_per_tensor;

        out[i] = inputs[tensor_idx][r * cols_per_tensor + local_c];
    }
}

__global__ void concat_axis1_bwd(float *grad_out, float **grad_inputs, int num_tensors, int rows, int cols_per_tensor)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total_cols = num_tensors * cols_per_tensor;
    int size = rows * total_cols;

    if (i < size)
    {
        int r = i / total_cols;
        int c = i % total_cols;
        int tensor_idx = c / cols_per_tensor;
        int local_c = c % cols_per_tensor;

        atomicAdd(&grad_inputs[tensor_idx][r * cols_per_tensor + local_c], grad_out[i]);
    }
}

__global__ void layernorm_fwd(float *x, float *gamma, float *beta, float *out, int R, int C, float eps)
{
    int i = blockIdx.x; // One row per block
    if (i < R)
    {
        // 1. Calculate Mean
        float sum = 0;
        for (int j = 0; j < C; j++)
            sum += x[i * C + j];
        float mean = sum / C;

        // 2. Calculate Variance
        float sq_diff_sum = 0;
        for (int j = 0; j < C; j++)
        {
            float diff = x[i * C + j] - mean;
            sq_diff_sum += diff * diff;
        }
        float var = sq_diff_sum / C;
        float std_inv = 1.0f / sqrtf(var + eps);

        // 3. Normalize and Scale/Shift
        for (int j = 0; j < C; j++)
        {
            float x_hat = (x[i * C + j] - mean) * std_inv;
            out[i * C + j] = x_hat * gamma[j] + beta[j];
        }
    }
}

__global__ void layernorm_bwd(float *x, float *gamma, float *go, float *gx, float *gGamma, float *gBeta, int R, int C, float eps)
{
    int i = blockIdx.x;
    if (i < R)
    {
        // Recalculate stats for the row
        float sum = 0;
        for (int j = 0; j < C; j++)
            sum += x[i * C + j];
        float mean = sum / C;

        float sq_diff_sum = 0;
        for (int j = 0; j < C; j++)
        {
            float diff = x[i * C + j] - mean;
            sq_diff_sum += diff * diff;
        }
        float std_inv = 1.0f / sqrtf((sq_diff_sum / C) + eps);

        float dl_dxhat_sum = 0;
        float dl_dxhat_xhat_sum = 0;

        // First pass: Calc gradients for gamma/beta and build sums for input grad
        for (int j = 0; j < C; j++)
        {
            float x_hat = (x[i * C + j] - mean) * std_inv;
            float dl_dxhat = go[i * C + j] * gamma[j];

            dl_dxhat_sum += dl_dxhat;
            dl_dxhat_xhat_sum += dl_dxhat * x_hat;

            atomicAdd(&gGamma[j], go[i * C + j] * x_hat);
            atomicAdd(&gBeta[j], go[i * C + j]);
        }

        // Second pass: Final gradient for x
        for (int j = 0; j < C; j++)
        {
            float x_hat = (x[i * C + j] - mean) * std_inv;
            float dl_dxhat = go[i * C + j] * gamma[j];
            gx[i * C + j] += (std_inv / C) * (C * dl_dxhat - dl_dxhat_sum - x_hat * dl_dxhat_xhat_sum);
        }
    }
}

__global__ void embedding_fwd(int *indices, float *wte, float *wpe, float *out, int T, int D)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < T * D)
    {
        int t = idx / D;
        int d = idx % D;
        int token_id = indices[t];

        // Sum token embedding and position embedding
        out[idx] = wte[token_id * D + d] + wpe[t * D + d];
    }
}

// Corresponding Backward (Atomic add gradients back to embedding weights)
__global__ void embedding_bwd(int *indices, float *go, float *gwte, float *gwpe, int T, int D)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < T * D)
    {
        int t = idx / D;
        int d = idx % D;
        int token_id = indices[t];

        atomicAdd(&gwte[token_id * D + d], go[idx]);
        atomicAdd(&gwpe[t * D + d], go[idx]);
    }
}

__global__ void cross_entropy_fwd(float *logits, int *targets, float *loss, int T, int V)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < T)
    {
        // Only print for the first index so we don't flood the terminal
        // if (t == 0)
        // {
        //     printf("[GPU KERNEL] T=%d, V=%d, target[0]=%d, logit[0]=%f\n", T, V, targets[0], logits[0]);
        // }
        // 1. Find max for numerical stability
        float max_val = -1e30f;
        float sum_logits = 0.0f; // Track sum for label smoothing
        for (int v = 0; v < V; v++)
        {
            float val = logits[t * V + v];
            if (val > max_val)
                max_val = val;
            sum_logits += val;
        }

        // 2. Compute Log-Sum-Exp
        float lse_sum = 0.0f;
        for (int v = 0; v < V; v++)
        {
            lse_sum += expf(logits[t * V + v] - max_val);
        }
        float lse = max_val + logf(lse_sum + 1e-12f);

        // 3. Label Smoothing Logic
        // We act as if the target is 90% likely and the other 10%
        // is spread across all other moves.
        float epsilon = 0.1f;
        int target_idx = targets[t];
        float logit_target = logits[t * V + target_idx];

        // Standard CE Loss
        float ce_loss = lse - logit_target;

        // Smoothing term: lse - average_logit
        float smooth_loss = lse - (sum_logits / V);

        // Final blended loss
        loss[t] = (1.0f - epsilon) * ce_loss + epsilon * smooth_loss;
    }
}
__global__ void cross_entropy_bwd(float *logits, int *targets, float *grad_logits, int T, int V, float grad_output)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < T * V)
    {
        int t = idx / V;
        int v = idx % V;
        int target_idx = targets[t];

        // 1. Recompute max for stable softmax
        float max_val = -1e30f;
        for (int i = 0; i < V; i++)
        {
            float val = logits[t * V + i];
            if (val > max_val)
                max_val = val;
        }

        // 2. Recompute sum of exponentials
        float sum_exp = 0.0f;
        for (int i = 0; i < V; i++)
        {
            sum_exp += expf(logits[t * V + i] - max_val);
        }

        // 3. Current Softmax probability
        float softmax = expf(logits[idx] - max_val) / (sum_exp + 1e-12f);

        // 4. Label Smoothing Logic
        float epsilon = 0.1f;
        float indicator = (v == target_idx) ? 1.0f : 0.0f;

        // target_prob for 4098 classes with eps 0.1:
        // Correct move: 0.90002
        // Wrong moves:  0.000024
        float target_prob = (1.0f - epsilon) * indicator + (epsilon / V);

        // 5. Scaling Correction
        // We multiply by grad_output (usually 1.0) and do NOT divide by T here
        // if you want the full gradient signal to reach Adam.
        // Usually, the loss function in the autograd system handles the 1/T scaling.
        grad_logits[idx] = (softmax - target_prob) * grad_output;

        // DEBUG: Print gradient for the first move of the first sequence
        if (idx == target_idx)
        {
            // printf("[GRAD DEBUG] target_prob: %f, softmax: %f, grad: %f\n", target_prob, softmax, grad_logits[idx]);
        }
    }
}

__global__ void adam_kernel(float *p, float *g, float *m, float *v,
                            int size, int t, float lr,
                            float b1, float b2, float eps)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        // 1. Safe Step Count
        // If t is 0, powf(b1, 0) is 1, leading to 1-1=0 denominator.
        // We force t to be at least 1 for the math.
        float step = (float)t + 1.0f;

        // 2. Gradient + Weight Decay
        float weight_decay = 0.001f;
        float grad = g[i];

        // 3. Update biased moments
        m[i] = b1 * m[i] + (1.0f - b1) * grad;
        v[i] = b2 * v[i] + (1.0f - b2) * (grad * grad);

        // 4. Bias correction (Mathematically safe now)
        float m_hat = m[i] / (1.0f - powf(b1, step));
        float v_hat = v[i] / (1.0f - powf(b2, step));

        // 5. Compute Update
        float delta = lr * m_hat / (sqrtf(v_hat) + eps);

        // 6. Apply Update + Weight Decay (AdamW style)
        // This is more stable than adding decay to the gradient directly
        float new_p = p[i] - delta - (lr * weight_decay * p[i]);

        // 7. Safety Clipping
        if (new_p > 10.0f)
            new_p = 10.0f;
        if (new_p < -10.0f)
            new_p = -10.0f;

        p[i] = new_p;

        // --- Optional Debug ---
        // if (i == 0 && t % 100 == 0) printf("Step %d | p: %f | g: %f\n", t, p[i], g[i]);
    }
}

__global__ void clip_grads_kernel(float *grad, float limit, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        // Clamp the gradient value between [-limit, limit]
        float g = grad[i];
        if (g > limit)
            grad[i] = limit;
        else if (g < -limit)
            grad[i] = -limit;
    }
}

__global__ void slice_fwd_kernel(float *full_data, float *slice_data, int start_idx, int total_elements)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total_elements)
    {
        slice_data[i] = full_data[start_idx + i];
    }
}

__global__ void slice_bwd_kernel(float *full_grad, float *slice_grad, int start_idx, int total_elements)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total_elements)
    {
        // Use atomicAdd because multiple slices could overlap the same memory
        atomicAdd(&full_grad[start_idx + i], slice_grad[i]);
    }
}

__global__ void abs_fwd(float *a, float *out, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        out[i] = fabsf(a[i]);
}

__global__ void abs_bwd(float *a, float *grad_out, float *grad_a, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        float val = a[i];
        float sign = (val > 0.0f) ? 1.0f : ((val < 0.0f) ? -1.0f : 0.0f);
        atomicAdd(&grad_a[i], grad_out[i] * sign);
    }
}

__global__ void compute_cost_matrix_kernel(
    float *pred_boxes,  // [num_queries * 4]
    float *gt_boxes,    // [num_gt * 4]
    float *cost_matrix, // [num_queries * num_gt]
    int num_queries,
    int num_gt)
{
    int q = blockIdx.y * blockDim.y + threadIdx.y; // Query index
    int g = blockIdx.x * blockDim.x + threadIdx.x; // GT index

    if (q < num_queries && g < num_gt)
    {
        float l1_loss = 0.0f;
        for (int i = 0; i < 4; i++)
        {
            l1_loss += fabsf(pred_boxes[q * 4 + i] - gt_boxes[g * 4 + i]);
        }
        cost_matrix[q * num_gt + g] = l1_loss;
    }
}

// --- Reduction Kernels ---

__global__ void sum_fwd_kernel(float *a, float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        // Atomically add each element to the first slot of the output
        atomicAdd(out, a[i]);
    }
}

__global__ void sum_bwd_kernel(float *grad_out, float *grad_a, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        // The gradient of a sum is 1.0.
        // We multiply the incoming gradient (grad_out[0]) by 1.0 and pass it back.
        atomicAdd(&grad_a[i], grad_out[0]);
    }
}

__global__ void mean_fwd_kernel(float *a, float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        // Add (value / n) to the output to get the mean
        atomicAdd(out, a[i] / (float)n);
    }
}

__global__ void mean_bwd_kernel(float *grad_out, float *grad_a, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        // The gradient of mean(x) is 1/n.
        atomicAdd(&grad_a[i], grad_out[0] / (float)n);
    }
}

__global__ void xavier_init_kernel(float* data, int size, int n_in, int n_out, unsigned long seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        curandState state;
        // Seed + i ensures every single weight gets a different random number
        curand_init(seed, i, 0, &state);
        
        float limit = sqrtf(6.0f / (float)(n_in + n_out));
        // curand_uniform is [0,1], we transform to [-limit, limit]
        data[i] = (curand_uniform(&state) * 2.0f - 1.0f) * limit;
    }
}

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

    DLLEXPORT void *cross_entropy_loss(void *lh, int *h_targets, int T, int V)
    {
        Tensor *logits = (Tensor *)lh;
        Tensor *out = new Tensor(1, 1); // Scalar loss
        out->_children = {logits};

        // Move targets to GPU
        int *d_targets;
        cudaMalloc(&d_targets, T * sizeof(int));
        cudaMemcpy(d_targets, h_targets, T * sizeof(int), cudaMemcpyHostToDevice);

        float *d_losses;
        cudaMalloc(&d_losses, T * sizeof(float));

        cross_entropy_fwd<<<(T + 255) / 256, 256>>>(logits->data_gpu, d_targets, d_losses, T, V);

        // Sum losses on CPU for the scalar result (simple for now)
        std::vector<float> h_losses(T);
        cudaMemcpy(h_losses.data(), d_losses, T * sizeof(float), cudaMemcpyDeviceToHost);
        float total_loss = 0;
        for (float l : h_losses)
            total_loss += l;
        total_loss /= T;

        cudaMemcpy(out->data_gpu, &total_loss, sizeof(float), cudaMemcpyHostToDevice);

        out->_backward = [out, logits, d_targets, T, V]()
        {
            float h_grad_out;
            cudaMemcpy(&h_grad_out, out->grad_gpu, sizeof(float), cudaMemcpyDeviceToHost);

            cross_entropy_bwd<<<(T * V + 255) / 256, 256>>>(
                logits->data_gpu, d_targets, logits->grad_gpu, T, V, h_grad_out);
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

    DLLEXPORT void tensor_xavier_init(void* ph, int n_in, int n_out, int seed) {
        Tensor* p = (Tensor*)ph;
        int size = p->rows * p->cols;
        
        int threads = 256;
        int blocks = (size + threads - 1) / threads;
        
        xavier_init_kernel<<<blocks, threads>>>(p->data_gpu, size, n_in, n_out, (unsigned long)seed);
        cudaDeviceSynchronize();
    }

    DLLEXPORT void tensor_zero_init(void* ph) {
        Tensor* p = (Tensor*)ph;
        int size = p->rows * p->cols;
        cudaMemset(p->data_gpu, 0, size * sizeof(float));
    }
}