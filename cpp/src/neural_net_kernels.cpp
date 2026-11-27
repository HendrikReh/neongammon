// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Mark Higgins
//
// NeuralNetwork compute kernels (scalar, NEON, Apple Accelerate BLAS, and
// Modal-only x86-64 AVX2), delta evaluation, transposed weights, and TD update.
#include "bgbot/neural_net.h"
#include "bgbot/board.h"
#include "bgbot/encoding.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <stdexcept>

// SIMD headers: NEON on Apple Silicon. The AVX2 kernels exist only for the
// Modal rollout image (Linux x86-64, scripts/modal_rollout.py).
#if defined(__x86_64__)
  #include <immintrin.h>
  #define BGBOT_USE_AVX2 1
#elif defined(__aarch64__) || defined(__arm64__)
  #include <arm_neon.h>
  #define BGBOT_USE_NEON 1
#endif

// Apple arm64 builds route batched evaluation through Accelerate's BLAS
// (BGBOT_USE_ACCELERATE, set by cpp/CMakeLists.txt together with
// ACCELERATE_NEW_LAPACK when the deployment target allows it; see
// NeuralNetwork::forward_group).
#if defined(BGBOT_USE_ACCELERATE)
  #include <Accelerate/Accelerate.h>
#endif

namespace bgbot {

// ======================== Utility ========================

// Fast sigmoid via lookup table with linear interpolation.
// Covers [-8, 8] with 4096 entries (512 per unit). Outside this range,
// sigmoid saturates to 0 or 1 (sigmoid(±8) ≈ 0.9997/0.0003).
// Max error with linear interpolation: < 1e-5 (negligible for play quality).
// Speed: ~3-5 cycles vs ~75 cycles for std::exp.
static constexpr int SIGMOID_TABLE_SIZE = 4096;
static constexpr float SIGMOID_RANGE = 8.0f;
static constexpr float SIGMOID_SCALE = SIGMOID_TABLE_SIZE / (2.0f * SIGMOID_RANGE);
static constexpr float SIGMOID_OFFSET = SIGMOID_RANGE;  // maps [-8,8] to [0, TABLE_SIZE]

struct SigmoidTable {
    float table[SIGMOID_TABLE_SIZE + 1];  // +1 for interpolation at upper bound

    SigmoidTable() {
        for (int i = 0; i <= SIGMOID_TABLE_SIZE; ++i) {
            float x = (static_cast<float>(i) / SIGMOID_SCALE) - SIGMOID_OFFSET;
            table[i] = 1.0f / (1.0f + std::exp(-x));
        }
    }
};

static const SigmoidTable sigmoid_lut;

static inline float fast_sigmoid(float x) {
    if (x >= SIGMOID_RANGE) return 1.0f;
    if (x <= -SIGMOID_RANGE) return 0.0f;
    float idx_f = (x + SIGMOID_OFFSET) * SIGMOID_SCALE;
    int idx = static_cast<int>(idx_f);
    float frac = idx_f - static_cast<float>(idx);
    return sigmoid_lut.table[idx] + frac * (sigmoid_lut.table[idx + 1] - sigmoid_lut.table[idx]);
}

// ======================== AVX2 Vectorized Sigmoid ========================
// Based on Cephes polynomial approximation of exp().
// Computes sigmoid(x) = 1/(1+exp(-x)) for 8 floats simultaneously.
// Max error vs std::exp: ~1-2 ULP (better than the LUT approach).
// Used for the hidden layer sigmoid in the column-major forward pass.
#if defined(BGBOT_USE_AVX2)
static inline __m256 exp256_ps(__m256 x) {
    // Cephes exp polynomial constants
    const __m256 one    = _mm256_set1_ps(1.0f);
    const __m256 half   = _mm256_set1_ps(0.5f);
    const __m256 exp_hi = _mm256_set1_ps(88.3762626647949f);
    const __m256 exp_lo = _mm256_set1_ps(-88.3762626647949f);
    const __m256 log2e  = _mm256_set1_ps(1.44269504088896341f);
    const __m256 c1     = _mm256_set1_ps(0.693359375f);
    const __m256 c2     = _mm256_set1_ps(-2.12194440e-4f);
    const __m256 p0     = _mm256_set1_ps(1.9875691500E-4f);
    const __m256 p1     = _mm256_set1_ps(1.3981999507E-3f);
    const __m256 p2     = _mm256_set1_ps(8.3334519073E-3f);
    const __m256 p3     = _mm256_set1_ps(4.1665795894E-2f);
    const __m256 p4     = _mm256_set1_ps(1.6666665459E-1f);
    const __m256 p5     = _mm256_set1_ps(5.0000001201E-1f);

    // Clamp input
    x = _mm256_min_ps(x, exp_hi);
    x = _mm256_max_ps(x, exp_lo);

    // exp(x) = 2^n * exp(f) where f = x - n*ln(2) and n = floor(x/ln(2) + 0.5)
    __m256 fx = _mm256_fmadd_ps(x, log2e, half);
    fx = _mm256_floor_ps(fx);

    // Reduce: x = x - fx * ln(2) (two-step for precision)
    __m256 tmp = _mm256_mul_ps(fx, c1);
    __m256 z   = _mm256_mul_ps(fx, c2);
    x = _mm256_sub_ps(x, tmp);
    x = _mm256_sub_ps(x, z);
    z = _mm256_mul_ps(x, x);

    // Horner polynomial evaluation: exp(f) ≈ 1 + f + f²*(p5 + f*(p4 + f*(p3 + f*(p2 + f*(p1 + f*p0)))))
    __m256 y = _mm256_fmadd_ps(p0, x, p1);
    y = _mm256_fmadd_ps(y, x, p2);
    y = _mm256_fmadd_ps(y, x, p3);
    y = _mm256_fmadd_ps(y, x, p4);
    y = _mm256_fmadd_ps(y, x, p5);
    y = _mm256_fmadd_ps(y, z, x);
    y = _mm256_add_ps(y, one);

    // Scale by 2^n: convert n to integer, shift into exponent bits
    __m256i imm0 = _mm256_cvttps_epi32(fx);
    imm0 = _mm256_add_epi32(imm0, _mm256_set1_epi32(0x7f));
    imm0 = _mm256_slli_epi32(imm0, 23);
    __m256 pow2n = _mm256_castsi256_ps(imm0);

    return _mm256_mul_ps(y, pow2n);
}

static inline __m256 sigmoid256_ps(__m256 x) {
    const __m256 one  = _mm256_set1_ps(1.0f);
    const __m256 zero = _mm256_setzero_ps();
    __m256 neg_x = _mm256_sub_ps(zero, x);
    __m256 exp_neg_x = exp256_ps(neg_x);
    return _mm256_div_ps(one, _mm256_add_ps(one, exp_neg_x));
}
#endif

// ======================== NEON Vectorized Sigmoid ========================
// The same Cephes exp() as exp256_ps above, 4 lanes wide. Used for the hidden
// layer across all arm64 forward paths so single-board forward() matches
// Accelerate forward_group() (and x86-64 AVX2 sigmoid256_ps) without LUT clamp
// drift between VR mean and single-board evaluations.
#if defined(BGBOT_USE_NEON)
static inline float32x4_t exp_neon_ps(float32x4_t x) {
    x = vminq_f32(x, vdupq_n_f32(88.3762626647949f));
    x = vmaxq_f32(x, vdupq_n_f32(-88.3762626647949f));

    // exp(x) = 2^n * exp(f), n = floor(x/ln2 + 0.5), f = x - n*ln2 (two-step)
    float32x4_t fx = vfmaq_f32(vdupq_n_f32(0.5f), x, vdupq_n_f32(1.44269504088896341f));
    fx = vrndmq_f32(fx);
    x = vfmsq_f32(x, fx, vdupq_n_f32(0.693359375f));
    x = vfmsq_f32(x, fx, vdupq_n_f32(-2.12194440e-4f));
    const float32x4_t z = vmulq_f32(x, x);

    float32x4_t y = vdupq_n_f32(1.9875691500E-4f);
    y = vfmaq_f32(vdupq_n_f32(1.3981999507E-3f), y, x);
    y = vfmaq_f32(vdupq_n_f32(8.3334519073E-3f), y, x);
    y = vfmaq_f32(vdupq_n_f32(4.1665795894E-2f), y, x);
    y = vfmaq_f32(vdupq_n_f32(1.6666665459E-1f), y, x);
    y = vfmaq_f32(vdupq_n_f32(5.0000001201E-1f), y, x);
    y = vfmaq_f32(x, y, z);
    y = vaddq_f32(y, vdupq_n_f32(1.0f));

    int32x4_t n = vcvtq_s32_f32(fx);
    n = vshlq_n_s32(vaddq_s32(n, vdupq_n_s32(0x7f)), 23);
    return vmulq_f32(y, vreinterpretq_f32_s32(n));
}

static inline void sigmoid_inplace_neon(float* v, size_t n) {
    const float32x4_t one = vdupq_n_f32(1.0f);
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        const float32x4_t e = exp_neon_ps(vnegq_f32(vld1q_f32(v + i)));
        vst1q_f32(v + i, vdivq_f32(one, vaddq_f32(one, e)));
    }
    for (; i < n; ++i) v[i] = 1.0f / (1.0f + std::exp(-v[i]));
}
#endif

// SIMD dot product: compute sum of a[i]*b[i] for i=0..n-1
// Both a and b must be at least n floats. No alignment required.
__attribute__((always_inline)) inline
float simd_dot(const float* __restrict a, const float* __restrict b, int n) {
#if defined(BGBOT_USE_AVX2)
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();

    int i = 0;
    // Process 16 floats per iteration (2 x 8-wide FMA)
    for (; i + 15 < n; i += 16) {
        sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), sum0);
        sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), sum1);
    }
    sum0 = _mm256_add_ps(sum0, sum1);
    // Process remaining 8-wide chunks
    for (; i + 7 < n; i += 8) {
        sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), sum0);
    }

    // Horizontal reduction: sum all 8 lanes
    __m128 hi = _mm256_extractf128_ps(sum0, 1);
    __m128 lo = _mm256_castps256_ps128(sum0);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    float dot = _mm_cvtss_f32(s);

    // Handle remaining elements (< 8)
    for (; i < n; ++i) {
        dot += a[i] * b[i];
    }
    return dot;
#elif defined(BGBOT_USE_NEON)
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);

    int i = 0;
    // Process 8 floats per iteration (2 x 4-wide FMA)
    for (; i + 7 < n; i += 8) {
        sum0 = vfmaq_f32(sum0, vld1q_f32(a + i),     vld1q_f32(b + i));
        sum1 = vfmaq_f32(sum1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    }
    sum0 = vaddq_f32(sum0, sum1);
    // Process remaining 4-wide chunk
    for (; i + 3 < n; i += 4) {
        sum0 = vfmaq_f32(sum0, vld1q_f32(a + i), vld1q_f32(b + i));
    }

    // Horizontal reduction: sum all 4 lanes
    float dot = vaddvq_f32(sum0);

    // Handle remaining elements (< 4)
    for (; i < n; ++i) {
        dot += a[i] * b[i];
    }
    return dot;
#else
    // Scalar fallback
    float dot = 0.0f;
    for (int i = 0; i < n; ++i) {
        dot += a[i] * b[i];
    }
    return dot;
#endif
}

// Keep old name as alias for compatibility within this file
#define avx2_dot simd_dot


// ======================== NeuralNetwork ========================

NeuralNetwork::NeuralNetwork(int n_hidden, int n_inputs, float eps, uint32_t seed)
    : n_hidden_(n_hidden), n_inputs_(n_inputs)
{
    hidden_weights_.resize(n_hidden_ * (n_inputs_ + 1));
    output_weights_.resize(NN_OUTPUTS * (n_hidden_ + 1));

    cached_inputs_.resize(n_inputs_);
    cached_hiddens_.resize(n_hidden_ + 1);
    cached_hprods_.resize(n_hidden_);

    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, eps);

    for (auto& w : hidden_weights_) w = dist(rng);
    for (auto& w : output_weights_) w = dist(rng);
}

// ======================== Forward pass (const) ========================

std::array<float, NN_OUTPUTS> NeuralNetwork::forward(const float* inputs) const {
    const int nh = n_hidden_;
    const int ni = n_inputs_;
    const int hid_stride = nh + 1;

    ensure_transposed_weights();

    // Stack buffer for hidden layer (512 covers all current architectures: max 400)
    float hiddens_stack[512];
    std::vector<float> hiddens_heap;
    float* hiddens;
    if (nh <= 512) {
        hiddens = hiddens_stack;
    } else {
        hiddens_heap.resize(nh);
        hiddens = hiddens_heap.data();
    }

    // Initialize hidden layer with biases (contiguous memcpy, cache-friendly)
    std::memcpy(hiddens, hidden_biases_.data(), nh * sizeof(float));

    // Collect non-zero inputs into compact arrays to avoid branch mispredictions
    // and enable multi-input processing (fewer passes over hiddens array).
    int nz_idx_stack[256];  // stack fast path covers all standard encoders (<= 244 inputs)
    float nz_val_stack[256];
    std::vector<int> nz_idx_heap;
    std::vector<float> nz_val_heap;
    int* nz_idx = nz_idx_stack;
    float* nz_val = nz_val_stack;
    if (ni > 256) {
        nz_idx_heap.resize(ni);
        nz_val_heap.resize(ni);
        nz_idx = nz_idx_heap.data();
        nz_val = nz_val_heap.data();
    }
    int n_nz = 0;
    for (int i = 0; i < ni; ++i) {
        if (inputs[i] != 0.0f) {
            nz_idx[n_nz] = i;
            nz_val[n_nz] = inputs[i];
            n_nz++;
        }
    }

    // Process groups of non-zero inputs together to reduce hiddens array passes.
    // Each group does multiple FMAs per hiddens element per pass.
    int d = 0;
#if defined(BGBOT_USE_AVX2)
    // Process 4 inputs at a time (4 FMAs per hiddens load/store)
    for (; d + 3 < n_nz; d += 4) {
        const float* col0 = &hidden_weights_T_[nz_idx[d] * nh];
        const float* col1 = &hidden_weights_T_[nz_idx[d + 1] * nh];
        const float* col2 = &hidden_weights_T_[nz_idx[d + 2] * nh];
        const float* col3 = &hidden_weights_T_[nz_idx[d + 3] * nh];
        __m256 vv0 = _mm256_set1_ps(nz_val[d]);
        __m256 vv1 = _mm256_set1_ps(nz_val[d + 1]);
        __m256 vv2 = _mm256_set1_ps(nz_val[d + 2]);
        __m256 vv3 = _mm256_set1_ps(nz_val[d + 3]);
        int h = 0;
        for (; h + 7 < nh; h += 8) {
            __m256 acc = _mm256_loadu_ps(&hiddens[h]);
            acc = _mm256_fmadd_ps(vv0, _mm256_loadu_ps(&col0[h]), acc);
            acc = _mm256_fmadd_ps(vv1, _mm256_loadu_ps(&col1[h]), acc);
            acc = _mm256_fmadd_ps(vv2, _mm256_loadu_ps(&col2[h]), acc);
            acc = _mm256_fmadd_ps(vv3, _mm256_loadu_ps(&col3[h]), acc);
            _mm256_storeu_ps(&hiddens[h], acc);
        }
        for (; h < nh; ++h) {
            hiddens[h] += nz_val[d] * col0[h] + nz_val[d + 1] * col1[h]
                        + nz_val[d + 2] * col2[h] + nz_val[d + 3] * col3[h];
        }
    }
    // Process remaining pairs
    for (; d + 1 < n_nz; d += 2) {
        const float* col0 = &hidden_weights_T_[nz_idx[d] * nh];
        const float* col1 = &hidden_weights_T_[nz_idx[d + 1] * nh];
        __m256 vv0 = _mm256_set1_ps(nz_val[d]);
        __m256 vv1 = _mm256_set1_ps(nz_val[d + 1]);
        int h = 0;
        for (; h + 7 < nh; h += 8) {
            __m256 acc = _mm256_loadu_ps(&hiddens[h]);
            acc = _mm256_fmadd_ps(vv0, _mm256_loadu_ps(&col0[h]), acc);
            acc = _mm256_fmadd_ps(vv1, _mm256_loadu_ps(&col1[h]), acc);
            _mm256_storeu_ps(&hiddens[h], acc);
        }
        for (; h < nh; ++h) {
            hiddens[h] += nz_val[d] * col0[h] + nz_val[d + 1] * col1[h];
        }
    }
#endif
    for (; d < n_nz; ++d) {
        const float vi = nz_val[d];
        const float* col = &hidden_weights_T_[nz_idx[d] * nh];
        int h = 0;
#if defined(BGBOT_USE_AVX2)
        __m256 vvi = _mm256_set1_ps(vi);
        for (; h + 7 < nh; h += 8) {
            __m256 acc = _mm256_loadu_ps(&hiddens[h]);
            acc = _mm256_fmadd_ps(vvi, _mm256_loadu_ps(&col[h]), acc);
            _mm256_storeu_ps(&hiddens[h], acc);
        }
#elif defined(BGBOT_USE_NEON)
        float32x4_t vvi = vdupq_n_f32(vi);
        for (; h + 7 < nh; h += 8) {
            float32x4_t acc0 = vld1q_f32(&hiddens[h]);
            float32x4_t wt0  = vld1q_f32(&col[h]);
            acc0 = vfmaq_f32(acc0, vvi, wt0);
            vst1q_f32(&hiddens[h], acc0);
            float32x4_t acc1 = vld1q_f32(&hiddens[h + 4]);
            float32x4_t wt1  = vld1q_f32(&col[h + 4]);
            acc1 = vfmaq_f32(acc1, vvi, wt1);
            vst1q_f32(&hiddens[h + 4], acc1);
        }
#endif
        for (; h < nh; ++h) {
            hiddens[h] += vi * col[h];
        }
    }

    // Apply sigmoid to hidden layer
    {
        int h = 0;
#if defined(BGBOT_USE_AVX2)
        for (; h + 7 < nh; h += 8) {
            __m256 v = _mm256_loadu_ps(&hiddens[h]);
            _mm256_storeu_ps(&hiddens[h], sigmoid256_ps(v));
        }
#elif defined(BGBOT_USE_NEON)
        sigmoid_inplace_neon(hiddens, static_cast<size_t>(nh));
        h = nh;
#endif
        for (; h < nh; ++h) {
            hiddens[h] = fast_sigmoid(hiddens[h]);
        }
    }

    // Output layer: 5 dot products with SIMD
    std::array<float, NN_OUTPUTS> outputs;
    const float* w0 = &output_weights_[0 * hid_stride];
    const float* w1 = &output_weights_[1 * hid_stride];
    const float* w2 = &output_weights_[2 * hid_stride];
    const float* w3 = &output_weights_[3 * hid_stride];
    const float* w4 = &output_weights_[4 * hid_stride];

    float o0 = w0[nh];
    float o1 = w1[nh];
    float o2 = w2[nh];
    float o3 = w3[nh];
    float o4 = w4[nh];
#if defined(BGBOT_USE_AVX2)
    {
        __m256 s0 = _mm256_setzero_ps();
        __m256 s1 = _mm256_setzero_ps();
        __m256 s2 = _mm256_setzero_ps();
        __m256 s3 = _mm256_setzero_ps();
        __m256 s4 = _mm256_setzero_ps();
        int h = 0;
        for (; h + 7 < nh; h += 8) {
            __m256 z = _mm256_loadu_ps(&hiddens[h]);
            s0 = _mm256_fmadd_ps(_mm256_loadu_ps(&w0[h]), z, s0);
            s1 = _mm256_fmadd_ps(_mm256_loadu_ps(&w1[h]), z, s1);
            s2 = _mm256_fmadd_ps(_mm256_loadu_ps(&w2[h]), z, s2);
            s3 = _mm256_fmadd_ps(_mm256_loadu_ps(&w3[h]), z, s3);
            s4 = _mm256_fmadd_ps(_mm256_loadu_ps(&w4[h]), z, s4);
        }
        // Horizontal reduce each accumulator
        auto hsum = [](__m256 v) -> float {
            __m128 hi = _mm256_extractf128_ps(v, 1);
            __m128 lo = _mm256_castps256_ps128(v);
            __m128 s = _mm_add_ps(lo, hi);
            s = _mm_hadd_ps(s, s);
            s = _mm_hadd_ps(s, s);
            return _mm_cvtss_f32(s);
        };
        o0 += hsum(s0);
        o1 += hsum(s1);
        o2 += hsum(s2);
        o3 += hsum(s3);
        o4 += hsum(s4);
        for (; h < nh; ++h) {
            const float z = hiddens[h];
            o0 += w0[h] * z;
            o1 += w1[h] * z;
            o2 += w2[h] * z;
            o3 += w3[h] * z;
            o4 += w4[h] * z;
        }
    }
#else
    for (int h = 0; h < nh; ++h) {
        const float z = hiddens[h];
        o0 += w0[h] * z;
        o1 += w1[h] * z;
        o2 += w2[h] * z;
        o3 += w3[h] * z;
        o4 += w4[h] * z;
    }
#endif

    outputs[0] = fast_sigmoid(o0);
    outputs[1] = fast_sigmoid(o1);
    outputs[2] = fast_sigmoid(o2);
    outputs[3] = fast_sigmoid(o3);
    outputs[4] = fast_sigmoid(o4);

    return outputs;
}

// ======================== Batch forward (const) ========================

void NeuralNetwork::forward_batch(
    const float* inputs_array,
    std::array<float, NN_OUTPUTS>* outputs_array,
    int count) const
{
    const int nh = n_hidden_;
    const int ni = n_inputs_;
    const int hid_stride = nh + 1;

    ensure_transposed_weights();

    float hiddens_stack[512];
    std::vector<float> hiddens_heap;
    float* hiddens;
    if (nh <= 512) {
        hiddens = hiddens_stack;
    } else {
        hiddens_heap.resize(nh);
        hiddens = hiddens_heap.data();
    }

    int nz_idx_stack[256];
    float nz_val_stack[256];
    std::vector<int> nz_idx_heap;
    std::vector<float> nz_val_heap;
    int* nz_idx = nz_idx_stack;
    float* nz_val = nz_val_stack;
    if (ni > 256) {
        nz_idx_heap.resize(ni);
        nz_val_heap.resize(ni);
        nz_idx = nz_idx_heap.data();
        nz_val = nz_val_heap.data();
    }

    for (int b = 0; b < count; ++b) {
        const float* inputs = inputs_array + b * ni;

        std::memcpy(hiddens, hidden_biases_.data(), nh * sizeof(float));

        // Collect non-zero inputs
        int n_nz = 0;
        for (int i = 0; i < ni; ++i) {
            if (inputs[i] != 0.0f) {
                nz_idx[n_nz] = i;
                nz_val[n_nz] = inputs[i];
                n_nz++;
            }
        }

        int d = 0;
#if defined(BGBOT_USE_AVX2)
        for (; d + 3 < n_nz; d += 4) {
            const float* col0 = &hidden_weights_T_[nz_idx[d] * nh];
            const float* col1 = &hidden_weights_T_[nz_idx[d + 1] * nh];
            const float* col2 = &hidden_weights_T_[nz_idx[d + 2] * nh];
            const float* col3 = &hidden_weights_T_[nz_idx[d + 3] * nh];
            __m256 vv0 = _mm256_set1_ps(nz_val[d]);
            __m256 vv1 = _mm256_set1_ps(nz_val[d + 1]);
            __m256 vv2 = _mm256_set1_ps(nz_val[d + 2]);
            __m256 vv3 = _mm256_set1_ps(nz_val[d + 3]);
            int h = 0;
            for (; h + 7 < nh; h += 8) {
                __m256 acc = _mm256_loadu_ps(&hiddens[h]);
                acc = _mm256_fmadd_ps(vv0, _mm256_loadu_ps(&col0[h]), acc);
                acc = _mm256_fmadd_ps(vv1, _mm256_loadu_ps(&col1[h]), acc);
                acc = _mm256_fmadd_ps(vv2, _mm256_loadu_ps(&col2[h]), acc);
                acc = _mm256_fmadd_ps(vv3, _mm256_loadu_ps(&col3[h]), acc);
                _mm256_storeu_ps(&hiddens[h], acc);
            }
            for (; h < nh; ++h) {
                hiddens[h] += nz_val[d] * col0[h] + nz_val[d + 1] * col1[h]
                            + nz_val[d + 2] * col2[h] + nz_val[d + 3] * col3[h];
            }
        }
        for (; d + 1 < n_nz; d += 2) {
            const float* col0 = &hidden_weights_T_[nz_idx[d] * nh];
            const float* col1 = &hidden_weights_T_[nz_idx[d + 1] * nh];
            __m256 vv0 = _mm256_set1_ps(nz_val[d]);
            __m256 vv1 = _mm256_set1_ps(nz_val[d + 1]);
            int h = 0;
            for (; h + 7 < nh; h += 8) {
                __m256 acc = _mm256_loadu_ps(&hiddens[h]);
                acc = _mm256_fmadd_ps(vv0, _mm256_loadu_ps(&col0[h]), acc);
                acc = _mm256_fmadd_ps(vv1, _mm256_loadu_ps(&col1[h]), acc);
                _mm256_storeu_ps(&hiddens[h], acc);
            }
            for (; h < nh; ++h) {
                hiddens[h] += nz_val[d] * col0[h] + nz_val[d + 1] * col1[h];
            }
        }
#endif
        for (; d < n_nz; ++d) {
            const float vi = nz_val[d];
            const float* col = &hidden_weights_T_[nz_idx[d] * nh];
            int h = 0;
#if defined(BGBOT_USE_AVX2)
            __m256 vvi = _mm256_set1_ps(vi);
            for (; h + 7 < nh; h += 8) {
                __m256 acc = _mm256_loadu_ps(&hiddens[h]);
                acc = _mm256_fmadd_ps(vvi, _mm256_loadu_ps(&col[h]), acc);
                _mm256_storeu_ps(&hiddens[h], acc);
            }
#elif defined(BGBOT_USE_NEON)
            float32x4_t vvi = vdupq_n_f32(vi);
            for (; h + 7 < nh; h += 8) {
                float32x4_t acc0 = vld1q_f32(&hiddens[h]);
                float32x4_t wt0  = vld1q_f32(&col[h]);
                acc0 = vfmaq_f32(acc0, vvi, wt0);
                vst1q_f32(&hiddens[h], acc0);
                float32x4_t acc1 = vld1q_f32(&hiddens[h + 4]);
                float32x4_t wt1  = vld1q_f32(&col[h + 4]);
                acc1 = vfmaq_f32(acc1, vvi, wt1);
                vst1q_f32(&hiddens[h + 4], acc1);
            }
#endif
            for (; h < nh; ++h) {
                hiddens[h] += vi * col[h];
            }
        }
        // Apply sigmoid to hidden layer
        {
            int h = 0;
#if defined(BGBOT_USE_AVX2)
            for (; h + 7 < nh; h += 8) {
                __m256 v = _mm256_loadu_ps(&hiddens[h]);
                _mm256_storeu_ps(&hiddens[h], sigmoid256_ps(v));
            }
#elif defined(BGBOT_USE_NEON)
            sigmoid_inplace_neon(hiddens, static_cast<size_t>(nh));
            h = nh;
#endif
            for (; h < nh; ++h) {
                hiddens[h] = fast_sigmoid(hiddens[h]);
            }
        }

        // Output layer with SIMD
        auto& outputs = outputs_array[b];
        const float* w0 = &output_weights_[0 * hid_stride];
        const float* w1 = &output_weights_[1 * hid_stride];
        const float* w2 = &output_weights_[2 * hid_stride];
        const float* w3 = &output_weights_[3 * hid_stride];
        const float* w4 = &output_weights_[4 * hid_stride];

        float o0 = w0[nh];
        float o1 = w1[nh];
        float o2 = w2[nh];
        float o3 = w3[nh];
        float o4 = w4[nh];
#if defined(BGBOT_USE_AVX2)
        {
            __m256 s0 = _mm256_setzero_ps();
            __m256 s1 = _mm256_setzero_ps();
            __m256 s2 = _mm256_setzero_ps();
            __m256 s3 = _mm256_setzero_ps();
            __m256 s4 = _mm256_setzero_ps();
            int h = 0;
            for (; h + 7 < nh; h += 8) {
                __m256 z = _mm256_loadu_ps(&hiddens[h]);
                s0 = _mm256_fmadd_ps(_mm256_loadu_ps(&w0[h]), z, s0);
                s1 = _mm256_fmadd_ps(_mm256_loadu_ps(&w1[h]), z, s1);
                s2 = _mm256_fmadd_ps(_mm256_loadu_ps(&w2[h]), z, s2);
                s3 = _mm256_fmadd_ps(_mm256_loadu_ps(&w3[h]), z, s3);
                s4 = _mm256_fmadd_ps(_mm256_loadu_ps(&w4[h]), z, s4);
            }
            auto hsum = [](__m256 v) -> float {
                __m128 hi = _mm256_extractf128_ps(v, 1);
                __m128 lo = _mm256_castps256_ps128(v);
                __m128 s = _mm_add_ps(lo, hi);
                s = _mm_hadd_ps(s, s);
                s = _mm_hadd_ps(s, s);
                return _mm_cvtss_f32(s);
            };
            o0 += hsum(s0);
            o1 += hsum(s1);
            o2 += hsum(s2);
            o3 += hsum(s3);
            o4 += hsum(s4);
            for (; h < nh; ++h) {
                const float z = hiddens[h];
                o0 += w0[h] * z;
                o1 += w1[h] * z;
                o2 += w2[h] * z;
                o3 += w3[h] * z;
                o4 += w4[h] * z;
            }
        }
#else
        for (int h = 0; h < nh; ++h) {
            const float z = hiddens[h];
            o0 += w0[h] * z;
            o1 += w1[h] * z;
            o2 += w2[h] * z;
            o3 += w3[h] * z;
            o4 += w4[h] * z;
        }
#endif
        outputs[0] = fast_sigmoid(o0);
        outputs[1] = fast_sigmoid(o1);
        outputs[2] = fast_sigmoid(o2);
        outputs[3] = fast_sigmoid(o3);
        outputs[4] = fast_sigmoid(o4);
    }
}

// ======================== Incremental (delta) evaluation ========================

void NeuralNetwork::save_hidden_base(const float* inputs, float* saved_base, float* saved_inputs) const {
    ensure_transposed_weights();
    std::memcpy(saved_inputs, inputs, n_inputs_ * sizeof(float));
    std::memcpy(saved_base, hidden_biases_.data(), n_hidden_ * sizeof(float));
    // Accumulate in input order. The saved pre-sigmoid sums also serve as
    // the unmodified input to the output pass when a prediction is needed.
    for (int i = 0; i < n_inputs_; ++i) {
        const float value = inputs[i];
        if (value == 0.0f) continue;
        const float* weights = hidden_weights_T_.data() + i * n_hidden_;
        int h = 0;
#if defined(BGBOT_USE_AVX2)
        const __m256 v = _mm256_set1_ps(value);
        for (; h + 7 < n_hidden_; h += 8) {
            const __m256 product = _mm256_mul_ps(v, _mm256_loadu_ps(weights + h));
            const __m256 sum = _mm256_add_ps(_mm256_loadu_ps(saved_base + h), product);
            _mm256_storeu_ps(saved_base + h, sum);
        }
#elif defined(BGBOT_USE_NEON)
        const float32x4_t v = vdupq_n_f32(value);
        for (; h + 7 < n_hidden_; h += 8) {
            vst1q_f32(saved_base + h, vfmaq_f32(vld1q_f32(saved_base + h), v, vld1q_f32(weights + h)));
            vst1q_f32(saved_base + h + 4, vfmaq_f32(vld1q_f32(saved_base + h + 4), v, vld1q_f32(weights + h + 4)));
        }
        for (; h + 3 < n_hidden_; h += 4) {
            vst1q_f32(saved_base + h, vfmaq_f32(vld1q_f32(saved_base + h), v, vld1q_f32(weights + h)));
        }
#endif
        for (; h < n_hidden_; ++h) saved_base[h] += value * weights[h];
    }
}

std::array<float, NN_OUTPUTS> NeuralNetwork::forward_save_base(
    const float* inputs, float* saved_base, float* saved_inputs) const
{
    save_hidden_base(inputs, saved_base, saved_inputs);
    const int nh = n_hidden_;
    const int hid_stride = nh + 1;

    float hiddens_stack[512];
    std::vector<float> hiddens_heap;
    float* hiddens = hiddens_stack;
    if (nh > 512) {
        hiddens_heap.resize(nh);
        hiddens = hiddens_heap.data();
    }

    // Apply sigmoid to hidden layer
    {
        int h = 0;
#if defined(BGBOT_USE_AVX2)
        for (; h + 7 < nh; h += 8) {
            __m256 v = _mm256_loadu_ps(&saved_base[h]);
            _mm256_storeu_ps(&hiddens[h], sigmoid256_ps(v));
        }
#elif defined(BGBOT_USE_NEON)
        std::memcpy(hiddens, saved_base, static_cast<size_t>(nh) * sizeof(float));
        sigmoid_inplace_neon(hiddens, static_cast<size_t>(nh));
        h = nh;
#endif
        for (; h < nh; ++h) {
            hiddens[h] = fast_sigmoid(saved_base[h]);
        }
    }

    // Output layer with SIMD
    std::array<float, NN_OUTPUTS> outputs;
    const float* w0 = &output_weights_[0 * hid_stride];
    const float* w1 = &output_weights_[1 * hid_stride];
    const float* w2 = &output_weights_[2 * hid_stride];
    const float* w3 = &output_weights_[3 * hid_stride];
    const float* w4 = &output_weights_[4 * hid_stride];

    float o0 = w0[nh];
    float o1 = w1[nh];
    float o2 = w2[nh];
    float o3 = w3[nh];
    float o4 = w4[nh];
#if defined(BGBOT_USE_AVX2)
    {
        __m256 s0 = _mm256_setzero_ps();
        __m256 s1 = _mm256_setzero_ps();
        __m256 s2 = _mm256_setzero_ps();
        __m256 s3 = _mm256_setzero_ps();
        __m256 s4 = _mm256_setzero_ps();
        int h = 0;
        for (; h + 7 < nh; h += 8) {
            __m256 z = _mm256_loadu_ps(&hiddens[h]);
            s0 = _mm256_fmadd_ps(_mm256_loadu_ps(&w0[h]), z, s0);
            s1 = _mm256_fmadd_ps(_mm256_loadu_ps(&w1[h]), z, s1);
            s2 = _mm256_fmadd_ps(_mm256_loadu_ps(&w2[h]), z, s2);
            s3 = _mm256_fmadd_ps(_mm256_loadu_ps(&w3[h]), z, s3);
            s4 = _mm256_fmadd_ps(_mm256_loadu_ps(&w4[h]), z, s4);
        }
        auto hsum = [](__m256 v) -> float {
            __m128 hi = _mm256_extractf128_ps(v, 1);
            __m128 lo = _mm256_castps256_ps128(v);
            __m128 s = _mm_add_ps(lo, hi);
            s = _mm_hadd_ps(s, s);
            s = _mm_hadd_ps(s, s);
            return _mm_cvtss_f32(s);
        };
        o0 += hsum(s0);
        o1 += hsum(s1);
        o2 += hsum(s2);
        o3 += hsum(s3);
        o4 += hsum(s4);
        for (; h < nh; ++h) {
            const float z = hiddens[h];
            o0 += w0[h] * z;
            o1 += w1[h] * z;
            o2 += w2[h] * z;
            o3 += w3[h] * z;
            o4 += w4[h] * z;
        }
    }
#else
    for (int h = 0; h < nh; ++h) {
        const float z = hiddens[h];
        o0 += w0[h] * z;
        o1 += w1[h] * z;
        o2 += w2[h] * z;
        o3 += w3[h] * z;
        o4 += w4[h] * z;
    }
#endif
    outputs[0] = fast_sigmoid(o0);
    outputs[1] = fast_sigmoid(o1);
    outputs[2] = fast_sigmoid(o2);
    outputs[3] = fast_sigmoid(o3);
    outputs[4] = fast_sigmoid(o4);
    return outputs;
}

std::array<float, NN_OUTPUTS> NeuralNetwork::forward_from_base(
    const float* inputs, const float* saved_base, const float* saved_inputs) const
{
    const int ni = n_inputs_;

    // Collect changed inputs into a compact list.
    // In backgammon, a move typically changes 4-12 of 196/244 inputs.
    int delta_indices[INPUT_DELTA_LIMIT];
    float delta_vals[INPUT_DELTA_LIMIT];
    int n_deltas = 0;

    int i = 0;
#if defined(BGBOT_USE_AVX2)
    for (; i + 7 < ni && n_deltas < INPUT_DELTA_LIMIT; i += 8) {
        const __m256 current = _mm256_loadu_ps(inputs + i);
        const __m256 previous = _mm256_loadu_ps(saved_inputs + i);
        const __m256 delta = _mm256_sub_ps(current, previous);
        unsigned mask = _mm256_movemask_ps(_mm256_cmp_ps(delta, _mm256_setzero_ps(), _CMP_NEQ_UQ));
        while (mask && n_deltas < INPUT_DELTA_LIMIT) {
            const int lane = __builtin_ctz(mask);
            delta_indices[n_deltas] = i + lane;
            delta_vals[n_deltas++] = inputs[i + lane] - saved_inputs[i + lane];
            mask &= mask - 1;
        }
    }
#endif
    for (; i < ni && n_deltas < INPUT_DELTA_LIMIT; ++i) {
        const float d = inputs[i] - saved_inputs[i];
        if (d != 0.0f) {
            delta_indices[n_deltas] = i;
            delta_vals[n_deltas++] = d;
        }
    }

    // The sparse list may have filled before the last changed input. Use
    // the complete input vector for such widely separated candidates.
    if (n_deltas == INPUT_DELTA_LIMIT) return forward(inputs);

    return forward_with_deltas(saved_base, delta_indices, delta_vals, n_deltas);
}

std::array<float, NN_OUTPUTS> NeuralNetwork::forward_board_from_base(
    const Board& board, const Board& base, const float* saved_hidden, const float* saved_inputs) const {
    const bool tesauro = n_inputs_ == TESAURO_INPUTS;
    if (!tesauro && n_inputs_ != EXTENDED_CONTACT_INPUTS)
        throw std::invalid_argument("Board inference requires Tesauro or extended contact inputs");
    int indices[INPUT_DELTA_LIMIT];
    float values[INPUT_DELTA_LIMIT];
    const int count = board_input_deltas(board, base, saved_inputs, tesauro, indices, values);
    if (count == INPUT_DELTA_LIMIT) {
        if (tesauro) return forward(compute_tesauro_inputs(board));
        return forward(compute_extended_contact_inputs(board).data());
    }
    return forward_with_deltas(saved_hidden, indices, values, count);
}

std::array<float, NN_OUTPUTS> NeuralNetwork::forward_with_deltas(
    const float* saved_base, const int* delta_indices, const float* delta_vals, int n_deltas) const
{
    ensure_transposed_weights();

    const int nh = n_hidden_;
    const int hid_stride = nh + 1;

    // Copy saved pre-sigmoid hidden sums, then apply input deltas
    float hiddens_stack[512];
    std::vector<float> hiddens_heap;
    float* hiddens;
    if (nh <= 512) {
        hiddens = hiddens_stack;
    } else {
        hiddens_heap.resize(nh);
        hiddens = hiddens_heap.data();
    }

    std::memcpy(hiddens, saved_base, nh * sizeof(float));

    const float* weights = hidden_weights_T_.data();
    int d = 0;
#if defined(BGBOT_USE_AVX2)
    // Modal only (Linux x86-64): retain one hidden tile in registers across four deltas.
    for (; d + 3 < n_deltas; d += 4) {
        const float* w0 = weights + delta_indices[d] * nh;
        const float* w1 = weights + delta_indices[d + 1] * nh;
        const float* w2 = weights + delta_indices[d + 2] * nh;
        const float* w3 = weights + delta_indices[d + 3] * nh;
        const __m256 v0 = _mm256_set1_ps(delta_vals[d]);
        const __m256 v1 = _mm256_set1_ps(delta_vals[d + 1]);
        const __m256 v2 = _mm256_set1_ps(delta_vals[d + 2]);
        const __m256 v3 = _mm256_set1_ps(delta_vals[d + 3]);
        int h = 0;
        for (; h + 7 < nh; h += 8) {
            __m256 z = _mm256_loadu_ps(hiddens + h);
            z = _mm256_fmadd_ps(v0, _mm256_loadu_ps(w0 + h), z);
            z = _mm256_fmadd_ps(v1, _mm256_loadu_ps(w1 + h), z);
            z = _mm256_fmadd_ps(v2, _mm256_loadu_ps(w2 + h), z);
            z = _mm256_fmadd_ps(v3, _mm256_loadu_ps(w3 + h), z);
            _mm256_storeu_ps(hiddens + h, z);
        }
        for (; h < nh; ++h) {
            hiddens[h] += delta_vals[d] * w0[h];
            hiddens[h] += delta_vals[d + 1] * w1[h];
            hiddens[h] += delta_vals[d + 2] * w2[h];
            hiddens[h] += delta_vals[d + 3] * w3[h];
        }
    }
#elif defined(BGBOT_USE_NEON)
    // Apple Silicon: keep hidden tile in NEON registers across four successive deltas.
    for (; d + 3 < n_deltas; d += 4) {
        const float* w0 = weights + delta_indices[d] * nh;
        const float* w1 = weights + delta_indices[d + 1] * nh;
        const float* w2 = weights + delta_indices[d + 2] * nh;
        const float* w3 = weights + delta_indices[d + 3] * nh;
        const float32x4_t v0 = vdupq_n_f32(delta_vals[d]);
        const float32x4_t v1 = vdupq_n_f32(delta_vals[d + 1]);
        const float32x4_t v2 = vdupq_n_f32(delta_vals[d + 2]);
        const float32x4_t v3 = vdupq_n_f32(delta_vals[d + 3]);
        int h = 0;
        for (; h + 7 < nh; h += 8) {
            float32x4_t z0 = vld1q_f32(hiddens + h);
            float32x4_t z1 = vld1q_f32(hiddens + h + 4);
            z0 = vfmaq_f32(z0, v0, vld1q_f32(w0 + h));
            z1 = vfmaq_f32(z1, v0, vld1q_f32(w0 + h + 4));
            z0 = vfmaq_f32(z0, v1, vld1q_f32(w1 + h));
            z1 = vfmaq_f32(z1, v1, vld1q_f32(w1 + h + 4));
            z0 = vfmaq_f32(z0, v2, vld1q_f32(w2 + h));
            z1 = vfmaq_f32(z1, v2, vld1q_f32(w2 + h + 4));
            z0 = vfmaq_f32(z0, v3, vld1q_f32(w3 + h));
            z1 = vfmaq_f32(z1, v3, vld1q_f32(w3 + h + 4));
            vst1q_f32(hiddens + h, z0);
            vst1q_f32(hiddens + h + 4, z1);
        }
        for (; h + 3 < nh; h += 4) {
            float32x4_t z = vld1q_f32(hiddens + h);
            z = vfmaq_f32(z, v0, vld1q_f32(w0 + h));
            z = vfmaq_f32(z, v1, vld1q_f32(w1 + h));
            z = vfmaq_f32(z, v2, vld1q_f32(w2 + h));
            z = vfmaq_f32(z, v3, vld1q_f32(w3 + h));
            vst1q_f32(hiddens + h, z);
        }
        for (; h < nh; ++h) {
            hiddens[h] += delta_vals[d] * w0[h];
            hiddens[h] += delta_vals[d + 1] * w1[h];
            hiddens[h] += delta_vals[d + 2] * w2[h];
            hiddens[h] += delta_vals[d + 3] * w3[h];
        }
    }
#endif
    for (; d < n_deltas; ++d) {
        const int col = delta_indices[d];
        const float dv = delta_vals[d];
        const float* col_weights = weights + col * nh;

        // Apply: hiddens[h] += dv * col_weights[h] using SIMD FMA
        int h = 0;
#if defined(BGBOT_USE_AVX2)
        __m256 vdv = _mm256_set1_ps(dv);
        for (; h + 15 < nh; h += 16) {
            __m256 vh0 = _mm256_loadu_ps(&hiddens[h]);
            __m256 vc0 = _mm256_loadu_ps(&col_weights[h]);
            vh0 = _mm256_fmadd_ps(vdv, vc0, vh0);
            _mm256_storeu_ps(&hiddens[h], vh0);
            __m256 vh1 = _mm256_loadu_ps(&hiddens[h + 8]);
            __m256 vc1 = _mm256_loadu_ps(&col_weights[h + 8]);
            vh1 = _mm256_fmadd_ps(vdv, vc1, vh1);
            _mm256_storeu_ps(&hiddens[h + 8], vh1);
        }
        for (; h + 7 < nh; h += 8) {
            __m256 vh = _mm256_loadu_ps(&hiddens[h]);
            __m256 vc = _mm256_loadu_ps(&col_weights[h]);
            vh = _mm256_fmadd_ps(vdv, vc, vh);
            _mm256_storeu_ps(&hiddens[h], vh);
        }
#elif defined(BGBOT_USE_NEON)
        float32x4_t vdv = vdupq_n_f32(dv);
        for (; h + 7 < nh; h += 8) {
            float32x4_t vh0 = vld1q_f32(&hiddens[h]);
            float32x4_t vc0 = vld1q_f32(&col_weights[h]);
            vh0 = vfmaq_f32(vh0, vdv, vc0);
            vst1q_f32(&hiddens[h], vh0);
            float32x4_t vh1 = vld1q_f32(&hiddens[h + 4]);
            float32x4_t vc1 = vld1q_f32(&col_weights[h + 4]);
            vh1 = vfmaq_f32(vh1, vdv, vc1);
            vst1q_f32(&hiddens[h + 4], vh1);
        }
        for (; h + 3 < nh; h += 4) {
            float32x4_t vh = vld1q_f32(&hiddens[h]);
            float32x4_t vc = vld1q_f32(&col_weights[h]);
            vh = vfmaq_f32(vh, vdv, vc);
            vst1q_f32(&hiddens[h], vh);
        }
#endif
        for (; h < nh; ++h) {
            hiddens[h] += dv * col_weights[h];
        }
    }

    // Apply sigmoid to updated hidden sums
    {
        int h = 0;
#if defined(BGBOT_USE_AVX2)
        for (; h + 7 < nh; h += 8) {
            __m256 v = _mm256_loadu_ps(&hiddens[h]);
            _mm256_storeu_ps(&hiddens[h], sigmoid256_ps(v));
        }
#elif defined(BGBOT_USE_NEON)
        sigmoid_inplace_neon(hiddens, static_cast<size_t>(nh));
        h = nh;
#endif
        for (; h < nh; ++h) {
            hiddens[h] = fast_sigmoid(hiddens[h]);
        }
    }

    // Output layer with SIMD
    std::array<float, NN_OUTPUTS> outputs;
    const float* w0 = &output_weights_[0 * hid_stride];
    const float* w1 = &output_weights_[1 * hid_stride];
    const float* w2 = &output_weights_[2 * hid_stride];
    const float* w3 = &output_weights_[3 * hid_stride];
    const float* w4 = &output_weights_[4 * hid_stride];

    float o0 = w0[nh];
    float o1 = w1[nh];
    float o2 = w2[nh];
    float o3 = w3[nh];
    float o4 = w4[nh];
#if defined(BGBOT_USE_AVX2)
    {
        __m256 s0 = _mm256_setzero_ps();
        __m256 s1 = _mm256_setzero_ps();
        __m256 s2 = _mm256_setzero_ps();
        __m256 s3 = _mm256_setzero_ps();
        __m256 s4 = _mm256_setzero_ps();
        int h = 0;
        for (; h + 7 < nh; h += 8) {
            __m256 z = _mm256_loadu_ps(&hiddens[h]);
            s0 = _mm256_fmadd_ps(_mm256_loadu_ps(&w0[h]), z, s0);
            s1 = _mm256_fmadd_ps(_mm256_loadu_ps(&w1[h]), z, s1);
            s2 = _mm256_fmadd_ps(_mm256_loadu_ps(&w2[h]), z, s2);
            s3 = _mm256_fmadd_ps(_mm256_loadu_ps(&w3[h]), z, s3);
            s4 = _mm256_fmadd_ps(_mm256_loadu_ps(&w4[h]), z, s4);
        }
        auto hsum = [](__m256 v) -> float {
            __m128 hi = _mm256_extractf128_ps(v, 1);
            __m128 lo = _mm256_castps256_ps128(v);
            __m128 s = _mm_add_ps(lo, hi);
            s = _mm_hadd_ps(s, s);
            s = _mm_hadd_ps(s, s);
            return _mm_cvtss_f32(s);
        };
        o0 += hsum(s0);
        o1 += hsum(s1);
        o2 += hsum(s2);
        o3 += hsum(s3);
        o4 += hsum(s4);
        for (; h < nh; ++h) {
            const float z = hiddens[h];
            o0 += w0[h] * z;
            o1 += w1[h] * z;
            o2 += w2[h] * z;
            o3 += w3[h] * z;
            o4 += w4[h] * z;
        }
    }
#else
    for (int h = 0; h < nh; ++h) {
        const float z = hiddens[h];
        o0 += w0[h] * z;
        o1 += w1[h] * z;
        o2 += w2[h] * z;
        o3 += w3[h] * z;
        o4 += w4[h] * z;
    }
#endif
    outputs[0] = fast_sigmoid(o0);
    outputs[1] = fast_sigmoid(o1);
    outputs[2] = fast_sigmoid(o2);
    outputs[3] = fast_sigmoid(o3);
    outputs[4] = fast_sigmoid(o4);
    return outputs;
}

// ======================== Group evaluation ========================

#if defined(BGBOT_USE_ACCELERATE)

// Dense path: H = sigmoid(X · W_hᵀ + b_h), O = sigmoid(H · W_oᵀ + b_o), each
// layer one cblas_sgemm. Accelerate runs these on the CPU's matrix unit, which
// beats the sparse column-sum kernels even though ~77% of the inputs are zero.
void NeuralNetwork::forward_group(const float* inputs, int count,
                                  std::array<float, NN_OUTPUTS>* outputs) const
{
    if (count <= 0) return;
    const int nh = n_hidden_;
    const int ni = n_inputs_;

    ensure_transposed_weights();

    thread_local std::vector<float> hidden;
    thread_local std::vector<float> out;
    hidden.resize(static_cast<size_t>(count) * nh);
    out.resize(static_cast<size_t>(count) * NN_OUTPUTS);

    // Hidden layer: seed every row with the biases, then accumulate X · W_hᵀ
    // (hidden_weights_T_ is W_hᵀ, [ni x nh] row-major).
    for (int b = 0; b < count; ++b) {
        std::memcpy(&hidden[static_cast<size_t>(b) * nh], hidden_biases_.data(),
                    nh * sizeof(float));
    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                count, nh, ni,
                1.0f, inputs, ni,
                hidden_weights_T_.data(), nh,
                1.0f, hidden.data(), nh);
    sigmoid_inplace_neon(hidden.data(), hidden.size());

    // Output layer: output_weights_ is [NN_OUTPUTS x (nh+1)] row-major with the
    // bias in the last column, so it is W_o with leading dimension nh+1.
    const int hid_stride = nh + 1;
    for (int b = 0; b < count; ++b) {
        for (int o = 0; o < NN_OUTPUTS; ++o) {
            out[b * NN_OUTPUTS + o] = output_weights_[o * hid_stride + nh];
        }
    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                count, NN_OUTPUTS, nh,
                1.0f, hidden.data(), nh,
                output_weights_.data(), hid_stride,
                1.0f, out.data(), NN_OUTPUTS);
    for (int b = 0; b < count; ++b) {
        for (int o = 0; o < NN_OUTPUTS; ++o) {
            outputs[b][o] = fast_sigmoid(out[b * NN_OUTPUTS + o]);
        }
    }
}

#else

void NeuralNetwork::forward_group(const float* inputs, int count,
                                  std::array<float, NN_OUTPUTS>* outputs) const
{
    if (count <= 0) return;
    const int ni = n_inputs_;
    thread_local std::vector<float> saved_base, saved_inputs;
    if (static_cast<int>(saved_base.size()) < n_hidden_) saved_base.resize(n_hidden_);
    if (static_cast<int>(saved_inputs.size()) < ni) saved_inputs.resize(ni);

    outputs[0] = forward_save_base(inputs, saved_base.data(), saved_inputs.data());
    for (int b = 1; b < count; ++b) {
        outputs[b] = forward_from_base(inputs + static_cast<size_t>(b) * ni,
                                       saved_base.data(), saved_inputs.data());
    }
}

#endif

void evaluate_board_group(const NeuralNetwork& nn, bool purerace,
                          const std::vector<Board>& boards, const int* idx, int n,
                          std::array<float, NN_OUTPUTS>* outputs)
{
    if (n <= 0) return;
    const int ni = nn.n_inputs();
    const int slot_width = purerace ? TESAURO_INPUTS : EXTENDED_CONTACT_INPUTS;
    thread_local std::vector<float> encoded;
    encoded.resize(static_cast<size_t>(n) * ni);
    if (purerace) {
        auto inp0 = compute_tesauro_inputs(boards[idx[0]]);
        std::copy_n(inp0.data(), ni, encoded.data());
    } else {
        auto inp0 = compute_extended_contact_inputs(boards[idx[0]]);
        std::copy_n(inp0.data(), ni, encoded.data());
    }
    if (ni == slot_width) {
        const Board& base = boards[idx[0]];
        for (int j = 1; j < n; ++j) {
            encode_board_from_base(boards[idx[j]], base, encoded.data(), purerace,
                                   &encoded[static_cast<size_t>(j) * ni]);
        }
    } else {
        for (int j = 1; j < n; ++j) {
            float* row = &encoded[static_cast<size_t>(j) * ni];
            if (purerace) {
                auto inp = compute_tesauro_inputs(boards[idx[j]]);
                std::copy_n(inp.data(), ni, row);
            } else {
                auto inp = compute_extended_contact_inputs(boards[idx[j]]);
                std::copy_n(inp.data(), ni, row);
            }
        }
    }
    nn.forward_group(encoded.data(), n, outputs);
}

// ======================== Transposed Weights ========================

void NeuralNetwork::build_transposed_weights() const {
    const int nh = n_hidden_;
    const int ni = n_inputs_;
    const int inp_stride = ni + 1;

    hidden_weights_T_.resize(ni * nh);
    for (int i = 0; i < ni; ++i) {
        for (int h = 0; h < nh; ++h) {
            hidden_weights_T_[i * nh + h] = hidden_weights_[h * inp_stride + i];
        }
    }

    // Cache bias vector (contiguous) for fast memcpy initialization
    hidden_biases_.resize(nh);
    for (int h = 0; h < nh; ++h) {
        hidden_biases_[h] = hidden_weights_[h * inp_stride + ni];
    }
}

// ======================== Equity ========================
// NeuralNetwork::compute_equity is now inlined in neural_net.h,
// delegating to the free function compute_equity() in strategy.h.

// ======================== Forward with gradient caching ========================

std::array<float, NN_OUTPUTS> NeuralNetwork::forward_with_gradients(const float* inputs) {
    const int nh = n_hidden_;
    const int ni = n_inputs_;
    const int inp_stride = ni + 1;
    const int hid_stride = nh + 1;

    std::copy(inputs, inputs + ni, cached_inputs_.begin());

    for (int h = 0; h < nh; ++h) {
        const float* w = &hidden_weights_[h * inp_stride];
        float sum = w[ni];
        for (int i = 0; i < ni; ++i) {
            sum += w[i] * inputs[i];
        }
        cached_hiddens_[h] = fast_sigmoid(sum);
    }
    cached_hiddens_[nh] = 1.0f;

    for (int o = 0; o < NN_OUTPUTS; ++o) {
        const float* w = &output_weights_[o * hid_stride];
        float sum = w[nh];
        for (int h = 0; h < nh; ++h) {
            sum += w[h] * cached_hiddens_[h];
        }
        cached_outputs_[o] = fast_sigmoid(sum);
    }

    for (int o = 0; o < NN_OUTPUTS; ++o) {
        cached_prods_[o] = cached_outputs_[o] * (1.0f - cached_outputs_[o]);
    }
    for (int h = 0; h < nh; ++h) {
        cached_hprods_[h] = cached_hiddens_[h] * (1.0f - cached_hiddens_[h]);
    }

    return cached_outputs_;
}

// ======================== TD weight update ========================

void NeuralNetwork::td_update(
    const std::array<float, NN_OUTPUTS>& targets, float alpha)
{
    const int nh = n_hidden_;
    const int ni = n_inputs_;

    std::array<float, NN_OUTPUTS> delta_prod;
    for (int o = 0; o < NN_OUTPUTS; ++o) {
        delta_prod[o] = (targets[o] - cached_outputs_[o]) * cached_prods_[o];
    }

    // Step 1: Compute bp_error using PRE-update output weights
    float bp_error_stack[256];
    std::vector<float> bp_error_heap;
    float* bp_error;
    if (nh <= 256) {
        bp_error = bp_error_stack;
    } else {
        bp_error_heap.resize(nh);
        bp_error = bp_error_heap.data();
    }

    const int hid_stride = nh + 1;
    for (int h = 0; h < nh; ++h) {
        float sum = 0.0f;
        for (int o = 0; o < NN_OUTPUTS; ++o) {
            sum += delta_prod[o] * output_weights_[o * hid_stride + h];
        }
        bp_error[h] = sum;
    }

    // Step 2: Update output weights
    for (int o = 0; o < NN_OUTPUTS; ++o) {
        float scale = alpha * delta_prod[o];
        float* w = &output_weights_[o * hid_stride];
        for (int h = 0; h <= nh; ++h) {
            w[h] += scale * cached_hiddens_[h];
        }
    }

    // Step 3: Update hidden weights
    const int inp_stride = ni + 1;
    for (int h = 0; h < nh; ++h) {
        float scale = alpha * bp_error[h] * cached_hprods_[h];
        if (scale == 0.0f) continue;
        float* w = &hidden_weights_[h * inp_stride];
        for (int i = 0; i < ni; ++i) {
            w[i] += scale * cached_inputs_[i];
        }
        w[ni] += scale;
    }

    // Mark transposed weight cache as stale — will be rebuilt on next forward().
    transposed_weights_valid_ = false;
}


bool accelerate_enabled() noexcept {
#if defined(BGBOT_USE_ACCELERATE)
    return true;
#else
    return false;
#endif
}

} // namespace bgbot
