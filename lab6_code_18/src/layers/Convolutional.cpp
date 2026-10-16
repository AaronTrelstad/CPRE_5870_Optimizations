#include "Convolutional.h"
#include "../ThreadPool.h"

namespace ML {

    void ConvolutionalLayer::computeNaive(const LayerData& dataIn) const {
        const dimVec& in_dims = getInputParams().dims;
        const dimVec& out_dims = getOutputParams().dims;
        const dimVec& weight_dims = getWeightParams().dims;

        size H = in_dims[0], W = in_dims[1], C = in_dims[2];
        size P = out_dims[0], Q = out_dims[1], M = out_dims[2];
        size R = weight_dims[0], S = weight_dims[1];
        size U = (H - R) / (P - 1);

        fp32* output = (fp32*)getOutputData().raw();
        const fp32* input = (const fp32*)dataIn.raw();
        const fp32* weights = (const fp32*)getWeightData().raw();
        const fp32* biases = (const fp32*)getBiasData().raw();
        ActivationType act = getActivationType();

        for (size p = 0; p < P; p++) {
            for (size q = 0; q < Q; q++) {
                fp32 acc[128];
                for (size m = 0; m < M; m++) acc[m] = biases[m];

                for (size r = 0; r < R; r++) {
                    for (size s = 0; s < S; s++) {
                        const fp32* in_ptr = input + (p * U + r) * W * C + (q * U + s) * C;
                        const fp32* w_ptr = weights + r * S * C * M + s * C * M;
                        for (size c = 0; c < C; c++) {
                            fp32 in_val = in_ptr[c];
                            const fp32* w_row = w_ptr + c * M;
                            for (size m = 0; m < M; m++) {
                                acc[m] += in_val * w_row[m];
                            }
                        }
                    }
                }

                fp32* out_ptr = output + p * Q * M + q * M;
                for (size m = 0; m < M; m++)
                    out_ptr[m] = Activation::apply(acc[m], act);
            }
        }
    }

    void ConvolutionalLayer::computeThreaded(const LayerData& dataIn) const {
        // Only use threading if enough work to amortize pool overhead
        if (getCachedMACs() < 100000) {
            computeNaive(dataIn);
        } else {
            computeCombined(dataIn);
        }
    }

    void ConvolutionalLayer::computeTiled(const LayerData& dataIn) const {
        // Skip tiling if working set fits in L1 (32KB)
        size weight_bytes = getWeightParams().flat_count() * sizeof(fp32);
        if (weight_bytes < 8192) {
            computeNaive(dataIn);
            return;
        }

        const dimVec& in_dims = getInputParams().dims;
        const dimVec& out_dims = getOutputParams().dims;
        const dimVec& weight_dims = getWeightParams().dims;

        size H = in_dims[0], W = in_dims[1], C = in_dims[2];
        size P = out_dims[0], Q = out_dims[1], M = out_dims[2];
        size R = weight_dims[0], S = weight_dims[1];
        size U = (H - R) / (P - 1);

        fp32* output = (fp32*)getOutputData().raw();
        const fp32* input = (const fp32*)dataIn.raw();
        const fp32* weights = (const fp32*)getWeightData().raw();
        const fp32* biases = (const fp32*)getBiasData().raw();
        ActivationType act = getActivationType();

        // Tile sizes tuned for L1 cache (32KB on Zedboard)
        const size TILE_M = 16;
        const size TILE_P = 4;
        const size TILE_Q = 4;

        for (size m0 = 0; m0 < M; m0 += TILE_M) {
            size m_end = (m0 + TILE_M < M) ? m0 + TILE_M : M;
            size m_len = m_end - m0;

            for (size p0 = 0; p0 < P; p0 += TILE_P) {
                size p_end = (p0 + TILE_P < P) ? p0 + TILE_P : P;

                for (size q0 = 0; q0 < Q; q0 += TILE_Q) {
                    size q_end = (q0 + TILE_Q < Q) ? q0 + TILE_Q : Q;

                    // Init accumulators with biases
                    fp32 acc[TILE_P * TILE_Q * TILE_M];
                    for (size p = p0; p < p_end; p++) {
                        for (size q = q0; q < q_end; q++) {
                            fp32* a = acc + (p - p0) * TILE_Q * TILE_M + (q - q0) * TILE_M;
                            for (size m = 0; m < m_len; m++) a[m] = biases[m0 + m];
                        }
                    }

                    // Convolution over filter window
                    for (size r = 0; r < R; r++) {
                        for (size s = 0; s < S; s++) {
                            for (size c = 0; c < C; c++) {
                                const fp32* w_ptr = weights + r * S * C * M + s * C * M + c * M + m0;
                                for (size p = p0; p < p_end; p++) {
                                    for (size q = q0; q < q_end; q++) {
                                        fp32 in_val = input[(p * U + r) * W * C + (q * U + s) * C + c];
                                        fp32* a = acc + (p - p0) * TILE_Q * TILE_M + (q - q0) * TILE_M;
                                        for (size m = 0; m < m_len; m++) {
                                            a[m] += in_val * w_ptr[m];
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Write with activation
                    for (size p = p0; p < p_end; p++) {
                        for (size q = q0; q < q_end; q++) {
                            fp32* out_ptr = output + p * Q * M + q * M + m0;
                            fp32* a = acc + (p - p0) * TILE_Q * TILE_M + (q - q0) * TILE_M;
                            for (size m = 0; m < m_len; m++)
                                out_ptr[m] = Activation::apply(a[m], act);
                        }
                    }
                }
            }
        }
    }

    void ConvolutionalLayer::computeCombined(const LayerData& dataIn) const {
        // Fall back to tiled-only if not enough work for threading
        if (getCachedMACs() < 100000) {
            computeTiled(dataIn);
            return;
        }

        const dimVec& in_dims = getInputParams().dims;
        const dimVec& out_dims = getOutputParams().dims;
        const dimVec& weight_dims = getWeightParams().dims;

        size H = in_dims[0], W = in_dims[1], C = in_dims[2];
        size P = out_dims[0], Q = out_dims[1], M = out_dims[2];
        size R = weight_dims[0], S = weight_dims[1];
        size U = (H - R) / (P - 1);

        fp32* output = (fp32*)getOutputData().raw();
        const fp32* input = (const fp32*)dataIn.raw();
        const fp32* weights = (const fp32*)getWeightData().raw();
        const fp32* biases = (const fp32*)getBiasData().raw();
        ActivationType act = getActivationType();

        const size TILE_Q = 4;
        const size MAX_M = 128;

        // Use thread pool — threads are created once and reused across layers
        // Accumulate all M channels at once so each input value is loaded exactly once
        ThreadPool::instance().parallelFor(P, [&](size p_start, size p_end_local) {
            for (size p = p_start; p < p_end_local; p++) {
                for (size q0 = 0; q0 < Q; q0 += TILE_Q) {
                    size q_end = (q0 + TILE_Q < Q) ? q0 + TILE_Q : Q;
                    size q_len = q_end - q0;

                    // Init accumulators with biases — full M width
                    fp32 acc[TILE_Q * MAX_M];
                    for (size q = 0; q < q_len; q++) {
                        fp32* a = acc + q * M;
                        for (size m = 0; m < M; m++) a[m] = biases[m];
                    }

                    // Convolution — each input value broadcasts to all M channels
                    for (size r = 0; r < R; r++) {
                        for (size s = 0; s < S; s++) {
                            for (size c = 0; c < C; c++) {
                                const fp32* w_ptr = weights + r * S * C * M + s * C * M + c * M;
                                for (size q = 0; q < q_len; q++) {
                                    fp32 in_val = input[(p * U + r) * W * C + ((q0 + q) * U + s) * C + c];
                                    fp32* a = acc + q * M;
                                    for (size m = 0; m < M; m++) {
                                        a[m] += in_val * w_ptr[m];
                                    }
                                }
                            }
                        }
                    }

                    // Write with activation
                    for (size q = 0; q < q_len; q++) {
                        fp32* out_ptr = output + p * Q * M + (q0 + q) * M;
                        fp32* a = acc + q * M;
                        for (size m = 0; m < M; m++)
                            out_ptr[m] = Activation::apply(a[m], act);
                    }
                }
            }
        });
    }

    size ConvolutionalLayer::getNumMACs() const {
        const dimVec& out_dims = getOutputParams().dims;
        const dimVec& weight_dims = getWeightParams().dims;

        const size P = out_dims[0], Q = out_dims[1];
        const size R = weight_dims[0], S = weight_dims[1];
        const size C = weight_dims[2], M = weight_dims[3];

        return P * Q * M * R * S * C;
    }
} 
