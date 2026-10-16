#include "ConvPool.h"
#include "../Config.h"
#include "../ThreadPool.h"

namespace ML {

    // Fused Conv+ReLU+Pool: compute all 4 conv outputs for each 2x2 pool window
    // simultaneously, sharing weight loads across the 4 positions.
    // Each weight element is loaded once (not 4x), and the intermediate conv
    // output is never written to memory.
    void ConvPoolLayer::computeNaive(const LayerData& dataIn) {
        const dimVec& in_dims = getInputParams().dims;
        const dimVec& out_dims = getOutputParams().dims;
        const dimVec& weight_dims = getWeightParams().dims;

        size W = in_dims[1], C = in_dims[2];
        size PH = out_dims[0], PW = out_dims[1], M = out_dims[2];
        size R = weight_dims[0], S = weight_dims[1];

        fp32* output = (fp32*)getOutputData().raw();
        const fp32* input = (const fp32*)dataIn.raw();
        const fp32* weights = (const fp32*)getWeightData().raw();
        const fp32* biases = (const fp32*)getBiasData().raw();
        ActivationType act = getActivationType();

        for (size ph = 0; ph < PH; ph++) {
            for (size pw = 0; pw < PW; pw++) {
                // 4 accumulators for the 2x2 conv outputs — all in registers
                fp32 a00[Config::CONV_MAX_M], a01[Config::CONV_MAX_M];
                fp32 a10[Config::CONV_MAX_M], a11[Config::CONV_MAX_M];
                for (size m = 0; m < M; m++) {
                    a00[m] = a01[m] = a10[m] = a11[m] = biases[m];
                }

                // Single pass over filter — weight loaded once, applied to 4 inputs
                for (size r = 0; r < R; r++) {
                    for (size s = 0; s < S; s++) {
                        const fp32* w_ptr = weights + r * S * C * M + s * C * M;
                        // The 4 input pointers for this (r,s) filter position
                        const fp32* i00 = input + (2*ph   + r) * W*C + (2*pw   + s) * C;
                        const fp32* i01 = input + (2*ph   + r) * W*C + (2*pw+1 + s) * C;
                        const fp32* i10 = input + (2*ph+1 + r) * W*C + (2*pw   + s) * C;
                        const fp32* i11 = input + (2*ph+1 + r) * W*C + (2*pw+1 + s) * C;

                        for (size c = 0; c < C; c++) {
                            fp32 v00 = i00[c], v01 = i01[c];
                            fp32 v10 = i10[c], v11 = i11[c];
                            const fp32* w_row = w_ptr + c * M;
                            for (size m = 0; m < M; m++) {
                                fp32 w = w_row[m];
                                a00[m] += v00 * w;
                                a01[m] += v01 * w;
                                a10[m] += v10 * w;
                                a11[m] += v11 * w;
                            }
                        }
                    }
                }

                // Activate and max-pool in one pass
                fp32* out_ptr = output + ph * PW * M + pw * M;
                for (size m = 0; m < M; m++) {
                    fp32 v00 = Activation::apply(a00[m], act);
                    fp32 v01 = Activation::apply(a01[m], act);
                    fp32 v10 = Activation::apply(a10[m], act);
                    fp32 v11 = Activation::apply(a11[m], act);
                    fp32 mx = v00;
                    if (v01 > mx) mx = v01;
                    if (v10 > mx) mx = v10;
                    if (v11 > mx) mx = v11;
                    out_ptr[m] = mx;
                }
            }
        }
    }

    void ConvPoolLayer::computeThreaded(const LayerData& dataIn) {
        if (getCachedMACs() < Config::THREADING_MAC_THRESHOLD) {
            computeNaive(dataIn);
        } else {
            computeCombined(dataIn);
        }
    }

    void ConvPoolLayer::computeTiled(const LayerData& dataIn) {
        const dimVec& in_dims = getInputParams().dims;
        const dimVec& out_dims = getOutputParams().dims;
        const dimVec& weight_dims = getWeightParams().dims;

        size W = in_dims[1], C = in_dims[2];
        size PH = out_dims[0], PW = out_dims[1], M = out_dims[2];
        size R = weight_dims[0], S = weight_dims[1];

        fp32* output = (fp32*)getOutputData().raw();
        const fp32* input = (const fp32*)dataIn.raw();
        const fp32* weights = (const fp32*)getWeightData().raw();
        const fp32* biases = (const fp32*)getBiasData().raw();
        ActivationType act = getActivationType();

        const size TILE_M = Config::CONV_TILE_M;

        for (size ph = 0; ph < PH; ph++) {
            for (size pw = 0; pw < PW; pw++) {
                for (size m0 = 0; m0 < M; m0 += TILE_M) {
                    size m_end = (m0 + TILE_M < M) ? m0 + TILE_M : M;
                    size m_len = m_end - m0;

                    fp32 a00[Config::CONV_TILE_M], a01[Config::CONV_TILE_M];
                    fp32 a10[Config::CONV_TILE_M], a11[Config::CONV_TILE_M];
                    for (size m = 0; m < m_len; m++) {
                        a00[m] = a01[m] = a10[m] = a11[m] = biases[m0 + m];
                    }

                    for (size r = 0; r < R; r++) {
                        for (size s = 0; s < S; s++) {
                            const fp32* w_ptr = weights + r * S * C * M + s * C * M + m0;
                            const fp32* i00 = input + (2*ph   + r) * W*C + (2*pw   + s) * C;
                            const fp32* i01 = input + (2*ph   + r) * W*C + (2*pw+1 + s) * C;
                            const fp32* i10 = input + (2*ph+1 + r) * W*C + (2*pw   + s) * C;
                            const fp32* i11 = input + (2*ph+1 + r) * W*C + (2*pw+1 + s) * C;

                            for (size c = 0; c < C; c++) {
                                fp32 v00 = i00[c], v01 = i01[c];
                                fp32 v10 = i10[c], v11 = i11[c];
                                const fp32* w_row = w_ptr + c * M;
                                for (size m = 0; m < m_len; m++) {
                                    fp32 w = w_row[m];
                                    a00[m] += v00 * w;
                                    a01[m] += v01 * w;
                                    a10[m] += v10 * w;
                                    a11[m] += v11 * w;
                                }
                            }
                        }
                    }

                    fp32* out_ptr = output + ph * PW * M + pw * M + m0;
                    for (size m = 0; m < m_len; m++) {
                        fp32 v00 = Activation::apply(a00[m], act);
                        fp32 v01 = Activation::apply(a01[m], act);
                        fp32 v10 = Activation::apply(a10[m], act);
                        fp32 v11 = Activation::apply(a11[m], act);
                        fp32 mx = v00;
                        if (v01 > mx) mx = v01;
                        if (v10 > mx) mx = v10;
                        if (v11 > mx) mx = v11;
                        out_ptr[m] = mx;
                    }
                }
            }
        }
    }

    void ConvPoolLayer::computeCombined(const LayerData& dataIn) {
        if (getCachedMACs() < Config::THREADING_MAC_THRESHOLD) {
            computeNaive(dataIn);
            return;
        }

        const dimVec& in_dims = getInputParams().dims;
        const dimVec& out_dims = getOutputParams().dims;
        const dimVec& weight_dims = getWeightParams().dims;

        size W = in_dims[1], C = in_dims[2];
        size PH = out_dims[0], PW = out_dims[1], M = out_dims[2];
        size R = weight_dims[0], S = weight_dims[1];

        fp32* output = (fp32*)getOutputData().raw();
        const fp32* input = (const fp32*)dataIn.raw();
        const fp32* weights = (const fp32*)getWeightData().raw();
        const fp32* biases = (const fp32*)getBiasData().raw();
        ActivationType act = getActivationType();

        const size TILE_M = Config::CONV_TILE_M;

        // Spatial parallelism over PH rows, with M-tiling inside each thread
        // to keep accumulators small (4 x TILE_M = 256B vs 4 x M = 2KB).
        // Input values stay L1-hot across M-tiles at the same spatial position.
        ThreadPool::instance().parallelFor(PH, [&](size ph_start, size ph_end) {
            for (size ph = ph_start; ph < ph_end; ph++) {
                for (size pw = 0; pw < PW; pw++) {
                    for (size m0 = 0; m0 < M; m0 += TILE_M) {
                        size m_end = (m0 + TILE_M < M) ? m0 + TILE_M : M;
                        size m_len = m_end - m0;

                        fp32 a00[Config::CONV_TILE_M], a01[Config::CONV_TILE_M];
                        fp32 a10[Config::CONV_TILE_M], a11[Config::CONV_TILE_M];
                        for (size m = 0; m < m_len; m++) {
                            a00[m] = a01[m] = a10[m] = a11[m] = biases[m0 + m];
                        }

                        for (size r = 0; r < R; r++) {
                            for (size s = 0; s < S; s++) {
                                const fp32* w_ptr = weights + r * S * C * M + s * C * M + m0;
                                const fp32* i00 = input + (2*ph   + r) * W*C + (2*pw   + s) * C;
                                const fp32* i01 = input + (2*ph   + r) * W*C + (2*pw+1 + s) * C;
                                const fp32* i10 = input + (2*ph+1 + r) * W*C + (2*pw   + s) * C;
                                const fp32* i11 = input + (2*ph+1 + r) * W*C + (2*pw+1 + s) * C;

                                for (size c = 0; c < C; c++) {
                                    fp32 v00 = i00[c], v01 = i01[c];
                                    fp32 v10 = i10[c], v11 = i11[c];
                                    const fp32* w_row = w_ptr + c * M;
                                    for (size m = 0; m < m_len; m++) {
                                        fp32 w = w_row[m];
                                        a00[m] += v00 * w;
                                        a01[m] += v01 * w;
                                        a10[m] += v10 * w;
                                        a11[m] += v11 * w;
                                    }
                                }
                            }
                        }

                        fp32* out_ptr = output + ph * PW * M + pw * M + m0;
                        for (size m = 0; m < m_len; m++) {
                            fp32 v00 = Activation::apply(a00[m], act);
                            fp32 v01 = Activation::apply(a01[m], act);
                            fp32 v10 = Activation::apply(a10[m], act);
                            fp32 v11 = Activation::apply(a11[m], act);
                            fp32 mx = v00;
                            if (v01 > mx) mx = v01;
                            if (v10 > mx) mx = v10;
                            if (v11 > mx) mx = v11;
                            out_ptr[m] = mx;
                        }
                    }
                }
            }
        });
    }

    size ConvPoolLayer::getNumMACs() const {
        const dimVec& out_dims = getOutputParams().dims;
        const dimVec& weight_dims = getWeightParams().dims;

        const size CP = out_dims[0] * 2;
        const size CQ = out_dims[1] * 2;
        const size R = weight_dims[0], S = weight_dims[1];
        const size C = weight_dims[2], M = weight_dims[3];

        return CP * CQ * M * R * S * C;
    }
}
