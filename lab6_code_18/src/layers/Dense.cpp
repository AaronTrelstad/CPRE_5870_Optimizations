#include "Dense.h"
#include "../ThreadPool.h"

namespace ML {
    void DenseLayer::computeNaive(const LayerData& dataIn) const {
        size in_size = getInputParams().flat_count();
        size out_size = getOutputParams().flat_count();

        fp32* output = (fp32*)getOutputData().raw();
        const fp32* input = (const fp32*)dataIn.raw();
        const fp32* weights = (const fp32*)getWeightData().raw();
        const fp32* biases = (const fp32*)getBiasData().raw();
        ActivationType act = getActivationType();

        fp32 acc[256];
        for (size o = 0; o < out_size; o++) acc[o] = biases[o];

        for (size i = 0; i < in_size; i++) {
            fp32 in_val = input[i];
            const fp32* w_row = weights + i * out_size;
            for (size o = 0; o < out_size; o++) {
                acc[o] += in_val * w_row[o];
            }
        }

        for (size o = 0; o < out_size; o++) {
            output[o] = Activation::apply(acc[o], act);
        }
    }

    void DenseLayer::computeThreaded(const LayerData& dataIn) const {
        if (getCachedMACs() < 100000) {
            computeNaive(dataIn);
        } else {
            computeCombined(dataIn);
        }
    }

    void DenseLayer::computeTiled(const LayerData& dataIn) const {
        // Skip tiling if weight matrix fits in L1
        size weight_bytes = getInputParams().flat_count() * getOutputParams().flat_count() * sizeof(fp32);
        if (weight_bytes < 8192) {
            computeNaive(dataIn);
            return;
        }

        size in_size = getInputParams().flat_count();
        size out_size = getOutputParams().flat_count();

        fp32* output = (fp32*)getOutputData().raw();
        const fp32* input = (const fp32*)dataIn.raw();
        const fp32* weights = (const fp32*)getWeightData().raw();
        const fp32* biases = (const fp32*)getBiasData().raw();
        ActivationType act = getActivationType();

        // Tile sizes for L1 cache fitting
        const size TILE_O = 16;  // Output tile
        const size TILE_I = 64;  // Input tile

        // Init output with biases
        for (size o = 0; o < out_size; o++) output[o] = biases[o];

        // Tiled multiply-accumulate
        for (size i0 = 0; i0 < in_size; i0 += TILE_I) {
            size i_end = (i0 + TILE_I < in_size) ? i0 + TILE_I : in_size;

            for (size o0 = 0; o0 < out_size; o0 += TILE_O) {
                size o_end = (o0 + TILE_O < out_size) ? o0 + TILE_O : out_size;

                for (size i = i0; i < i_end; i++) {
                    fp32 in_val = input[i];
                    const fp32* w_row = weights + i * out_size + o0;
                    fp32* out_ptr = output + o0;
                    for (size o = 0; o < o_end - o0; o++) {
                        out_ptr[o] += in_val * w_row[o];
                    }
                }
            }
        }

        // Apply activation
        for (size o = 0; o < out_size; o++) {
            output[o] = Activation::apply(output[o], act);
        }
    }

    void DenseLayer::computeCombined(const LayerData& dataIn) const {
        // Fall back to tiled-only if not enough work for threading
        if (getCachedMACs() < 100000) {
            computeTiled(dataIn);
            return;
        }

        size in_size = getInputParams().flat_count();
        size out_size = getOutputParams().flat_count();

        fp32* output = (fp32*)getOutputData().raw();
        const fp32* input = (const fp32*)dataIn.raw();
        const fp32* weights = (const fp32*)getWeightData().raw();
        const fp32* biases = (const fp32*)getBiasData().raw();
        ActivationType act = getActivationType();

        const size TILE_I = 64;

        // Use thread pool — partition output neurons, tiled input access
        ThreadPool::instance().parallelFor(out_size, [&](size o_start, size o_end_local) {
            size o_len = o_end_local - o_start;

            // Init with biases
            for (size o = 0; o < o_len; o++) output[o_start + o] = biases[o_start + o];

            // Tiled MAC
            for (size i0 = 0; i0 < in_size; i0 += TILE_I) {
                size i_end = (i0 + TILE_I < in_size) ? i0 + TILE_I : in_size;
                for (size i = i0; i < i_end; i++) {
                    fp32 in_val = input[i];
                    const fp32* w_row = weights + i * out_size + o_start;
                    fp32* out_ptr = output + o_start;
                    for (size o = 0; o < o_len; o++) {
                        out_ptr[o] += in_val * w_row[o];
                    }
                }
            }

            // Activation
            for (size o = 0; o < o_len; o++) {
                output[o_start + o] = Activation::apply(output[o_start + o], act);
            }
        });
    }

    size DenseLayer::getNumMACs() const {
        const dimVec& in_dims = getInputParams().dims;
        const dimVec& out_dims = getOutputParams().dims;

        const size N = in_dims[0];
        const size M = out_dims[0];

        return N * M;
    }
}
