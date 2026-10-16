#include "MaxPooling.h"

namespace ML {
    void MaxPoolingLayer::computeNaive(const LayerData& dataIn) {
        const dimVec& in_dims = getInputParams().dims;
        const dimVec& out_dims = getOutputParams().dims;

        size w_in = in_dims[1];
        size h_out = out_dims[0];
        size w_out = out_dims[1];
        size C = out_dims[2];

        const fp32* input = (const fp32*)dataIn.raw();
        fp32* output = (fp32*)getOutputData().raw();
        size in_row_stride = w_in * C;

        // HWC-order iteration: sequential memory access
        for (size h = 0; h < h_out; h++) {
            const fp32* row0 = input + (2 * h) * in_row_stride;
            const fp32* row1 = input + (2 * h + 1) * in_row_stride;
            fp32* out_row = output + h * w_out * C;

            for (size w = 0; w < w_out; w++) {
                const fp32* r0c0 = row0 + (2 * w) * C;
                const fp32* r0c1 = row0 + (2 * w + 1) * C;
                const fp32* r1c0 = row1 + (2 * w) * C;
                const fp32* r1c1 = row1 + (2 * w + 1) * C;
                fp32* out_ptr = out_row + w * C;

                for (size c = 0; c < C; c++) {
                    fp32 maxVal = r0c0[c];
                    if (r0c1[c] > maxVal) maxVal = r0c1[c];
                    if (r1c0[c] > maxVal) maxVal = r1c0[c];
                    if (r1c1[c] > maxVal) maxVal = r1c1[c];
                    out_ptr[c] = maxVal;
                }
            }
        }
    }

    void MaxPoolingLayer::computeThreaded(const LayerData& dataIn) { computeNaive(dataIn); }
    void MaxPoolingLayer::computeTiled(const LayerData& dataIn) { computeNaive(dataIn); }
    void MaxPoolingLayer::computeCombined(const LayerData& dataIn) { computeNaive(dataIn); }

    size MaxPoolingLayer::getNumMACs() const {
        return 0;
    }
}