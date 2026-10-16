#include "Softmax.h"

namespace ML {
    void SoftmaxLayer::computeNaive(const LayerData& dataIn) {
        std::size_t N = getInputParams().flat_count();

        // Add numerical stability to avoid overflow: https://stackoverflow.com/questions/42599498/numerically-stable-softmax
        fp32 maxVal = dataIn.get<fp32>(0);
        for (std::size_t i = 1; i < N; i++) {
            if (dataIn.get<fp32>(i) > maxVal) maxVal = dataIn.get<fp32>(i);
        }

        fp32 sum = 0.0f;
        for (std::size_t i = 0; i < N; i++) {
            fp32 val = std::exp(dataIn.get<fp32>(i) - maxVal);
            getOutputData().get<fp32>(i) = val;
            sum += val;
        }

        for (std::size_t i = 0; i < N; i++) {
            getOutputData().get<fp32>(i) /= sum;
        }
    }

    void SoftmaxLayer::computeThreaded(const LayerData& dataIn) { computeNaive(dataIn); }
    void SoftmaxLayer::computeTiled(const LayerData& dataIn) { computeNaive(dataIn); }
    void SoftmaxLayer::computeCombined(const LayerData& dataIn) { computeNaive(dataIn); }

    size SoftmaxLayer::getNumMACs() const {
        return 0;
    }
}

