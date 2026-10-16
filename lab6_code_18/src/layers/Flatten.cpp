#include "Flatten.h"

namespace ML {
    void FlattenLayer::computeNaive(const LayerData& dataIn) {
        std::memcpy(getOutputData().raw(), dataIn.raw(), getOutputParams().byte_size());
    }

    void FlattenLayer::computeThreaded(const LayerData& dataIn) { computeNaive(dataIn); }
    void FlattenLayer::computeTiled(const LayerData& dataIn) { computeNaive(dataIn); }
    void FlattenLayer::computeCombined(const LayerData& dataIn) { computeNaive(dataIn); }

    size FlattenLayer::getNumMACs() const {
        return 0;
    }
}
