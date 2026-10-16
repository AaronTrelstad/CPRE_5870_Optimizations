#include "Flatten.h"

namespace ML {
    void FlattenLayer::computeNaive(const LayerData& dataIn) const {
        std::memcpy(getOutputData().raw(), dataIn.raw(), getOutputParams().byte_size());
    }

    void FlattenLayer::computeThreaded(const LayerData& dataIn) const { computeNaive(dataIn); }
    void FlattenLayer::computeTiled(const LayerData& dataIn) const { computeNaive(dataIn); }
    void FlattenLayer::computeCombined(const LayerData& dataIn) const { computeNaive(dataIn); }

    size FlattenLayer::getNumMACs() const {
        return 0;
    }
}
