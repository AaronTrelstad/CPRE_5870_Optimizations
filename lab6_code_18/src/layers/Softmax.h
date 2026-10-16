#pragma once

#include "../Types.h"
#include "../Utils.h"
#include "Activation.h"
#include "Layer.h"

namespace ML {
class SoftmaxLayer : public Layer {
    public:
    SoftmaxLayer(const LayerParams inParams, const LayerParams outParams)
        : Layer(inParams, outParams, LayerType::SOFTMAX) {}

    // Virtual functions
    virtual void computeNaive(const LayerData& dataIn) const override;
    virtual void computeThreaded(const LayerData& dataIn) const override;
    virtual void computeTiled(const LayerData& dataIn) const override;
    virtual void computeCombined(const LayerData& dataIn) const override;
    virtual size getNumMACs() const override;
};
}