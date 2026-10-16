#pragma once

#include "../Types.h"
#include "../Utils.h"
#include "../Config.h"
#include "Activation.h"
#include "Layer.h"

namespace ML {

// Fused Convolution + ReLU + 2x2 MaxPool layer.
// Instead of writing the full conv output then reading it back for pooling,
// this computes each 2x2 block of conv outputs on the fly and max-pools
// them immediately. The intermediate conv feature map is never materialized.
//
// Input:  HxWxC       (e.g. 60x60x32)
// Output: (H-R+1)/2 x (W-S+1)/2 x M  (e.g. 28x28x32)
// The conv output would be (H-R+1) x (W-S+1) x M, then 2x2 pool halves it.
class ConvPoolLayer : public Layer {
   public:
    ConvPoolLayer(const LayerParams inParams, const LayerParams outParams,
                  const LayerParams weightParams, const LayerParams biasParams,
                  ActivationType activation = ActivationType::RELU)
        : Layer(inParams, outParams, LayerType::CONVOLUTIONAL),
          weightParam(weightParams), weightData(weightParams),
          biasParam(biasParams), biasData(biasParams),
          activation(activation) {}

    const LayerParams& getWeightParams() const { return weightParam; }
    const LayerParams& getBiasParams() const { return biasParam; }
    const LayerData& getWeightData() const { return weightData; }
    const LayerData& getBiasData() const { return biasData; }
    const ActivationType getActivationType() const { return activation; }

    virtual void allocLayer() override {
        Layer::allocLayer();
        biasData.loadData();
        weightData.loadData();
    }

    virtual void freeLayer() override {
        Layer::freeLayer();
        weightData.freeData();
        biasData.freeData();
    }

    virtual void computeNaive(const LayerData& dataIn) override;
    virtual void computeThreaded(const LayerData& dataIn) override;
    virtual void computeTiled(const LayerData& dataIn) override;
    virtual void computeCombined(const LayerData& dataIn) override;
    virtual size getNumMACs() const override;

   private:
    LayerParams weightParam;
    LayerData weightData;
    LayerParams biasParam;
    LayerData biasData;
    ActivationType activation;
};

}
