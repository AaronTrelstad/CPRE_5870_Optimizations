#pragma once

#include "../Types.h"
#include "../Utils.h"
#include "Activation.h"
#include "Layer.h"

namespace ML {
class ConvolutionalLayer : public Layer {
   public:
    ConvolutionalLayer(const LayerParams inParams, const LayerParams outParams,
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
