#pragma once

#include "Types.h"
#include "Utils.h"
#include "layers/Layer.h"

#include <string>
#include <vector>

namespace ML {

class Model;

class Profiler {
public:
    explicit Profiler(const Model& model);

    void startLayer(size layerNum);
    void stopLayer(size layerNum);
    void printReport() const;

private:
    struct LayerStats {
        std::string name;
        fp32 time_ms = 0.0;
        size macs = 0;
    };

    Timer timer;
    std::vector<LayerStats> stats;
    fp32 total_time = 0;
};

}