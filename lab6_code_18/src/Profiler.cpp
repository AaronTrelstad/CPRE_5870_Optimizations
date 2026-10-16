#include "Profiler.h"
#include "Model.h"

#include <cassert>
#include <iomanip>
#include <iostream>
#include <string>

namespace ML {
    Profiler::Profiler(const Model& model) : timer("Profiler"), stats(model.getNumLayers()) {
        for (size layer = 0; layer < stats.size(); layer++) {
            stats[layer].name = std::to_string(layer + 1) + ": " + std::string(Layer::to_string(model.getLayer(layer).getLType()));
            stats[layer].macs = model.getLayer(layer).getNumMACs();
        }
    }

    void Profiler::startLayer(size layerNum) {
        assert(layerNum < stats.size());

        timer.start();
    }

    void Profiler::stopLayer(size layerNum) {
        assert(layerNum < stats.size());

        timer.stop();

        stats[layerNum].time_ms = timer.milliseconds;
        total_time += timer.milliseconds;
    }

    void Profiler::printReport() const {
        std::cout << "\n--- Profiler Report ---\n";

        std::cout << std::left
                << std::setw(20) << "Layer"
                << std::right
                << std::setw(15) << "Time (ms)"
                << std::setw(20) << "MACs"
                << '\n';

        std::cout << std::string(55, '-') << '\n';

        for (const LayerStats& stat : stats) {
            std::cout << std::left
                    << std::setw(20) << stat.name
                    << std::right
                    << std::setw(15) << stat.time_ms
                    << std::setw(20) << stat.macs
                    << '\n';
        }

        std::cout << "Total time (ms): " << total_time << '\n';
    }
}