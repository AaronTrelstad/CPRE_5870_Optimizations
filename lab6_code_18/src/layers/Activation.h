#pragma once

#include <cmath>

#include "../Types.h"

namespace ML {

enum class ActivationType { NONE, RELU };

class Activation {
    public:
    static fp32 apply(fp32 x, ActivationType type) {
        switch (type) {
            case ActivationType::RELU:
                return (x > 0.0f) ? x : 0.0f;
            case ActivationType::NONE:
            default:
                return x;
        }
     }
};

}