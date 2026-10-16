#pragma once

#include <cstddef>

namespace ML {

namespace Config {
constexpr bool FANCY_LOGGING = true;
constexpr float EPSILON = 0.001;

// Minimum MACs before enabling threading/combined paths
constexpr std::size_t THREADING_MAC_THRESHOLD = 100000;

// Weight working-set threshold (bytes) below which tiling is skipped
constexpr std::size_t TILING_BYTE_THRESHOLD = 8192;

// Convolution tile sizes
constexpr std::size_t CONV_TILE_M = 16;
constexpr std::size_t CONV_TILE_P = 4;
constexpr std::size_t CONV_TILE_Q = 4;
constexpr std::size_t CONV_MAX_M = 128;

// Dense tile sizes
constexpr std::size_t DENSE_TILE_O = 16;
constexpr std::size_t DENSE_TILE_I = 64;
constexpr std::size_t DENSE_MAX_OUT = 256;
}
}
