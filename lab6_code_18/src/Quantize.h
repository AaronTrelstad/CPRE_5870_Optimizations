#pragma once

#include <cmath>
#include <cstring>
#include <vector>

#include "Config.h"
#include "Types.h"
#include "Utils.h"

namespace ML {

struct QuantParams {
    fp32 scale;
    i8 zero_point;
};

namespace Quantize {
    inline fp32 muLawEncode(fp32 x, fp32 max_abs, fp32 mu) {
        if (max_abs == 0.0f) return 0.0f;
        fp32 compressed = logf(1.0f + mu * fabsf(x) / max_abs) / logf(1.0f + mu);
        return (x >= 0.0f ? 1.0f : -1.0f) * compressed * max_abs;
    }

    inline fp32 muLawDecode(fp32 x, fp32 max_abs, fp32 mu) {
        if (max_abs == 0.0f) return 0.0f;
        fp32 expanded = (powf(1.0f + mu, fabsf(x) / max_abs) - 1.0f) / mu;
        return (x >= 0.0f ? 1.0f : -1.0f) * expanded * max_abs;
    }

    inline fp32 compress(fp32 x, fp32 max_abs) {
        if (Config::QUANT_METHOD == QUANT_LOG)
            return muLawEncode(x, max_abs, (fp32)Config::QUANT_MAX);
        return x;
    }

    inline QuantParams computeInputParams(const fp32* data, size count) {
        fp32 sum = 0.0f;
        for (size i = 0; i < count; i++) sum += data[i];
        fp32 avg = sum / (fp32)count;

        fp32 max_dev = 0.0f;
        for (size i = 0; i < count; i++) {
            fp32 d = fabsf(data[i] - avg);
            if (d > max_dev) max_dev = d;
        }

        fp32 scale = (max_dev > 0.0f) ? (fp32)Config::QUANT_MAX / max_dev : 1.0f;
        i8 zp = (i8)clamp((int)(-roundf(avg * scale)), (int)Config::QUANT_MIN, (int)Config::QUANT_MAX);

        QuantParams p;
        p.scale = scale;
        p.zero_point = zp;
        return p;
    }

    inline QuantParams computeAndQuantizeInput(const fp32* src, size count, std::vector<i8>& dst) {
        fp32 sum = 0.0f;
        for (size i = 0; i < count; i++) sum += src[i];
        fp32 avg = sum / (fp32)count;

        fp32 max_dev = 0.0f;
        for (size i = 0; i < count; i++) {
            fp32 d = fabsf(src[i] - avg);
            if (d > max_dev) max_dev = d;
        }

        fp32 scale = (max_dev > 0.0f) ? (fp32)Config::QUANT_MAX / max_dev : 1.0f;
        i8 zp = (i8)clamp((int)(-roundf(avg * scale)), (int)Config::QUANT_MIN, (int)Config::QUANT_MAX);

        dst.resize(count);
        for (size i = 0; i < count; i++) {
            fp32 v = compress(src[i], max_dev);
            int val = (int)roundf(scale * v) + zp;
            dst[i] = (i8)clamp(val, (int)Config::QUANT_MIN, (int)Config::QUANT_MAX);
        }

        QuantParams p;
        p.scale = scale;
        p.zero_point = zp;
        return p;
    }

    inline void quantizeToInt8(const fp32* src, size count, const QuantParams& params,
                               std::vector<i8>& dst, fp32 max_abs = 0.0f) {
        dst.resize(count);
        for (size i = 0; i < count; i++) {
            fp32 v = compress(src[i], max_abs);
            int val = (int)roundf(params.scale * v) + params.zero_point;
            dst[i] = (i8)clamp(val, (int)Config::QUANT_MIN, (int)Config::QUANT_MAX);
        }
    }

    inline void quantizeBiasToInt32(const fp32* src, size count, fp32 Sb, std::vector<i32>& dst) {
        dst.resize(count);
        for (size i = 0; i < count; i++) {
            dst[i] = (i32)roundf(Sb * src[i]);
        }
    }

    inline void computeWeightSums(const std::vector<i8>& qw, size filters_per_output,
                                   size num_outputs, size stride, std::vector<i32>& sums) {
        sums.assign(num_outputs, 0);
        for (size o = 0; o < num_outputs; o++) {
            for (size i = 0; i < filters_per_output; i++) {
                sums[o] += (i32)qw[i * stride + o];
            }
        }
    }

    inline void unpackWeights(const ui8* packed, size count, int bits, std::vector<i8>& dst) {
        dst.resize(count);
        if (bits >= 8) {
            std::memcpy(dst.data(), packed, count);
        } else if (bits == 4) {
            for (size i = 0; i < count; i++) {
                ui8 byte = packed[i / 2];
                i8 val = (i % 2 == 0) ? (i8)(byte << 4) >> 4 : (i8)byte >> 4;
                dst[i] = val;
            }
        } else if (bits == 2) {
            for (size i = 0; i < count; i++) {
                ui8 byte = packed[i / 4];
                int shift = (i % 4) * 2;
                i8 val = (byte >> shift) & 0x03;
                if (val & 0x02) val |= (i8)0xFC; 
                dst[i] = val;
            }
        }
    }

} 
} 
