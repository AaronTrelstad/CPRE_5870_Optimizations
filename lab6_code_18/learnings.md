# Lab 6 Learnings: Multithreading, Tiling, and SIMD for CNN Optimization

## 1. Why Convolution Dominates Latency

In our CNN model, the 6 convolutional layers account for ~95% of total inference time. Conv2 alone (5x5x32x32 on 60x60 input) performs ~80M multiply-accumulate (MAC) operations. Dense layers contribute far less despite having many weights because they process 1D vectors rather than 2D spatial feature maps.

**Key takeaway:** Always profile first. Optimizing the bottleneck layer gives the biggest returns.

---

## 2. Cache Hierarchy and Memory Access Patterns

### Cache Sizes (Zedboard - ARM Cortex-A9)
| Cache Level | Size | Latency | Notes |
|-------------|------|---------|-------|
| L1 Data | 32 KB | ~1 cycle | Per-core, private |
| L1 Instruction | 32 KB | ~1 cycle | Per-core, private |
| L2 | 512 KB | ~10 cycles | Shared between 2 cores |
| L3 | None | N/A | Zedboard has no L3 |
| DRAM | 512 MB | ~100 cycles | DDR3 |

### Why Naive Convolution Has Poor Cache Behavior

The naive convolution loop order is: `for p, for q, for r, for s, for c, for m`. On each `(r,s,c)` iteration, we access M weight values. For Conv2 where M=32:
- Weight slice accessed per (r,s,c): 32 floats = 128 bytes
- Total weights for Conv2: 5x5x32x32 = 25,600 floats = 100 KB

The entire weight tensor (100 KB) does NOT fit in L1 cache (32 KB). As we iterate over different `(r,s,c)` combinations, weights get evicted and must be reloaded from L2/DRAM.

### How Tiling Fixes This

Tiling (also called "loop blocking") restructures computation so that the working set fits in cache. We introduce three tile dimensions:

- **TILE_M = 16**: Process 16 output channels at a time. Weight slice per (r,s,c) = 16 floats = 64 bytes - fits easily in L1.
- **TILE_P = 4**: Process 4 output rows at a time. The same weight tile is reused across all 4 rows.
- **TILE_Q = 4**: Process 4 output columns at a time. Same weight tile reused across columns.

The accumulator array size is: TILE_P x TILE_Q x TILE_M = 4 x 4 x 16 = 256 floats = 1 KB. Combined with the weight tile being accessed (64 bytes per inner iteration), the working set stays well within L1.

**Result:** Tiling gave us a 1.39x overall speedup (1.63x on Conv2 alone).

### Access Pattern Analysis

**Weights (tiled):** For each (r,s,c) we access a contiguous slice of 16 floats starting at `w_ptr[m0]`. This is sequential memory access = excellent spatial locality. The same 16 weights are reused across TILE_P x TILE_Q = 16 output positions = excellent temporal locality.

**Input (tiled):** For each spatial tile (p0..p_end, q0..q_end), we read overlapping input patches. Adjacent output positions share most of their input receptive fields (stride=1), so input values loaded for position (p,q) are likely still in cache for (p,q+1).

**Output (tiled):** We accumulate into a small local array (1 KB) that stays in registers/L1, then write results to the output buffer once at the end. This avoids repeatedly reading/writing the output array.

---

## 3. Multithreading

### Concept
Modern CPUs have multiple cores that can execute instructions simultaneously. By splitting work across threads, we can achieve near-linear speedup up to the number of physical cores.

### Parallelization Strategy

**Convolution:** We parallelize over the P (output height) dimension. Each thread handles a contiguous range of output rows. This works because:
1. Each output row's computation is independent - no data dependencies between rows
2. Contiguous row assignment means each thread writes to a contiguous memory region, avoiding false sharing
3. Input data is read-only and shared across threads (no synchronization needed)

**Dense:** We parallelize over output neurons. Each thread computes a subset of output values. Each output neuron's dot product is fully independent.

### Thread Count Considerations

| Threads | Expected Behavior |
|---------|-------------------|
| 2 | Good speedup on dual-core (Zedboard) |
| 4 | Matches typical core count; good on quad-core CPUs |
| 8 | May exceed physical cores; hyperthreading helps slightly |
| 16-64 | Diminishing returns; thread creation overhead starts to dominate |

**Optimal thread count** generally equals the number of physical cores. Beyond that, threads compete for the same execution resources, and the overhead of creating/joining threads exceeds the parallelism benefit.

### Why Not Both Tiling AND Threading?

You can combine them! A production implementation would:
1. Partition output rows across threads (coarse-grained parallelism)
2. Within each thread, use tiling for cache efficiency (fine-grained optimization)

This gives the best of both worlds: multi-core utilization AND cache-friendly access patterns.

---

## 4. SIMD (Single Instruction, Multiple Data)

### Concept
SIMD instructions process multiple data elements in a single CPU instruction. Instead of adding two floats one at a time, SIMD adds 4, 8, or 16 floats simultaneously.

| Extension | Width (bits) | FP32 elements | Year |
|-----------|-------------|---------------|------|
| SSE2 | 128 | 4 | 2001 |
| AVX | 256 | 8 | 2011 |
| AVX-512 | 512 | 16 | 2013 |
| ARM NEON | 128 | 4 | (Zedboard) |

### How the Compiler Uses SIMD

With `-O3 -march=native`, the compiler automatically vectorizes loops where possible (auto-vectorization). The inner loop of convolution:
```cpp
for (size m = 0; m < M; m++) {
    acc[m] += in_val * w_row[m];
}
```
This is an ideal SIMD target: the same operation (multiply-add) applied to consecutive memory locations. The compiler can emit `vmulps`/`vaddps` (AVX) or `fmla` (NEON) instructions to process 4-8 elements per cycle.

### SIMD Flags and Their Effects

- **No SIMD flags (-O3 only):** Compiler uses scalar instructions (`mulss`, `addss`) - one float per instruction
- **-msse2:** Uses 128-bit XMM registers, 4 floats per instruction (`mulps`, `addps`)
- **-mavx:** Uses 256-bit YMM registers, 8 floats per instruction (`vmulps`, `vaddps`)
- **-march=native:** Uses all SIMD extensions available on the current CPU (may include AVX2, FMA, etc.)

### Assembly Code Examples

**Scalar (no SIMD):**
```asm
mulss  xmm0, [rdi]     ; multiply single float
addss  xmm1, xmm0      ; add single float
```

**SSE2:**
```asm
mulps  xmm0, [rdi]     ; multiply 4 floats
addps  xmm1, xmm0      ; add 4 floats
```

**AVX:**
```asm
vmulps ymm0, ymm1, [rdi]  ; multiply 8 floats
vaddps ymm2, ymm2, ymm0   ; add 8 floats
```

---

## 5. Combining Optimizations

The "Need for Speed" implementation should combine:
1. **Tiling** for cache efficiency (keeps working set in L1)
2. **Multithreading** for multi-core utilization (partition spatial dimensions)
3. **SIMD** via compiler flags for instruction-level parallelism

These optimizations are orthogonal and multiplicative:
- Tiling: ~1.4-1.6x speedup (cache efficiency)
- Threading: ~2-4x speedup (depends on core count)
- SIMD: ~2-4x speedup (depends on vector width)
- Combined: potentially 5-25x over naive baseline

---

## 6. Practical Insights

### Block Size Selection (Tiling)
- **Too small (2):** Loop overhead dominates; not enough work per tile to amortize
- **Too large (32):** Working set exceeds L1 cache; back to cache miss territory
- **Sweet spot (8-16):** Accumulator + weight tile fits in L1; enough iterations for pipeline efficiency

### Thread Count Selection
- Match to physical core count for compute-bound work
- Going beyond physical cores adds overhead without benefit for CPU-bound tasks
- For the Zedboard (2 cores), 2 threads is optimal
- For lab machines, 4-8 threads is typically optimal

### Data Layout Matters
The weight tensor layout `[R][S][C][M]` means the M dimension is contiguous in memory. This is ideal for our inner loop which iterates over M, giving sequential memory access. If weights were stored as `[R][S][M][C]`, we'd need to either:
- Transpose the weights (one-time cost, amortized over many inferences)
- Accept poor spatial locality (ongoing performance penalty)

This is why the lab suggests considering "reordering the static data files."
