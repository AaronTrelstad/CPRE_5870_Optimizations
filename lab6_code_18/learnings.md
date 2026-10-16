# Lab 6 Learnings: CNN Optimization — Tiling, Multithreading, SIMD, and Profiling

## Table of Contents
1. [Profiling First: Identifying the Bottleneck](#1-profiling-first-identifying-the-bottleneck)
2. [Cache Hierarchy and Memory Access Patterns](#2-cache-hierarchy-and-memory-access-patterns)
3. [Optimization 1: Cache Tiling](#3-optimization-1-cache-tiling)
4. [Optimization 2: Multithreading](#4-optimization-2-multithreading)
5. [Optimization 3: Thread Pool](#5-optimization-3-thread-pool)
6. [Optimization 4: Combined Tiling + Threading](#6-optimization-4-combined-tiling--threading)
7. [Optimization 5: Input Reuse (Eliminating Redundant Loads)](#7-optimization-5-input-reuse-eliminating-redundant-loads)
8. [Optimization 6: Intelligent Dispatch](#8-optimization-6-intelligent-dispatch)
9. [Optimization 7: SIMD-Aligned Partitioning](#9-optimization-7-simd-aligned-partitioning)
10. [Optimization 8: MaxPooling Loop Reorder](#10-optimization-8-maxpooling-loop-reorder)
11. [SIMD Auto-Vectorization](#11-simd-auto-vectorization)
12. [Perf Profiling and Hardware Counter Analysis](#12-perf-profiling-and-hardware-counter-analysis)
13. [Removing Dead Code (Quantization)](#13-removing-dead-code-quantization)
14. [Performance Summary](#14-performance-summary)
15. [Key Takeaways](#15-key-takeaways)

---

## 1. Profiling First: Identifying the Bottleneck

Before optimizing anything, we profiled the naive baseline to understand where time was spent.

### Model Architecture
| Layer | Type | Input | Output | MACs |
|-------|------|-------|--------|------|
| L1 | Conv 5x5 | 64x64x3 | 60x60x32 | 8.6M |
| L2 | Conv 5x5 | 60x60x32 | 56x56x32 | **80.3M** |
| L3 | MaxPool 2x2 | 56x56x32 | 28x28x32 | 0 |
| L4 | Conv 3x3 | 28x28x32 | 26x26x64 | 12.5M |
| L5 | Conv 3x3 | 26x26x64 | 24x24x64 | 21.2M |
| L6 | MaxPool 2x2 | 24x24x64 | 12x12x64 | 0 |
| L7 | Conv 3x3 | 12x12x64 | 10x10x64 | 3.7M |
| L8 | Conv 3x3 | 10x10x64 | 8x8x128 | 4.7M |
| L9 | MaxPool 2x2 | 8x8x128 | 4x4x128 | 0 |
| L10 | Flatten | 4x4x128 | 2048 | 0 |
| L11 | Dense | 2048 | 256 | 524K |
| L12 | Dense | 256 | 200 | 51K |
| L13 | Softmax | 200 | 200 | 0 |

### Naive Baseline Timing
| Layer | Time (ms) | % of Total |
|-------|-----------|------------|
| Conv2 (L2) | 15.9 | 69% |
| Conv5 (L5) | 2.8 | 12% |
| Conv1 (L1) | 1.7 | 7% |
| Conv4 (L4) | 1.5 | 6% |
| All others | 1.2 | 5% |
| **Total** | **23.1** | **100%** |

**Key finding:** Conv2 alone is 69% of total inference time because it has the highest MAC count (80.3M) and the largest weight tensor (5x5x32x32 = 100KB, which doesn't fit in L1 cache).

**Key takeaway:** Always profile first. Optimizing the bottleneck layer gives the biggest returns. Optimizing Softmax or Flatten would be wasted effort.

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
- Total weights for Conv2: 5x5x32x32 = 25,600 floats = **100 KB**

The entire weight tensor (100 KB) does NOT fit in L1 cache (32 KB). As we iterate over different `(r,s,c)` combinations, weights get evicted and must be reloaded from L2/DRAM. Each reload costs ~10-100x more than an L1 hit.

### Data Layout: HWC Format

Our tensors use HWC (Height-Width-Channel) layout: `tensor[h][w][c]`. This means:
- Channels are contiguous in memory: `tensor[0][0][0], tensor[0][0][1], ..., tensor[0][0][C-1]`
- Moving to the next column: stride = C elements
- Moving to the next row: stride = W*C elements

The weight tensor is `[R][S][C][M]`, meaning the M (output channel) dimension is contiguous. This is ideal for our inner loop which iterates over M, giving sequential memory access and good spatial locality.

---

## 3. Optimization 1: Cache Tiling

### Concept

Tiling (also called "loop blocking") restructures computation so that the working set fits in cache. Instead of processing the entire output at once (which requires the full weight tensor), we process small tiles that reuse cached data.

### Tile Size Selection

We chose tile sizes to keep the accumulator + weight slice within L1 (32KB):

- **TILE_M = 16**: Process 16 output channels at a time. Weight slice per (r,s,c) = 16 floats = 64 bytes.
- **TILE_P = 4**: Process 4 output rows at a time. Same weight tile reused across rows.
- **TILE_Q = 4**: Process 4 output columns at a time. Same weight tile reused across columns.

Accumulator size: `TILE_P * TILE_Q * TILE_M = 4 * 4 * 16 = 256 floats = 1 KB`

This is well within L1. The weight tile being accessed (64 bytes per inner iteration) plus the accumulator (1 KB) totals ~1.1 KB — tiny compared to 32 KB L1.

### Why These Specific Values?

| TILE_M | Accumulator Size | Weight Access | Fits L1? |
|--------|-----------------|---------------|----------|
| 4 | 256 bytes | 16 bytes/iter | Yes, but loop overhead dominates |
| 8 | 512 bytes | 32 bytes/iter | Yes, good balance |
| **16** | **1 KB** | **64 bytes/iter** | **Yes, sweet spot** |
| 32 | 2 KB | 128 bytes/iter | Yes, but less reuse across tiles |
| 64 | 4 KB | 256 bytes/iter | Borderline, less room for other data |

### Dense Layer Tiling

For Dense layers, we tile over both input and output dimensions:
- **TILE_O = 16**: Output neuron tile (accumulator width)
- **TILE_I = 64**: Input tile (how many input values we process before moving to next output tile)

This ensures the weight sub-matrix being accessed (`TILE_I * TILE_O = 64 * 16 = 1024 floats = 4 KB`) fits in L1.

### Result

| Layer | Naive (ms) | Tiled (ms) | Speedup |
|-------|-----------|-----------|---------|
| Conv2 | 15.9 | 10.0 | 1.59x |
| Total | 23.1 | 16.9 | **1.37x** |

---

## 4. Optimization 2: Multithreading

### Concept

Modern CPUs have multiple cores that can execute instructions simultaneously. By splitting work across threads, we achieve near-linear speedup up to the number of physical cores.

### Parallelization Strategy

**Convolution:** Parallelize over the P (output height) dimension. Each thread handles a contiguous range of output rows.

Why this works:
1. Each output row's computation is **independent** — no data dependencies between rows
2. Contiguous row assignment means each thread writes to a **contiguous memory region**, avoiding false sharing (where threads invalidate each other's cache lines)
3. Input data is **read-only** and shared across threads (no synchronization needed)
4. Weights are read-only and shared — no duplication needed

**Dense:** Parallelize over output neurons. Each thread computes a subset of output values. Each output neuron's dot product is fully independent.

### Thread Count Considerations

| Threads | Expected Behavior |
|---------|-------------------|
| 2 | Good speedup on dual-core (Zedboard) |
| 4 | Matches typical quad-core CPUs |
| 8+ | Diminishing returns; overhead grows |

Optimal thread count generally equals the number of **physical cores**. Beyond that, threads compete for the same execution resources, and thread creation/joining overhead starts to dominate.

### Initial Result (with raw std::thread per layer call)

Threaded mode achieved ~5.6x speedup, but each `parallelFor` call created and destroyed threads. For 13 layers, that's 13 thread creation/destruction cycles — each costing ~0.05-0.1ms in OS overhead (syscalls, stack allocation, scheduler).

---

## 5. Optimization 3: Thread Pool

### Problem: Thread Creation Overhead

Creating `std::thread` objects is expensive:
1. OS must allocate a stack (typically 8MB virtual, 4KB physical initially)
2. Kernel creates a new task/thread structure
3. Scheduler must be notified
4. Thread must be joined when work completes

For 13 layer calls per inference, this overhead adds up. On our system, thread creation/destruction was ~0.05ms per call, totaling ~0.65ms of pure overhead.

### Solution: Persistent Thread Pool

We implemented a singleton `ThreadPool` that creates threads **once** at startup and reuses them for every `parallelFor` call:

```
ThreadPool::instance().parallelFor(count, [&](size start, size end) {
    // Each thread processes elements [start, end)
});
```

### Synchronization Design: Generation Counter

The key challenge: how do workers know when new work is available?

**Naive approach (broken):** Use a boolean `workReady` flag. Problem: between consecutive `parallelFor` calls, a worker might miss the flag transition `false → true → false` if it's slow to wake up.

**Our approach:** Use a monotonically increasing `generation` counter. Each worker tracks its own `myGen`. When `generation != myGen`, the worker knows there's new work. This is immune to missed transitions because generation only increases.

```cpp
// Worker loop
while (true) {
    unique_lock<mutex> lock(mtx);
    startCV.wait(lock, [&]{ return generation != myGen || shutdown; });
    if (shutdown) return;
    myGen = generation;  // Acknowledge this generation
    lock.unlock();

    if (id < numChunks) {
        currentFunc(boundaries[id], boundaries[id + 1]);
        // Signal completion
        lock_guard<mutex> lk(mtx);
        finishedCount++;
        doneCV.notify_one();
    }
}
```

### Race Condition Fix: Only Count Workers That Did Work

We discovered a subtle race condition: workers with `id >= numChunks` (no work assigned) were still incrementing `finishedCount`. If they did this **after** the main thread returned and started a new `parallelFor`, the count for the new call would be corrupted — causing the main thread to return before workers finished.

**Fix:** Only increment `finishedCount` inside the `if (id < numChunks)` block.

### Main Thread Participates

The main thread does chunk 0 itself instead of sleeping. This avoids wasting a core — with N threads total, we get N workers (N-1 pool threads + 1 main thread).

### Result

Thread pool eliminated per-call overhead:

| Mode | Before Pool (ms) | After Pool (ms) |
|------|------------------|-----------------|
| Threaded | 4.1 | 2.96 |
| Combined | 3.8 | 2.80 |

---

## 6. Optimization 4: Combined Tiling + Threading

### Concept

Tiling and threading are **orthogonal** optimizations:
- Threading splits work across cores (coarse-grained parallelism)
- Tiling keeps each core's working set in L1 cache (fine-grained optimization)

Combining them: partition output rows across threads, then within each thread, use tiled loops for cache efficiency.

### Conv Combined Implementation

```
parallelFor(P, [&](size p_start, size p_end) {
    // Each thread processes its assigned output rows
    for each row p in [p_start, p_end):
        for each q tile:
            initialize accumulator with biases
            for r, s, c (filter window):
                for q in tile:
                    acc[q][m] += input[p,q,r,s,c] * weight[r,s,c,m]
            write output with activation
});
```

### Dense Combined Implementation

```
parallelFor(out_size, [&](size o_start, size o_end) {
    // Each thread handles a range of output neurons
    init output[o_start..o_end] with biases
    for each input tile:
        for each input i in tile:
            broadcast input[i] to output[o_start..o_end]
    apply activation
});
```

---

## 7. Optimization 5: Input Reuse (Eliminating Redundant Loads)

### Problem: M-Tile Loop Causes Redundant Input Reads

In the initial combined Conv implementation, the loop structure was:

```
for m0 = 0 to M step TILE_M:        // outer: output channel tile
    for p:                            // output row
        for q0:                       // output col tile
            for r, s, c:             // filter window
                for q in tile:
                    in_val = input[p,q,r,s,c]   // SAME input value
                    for m in tile:
                        acc[m] += in_val * weight[r,s,c,m0+m]
```

The input value `input[p,q,r,s,c]` is **identical** regardless of `m0`. With the `m0` loop on the outside, each input value is loaded `M/TILE_M` times:
- M=32, TILE_M=16: 2x redundant loads
- M=64, TILE_M=16: 4x redundant loads
- M=128, TILE_M=16: **8x redundant loads**

### Solution: Accumulate All M Channels At Once

By removing the `m0` tile loop and using a full-width accumulator:

```
for p:
    for q0:
        init acc[TILE_Q * M] with biases     // full M width
        for r, s, c:
            for q in tile:
                in_val = input[p,q,r,s,c]     // loaded ONCE
                for m in 0..M:
                    acc[q][m] += in_val * weight[r,s,c,m]
        write output
```

Each input value is now loaded exactly **once**.

### Why This Works

The accumulator grows from `TILE_Q * TILE_M` to `TILE_Q * M`:
| M | Old Accumulator | New Accumulator | Fits L1? |
|---|----------------|-----------------|----------|
| 32 | 256 bytes | 512 bytes | Yes |
| 64 | 256 bytes | 1 KB | Yes |
| 128 | 256 bytes | 2 KB | Yes |

Even for M=128 (Conv6), the accumulator is only 2 KB — easily fits in 32 KB L1.

### Result

| Layer | Before (ms) | After (ms) | Speedup |
|-------|------------|-----------|---------|
| Conv1 (M=32) | 0.239 | 0.174 | 1.37x |
| Conv2 (M=32) | 1.777 | 1.222 | 1.45x |
| Conv4 (M=64) | 0.564 | 0.346 | 1.63x |
| Conv8 (M=128) | 0.237 | 0.137 | 1.73x |

Higher M = bigger savings, as expected (M/TILE_M more redundant loads eliminated).

---

## 8. Optimization 6: Intelligent Dispatch

### Problem: Small Layers Don't Benefit From Threading/Tiling

Thread pool dispatch has fixed overhead (~0.05ms for mutex acquire, CV signal, context switches). For small layers:
- Dense2 (51K MACs) takes ~0.007ms naive — threading adds overhead, not speedup
- Small Conv layers with tiny weight tensors already fit in L1 — tiling just adds loop overhead

### Solution: MAC-Based and Size-Based Thresholds

```cpp
void computeThreaded(const LayerData& dataIn) const {
    if (getCachedMACs() < 100000) {
        computeNaive(dataIn);    // Too small for threading overhead
    } else {
        computeCombined(dataIn);
    }
}

void computeTiled(const LayerData& dataIn) const {
    size weight_bytes = getWeightParams().flat_count() * sizeof(fp32);
    if (weight_bytes < 8192) {
        computeNaive(dataIn);    // Already fits in L1, tiling adds no benefit
    } else {
        // ... tiled implementation
    }
}

void computeCombined(const LayerData& dataIn) const {
    if (getCachedMACs() < 100000) {
        computeTiled(dataIn);    // Fall back to tiled-only
    } else {
        // ... tiled + threaded implementation
    }
}
```

### Caching the MAC Count

`getNumMACs()` performs multiplication of dimension values. Rather than calling it every inference, we compute it **once** in `allocLayer()` and store it in a `cachedMACs` member:

```cpp
virtual void allocLayer() {
    outData.allocData();
    cachedMACs = getNumMACs();  // Computed once, read many times
}
```

### Result

Dense2 in THREADED mode: 0.062ms (threaded) → 0.007ms (falls back to naive). Small but clean — no wasted work.

---

## 9. Optimization 7: SIMD-Aligned Partitioning

### Problem Discovered via perf annotate

Using `perf annotate` on the Dense layer lambda, we found:

```asm
33.71%  vfmadd213ss  (%r10),%xmm11,%xmm15    ; SCALAR fused multiply-add
```

The `ss` suffix means **scalar single** — processing 1 float at a time. Despite the compiler auto-vectorizing the main loop body, the **scalar cleanup loop** was dominating.

**Root cause:** The thread pool splits 256 output neurons across, say, 12 threads, giving each ~21 neurons. The compiler vectorizes with AVX (8 floats), so 21 = 16 (vectorized) + 5 (scalar cleanup). But scalar instructions are 8x slower per element, so the 5 scalar iterations took disproportionate time.

### Solution: 8-Aligned Chunk Boundaries for Dense

Instead of letting the thread pool decide chunk sizes, Dense now computes its own 8-aligned partitioning:

```cpp
size chunk_aligned = ((chunk_base + 7) / 8) * 8;  // Round up to multiple of 8

parallelFor(nThreads, [&](size t_start, size t_end) {
    for (size t = t_start; t < t_end; t++) {
        size o_start = t * chunk_aligned;      // Always 8-aligned
        size o_end = (t+1 < nThreads) ? (t+1) * chunk_aligned : out_size;
        // ... inner loop now has o_len that's a multiple of 8
    }
});
```

### Why Not Align in ThreadPool Globally?

We tried aligning all chunk boundaries in the ThreadPool itself, but this **regressed** Conv performance badly. For Conv, the parallelFor is over P (output rows, e.g., P=56), and rounding to 8 created severe load imbalance — some threads got 8 rows, others got 0. Conv doesn't need alignment because its inner M loop is always full-width and already a multiple of 8.

**Lesson:** Alignment is layer-specific, not a one-size-fits-all ThreadPool concern.

### Result

Dense1: 0.094ms → 0.071ms (25% faster from eliminating scalar cleanup).

---

## 10. Optimization 8: MaxPooling Loop Reorder

### Problem: C-H-W Loop Order Causes Strided Access

The original MaxPooling iterated `for c, for h, for w`, accessing elements at stride C. In HWC layout, this means jumping across the entire row width for each element.

### Solution: H-W-C Loop Order

Reordered to `for h, for w, for c` with raw pointer access:

```cpp
for (size h = 0; h < h_out; h++) {
    const fp32* row0 = input + (2*h) * w_in * C;
    const fp32* row1 = input + (2*h+1) * w_in * C;
    for (size w = 0; w < w_out; w++) {
        for (size c = 0; c < C; c++) {
            // Access is now stride-1 in C (contiguous)
            out_ptr[c] = max(r0c0[c], r0c1[c], r1c0[c], r1c1[c]);
        }
    }
}
```

This also replaced `get<fp32>(index)` (which does bounds checking and pointer arithmetic) with raw `fp32*` pointer access.

### Result

MaxPooling: ~0.12ms → ~0.007ms per call (**17x speedup** on that layer). Small absolute time, but demonstrates how loop order interacts with memory layout.

---

## 11. SIMD Auto-Vectorization

### How the Compiler Uses SIMD

With `-O3 -march=native -funroll-loops`, the compiler automatically vectorizes loops where it detects:
1. **No loop-carried dependencies** (each iteration is independent)
2. **Contiguous memory access** (stride-1 in the inner loop)
3. **Simple arithmetic** (multiply, add, FMA)

Our inner convolution loop is an ideal SIMD target:
```cpp
for (size m = 0; m < M; m++) {
    acc[m] += in_val * w_ptr[m];
}
```

### What perf annotate Showed Us

The Conv combined lambda disassembly confirmed excellent vectorization:

```asm
vbroadcastss %xmm1, %ymm0           ; broadcast input value to all 8 lanes
vmovups      (%rcx,%r10,1), %ymm9   ; load 8 weights
vfmadd213ps  (%rax,%r10,1), %ymm0, %ymm9  ; acc[0:7] += in_val * w[0:7]
vmovups      %ymm9, (%rax,%r10,1)   ; store 8 results
```

- `vbroadcastss`: Broadcasts the scalar `in_val` to all 8 positions in a 256-bit YMM register
- `vfmadd213ps`: **Fused multiply-add** on 8 floats simultaneously (1 instruction instead of separate multiply + add)
- `ymm` registers: 256-bit AVX2, processing 8 fp32 values per instruction
- The loop is unrolled 8x by the compiler, processing 64 floats per iteration

### SIMD Width Summary

| Extension | Register | Width | FP32/instruction | Available |
|-----------|----------|-------|-------------------|-----------|
| SSE2 | xmm | 128-bit | 4 | Everywhere |
| AVX | ymm | 256-bit | 8 | Lab machines |
| AVX2+FMA | ymm | 256-bit | 8 (fused) | Lab machines |
| AVX-512 | zmm | 512-bit | 16 | Some servers |
| ARM NEON | q | 128-bit | 4 | Zedboard |

### Compiler Flags

- `-O3`: Enables auto-vectorization and aggressive optimization
- `-march=native`: Uses all SIMD extensions available on the current CPU
- `-funroll-loops`: Unrolls loops to reduce branch overhead and expose more ILP

---

## 12. Perf Profiling and Hardware Counter Analysis

### Using perf

```bash
# Profile the program
perf record -g ./build/ml

# See hottest functions
perf report

# See hottest INSTRUCTIONS inside a function
perf report --symbol-filter=computeCombined
# Then press 'a' to annotate

# Hardware counter analysis
perf stat -e cycles,instructions,cache-misses,cache-references,\
L1-dcache-load-misses,branch-misses,branches ./build/ml
```

### Our Hardware Counter Results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **IPC** | 2.38 | Excellent — CPU executes 2.4 instructions/cycle (not stalled) |
| **L1 cache miss rate** | 4.23% | Very good — tiling keeps data in L1 |
| **LLC miss rate** | 3.50% | Almost everything fits in cache hierarchy |
| **Branch mispredict** | 0.16% | Negligible — loop trip counts are predictable |
| **CPUs utilized** | 2.43 | Thread pool spreads work across cores |

### What Each Metric Tells You

**IPC (Instructions Per Cycle):**
- < 1.0: CPU is stalled — likely memory-bound (cache misses) or branch-bound
- 1.0–2.0: Decent utilization
- 2.0–4.0: Good — compute-bound, doing useful work
- Our 2.38: The CPU is busy doing FMA operations, not waiting for memory

**L1 Cache Miss Rate:**
- > 10%: Tiling isn't working or access patterns are strided
- 5–10%: Acceptable
- < 5%: Excellent — our 4.23% confirms tiling is effective

**Branch Misprediction:**
- > 5%: Consider eliminating branches (branchless code, lookup tables)
- < 1%: Not a concern — our 0.16% means loop branches are perfectly predicted

**Key insight:** With IPC=2.38 and low cache misses, we are **compute-bound**. Further speedup requires either algorithmic improvements (fewer MACs) or wider SIMD (AVX-512).

### perf report Function Breakdown

| Function | Self % | What It Tells Us |
|----------|--------|------------------|
| Conv computeCombined lambda | 37.5% | The actual convolution work — this is useful compute |
| Conv computeTiled | 20.9% | Time in the TILED test mode (separate run) |
| Conv computeNaive | 11.4% | Time in the NAIVE test mode (separate run) |
| Dense computeCombined lambda | 7.7% | Dense layer work |
| Kernel `[unknown]` | 18% | Thread synchronization, context switches |

---

## 13. Removing Dead Code (Quantization)

### What We Removed

The original code had int8 quantization paths gated by `Config::QUANT_BITS`:
```cpp
if (Config::QUANT_BITS == 32) {
    // fp32 path
} else {
    // int8 quantized path with scale factors, zero points, etc.
}
```

Since we always use 32-bit fp32, the quantized path was dead code. We removed:

- **Config.h:** `QuantMethod` enum, `QUANT_BITS`, `QUANT_MAX`, `QUANT_MIN`, `QUANT_METHOD`, `ENABLE_SIMD`
- **Convolutional.h / Dense.h:** `#include "Quantize.h"`, quantization members (`qWeights`, `wSums`, `wScale`), helper methods (`loadQuantizedWeights`, `loadScale`, `scaleIndex`)
- **Convolutional.cpp / Dense.cpp:** Entire `else` branches with int8 accumulation logic
- **allocLayer():** Simplified from conditional `if (QUANT_BITS == 32) loadData() else loadQuantizedWeights()` to just `weightData.loadData()`

### Why This Matters

1. **Readability:** Half the code in computeNaive was an unused quantization path
2. **Compile time:** Fewer includes and less template instantiation
3. **Maintainability:** One code path to understand and optimize instead of two
4. **Header cleanup:** Removed `<fstream>`, `<vector>`, `"Config.h"`, `"Quantize.h"` includes that were only needed for quantization

---

## 14. Performance Summary

### Overall Progression

| Optimization | Total (ms) | Speedup vs Naive |
|-------------|-----------|-----------------|
| Naive baseline | 23.1 | 1.0x |
| + Tiling | 16.9 | 1.37x |
| + Threading (raw std::thread) | 4.1 | 5.6x |
| + Thread Pool | 2.96 | 7.8x |
| + Combined (tiling + threading) | 2.80 | 8.3x |
| + Input reuse | 2.35 | 9.8x |
| + SIMD-aligned Dense | **2.09** | **11.1x** |

### Final Per-Layer Breakdown (Tiled+Threaded)

| Layer | Naive (ms) | Optimized (ms) | Layer Speedup |
|-------|-----------|----------------|---------------|
| Conv1 | 1.68 | 0.184 | 9.1x |
| Conv2 | 15.9 | 1.088 | 14.6x |
| Conv3 | 1.48 | 0.195 | 7.6x |
| Conv4 | 2.86 | 0.311 | 9.2x |
| Conv5 | 0.48 | 0.096 | 5.0x |
| Conv6 | 0.51 | 0.111 | 4.6x |
| Dense1 | 0.20 | 0.071 | 2.8x |
| Dense2 | 0.02 | 0.017 | 1.2x |

---

## 15. Key Takeaways

### Optimization Principles

1. **Profile before optimizing.** Conv2 was 69% of time — that's where effort should go.

2. **Optimizations are multiplicative when orthogonal:**
   - Tiling improves cache behavior (1.4x)
   - Threading uses multiple cores (4-8x)
   - SIMD processes multiple values per instruction (auto-vectorized)
   - Combined: 11x total

3. **Overhead matters for small workloads.** Thread pool dispatch (~0.05ms) exceeds Dense2's compute time (~0.007ms). Use thresholds to fall back to simpler code.

4. **Memory access patterns matter as much as compute.** MaxPooling got 17x faster just from reordering loops to match HWC layout (stride-1 vs stride-C access).

5. **The compiler is smart but needs help.** Auto-vectorization works well when:
   - Inner loops have no dependencies
   - Memory access is contiguous
   - Loop trip counts are multiples of SIMD width (8 for AVX)

6. **SIMD alignment at partition boundaries** prevents scalar cleanup loops from dominating. But alignment must be applied per-layer — global alignment causes load imbalance.

7. **Input reuse** is a key optimization for convolution. Broadcasting one input value across all output channels reduces memory traffic by M/TILE_M (up to 8x for M=128).

8. **Thread pool with generation counter** avoids race conditions. Each worker tracks its own generation, and only workers that actually perform work signal completion.

### What perf Tells You About "Done-ness"

When hardware counters show:
- IPC > 2.0 (compute-bound, not stalled)
- L1 miss rate < 5% (tiling works)
- Branch mispredict < 1% (loops are predictable)

...you've extracted most of the available performance. Further gains require algorithmic changes (Winograd, FFT-based convolution) or hardware changes (AVX-512, GPU).
