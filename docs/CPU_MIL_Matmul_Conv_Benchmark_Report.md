# Comprehensive CPU Matrix Multiplication and Convolution Benchmark Report

## 1. Executive Summary

This report establishes the first complete, rigorous performance baseline for Apple Silicon **CPU matrix multiplication and convolution** using CoreML/MIL (`MLComputeUnitsCPUOnly`), spanning multiple chip generations:
- **Apple M4 Pro** (macOS, 10 P-cores / 2 Performance clusters)
- **Apple A18 Pro** (iPhone 16 Pro, 2 P-cores / 1 Performance cluster)
- **Apple A19 Pro** (iPhone 17 Pro, 2 P-cores / 1 Performance cluster)
- **Apple A20 Pro** (iPhone 18 Pro, 2 P-cores / 1 Performance cluster)

We evaluate **FP16, INT8, and FP8** data formats, trace the exact low-level instruction execution pathways inside `libBNNS.dylib`, and cross-compare CoreML's software behavior against raw, hand-written **SME and SME2 assembly microkernels**.

---

## 2. Benchmark Workloads & Methodology

### CoreML/MIL Workloads
1. **Matrix Multiplication (GEMM)**:
   - Shape: $[1, 1, 1024, 1024] \times [1, 1, 1024, 1024]$ across $L=20$ chained layers.
   - Workload density: 42.95 GOPs per evaluation pass.
   - Executed via `measure_matmul_coreml` (macOS) and `NativeCoreMLMatmulProbe` inside `ANECapacityApp` (iOS).
2. **Convolution (Conv2D)**:
   - Shape: $[1, 128, 256, 256]$ input, 20 sequential layers of $3 \times 3$ convolutions ($C_{in} = C_{out} = 128$, stride 1).
   - Workload density: 386.55 GOPs per evaluation pass.
   - Executed via `measure_conv_coreml` (macOS).

### Quantization Formats in MIL
- **FP16**: Dense 16-bit floating-point weights and activations.
- **FP8 (Weight-Only)**: Float8-E4M3 raw bytes expanded via a 256-entry IEEE Float8-to-FP16 lookup table (`constexpr_lut_to_dense`).
- **INT8 (W8A8 QDQ)**: Weights compressed via `constexpr_blockwise_shift_scale`; activations wrapped in explicit runtime `quantize` and `dequantize` pairs around each layer (`act_scale = 1/32`).
- **INT8W (Weight-Only)**: Same INT8 weight constants, but without per-layer activation QDQ pairs.

---

## 3. Measured CoreML / MIL CPU Performance

### 3.1 Matrix Multiplication Throughput ($1024^3 \times L20$)

| Hardware Target | CPU Cluster Topology | FP16 Throughput | FP8 (LUT-dequant) | INT8 (W8A8 QDQ) | INT8W (Weight-Only) |
|:---|:---:|---:|---:|---:|---:|
| **Apple M4 Pro** (Mac) | 2 P-clusters (10 P-cores) | **6.62 TOPS** (6.49 ms) | **6.63 TOPS** (6.48 ms) | **4.89 TOPS** (8.79 ms) | **6.61 TOPS** (6.50 ms) |
| **Apple A20 Pro** (iPhone 18 Pro) | 1 P-cluster (2 P-cores) | **4.04 TOPS** (10.63 ms) | **4.06 TOPS** (10.58 ms) | **3.55 TOPS** (12.10 ms) | — |
| **Apple A19 Pro** (iPhone 17 Pro) | 1 P-cluster (2 P-cores) | **3.19 TOPS** (13.46 ms) | **3.32 TOPS** (12.94 ms) | **2.93 TOPS** (14.66 ms) | — |
| **Apple A18 Pro** (iPhone 16 Pro) | 1 P-cluster (2 P-cores) | **3.13 TOPS** (13.72 ms) | **3.16 TOPS** (13.59 ms) | **3.00 TOPS** (14.32 ms) | — |

*All runs verified with deterministic weights and finite, non-degenerate output checks.*

### 3.2 Convolution Throughput (M4 Pro Baseline)

| Workload | CoreML CPU FP16 | CoreML GPU FP16 | CoreML ANE FP16 |
|:---|---:|---:|---:|
| **Conv2D $[1,128,256,256]$ / $3\times3$, $L=20$** | **3.09 TOPS** (8.0 FPS) | 9.92 TOPS (25.7 FPS) | 19.10 TOPS (49.4 FPS) |

---

## 4. Reverse-Engineering the CoreML CPU Execution Path

Live profiling via `sample` and interactive instruction-stepping via `lldb` revealed two critical architectural discoveries about how CoreML executes CPU workloads:

```
[CoreML Prediction Call]
       │
       ▼
[MLE5Engine / Espresso E5RT]
       │
       ▼
[BNNSGraphContextExecute_v2 (libBNNS.dylib)]
       │
       ▼
[dispatch_apply_with_attr (libdispatch.dylib)]   <-- MULTI-THREADED ACROSS CPU CORES
       │
       ├──> Thread 0 (Main Thread): Tile Slice 0 ──> AMX Microkernel (0x18f0623a4)
       ├──> Thread 1 (Worker Thread): Tile Slice 1 ──> AMX Microkernel (0x18f0623a4)
       └──> Thread 2 (Worker Thread): Tile Slice 2 ──> AMX Microkernel (0x18f0623a4)
```

### Discovery 1: CoreML CPU is Fully Multi-Threaded Across Cores
When executing a model under `MLComputeUnitsCPUOnly`, `libBNNS.dylib` partitions matrix multiplication across worker threads via `dispatch_apply_with_attr`. Threads on `com.apple.root.user-interactive-qos` concurrently process tile blocks.

### Discovery 2: The Underlying Compute Engine is Apple AMX, Not SME2
Disassembly of the actual hot compute kernel (`0x18f0623a4`) hit by live execution proves that BNNS executes **Apple's undocumented AMX coprocessor**, not SME2:
- `0x00201009..0d` $\to$ AMX `ldx` (load X staging register)
- `0x00201029` $\to$ AMX `ldy` (load Y staging register)
- `0x0020108a` $\to$ AMX `ldz` (load Z accumulator tile)
- `0x00201268..6d` $\to$ AMX op 19 (outer-product multiply-accumulate into Z)

Even though `libBNNS.dylib` contains real SME2 instructions (`smstart za`, `zt0`, `luti4`, `fmopa`), the runtime scheduler routes dense matmul directly to the legacy AMX block.

---

## 5. Architectural Explanation of Format Behaviors

1. **Why FP8 Matches FP16 Exactly (~4.05 TOPS on A20 Pro, ~6.63 TOPS on Mac)**:
   - In CoreML, FP8 weights are expanded at runtime by `constexpr_lut_to_dense` to FP16 before reaching the matrix multiply op.
   - The AMX coprocessor executes a normal FP16 matmul. FP8 delivers 50% model footprint reduction and memory bandwidth relief, but identical compute throughput.
2. **Why INT8W (Weight-Only) Matches FP16 Exactly (6.61 vs 6.62 TOPS)**:
   - `weights = dequantize(int8_weights, scale)` consists entirely of compile-time constants.
   - CoreML's MIL compiler **constant-folds** the dequantization pass during compilation, baking dense FP16 weights directly into the compiled model artifact.
3. **Why INT8 (W8A8 QDQ) Regresses (3.55 vs 4.04 TOPS on A20 Pro; 4.89 vs 6.62 TOPS on Mac)**:
   - To prevent constant folding, the W8A8 graph inserts runtime `quantize` and `dequantize` operations on the dynamic activation tensor.
   - Because AMX still carries out the inner matrix multiply using 16-bit accumulators, the per-layer quantization overhead results in an 8–12% throughput penalty on iOS and a 26% penalty on macOS.

---

## 6. Comparing CoreML (AMX) Against Raw SME / SME2 Microbenchmarks

Both Apple AMX and ARM SME matrix coprocessors are architected **per CPU cluster** (attached to the shared cluster L2 cache), rather than per CPU core.

### Empirical Scaling Across Threads (M4 Pro Benchmark)
Benchmarking raw SME FP32 FMOPA across threads proves the per-cluster coprocessor sharing:
- **1 Thread**: **1,996 GOPS** (~2.0 TOPS)
- **2 Threads**: **3,924 GOPS** (~3.92 TOPS) — *2× linear scaling (utilizing both P-clusters)*
- **3 to 10 Threads**: **Plateaus at ~4.1 TOPS** (saturating the 2 physical SME engines)

### Head-to-Head Comparison: CoreML (AMX) vs. SME2 Hardware (iPhone A20 Pro)

On the iPhone 18 Pro (A20 Pro), there is **exactly one Performance cluster** (2 P-cores sharing 1 matrix coprocessor block). We can directly compare the cluster's CoreML AMX throughput against its native SME2 hardware capability:

| Framework / Engine | Workload / Precision | Target Hardware | Coprocessor Architecture | Measured Performance (A20 Pro) |
|:---|:---|:---:|:---:|---:|
| **CoreML / MIL CPU** | FP16 Matmul ($1024^3$) | **AMX** (Multi-thread) | Shared Cluster Coprocessor | **4.04 TOPS** |
| **CoreML / MIL CPU** | FP8 Weight-Only ($1024^3$) | **AMX** (Multi-thread) | Shared Cluster Coprocessor | **4.06 TOPS** |
| **CoreML / MIL CPU** | INT8 W8A8 QDQ ($1024^3$) | **AMX** (Multi-thread) | Shared Cluster Coprocessor | **3.55 TOPS** |
| **Direct SME Assembly** | FP32 / FP16 widening | **SME** (`FEAT_SME`) | Shared Cluster Coprocessor | **2.43 TOPS** |
| **Direct SME Assembly** | FP16 $\to$ FP16 non-widening | **SME** (`FEAT_SME_F16F16`) | Shared Cluster Coprocessor | **4.85 TOPS** |
| **Direct SME Assembly** | INT8 $\to$ INT32 widening | **SME** (`FEAT_SME_I8I32`) | Shared Cluster Coprocessor | **4.85 TOPS** |
| **Direct SME2 Assembly** | FP8 $\to$ FP32 widening | **SME2** (`FEAT_SME_F8F32`) | Shared Cluster Coprocessor | **4.85 TOPS** |
| **Direct SME2 Assembly** | FP8 $\to$ FP16 widening | **SME2** (`FEAT_SME_F8F16`) | Shared Cluster Coprocessor | **9.69 TOPS** |

---

## 7. Conclusions & Insights

1. **Topology Governs CPU Throughput**:
   - The M4 Pro achieves **~6.6 TOPS** because it contains **2 Performance clusters** running 2 concurrent AMX blocks.
   - The iPhone A-series chips achieve **~3.1–4.0 TOPS** because they contain **1 Performance cluster** running 1 AMX block.
2. **CoreML CPU is Conservative**:
   - For dense matrix multiplication, CoreML continues to dispatch Apple's proprietary AMX coprocessor.
   - The hardware's newer, public ARM SME/SME2 extensions (`FEAT_SME_F16F16`, `FEAT_SME_F8F16`) offer **$2\times$ to $2.4\times$ higher throughput (4.85–9.69 TOPS)** on the identical silicon, but require direct assembly or lower-level interfaces until Apple's compiler updates its graph routing.

---

## 8. Acknowledgements & References

The standalone ARM SME / SME2 microkernels and hardware baseline measurements referenced in this report are derived from and build upon the open-source suite by [scalable-analyses/sme](https://github.com/scalable-analyses/sme).

