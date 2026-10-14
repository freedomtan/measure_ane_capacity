# Apple Silicon SME / SME2 / SME2.1 Kernels: Architecture, Dispatch, and Usage Guide

This document presents a comprehensive technical breakdown of Apple's Scaled Matrix Extension (**SME / SME2 / SME2.1**) kernels embedded within `libBNNS.dylib` (`Accelerate.framework`), CoreAI (`ODIE` / `AICode`), and hardware execution layers. All findings are empirically verified via reverse engineering and on-device execution across **Apple M4 Pro**, **iPhone 17 Pro (A19 Pro)**, and **iPhone 18 Pro (A20 Pro)**.

---

## 1. Overview & Hardware Support Matrix

Apple Silicon CPU matrix acceleration has evolved through three distinct eras:
1. **AMX (Apple Matrix Coprocessor)**: Proprietary, register-driven coprocessor instructions (`arm64_amx`, MLA1/MLA2/MLA3), accessible primarily via `vecLib` and Accelerate.
2. **SME1 & SME2 (A18 Pro / M4, A19 Pro)**: Standardized ARMv9.2-A architectural matrix state (`ZA` tiles, `SVCR.SM`, Streaming SVE 512-bit vectors).
3. **SME2.1 & FP8 (A20 Pro / iPhone 18 Pro)**: Introduction of `FEAT_SME2p1` and native widening byte-level floating point outer products (`FEAT_SME_F8F16`).

### Hardware Capabilities Across Generations

| Architecture Feature | sysctl Key | M4 Pro (Mac) | A19 Pro (iPhone 17 Pro) | A20 Pro (iPhone 18 Pro) |
| :--- | :--- | :---: | :---: | :---: |
| **SME Base** | `hw.optional.arm.FEAT_SME` | **1** | **1** | **1** |
| **SME2** | `hw.optional.arm.FEAT_SME2` | **1** | **1** | **1** |
| **SME2.1** | `hw.optional.arm.FEAT_SME2p1` | **1** | **1** | **1** |
| **FP16 Tile Accumulation** | `FEAT_SME_F16F16` | **1** | **1** | **1** |
| **BF16 Tile Accumulation** | `FEAT_SME_B16B16` | **1** | **1** | **1** |
| **Native FP8 Outer Product** | `hw.optional.arm.FEAT_SME_F8F16` | **0** | **0** | **1** |
| **SME2 Microbench Peak** | Assembly `FMOPA` | 4.30 TOPS | 4.30 TOPS | **9.69 TOPS** (FP8) |

---

## 2. The 162 SME Kernel Catalogue in `libBNNS.dylib`

Reverse engineering of `/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/libBNNS.dylib` reveals **162 dedicated SME / SME2 / SME2.1 kernels**. They fall into 8 primary workload categories:

### A. General Matrix Multiplication (GEMM / MatMul / SDPA)
- `KERNEL_GEMM_SME2`, `KERNEL_GEMM_SME2_DYNAMIC`
- `KERNEL_MATMUL_SME2`, `KERNEL_MATMUL_SME2_DYNAMIC`
- `KERNEL_MATMUL_SME2P1`, `KERNEL_MATMUL_SME2P1_DYNAMIC`
- `KERNEL_MATVEC_SME2`, `KERNEL_MATVEC_SME2_DYNAMIC`
- `KERNEL_EINSUM_STRIDED_GEMM_SME2_DYNAMIC`
- `KERNEL_EINSUM_STRIDED_MATMUL_SME2_DYNAMIC`
- `KERNEL_EINSUM_STRIDED_MATMUL_SME2P1_DYNAMIC`
- `KERNEL_SPARSE_MATMUL_SME2P1`, `KERNEL_SPARSE_MATMUL_SME2P1_DYNAMIC`
- `KERNEL_SDPA_SME2P1_GEMM`, `KERNEL_SDPA_SME2P1_DYNAMIC`, `KERNEL_SDPA_SME2P1_MULTI_TOKEN`
- `KERNEL_MATMUL_ELEMENTWISE_ACTIVATION_MATMUL_SME2`
- `KERNEL_MATMUL_SOFTMAX_MATMUL_SME`

### B. Quantized Matrix Computations (INT8 / Sub-8-bit)
- `KERNEL_INT8_MATVEC_SME2_1` *(Uses `sumopa` / `smopa`)*
- `KERNEL_CONV_CAPTURE_INT8_MATVEC_SME2_1`
- `KERNEL_CONV_CAPTURE_INT4_MATVEC_SME2_1`
- `KERNEL_CONV_CAPTURE_UINT4_MATMUL_SME2_1`
- `KERNEL_UINT4_LUT_TO_DENSE_MATMUL_SME`
- `KERNEL_MLIC_QUANTIZE_CONV_SME2`

### C. General & Transposed Convolutions
- `KERNEL_CONV_SME2`, `KERNEL_CONV_SME2_1`, `KERNEL_CONV_SME2_DYNAMIC`, `KERNEL_CONV_SME2_1_DYNAMIC`
- `KERNEL_CONV_3D_SME2_0`, `KERNEL_CONV_3D_SME2_1`
- `KERNEL_CONV_TRANS_1D_FP16_FP16_ACCUM_SME2_1`, `KERNEL_CONV_TRANS_1D_FP32_FP32_ACCUM_SME2_0`
- `KERNEL_CONV_TRANS_2D_FP16_FP16_ACCUM_SME2_1`, `KERNEL_CONV_TRANS_2D_FP32_FP32_ACCUM_SME2_0`
- `KERNEL_CONV_TRANS_2D_SME2`, `KERNEL_CONV_TRANS_2D_SME2_1`
- `KERNEL_CONV_TRANS_3D_SME2_0`, `KERNEL_CONV_TRANS_3D_SME2_1`
- `KERNEL_CONV_CAPTURE_ATTENTION_SME2_1`, `KERNEL_CONV_CAPTURE_FFN_SME2_1`

### D. Apple First-Party Audio & OS Workload Kernel Pools
Apple statically compiles hand-tuned SME2 kernel pools for production models:
- **FaceTime Voice Isolation (16 kernels)**:
  `KERNEL_CONV_KERNELPOOL_FACETIME_VOICE_ISOLATION_SME2_*` (e.g. `12WIDTH_NET_xI520`, `12x1x64OC`, `4BIT_PALETTIZED`).
- **Apple Music Sing (2 kernels)**:
  `KERNEL_CONV_KERNELPOOL_APPLE_MUSIC_SING_DEPTHWISE_OW128_IOC448_SME2`, `KERNEL_CONV_KERNELPOOL_APPLE_MUSIC_SING_SME2_S32_1_A_1_FP16`.
- **Telephony Audio (2 kernels)**:
  `KERNEL_CONV_KERNELPOOL_TELEPHONY_AUDIO_SME2_1_DEPTHWISE_32N_FP16`, `KERNEL_CONV_KERNELPOOL_TELEPHONY_AUDIO_SME2_1_VERTICAL_REDUCTION_FP16`.

### E. Speech Synthesis & LLM / Transformer Attention (SoundStorm & DepthFormer)
- Attention & KV-Cache:
  - `KERNEL_ATTENTION_SME`
  - `KERNEL_GASR_EINSUM_ATTENTION_SME2_1`, `KERNEL_GASR_EINSUM_CROSS_ATTENTION_SME2_1`
  - `KERNEL_HID_SSP_ATTENTION_SME2_1`
  - `KERNEL_TTS_NLP_MHA_SME`, `KERNEL_TTS_NLP_CROSS_MHA_SME`
  - `KERNEL_TTS_NLP_KV_CACHE_MHA_SME`, `KERNEL_TTS_NLP_KV_CACHE_CROSS_MHA_SME`
  - `KERNEL_TTS_NLP_KV_CACHE_DRAFT_MHA_SME` *(Draft speculative decoding)*
- SoundStorm FFN & SwiGLU:
  - `KERNEL_TTS_SOUNDSTORM_CONV_FC_SWIGLU_MODULE_SME`
  - `KERNEL_TTS_SOUNDSTORM_ENCODER_ROPE_MHA_SME` *(RoPE + MHA)*
  - `KERNEL_TTS_SOUNDSTORM_ENCODER_FFN_MF_SILU_SME`
  - `KERNEL_TTS_SOUNDSTORM_FFN_MF_NORM_X3_SME`

### F. Recurrent Neural Networks (LSTM / GRU / RNN)
- `KERNEL_GRU_SME2_0`, `KERNEL_GRU_SME2_1`, `KERNEL_GRU_FUSED_RESET_GATE_BACKWARD_SME2`
- `KERNEL_LSTM_SME2_0`, `KERNEL_LSTM_SME2_1`, `KERNEL_LSTM_MERGED_WEIGHTS_SME2_0`
- `KERNEL_RNN_SME2_0`, `KERNEL_RNN_SME2_1`, `KERNEL_RNN_MERGED_WEIGHTS_SME2_0`

### G. Vector & Streaming Arithmetic / Reductions
- `KERNEL_ARITHMETIC_ADD_*_DYNAMIC_SME2` (`FP16`, `BF16`, `FP32`, `RELU`)
- `KERNEL_ARITHMETIC_MUL_*_DYNAMIC_SME2`
- Scalar-Vector & Vector-Vector variants: `SVADD`, `SVMUL`, `VADD`, `VMUL`, `VSADD`, `VSMUL`.
- Reductions: `KERNEL_REDUCE_FP_SUM_SME`, `KERNEL_REDUCE_FP_SUM_SME2_DYNAMIC`, `KERNEL_REDUCE_MIN_MAX_SME`.

### H. Tensor Transformations & Memory
- `KERNEL_CAST_SME2_0`
- `KERNEL_CHANNELNORM_V2_SME2`, `KERNEL_CHANNELNORM_V2_SME2_1`
- `KERNEL_SPATIAL_NORM_SME2`
- `KERNEL_TRANSPOSE2D_SME2`, `KERNEL_TRANSPOSE2D_SME2_DYNAMIC`
- `KERNEL_COPY_DYNAMIC_SME_OR_NEON`, `KERNEL_COPY_SLICE_UPDATE_SME2_0`
- `KERNEL_VIO_POSE_SME` *(Vision Inertial Odometry tracking)*

---

## 3. How to Use SME Kernels: 3 Methods

### Method 1: The Modern Graph API (`BNNSGraphCompileFromFile`) [Recommended]

This is the canonical path introduced in macOS 15.0 / iOS 18.0 that replaces the deprecated `BNNSFilter*` APIs. It operates directly on compiled `.mlmodelc` bundles, builds the graph, and dispatches to `KERNEL_MATMUL_SME2P1` or `KERNEL_CONV_SME2`.

#### Implementation:
```objc
#import <Accelerate/Accelerate.h>

// 1. Configure options
bnns_graph_compile_options_t options = BNNSGraphCompileOptionsMakeDefault();
BNNSGraphCompileOptionsSetOptimizationPreference(options, BNNSGraphOptimizationPreferencePerformance);

// 2. Compile .mlmodelc directly to a runnable BNNS graph
bnns_graph_t graph = BNNSGraphCompileFromFile(mlmodelcPath, NULL, options);

// 3. Create execution context
bnns_graph_context_t context = BNNSGraphContextMake(graph);

// 4. Allocate page-aligned workspace memory (required by SME kernels)
size_t workspaceSize = BNNSGraphContextGetWorkspaceSize(context, NULL);
void *workspace = NULL;
if (workspaceSize > 0) {
    posix_memalign(&workspace, 16384, workspaceSize);
}

// 5. Populate graph argument bindings
bnns_graph_argument_t args[3];
args[xPos].data_ptr = input_buffer;
args[xPos].data_ptr_size = input_size_bytes;
args[wPos].data_ptr = weight_buffer;
args[wPos].data_ptr_size = weight_size_bytes;
args[yPos].data_ptr = output_buffer;
args[yPos].data_ptr_size = output_size_bytes;

// 6. Execute (Dispatches to KERNEL_MATMUL_SME2P1 on A19/A20)
int rc = BNNSGraphContextExecute(context, NULL, 3, args, workspaceSize, workspace);

// 7. Teardown
free(workspace);
BNNSGraphContextDestroy(context);
free(graph.data);
BNNSGraphCompileOptionsDestroy(options);
```

#### Physical Verification on iPhone 17 Pro:
Using the internal inspection symbols `BNNSGraphGetOperationCount` and `BNNSGraphGetOperationTypeName`:
```
[BNNSGraph FP16 weight-input] Compiled Op[0]: KERNEL_MATMUL_SME2P1
[BNNSGraph FP16 weight-input] Avg: 0.651 ms, Speed: 3.3010 TOPS

[BNNSGraph INT8 weight-input] Compiled Op[0]: KERNEL_CAST_GENERIC
[BNNSGraph INT8 weight-input] Compiled Op[1]: KERNEL_KBIT_DEQUANTIZE_SYMMETRIC_UINT8_INT8
[BNNSGraph INT8 weight-input] Compiled Op[2]: KERNEL_MATMUL_SME2P1
[BNNSGraph INT8 weight-input] Avg: 0.702 ms, Speed: 3.0597 TOPS
```

---

### Method 2: Hardware Engine Selection & Overrides

BNNS maintains internal engine preference logic:
$$\text{SME2.1 / SME2} \longrightarrow \text{AMX (Apple Matrix Coprocessor)} \longrightarrow \text{NEON}$$

You can steer kernel selection for debugging or benchmarking via internal environment variables:

1. **Force SME Hardware Generation (`ACCELERATE_C3AB8FF13720`)**:
   - `0x20000000`: Force SME1 capability word
   - `0x21000000`: Force SME2 capability word
   - `0x21000001`: Force SME2 + `FEAT_SME_F8F16`
2. **Cap Matrix Math Unit Generation (`ACCELERATE_83A752B02172`)**:
   - Setting to `0` disables AMX/MLA, forcing BNNS to route all matrix workloads to SME2.

#### Benchmark Isolation on iPhone 17 Pro ($1024 \times 1024$ FP16):
- **SME2 (`KERNEL_MATMUL_SME2P1`)**: **0.658 ms** (**3.265 TOPS**)
- **NEON Fallback (`KERNEL_MATMUL_NEON`)**: **10.006 ms** (**0.215 TOPS**)
- **Speedup**: **15.2× faster** than Neon vector units.

---

### Why `KERNEL_MATMUL_SME2P1` is NOT Used by Apple M4

Although the Apple M4 family (M4, M4 Pro, M4 Max) reports `hw.optional.arm.FEAT_SME = 1`, `hw.optional.arm.FEAT_SME2 = 1`, and `hw.optional.arm.FEAT_SME2p1 = 1`, BNNS Graph **never** selects `KERNEL_MATMUL_SME2P1` on M4. Instead, M4 executes `KERNEL_MATMUL_MLA3` (AMX).

Through disassembly and crash diagnosis of `libBNNS.dylib`, we identified the exact two architectural reasons for this separation:

#### Reason 1: The `FEAT_CSSC` Hardware Dependency
In `libBNNS.dylib`, the SME2.1 kernel generation (`0x21000000`) is compiled assuming the host microarchitecture is ARMv9.4-A (chipset IDs `0xbbd`, `0xbbe`, `0xbbf`, representing A19/A20).

Crucially, Apple's SME2.1 tiling loop prologue uses ARMv8.9-A / ARMv9.4-A **`FEAT_CSSC`** (Common Short Sequence Compression) scalar instructions to compute tile boundary clamp bounds:
```assembly
// From libBNNS.dylib at 0x18eccc954 (tiling prologue right before smstart):
0x18eccc950  sub   x9, x10, x22
0x18eccc954  umin  x10, x9, x20      <--- Instruction: umin (0x9ad46d2a, requires FEAT_CSSC)
...
0x18ecccd4c  smstart sm             <--- SME2 tile engine entry
```

**What is `FEAT_CSSC`?**
- `FEAT_CSSC` (Common Short Sequence Compression) is an ARM architectural extension introduced in ARMv8.9-A and ARMv9.4-A.
- It compresses multi-instruction scalar ALU sequences (such as `CMP` + `CSEL` for integer min/max, absolute values via `ABS`, and population count via `CNT`) into single-cycle, single-instruction primitives like `UMIN`, `UMAX`, `SMIN`, `SMAX`.
- **Generational Support**:
  - **Apple M4 / M4 Pro / M4 Max**: ARMv9.2-A core architecture. Lacks `FEAT_CSSC` (`sysctl hw.optional.arm.FEAT_CSSC: 0`).
  - **A19 Pro (iPhone 17 Pro) / A20 Pro**: ARMv9.4-A+ core architecture. Implements `FEAT_CSSC` (`sysctl hw.optional.arm.FEAT_CSSC: 1`).
- If SME2.1 is forced on M4 via environment override `ACCELERATE_C3AB8FF13720=0x21000000`, the M4 immediately crashes with `EXC_BAD_INSTRUCTION` (`SIGILL`) on `umin` before reaching the `smstart` instruction.

#### Reason 2: Asymmetric Compute Balance (AMX on Mac vs SME2 on Mobile)
On mobile chips (A18/A19 Pro), CPU core counts are small (2 P-cores, 4 E-cores), and the SME2 cluster co-processor provides the primary matrix throughput.

On Mac desktop chips (M4 Pro / M4 Max):
- M4 Pro features multiple Performance core clusters, each equipped with dedicated **AMX (Apple Matrix Coprocessor)** units (`MLA3`) connected directly to L1/L2 caches.
- Measuring pure CPU matrix compute on the host M4 Pro:
  - **M4 Pro CPU AMX FP16 (`KERNEL_MATMUL_MLA3`)**: **6.71 TOPS** (sustained across M4 Pro Performance clusters at $2048 \times 2048$, 2.56 ms).
  - **M4 Pro SME2 FP16 (`fmopa za.h`)**: **4.30 TOPS** (single-cluster peak limit).
- Because M4 Pro's multi-cluster AMX provides higher aggregate CPU throughput (**6.71 TOPS** vs. single-cluster SME2's **4.30 TOPS**), BNNS Graph's compiler heuristics on macOS naturally prefer `KERNEL_MATMUL_MLA3` (AMX). On mobile A19 Pro, where core counts are low, BNNS selects `KERNEL_MATMUL_SME2P1`.

---

### Method 3: Direct Assembly Invocation (Zero-Overhead Peak Compute)

When you need the absolute maximum hardware throughput without MLIR graph parsing or dequantization overhead—such as reaching the **9.69 TOPS FP8 peak on A20 Pro**—direct assembly is the method of choice.

#### Direct SME2 FP16 Outer Product Assembly:
```assembly
.arch armv9-a+sme+sme2
.text
.globl _sme2_fp16_gemm_block

_sme2_fp16_gemm_block:
    // Enter Streaming SVE Mode & Allocate ZA Storage
    smstart sm
    smstart za

    // Predicate 512-bit vectors (32 x 16-bit elements per vector)
    ptrue   p0.h
    ptrue   p1.h

    // Load matrices A and B into Z registers
    ld1h    { z0.h }, p0/z, [x0]
    ld1h    { z1.h }, p1/z, [x1]

    // SME2 2-Way Outer Product Accumulate: za0.h += z0.h * z1.h^T
    fmopa   za0.h, p0/m, p1/m, z0.h, z1.h

    // Exit Streaming Mode
    smstop  za
    smstop  sm
    ret
```

#### Direct SME2 FP8 Outer Product Assembly (`FEAT_SME_F8F16`, A20 Pro Exclusive):
```assembly
.arch armv9-a+sme+sme2+sme-f8f16
.text
.globl _sme2_fp8_gemm_block

_sme2_fp8_gemm_block:
    smstart sm
    smstart za

    // Predicate 64 bytes per 512-bit SVE vector
    ptrue   p0.b
    ptrue   p1.b

    // Load 64 FP8 bytes (E4M3 or E5M2)
    ld1b    { z0.b }, p0/z, [x0]
    ld1b    { z1.b }, p1/z, [x1]

    // Native FP8 Widening Outer Product into 16-bit Float Tile
    fmopa   za0.h, p0/m, p1/m, z0.b, z1.b

    smstop  za
    smstop  sm
    ret
```

---

## 4. Summary & Best Practices

1. **For Production App Development**:
   Use **CoreML / MIL** compiled to `.mlmodelc` and executed via `BNNSGraphCompileFromFile`. As demonstrated on the iPhone 17 Pro, it automatically targets **`KERNEL_MATMUL_SME2P1`** at over **3.3 TOPS**.
2. **For Quantized Models**:
   Weights quantized via CoreML weight compression (`constexpr_lut_to_dense` or `constexpr_blockwise_shift_scale`) will automatically dequantize to FP16 and stream through `KERNEL_MATMUL_SME2P1`.
3. **For Research / Peak Compute**:
   To exploit native FP8 matrix multipliers on A20 Pro hardware without FP16 memory expansion, direct assembly with `FEAT_SME_F8F16` (`fmopa za.h, ..., z.b, z.b`) yields **9.69 TOPS**.
