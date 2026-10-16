# Apple Silicon SME & SME2 Generational Benchmark Analysis

## Overview

This report documents the architectural capabilities and measured peak performance of Apple's Scalable Matrix Extension (SME) and Scalable Matrix Extension 2 (SME2) implementations across four consecutive hardware implementations:
- **iPhone 16 Pro** (Apple A18 Pro)
- **iPhone 17 Pro** (Apple A19 Pro)
- **iPhone 18 Pro** (Apple A20 Pro)
- **Mac mini M6** (Apple M6, `Mac18,5`, Super Core)

All measurements were taken on physical devices using native microbenchmarks under `User Interactive` Quality of Service (QoS).

---

## Architectural Feature Support by Generation

| Architecture Feature | Instruction / Capability | iPhone 16 Pro (A18 Pro) | iPhone 17 Pro (A19 Pro) | iPhone 18 Pro (A20 Pro) | Apple M6 (Mac18,5) | Introduction Milestone |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| **FEAT_SME** | Streaming SVE, ZA tile array (512-bit SVL) | **Supported** | **Supported** | **Supported** | **Supported** | Base SME |
| **FEAT_SME2** | Multi-vector SME2 ops (e.g. `bfdot`) | **Supported** | **Supported** | **Supported** | **Supported** | Base SME2 |
| **FEAT_SME2p1** | SME 2.1 extensions | Unsupported | **Supported** | **Supported** | **Supported** | **Apple A19 Pro / M6** |
| **FEAT_CSSC** | Common Short Sequence Compression | Unsupported | **Supported** | **Supported** | **Supported** | **Apple A19 Pro / M6** |
| **FEAT_SME_F16F16** | Non-widening FP16 $\to$ FP16 `FMOPA` | Unsupported | **Supported** | **Supported** | **Supported** | **Apple A19 Pro (iPhone 17 Pro)** |
| **FEAT_SME_B16B16** | Non-widening BF16 $\to$ BF16 `BFMOPA` | Unsupported | **Supported** | **Supported** | **Supported** | **Apple A19 Pro (iPhone 17 Pro)** |
| **FEAT_SME_F8F16** | Widening FP8 $\to$ FP16 `FMOPA` | Unsupported | Unsupported | **Supported** | **Supported** | **Apple A20 Pro / M6** |
| **FEAT_SME_F8F32** | Widening FP8 $\to$ FP32 `FMOPA` | Unsupported | Unsupported | **Supported** | **Supported** | **Apple A20 Pro / M6** |

> [!NOTE]
> All tested chips feature an SVL (Streaming Vector Length) of **512 bits** (64 bytes). Therefore, a full ZA matrix tile array contains:
> - 64 rows of 64 bytes = 4,096 bytes total.
> - Formatted as 32-bit tiles: 4 tiles (`za0.s`..`za3.s`) of $16 \times 16$ elements.
> - Formatted as 16-bit tiles: 2 tiles (`za0.h`, `za1.h`) of $32 \times 32$ elements.

---

## Benchmark Results (Single-Thread Peak, User Interactive QoS)

### Matrix Outer-Product (SME / SME2) Throughput

| Precision & Operation | Instruction Syntax | Destination Tile | iPhone 16 Pro (A18 Pro) | iPhone 17 Pro (A19 Pro) | iPhone 18 Pro (A20 Pro) | Apple M6 (Super Core) |
|:---|:---|:---:|---:|---:|---:|---:|
| **SME2 FP8 $\to$ FP16 (`FEAT_SME_F8F16`)** | `fmopa za[0..1].h, ..., Zn.b, Zm.b` | `.H` (16-bit) | *Unsupported* | *Unsupported* | **9,694.24 GOPS (~9.69 TOPS)** | **9,134.83 GOPS (~9.13 TOPS)** |
| **SME2 FP8 $\to$ FP32 (`FEAT_SME_F8F32`)** | `fmopa za[0..3].s, ..., Zn.b, Zm.b` | `.S` (32-bit) | *Unsupported* | *Unsupported* | **4,854.51 GOPS (~4.85 TOPS)** | **4,545.53 GOPS (~4.55 TOPS)** |
| **SME FP16 $\to$ FP16 (`FEAT_SME_F16F16`)** | `fmopa za[0..1].h, ..., Zn.h, Zm.h` | `.H` (16-bit) | *Unsupported* | **4,304.14 GOPS (~4.30 TOPS)** | **4,854.56 GOPS (~4.85 TOPS)** | **4,533.39 GOPS (~4.53 TOPS)** |
| **SME BF16 $\to$ BF16 (`FEAT_SME_B16B16`)** | `bfmopa za[0..1].h, ..., Zn.h, Zm.h` | `.H` (16-bit) | *Unsupported* | **4,303.26 GOPS (~4.30 TOPS)** | **4,854.84 GOPS (~4.85 TOPS)** | **4,553.24 GOPS (~4.55 TOPS)** |
| **SME INT8 $\to$ INT32 (`FEAT_SME_I8I32`)** | `smopa za[0..3].s, ..., Zn.b, Zm.b` | `.S` (32-bit) | **3,947.04 GOPS** | **4,303.19 GOPS** | **4,854.91 GOPS (~4.85 TOPS)** | **4,540.37 GOPS (~4.54 TOPS)** |
| **SME FP16 $\to$ FP32 (`FEAT_SME_F16F32`)** | `fmopa za[0..3].s, ..., Zn.h, Zm.h` | `.S` (32-bit) | 1,973.23 GOPS | 2,149.31 GOPS | 2,427.20 GOPS | 2,268.54 GOPS (~2.27 TOPS) |
| **SME BF16 $\to$ FP32 (`FEAT_SME_B16F32`)** | `bfmopa za[0..3].s, ..., Zn.h, Zm.h` | `.S` (32-bit) | 1,973.46 GOPS | 2,151.53 GOPS | 2,427.20 GOPS | 2,274.31 GOPS (~2.27 TOPS) |
| **SME FP32 $\to$ FP32 (4 tiles)** | `fmopa za[0..3].s, ..., Zn.s, Zm.s` | `.S` (32-bit) | 1,973.28 GOPS | 2,151.81 GOPS | 2,427.69 GOPS | 2,275.94 GOPS (~2.28 TOPS) |
| **SME FP64 $\to$ FP64** | `fmopa za0.d, ..., Zn.d, Zm.d` | `.D` (64-bit) | 493.43 GOPS | 538.20 GOPS | 606.79 GOPS | 568.51 GOPS (~0.57 TOPS) |
| **SME BF16 BFDOT** | `bfdot za.s[wN, #0], {Zn.h, ...}, ...` | `.S` (32-bit) | 607.39 GOPS | 661.97 GOPS | 747.23 GOPS | *Available in SME2* |

### Baseline Vector (NEON & SSVE) Throughput

| Kernel | Precision | iPhone 16 Pro (A18 Pro) | iPhone 17 Pro (A19 Pro) | iPhone 18 Pro (A20 Pro) | Apple M6 (Super Core) |
|:---|:---:|---:|---:|---:|---:|
| **NEON FP16 FMLA** | FP16 | 204.08 GOPS | 230.28 GOPS | 261.41 GOPS | 257.99 GOPS (1C) / 503.81 GOPS (2C) |
| **NEON FP32 FMLA** | FP32 | 101.85 GOPS | 115.02 GOPS | 130.41 GOPS | 129.11 GOPS (1C) / 252.03 GOPS (2C) |
| **NEON FP64 FMLA** | FP64 | 49.46 GOPS | 56.23 GOPS | 64.36 GOPS | 52.87 GOPS (1C) / 120.92 GOPS (2C) |
| **NEON BF16 BFMMLA** | BF16/FP32 | 61.66 GOPS | 65.51 GOPS | 78.73 GOPS | 76.30 GOPS (1C) / 149.41 GOPS (2C) |
| **Streaming-SVE FP32 FMLA** | FP32 | 30.83 GOPS | 33.62 GOPS | 37.93 GOPS | 35.69 GOPS |
| **Streaming-SVE FP64 FMLA** | FP64 | 15.42 GOPS | 16.80 GOPS | 18.96 GOPS | 17.84 GOPS |

---

## Architectural & Performance Analysis

### 1. The $2\times$ Step: Non-Widening Half-Precision in A19 Pro (`FEAT_SME_F16F16` & `B16B16`)
* **Prior Generations (A18 Pro)**: All half-precision outer products were forced to widen into 32-bit singleword accumulators (`za0.s`..`za3.s`). As a result, FP16 and BF16 peak throughput was identical to FP32 throughput (~1.97 TOPS).
* **A19 Pro Breakthrough**: With `FEAT_SME_F16F16` and `FEAT_SME_B16B16`, the engine can accumulate directly into the two $32 \times 32$ 16-bit tiles (`za0.h` and `za1.h`). Because each vector contains twice as many elements and does not require widening datapath splits, throughput **doubled immediately to 4.30 TOPS** (matching INT8 `SMOPA`).

### 2. The Next $2\times$ Step: Native FP8 Outer Products in A20 Pro (`FEAT_SME_F8F16`)
* **A20 Pro Breakthrough**: Apple introduced `FEAT_SME_F8F16` and `FEAT_SME_F8F32` on the CPU SME unit.
* When accumulating into `.H` tiles (`FEAT_SME_F8F16`), 8-bit inputs provide another $2\times$ density improvement, reaching **~9.70 TOPS on a single CPU core**.
* This is:
  - **$2\times$** the INT8 `SMOPA` rate.
  - **$2\times$** the non-widening FP16/BF16 rate (`F16F16`).
  - **$4\times$** the baseline FP32 / widening FP16 rate.

### 3. Generational Clock & Datapath Scaling
Beyond the structural ISA additions, clock frequency and execution efficiency improved monotonically across all three chips:
- **A18 Pro $\to$ A19 Pro**: ~9.0% frequency/throughput uplift on identical baseline kernels (e.g. FP32 FMOPA: 1,973.3 $\to$ 2,151.8 GOPS).
- **A19 Pro $\to$ A20 Pro**: ~12.8% frequency/throughput uplift on identical baseline kernels (e.g. FP32 FMOPA: 2,151.8 $\to$ 2,427.7 GOPS; INT8 SMOPA: 4,303.2 $\to$ 4,854.9 GOPS).

### 4. Integer vs. Floating Point Accumulation Asymmetry
While FP8 can target both 16-bit (`F8F16`) and 32-bit (`F8F32`) accumulators, INT8 in Arm SME is strictly limited to 32-bit integer accumulation (`I8I32`). There is no architectural `I8I16` instruction in SME or SME2, as 16-bit signed integer accumulation overflows after only 2 to 3 vector dot products without exponent scaling.

### 5. Apple M6 Desktop Scaling & Super Core Cluster Topology
On the Apple M6 desktop chip (`Mac18,5`):
- **Core Organization:** The CPU features 12 cores split across three performance levels: 2 Super cores (`perflevel0`, 192KB L1-I, 128KB L1-D, 20MB L2), 4 Performance cores (`perflevel1`, 128KB L1-I, 64KB L1-D, 20MB L2), and 6 Efficiency cores (`perflevel2`, 160KB L1-I, 96KB L1-D, 8MB L2).
- **Cluster Sharing:** Similar to the Apple AMX coprocessor on prior generations (M1–M4), the physical 512-bit SME matrix execution unit is shared across cores in a performance cluster. Running 2 concurrent threads on the 2 Super cores shares the ~4.54 TOPS (FP16/INT8) and ~9.13 TOPS (FP8) peak matrix pipe, while standard Neon SIMD execution scales linearly ($2\times$) across both cores (Neon FP16 reaches 503.81 GOPS on 2 cores).
- **BNNS Dispatch Milestone:** Unlike Apple M4 (which lacked `FEAT_SME2p1` and fell back to AMX `KERNEL_MATMUL_MLA3`), Apple M6 natively routes `bnns_graph_matmul` to `KERNEL_MATMUL_SME2P1`, realizing **4.17 TOPS** FP16 GEMM ($4096 \times 4096$) on the CPU.

---

## Acknowledgements & References

The standalone ARM SME / SME2 assembly microkernels (`fmopa`, `bfmopa`, `smopa`, `FEAT_SME_F16F16`, `FEAT_SME_B16B16`, `FEAT_SME_F8F16`, `FEAT_SME_F8F32`), vector loop structures, and benchmark harnesses used throughout this generational evaluation are based on and extended from:
* **Repository**: [https://github.com/scalable-analyses/sme](https://github.com/scalable-analyses/sme)
* **Citation**: Scalable Analyses SME Microbenchmarking Suite (`scalable-analyses/sme`).

