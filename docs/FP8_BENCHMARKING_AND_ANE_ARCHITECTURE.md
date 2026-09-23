# FP8 Benchmarking & Apple Neural Engine (ANE) Architecture Reference

This document details the technical implementation, mathematical stability constraints, hardware support matrix, and benchmarking methodology for **FP8 (Float8E4M3) Quantize-Dequantize (QDQ)** execution across Metal GPU and the Apple Neural Engine (ANE).

---

## 1. Executive Summary & Technical Background

* **Format**: `MPSDataTypeFloat8e4m3` (E4M3: 1 sign bit, 4 exponent bits, 3 mantissa bits, bias = 7; IEEE 754-style). `E5M2` is unsupported by MPSGraph quantization passes.
* **OS Requirement**: macOS 27.0+ / iOS 27.0+.
* **Core Constraint**: `mps.conv_2d` and `mps.matmul` do not accept raw FP8 tensor operands. All FP8 arithmetic in MPSGraph must follow a **Quantize-Dequantize (QDQ)** pattern where computation executes in FP16.
* **ANE Generation Matrix**:
  * **H13–H16 (A14–A17 Pro, M1–M4 Pro)**: Physical ANE MAC units lack FP8 ALUs. ANE compilation rejects FP8 MLIR; MPSGraph falls back to Metal GPU automatically.
  * **H18 (A19 / iPhone 18 Pro)**: ANE compiler accepts FP8 MLIR, but runtime dynamic activation QDQ outputs all zeros on physical silicon.
  * **H19 (A20 Pro / H19)**: Target architecture for validated native FP8 execution.
* **Decoupled Scale Pattern**: Physical FP8 storage magnitude ($2^{-1} = 0.5$) is decoupled from logical FP16 math magnitude ($2^{-5} = 0.03125$) using QDQ `scale = 0.0625` to avoid both the H18 underflow cliff and 20-layer compounding saturation.

---

## 2. FP8 Data Format Specification

MPSGraph currently restricts FP8 quantization to **E4M3**:

| Format | Sign | Exponent | Mantissa | Exponent Bias | Max Normal Value | Min Normal Positive | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Float8E4M3** | 1 | 4 | 3 | 7 | 448.0 | $2^{-6} \approx 0.015625$ | ✅ Supported (`MPSDataTypeFloat8e4m3`) |
| **Float8E5M2** | 1 | 5 | 2 | 15 | 57344.0 | $2^{-14} \approx 6.1 \times 10^{-5}$ | ❌ Rejected (`"Unsupported quantization scheme"`) |

---

## 3. Mandatory Quantize-Dequantize (QDQ) Flow in MPSGraph

Attempting to pass raw FP8 tensors directly into `convolution2DWithSourceTensor:weightsTensor:` or `matrixMultiplicationWithPrimaryTensor:secondaryTensor:` produces an MLIR dialect lowering error:

```text
'mps.conv_2d' op operand #0 must be tensor of mps native type values, but got 'tensor<...xf8E4M3FN>'
```

Because the underlying arithmetic pipeline requires FP16 or FP32 native types, MPSGraph supports two distinct QDQ topologies:

### Pattern A: Full W8A8 QDQ (Activations & Weights in FP8)
Both activations and weights reside in FP8 memory:
1. Input activation tensor is stored in FP8.
2. At each layer $l$, the activation is dequantized to FP16:
   $$\text{act}_{\text{FP16}} = \text{dequantize}(\text{act}_{\text{FP8}}, \text{scale})$$
3. Weight tensor is stored in FP8 and dequantized to FP16:
   $$W_{\text{FP16}} = \text{dequantize}(W_{\text{FP8}}, \text{scale})$$
4. Spatial convolution or matrix multiplication runs in FP16:
   $$\text{out}_{\text{FP16}} = \text{conv2D}(\text{act}_{\text{FP16}}, W_{\text{FP16}})$$
5. Output is requantized back to FP8 for the subsequent layer:
   $$\text{act}_{\text{FP8}}^{(l+1)} = \text{quantize}(\text{out}_{\text{FP16}}, \text{scale})$$

### Pattern B: Weight-Only QDQ (FP8 Weights, FP16 Activations)
1. Weights are compressed as 8-bit FP8 (cutting weight storage and bandwidth by 2×).
2. Weights are dequantized to FP16 once (constant-folded by the compiler).
3. Activations remain native FP16 throughout the chain:
   $$\text{out}_{\text{FP16}} = \text{conv2D}(\text{act}_{\text{FP16}}, W_{\text{FP16}})$$

---

## 4. Mathematical Stability vs. Hardware Underflow Cliff

### A. The Compounding Magnitude Problem
In a 20-layer chained convolution with a spatial reduction window:
$$C_i \times K \times K = 128 \times 3 \times 3 = 1152$$
If input activations and weights are populated with magnitude $\approx 1.0$, the expected value compounds by $\approx 1152\times$ per layer:
* **Layer 1**: Magnitude $\approx 10^3$
* **Layer 2**: Magnitude $\approx 10^6$ (overflows FP16 max 65,504 and FP8 max 448 to `NaN` / `Inf`)
* **Layers 3–20**: Benchmarking pure saturation propagation rather than real arithmetic.

To keep the per-layer root-mean-square (RMS) gain near unity ($\approx 1.0$):
$$\text{Expected Gain} = \frac{\sqrt{C_i \times K \times K}}{2^{\text{shift}}} = \frac{\sqrt{1152}}{32} \approx 1.06$$
Thus, the arithmetic domain **must** operate at logical magnitude:
$$\text{Logical Magnitude} = 2^{-5} = \frac{1}{32} \approx 0.03125$$

### B. The H18 ANE Underflow Cliff
On iPhone 18 Pro (H18 ANE), physically encoding dynamic activation FP8 bytes at $2^{-5} = 0.03125$ and dequantizing with $\text{scale} = 1.0$ hits a hardware underflow-to-zero cliff:
* 100% of the output tensor elements become exactly `0.0`.
* This triggers H17+ zero-skipping logic, inflating wall-clock measurement to an artificial ~65+ TOPS on degenerate output.
* However, physical FP8 bytes encoded at $2^{-1} = 0.5$ measure completely clean (linear response, zero underflow).

### C. The Decoupled Scale Solution
To simultaneously satisfy **20-layer math stability** and **H18 hardware underflow safety**, physical byte storage is decoupled from logical math magnitude:

$$\text{Dequantized FP16} = \text{Physical Stored Value} \times \text{scale}$$

$$\text{scale} = 2^{(\text{logical\_shift} - \text{physical\_shift})} = 2^{-5 - (-1)} = 2^{-4} = 0.0625$$

| Domain | Shift | Magnitude | Purpose |
| :--- | :---: | :---: | :--- |
| **Physical (FP8 Byte)** | $2^{-1}$ | $0.5$ (`0x30` / `0xB0`) | Prevents H18 ANE activation underflow cliff |
| **Logical (FP16 Math)** | $2^{-5}$ | $0.03125$ | Keeps 20-layer chained RMS growth stable ($\approx 1.06\times$) |
| **QDQ Scale Parameter** | $2^{-4}$ | $0.0625$ | Bridging scale for quantize/dequantize |

*(Note: On H19, physical shift `-1` with scale `1.0` is also supported).*

---

## 5. Apple Silicon Architecture Support Matrix

| Generation | Chips | ANE FP8 Support | Compilation & Runtime Behavior |
| :--- | :--- | :---: | :--- |
| **H13** | A14, M1 | ❌ None | MAC array is FP16/INT8 only. MPSGraph falls back to Metal GPU. |
| **H14** | A15, M2 | ❌ None | MAC array is FP16/INT8 only. MPSGraph falls back to Metal GPU. |
| **H15** | A16, M3 | ❌ None | MAC array is FP16/INT8 only. MPSGraph falls back to Metal GPU. |
| **H16 / H16g / H16s** | A17 Pro, M4 / Pro / Max | ❌ None | `ANECCompile` returns MLIR conversion error; falls back to Metal GPU. |
| **H18** | A19 (iPhone 18 Pro) | ⚠️ Partial | Compiles to ANE, but dynamic activation QDQ outputs all zeros on silicon. Weight-only QDQ works (constant-folded). |
| **H19** | A20 Pro | ✅ Native | Target architecture for native FP8 QDQ execution on ANE. |

---

## 6. Zero-Skip Detection & Output Verification Protocol

Beginning with the H17 architecture, the ANE includes hardware-level zero-skipping logic to reduce dynamic power. If an operation produces or consumes an all-zero tensor:
1. The hardware terminates execution early.
2. Latency drops drastically.
3. Wall-clock calculations report implausibly high TOPS metrics (~65+ TOPS) on garbage data.

### Verification Protocol
Both [`measure_conv_fp8`](file:///Users/freedom/work/measure_ane_capacity/measure_conv_fp8.m) and [`ANECapacityEngine.swift`](file:///Users/freedom/work/measure_ane_capacity/ANECapacityApp/ANECapacityApp/ANECapacityEngine.swift) enforce post-execution buffer validation:

```swift
let outputElementCount = outputBufferLen / elementSize
let zeroCount = countZeroElements(buffer: oBuf.contents(), elementCount: outputElementCount, dataType: mpsType)
if zeroCount == outputElementCount {
    print("⚠️ WARNING: Output tensor is 100% ZERO. Result may be inflated by zero-skipping!")
}
```

---

## 7. Offline Compilation to ANE Binary (`convert_fp8_to_hwx`)

The [`convert_fp8_to_hwx`](file:///Users/freedom/work/measure_ane_capacity/convert_fp8_to_hwx.m) utility allows offline cross-compilation of an MPSGraph FP8 QDQ model to an ANE Mach-O `.hwx` binary.

> [!IMPORTANT]
> **Architecture Compatibility**: `convert_fp8_to_hwx` **only works for H18 variants (e.g. `h18`, `h18g`) and H19**. Earlier architectures (H17 and below) lack hardware FP8 support, causing their respective ANE compilers to reject the FP8 MLIR graph during compilation (`"MLIR MPS to ANEC conversion failed"`).

```bash
# Build the tool
make convert_fp8_to_hwx

# Compile FP8 QDQ Conv for H19 (A20 Pro)
./convert_fp8_to_hwx --arch h19 --layers 20 --output hwx_output

# Compile FP8 QDQ Conv for H18 (A19)
./convert_fp8_to_hwx --arch h18 --layers 20 --output hwx_output
```

### Compilation Pipeline
1. **Device Descriptor**: Creates private `MPSGraphDeviceDescriptor` specifying the target architecture string (`h18`, `h18g`, `h19`).
2. **Graph Construction**: Constructs the FP8 QDQ convolution chain with decoupled scale.
3. **Compilation**: Sets `preferredDevice = 2` (ANE) and `enableCompileResourcesForPackage = YES`.
4. **Package Extraction**: Serializes to `.mpsgraphpackage`, extracts `binary_0.hwx`, and parses Mach-O header metadata (`CPU Type: 0x0100000c`, `Subtype: 11` for H19 / `Subtype: 10` for H18).

---

## 8. CLI Benchmark Reference

### Running FP8 Convolution Benchmark
```bash
# Build CLI
make measure_conv_fp8

# Full W8A8 QDQ (defaults: 20 layers, logical 2^-5, physical 2^-1)
./measure_conv_fp8

# Weight-Only QDQ
./measure_conv_fp8 --mode weight-only

# Single-layer diagnostic check
./measure_conv_fp8 --layers 1 --logical-shift -5 --physical-shift -1
```

### Running FP8 Matrix Multiplication Benchmark
```bash
# Build CLI
make measure_matmul_fp8

# Run GEMM benchmark (B=1, M=1024, K=1024, N=1024)
./measure_matmul_fp8
```
