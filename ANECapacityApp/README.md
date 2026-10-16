# ANECapacityApp: iOS Apple Neural Engine (ANE) Capacity Benchmark & Visualizer

A native SwiftUI application for iOS designed to measure the full hardware capacity of the **Apple Neural Engine (ANE)** and **GPU (Metal)** across configurable tensor shapes, precisions, and chained depths, with interactive **Swift Charts** for throughput (TOPS) and latency figures.

Based directly on [`measure_conv_universal.m`](../measure_conv_universal.m) from the `measure_ane_capacity` toolkit.

---

## Key Features

1. **Dual Execution Engines / Frameworks**:
   - **MetalPerformanceShadersGraph (MPSGraph)**: Direct compiled execution on Metal / ANE with private `_ANEClient` PMU performance counter telemetry.
   - **CoreML / MIL**: Direct on-device MIL (Model Intermediate Language) protobuf code generation and native FP8 model package synthesis with fused bias (`BiasEn=1`) and hardware zero-skipping elimination.
   - **Side-by-Side Benchmarking**: Compare MPSGraph vs. CoreML / MIL performance curves concurrently across identical tensor shapes.

2. **Dual Workload Operations**:
   - **Conv2D**: 2D spatial convolutions across configurable channel depths, spatial resolutions, kernel sizes, and chained layers.
   - **MatMul (GEMM)**: Dense Matrix Multiplication using native MPSGraph and MIL 1x1 conv mapping across matrix dimensions ($M, K, N$) and chained layers ($L$).

3. **Precision Support**:
   - **FP16 (Half Precision)**: High-throughput floating point convolution and GEMM on the ANE matrix compute array.
   - **INT8 (Simulated QDQ)**: Realistic quantization flow (`Int8` $\rightarrow$ `Dequant FP16` $\rightarrow$ `Compute` $\rightarrow$ `Requant Int8`) matching `measure_conv_universal.m` and `measure_matmul_universal.m`.
   - **FP8 (Float8E4M3 QDQ & Native MIL FP8)**: 8-bit floating-point execution with decoupled scaling, dithering protection, and zero-overhead fused hardware bias on H18 and H19 silicon.
   - **Deterministic Non-Canceling Initialization (H17+ Ready)**: Both weight and input tensors are populated with deterministic pseudo-random signs and alternating fused bias to prevent underflow and eliminate hardware zero-skipping artifacts on H17 (A18 Pro), H18 (A19 Pro), and H19 (A20 Pro).

4. **Automated Capacity Sweeps**:
   - **Conv2D Channel Capacity Sweep**: Holds $H=W=256$, sweeps channels $C \in [32, 64, 128, 256, 512, 1024]$ to find the ANE matrix tile saturation point.
   - **SRAM-Resident Sweep**: Holds $H=W=64$, keeping working buffers strictly resident in ANE on-chip L2 SRAM cache to eliminate DRAM latency bottlenecks.
   - **Spatial Dimension Sweep**: Holds $C=128$, sweeps spatial resolutions $H=W \in [64, 128, 256, 384, 512, 768]$.
   - **Chained Depth Sweep**: Sweeps $L \in [1, 5, 10, 20, 40, 80, 100]$ to characterize driver submission latency overhead vs. sustained hardware capacity.
   - **Pointwise Depth Sweep**: Sweeps $L \in [1 \dots 100]$ with $K=1\times 1$ to isolate weight reuse and kernel-memory stall differences between GEMM and $3\times 3$ convolutions.
   - **Kernel Size Sweep**: Compares $K=1\times1$ vs $K=3\times3$ vs $K=5\times5$.
   - **MatMul Dimension Sweep**: Sweeps square dimensions $M=K=N \in [128 \dots 4096]$ to map GEMM tile saturation on ANE.
   - **MatMul Rectangular Sweep**: Sweeps large batch/sequence length $M \in [1024, 2048, 4096, 8192]$ with fixed $K=N=1024$.
   - **Full Capacity Comparison**: Multi-variant sweeps across FP16, INT8, and FP8.

5. **Interactive Figures & Visualizations (Swift Charts)**:
   - **Throughput (TOPS) vs. Size**: Line and point chart with smooth interpolation and distinct backend colors (`[MPS]` vs `[MIL]`).
   - **Latency (ms) vs. Size**: Execution time curve per iteration.
   - **Peak TOPS Rule Mark**: Automatically highlights the maximum detected throughput with a callout banner.
   - **Interactive Scrubbing / Point Inspector**: Drag along the chart to inspect exact parameters ($C, H, W, K, L$ or $M, K, N, L$), latency, and TOPS at any point.
   - **Data Table & CSV Export**: Built-in iOS Share Sheet support to export results as standard CSV (including Backend, Operation type, $M, K, N$ metadata).

6. **Live Execution Console**:
   - Monospaced execution log console showing warmup status, compilation time, iteration progress, and live TOPS calculations.

---

## Silicon Throughput Formula

The benchmark uses the canonical floating-point and integer operations formulas:

### Conv2D Operations:
$$\text{Total Operations}_{\text{Conv2D}} = 2 \times B \times H \times W \times C_{in} \times C_{out} \times K^2 \times L$$

### MatMul (GEMM) Operations:
$$\text{Total Operations}_{\text{MatMul}} = 2 \times B \times M \times K \times N \times L$$

### Throughput & Latency:
$$\text{Throughput (TOPS)} = \frac{\text{Total Operations}}{\text{Average Execution Time (seconds)} \times 10^{12}}$$

$$\text{Latency} = \frac{\sum_{i=1}^{N} \text{time}_i}{N} \quad (\text{ms})$$

---

## Building and Running the iOS App

### Method 1: Xcode
Open `ANECapacityApp.xcodeproj` in Xcode:
```bash
open ANECapacityApp/ANECapacityApp.xcodeproj
```
Select your connected iPhone, iPad, or Mac target, and press **Run (Cmd+R)**.

### Method 2: Command Line (devicectl)
Launch the application headlessly on a connected physical iOS device using `xcrun devicectl`:
```bash
xcrun devicectl device process launch --device <UDID> com.freedom.ANECapacityApp [ARGUMENTS...]
```

---

## Command-Line Interface (CLI) & Headless Execution

`ANECapacityApp` includes a rich command-line interface accessible via Xcode Launch Arguments or `xcrun devicectl`. The CLI supports two execution paradigms:
1. **Targeted Micro-Probes**: Standalone diagnostics that run immediately, log detailed hardware/kernel telemetry to stdout, and exit (`exit(0)`).
2. **Automated Headless Sweeps (`--autorun`)**: Executes full parameter sweeps on ANE or GPU and populates the history/chart stores automatically.

### 1. Standalone Diagnostics & Micro-Probes

Pass any of the following arguments to execute targeted hardware and kernel tests:

| CLI Argument | Description / Component Tested | Output & Action |
|:---|:---|:---|
| `--sysctl-features` | Inspects CPU core topology (Super/P/E clusters), cache line/sizes, and ARMv9.x features (`FEAT_SME`, `FEAT_SME2`, `FEAT_SME2p1`, `FEAT_CSSC`, `FEAT_SME_F16F16`, `FEAT_SME_F8F16`, `FEAT_FP8`). | Prints active sysctl features and exits. |
| `--sme-microbench` | Validates streaming SVE and SME/SME2 extensions, executes microkernel tile accumulation, and runs single-thread GEMM benchmarks. | Verifies SME hardware support and runs microkernels. |
| `--sme2-fp8-sketch` | Evaluates the experimental FP8 SME2 matrix outer-product microkernel sketch. | Verifies FP8 assembly execution. |
| `--sme2-fp8-peak` | Measures raw single-core instruction roofline for `FEAT_SME_F8F16` (`FMOPA`) tile outer products. | Logs peak GOPS / TOPS. |
| `--native-bnns-graph-matmul` | Compiles and executes an Accelerate `BNNSGraph` GEMM, inspecting whether `KERNEL_MATMUL_SME2P1` or `KERNEL_MATMUL_MLA3` (AMX) is dispatched. | Logs dispatched kernel name, latency, and TOPS. |
| `--native-coreml-matmul` | Builds dynamic on-device MIL matmul chains and benchmarks CoreML matrix multiplication across dimensions and precisions. | Runs full CoreML GEMM sweep. |
| `--native-coreml-matmul-single <fp8\|int8\|fp16>` | Runs an isolated single-pass CoreML matrix multiplication benchmark. Optionally filter compute units with `--units <cpuAndGPU\|cpuOnly\|all>`. | Logs latency, throughput, and zero-check status. |
| `--native-cpu-matmul` | Executes direct CPU matrix multiplication microkernels across Neon and matrix coprocessors. | Benchmarks CPU GEMM throughput. |
| `--native-cpu-conv` | Executes direct CPU 2D convolution microkernels. | Benchmarks CPU convolution throughput. |
| `--native-int8-gemm` | Evaluates native INT8 matrix multiplication on GPU using `MPSNDArray` and quantized affine descriptors. | Validates GPU INT8 GEMM dispatch and non-zero output. |
| `--native-int8-conv` | Evaluates native INT8 convolution on Metal GPU vs QDQ flow. | Verifies GPU INT8 kernel dispatch. |
| `--supports-mxu` | Probes whether the Metal GPU device implements dedicated Matrix Execution Units (MXU). | Reports Metal device feature queries. |

#### Micro-Probe Examples:
```bash
# Query ARMv9.4 SME feature flags on device:
xcrun devicectl device process launch --device <UDID> com.freedom.ANECapacityApp --sysctl-features

# Benchmark SME2.1 BNNSGraph kernel selection:
xcrun devicectl device process launch --device <UDID> com.freedom.ANECapacityApp --native-bnns-graph-matmul

# Measure single-core FP8 SME2 peak instruction throughput:
xcrun devicectl device process launch --device <UDID> com.freedom.ANECapacityApp --sme2-fp8-peak

# Benchmark single CoreML FP8 GEMM on ANE:
xcrun devicectl device process launch --device <UDID> com.freedom.ANECapacityApp --native-coreml-matmul-single fp8 --units all
```

---

### 2. Automated Headless Benchmark Sweeps (`--autorun`)

Adding `--autorun` (or setting the environment variable `AUTORUN=1`) instructs the app to bypass manual UI interaction, configure the benchmark runner from arguments, execute the sweep, and record results:

```bash
xcrun devicectl device process launch --device <UDID> com.freedom.ANECapacityApp --autorun [OPTIONS]
```

#### Sweep Configuration Options:

| Category | Argument / Flag | Description / Values |
|:---|:---|:---|
| **Target Engine** | `--ane` | Target Apple Neural Engine (default if available). |
| | `--gpu` | Target Metal GPU. |
| **Backend** | `--backend <mps\|coreml\|both>` | Execution framework. Aliases: `--mps`, `--mpsgraph`, `--mil`, `--coreml`, `--both`. |
| **Precision** | `--fp8` | 8-bit Floating Point (`Float8E4M3` QDQ / Native MIL FP8). |
| | `--int8` | 8-bit Integer (Simulated QDQ flow). |
| | `--fp16` | 16-bit Floating Point (Native half precision). |
| | `--all` | Runs multi-precision comparison across FP16, INT8, and FP8. |
| **Sweep Type** | `--sweep <type>` | Pre-configured sweep profile: <br>• `channels`: Conv2D channel capacity sweep ($C \in [32\dots 1024]$)<br>• `spatial`: Spatial dimension sweep ($H=W \in [64\dots 768]$)<br>• `depth`: Chained layer depth sweep ($L \in [1\dots 100]$)<br>• `pointwisedepth`: Pointwise $1\times 1$ depth sweep ($L \in [1\dots 100]$)<br>• `kernels`: Kernel size sweep ($K \in [1, 3, 5]$)<br>• `sram`: SRAM-resident sweep ($H=W=64$, eliminates DRAM spill)<br>• `matmuldimensions`: Square GEMM sweep ($M=K=N \in [128\dots 4096]$)<br>• `matmuldepth`: Chained GEMM depth sweep<br>• `matmulasymmetric`: Rectangular GEMM sweep ($M \in [1024\dots 8192]$)<br>• `none`: Run single shape with custom parameters |
| | `--sram` | Quick shortcut for `--sweep sram`. |
| **Parameter Overrides** | `--channels <c1,c2,...>` | Custom comma-separated channel list (e.g. `--channels 128,192,256,320`). |
| | `--layers <N>` | Number of chained layers (e.g. `--layers 30`). |
| | `--dim <size>` | Sets spatial height and width (e.g. `--dim 256`). |
| | `--iterations <N>` | Number of benchmark iterations per point (default: 20). |
| **UI Tab Selection** | `-selectedTab <0..3>` | Sets the active tab on startup (0: Benchmark, 1: Figures, 2: History, 3: Info). |

#### Headless Sweep Examples:
```bash
# 1. Full ANE FP8 SRAM-resident sweep using CoreML/MIL:
xcrun devicectl device process launch --device <UDID> com.freedom.ANECapacityApp \
  --autorun --ane --backend coreml --fp8 --sram

# 2. Side-by-side ANE vs. GPU comparison across custom channel depths:
xcrun devicectl device process launch --device <UDID> com.freedom.ANECapacityApp \
  --autorun --ane --backend both --all --channels 128,192,256 --layers 25

# 3. Dense MatMul dimension sweep on GPU with 50 iterations per point:
xcrun devicectl device process launch --device <UDID> com.freedom.ANECapacityApp \
  --autorun --gpu --backend mps --fp16 --sweep matmuldimensions --iterations 50
```

---

### 3. Environment Variables

| Variable | Values | Description |
|:---|:---:|:---|
| `AUTORUN` | `1` | Alternative trigger for headless execution (equivalent to `--autorun`). |
| `ANE_CAPACITY_GPU_CAPTURE` | `1` | Enables automatic Metal GPU frame capture. Dumps a `.gputrace` package into the app's `Documents/` directory for in-depth GPU shader profiling in Xcode Instruments. |

---

## Project Structure

```text
ANECapacityApp/
├── ANECapacityApp.xcodeproj/
│   └── project.pbxproj            # Clean Xcode project (objectVersion = 77)
└── ANECapacityApp/
    ├── ANECapacityApp.swift       # @main SwiftUI entry point
    ├── ContentView.swift          # Main TabView layout (Benchmark, Figures, History, Info)
    ├── BenchmarkView.swift        # Sweep selection, backend picker, parameter sliders, controls
    ├── ChartsView.swift           # Interactive Swift Charts for TOPS & Latency ([MPS] vs [MIL])
    ├── HistoryView.swift          # Tabular run logs, backend filtering, CSV export
    ├── DeviceInfoView.swift       # Hardware capabilities, Metal specs, formula documentation
    ├── BenchmarkViewModel.swift   # Async task management, multi-backend dispatcher, CLI parsing
    ├── ANECapacityEngine.swift    # Core MPSGraph convolution engine (ANE/GPU, FP16/INT8/FP8)
    ├── MILCapacityEngine.swift    # CoreML / MIL capacity engine (ANE/GPU, FP16/INT8/FP8)
    ├── MILCapacityEngineBridge.h  # ObjC++ bridge for CoreML dynamic model evaluation
    ├── MILCapacityEngineBridge.mm # CoreML runtime evaluation and PMU performance telemetry
    ├── MILSpecBuilder.h           # Pure ObjC MIL protobuf specification builder
    ├── MILSpecBuilder.m           # MIL wire-format encoding & native FP8 package generator
    ├── ANEClientBridge.h          # Private _ANEClient header
    ├── ANEClientBridge.mm         # Private _ANEClient PMU performance counter interface
    ├── ANECapacityApp-Bridging-Header.h # Bridging header for private ANE and MIL telemetry
    ├── BenchmarkModels.swift      # Data models, BenchmarkBackend enum, dimensions, presets
    ├── Info.plist                 # iOS app bundle configuration
    └── Assets.xcassets/           # App icon & accent colors
```
