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

## Building the iOS App

### Method 1: Xcode
Open `ANECapacityApp.xcodeproj` in Xcode:
```bash
open ANECapacityApp/ANECapacityApp.xcodeproj
```
Select your connected iPhone or iPad, and press **Run (Cmd+R)**.

### Method 2: Command Line (devicectl)
```bash
# Automated headless run via devicectl
xcrun devicectl device process launch --device <UDID> com.freedom.ANECapacityApp --autorun --mil --fp8 --sram
```

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
