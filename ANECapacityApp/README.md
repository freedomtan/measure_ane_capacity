# ANECapacityApp: iOS Apple Neural Engine (ANE) Capacity Benchmark & Visualizer

A native SwiftUI application for iOS designed to measure the full hardware capacity of the **Apple Neural Engine (ANE)** and **GPU (Metal)** across configurable tensor shapes, precisions, and chained depths, with interactive **Swift Charts** for throughput (TOPS) and latency figures.

Based directly on [`measure_conv_universal.m`](../measure_conv_universal.m) from the `measure_ane_capacity` toolkit.

---

## Key Features

1. **Dual Workload Operations**:
   - **Conv2D**: 2D spatial convolutions across configurable channel depths, spatial resolutions, kernel sizes, and chained layers.
   - **MatMul (GEMM)**: Dense Matrix Multiplication using native MPSGraph `matrixMultiplication(primary:secondary:name:)` across matrix dimensions ($M, K, N$) and chained layers ($L$).

2. **Precision Support**:
   - **FP16 (Half Precision)**: High-throughput floating point convolution and GEMM on the ANE matrix compute array.
   - **INT8 (Simulated QDQ)**: Realistic quantization flow (`Int8` $\rightarrow$ `Dequant FP16` $\rightarrow$ `Compute` $\rightarrow$ `Requant Int8`) matching `measure_conv_universal.m` and `measure_matmul_universal.m`.
   - **Both Mode**: Runs FP16 and INT8 back-to-back to directly observe the INT8 speedup multiplier.
   - **Dense Non-Zero Initialization (H17+ Ready)**: Both weight and input tensors are initialized with bounded alternating non-zero values to bypass hardware zero-skipping and lossless zero-compression on H17 (A18 Pro) and H18 (A19 Pro), ensuring true dense silicon capacity is measured.

3. **Configurable Dimensions & Hyperparameters**:
   - **Conv2D Parameters**: Channels ($C_{in}, C_{out} \in [16 \dots 1024]$), Spatial ($H \times W \in [64 \times 64 \dots 1024 \times 1024]$), Kernels ($1 \times 1, 3 \times 3, 5 \times 5$), Layers ($L \in [1 \dots 40]$).
   - **MatMul Parameters**: $M, K, N \in [128, 256, 512, 1024, 2048]$, Chained Depth ($L \in [1 \dots 40]$).
   - **Batch Size ($B$)**: Configurable (default $B=1$).

4. **Automated Capacity Sweeps**:
   - **Conv2D Channel Capacity Sweep**: Holds $H=W=256$, sweeps channels $C \in [32, 64, 128, 256, 512, 1024]$ to find the ANE matrix tile saturation point.
   - **Conv2D Spatial Dimension Sweep**: Holds $C=128$, sweeps spatial resolutions $H=W \in [64, 128, 256, 384, 512, 768]$.
   - **Conv2D Chained Depth Sweep**: Sweeps $L \in [1, 5, 10, 20, 30, 40]$ to characterize driver submission latency overhead vs. sustained hardware capacity.
   - **Conv2D Kernel Size Sweep**: Compares $K=1\times1$ vs $K=3\times3$ vs $K=5\times5$.
   - **MatMul Dimension Sweep**: Sweeps $M=K=N \in [128, 256, 512, 1024, 2048]$ to map GEMM tile saturation on ANE.
   - **MatMul Depth Sweep**: Sweeps $L \in [1, 5, 10, 20, 30, 40]$ for dense matrix multiplications.
   - **Full Capacity Comparison**: Complete sweep for both FP16 and INT8.

5. **Interactive Figures & Visualizations (Swift Charts)**:
   - **Throughput (TOPS) vs. Size**: Line and point chart with smooth interpolation.
   - **Latency (ms) vs. Size**: Execution time curve per iteration.
   - **Peak TOPS Rule Mark**: Automatically highlights the maximum detected throughput with a callout banner.
   - **Interactive Scrubbing / Point Inspector**: Drag along the chart to inspect exact parameters ($C, H, W, K, L$ or $M, K, N, L$), latency, and TOPS at any point.
   - **Data Table & CSV Export**: Built-in iOS Share Sheet support to export results as standard CSV (including Operation type, $M, K, N$ metadata).

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

Where:
- $B$: Batch size
- $H, W$: Input feature map height and width
- $C_{in}, C_{out}$: Input and output channels
- $K$: Convolution kernel size ($K \times K$)
- $M, K, N$: Matrix multiplication row, inner, and column dimensions ($[B, M, K] \times [B, K, N] \rightarrow [B, M, N]$)
- $L$: Number of chained layers
- $N$: Iterations (timed using `CLOCK_MONOTONIC_RAW` after an initial warmup pass)

---

## Building the iOS App

### Method 1: Xcode
Open `ANECapacityApp.xcodeproj` in Xcode:
```bash
open ANECapacityApp/ANECapacityApp.xcodeproj
```
Select your connected iPhone or iPad, and press **Run (Cmd+R)**.

### Method 2: Command Line (Makefile)
From `~/work/measure_ane_capacity/`:

- **Build for Connected iOS Physical Device**:
  ```bash
  make app
  ```

> [!NOTE]
> Physical Apple Neural Engine (ANE) silicon and PMU performance counters require a physical iPhone or iPad device (not supported in the iOS Simulator).

---

## Project Structure

```text
ANECapacityApp/
├── ANECapacityApp.xcodeproj/
│   └── project.pbxproj            # Clean Xcode project (objectVersion = 77)
└── ANECapacityApp/
    ├── ANECapacityApp.swift       # @main SwiftUI entry point
    ├── ContentView.swift          # Main TabView layout (Benchmark, Figures, History, Info)
    ├── BenchmarkView.swift        # Sweep selection, custom parameter sliders, run controls
    ├── ChartsView.swift           # Interactive Swift Charts for TOPS & Latency
    ├── HistoryView.swift          # Tabular run logs, filtering, CSV export
    ├── DeviceInfoView.swift       # Hardware capabilities, Metal specs, formula documentation
    ├── BenchmarkViewModel.swift   # Async task management, state updates, CSV exporter
    ├── ANECapacityEngine.swift    # Core MPSGraph convolution engine (ANE/GPU, FP16/INT8)
    ├── BenchmarkModels.swift      # Data models, dimensions, presets, results
    ├── Info.plist                 # iOS app bundle configuration
    └── Assets.xcassets/           # App icon & accent colors
```
