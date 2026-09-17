# ANECapacityApp: iOS Apple Neural Engine (ANE) Capacity Benchmark & Visualizer

A native SwiftUI application for iOS designed to measure the full hardware capacity of the **Apple Neural Engine (ANE)** and **GPU (Metal)** across configurable tensor shapes, precisions, and chained depths, with interactive **Swift Charts** for throughput (TOPS) and latency figures.

Based directly on [`measure_conv_universal.m`](../measure_conv_universal.m) from the `measure_ane_capacity` toolkit.

---

## Key Features

1. **Precision Support**:
   - **FP16 (Half Precision)**: High-throughput floating point convolution on the ANE matrix compute array.
   - **INT8 (Simulated QDQ)**: Realistic quantization flow (`Int8` $\rightarrow$ `Conv` $\rightarrow$ `FP16 Dequant` $\rightarrow$ `Int8 Requant`) as in `measure_conv_universal.m`.
   - **Both Mode**: Runs FP16 and INT8 back-to-back to directly observe the INT8 speedup multiplier.
   - **Dense Non-Zero Initialization (H17+ Ready)**: Both weight and input tensors are initialized with bounded alternating non-zero values to bypass hardware zero-skipping and lossless zero-compression on H17 (A18 Pro) and H18 (A19 Pro), ensuring true dense silicon capacity is measured.

2. **Configurable Tensor Dimensions & Hyperparameters**:
   - **Channel Capacity ($C_{in}, C_{out}$)**: $16, 32, 64, 128, 256, 512, 1024$.
   - **Spatial Resolution ($H \times W$)**: $64 \times 64, 128 \times 128, 256 \times 256, 384 \times 384, 512 \times 512, 768 \times 768, 1024 \times 1024$.
   - **Chained Layers ($L$)**: $1, 5, 10, 20, 30, 40$ (amortizes driver enqueue and kernel launch overhead to uncover true peak silicon TOPS).
   - **Kernel Size ($K \times K$)**: $1 \times 1$ (GEMM / matrix-multiply equivalent), $3 \times 3$ (standard 2D spatial convolution), $5 \times 5$.
   - **Batch Size ($B$)**: Configurable (default $B=1$).

3. **Automated Capacity Sweeps**:
   - **Channel Capacity Sweep**: Holds $H=W=256$, sweeps channels $C \in [32, 64, 128, 256, 512, 1024]$ to find the ANE matrix tile saturation point.
   - **Spatial Dimension Sweep**: Holds $C=128$, sweeps spatial resolutions $H=W \in [64, 128, 256, 384, 512, 768]$ to test memory bandwidth limits vs. compute reuse.
   - **Chained Depth Sweep**: Sweeps $L \in [1, 5, 10, 20, 30, 40]$ to characterize driver submission latency overhead vs. sustained hardware capacity.
   - **Kernel Size Sweep**: Compares $K=1\times1$ vs $K=3\times3$ vs $K=5\times5$.
   - **Full Capacity Comparison**: Complete matrix sweep for both FP16 and INT8.

4. **Interactive Figures & Visualizations (Swift Charts)**:
   - **Throughput (TOPS) vs. Size**: Line and point chart with smooth interpolation and subtle area gradient.
   - **Latency (ms) vs. Size**: Execution time curve per iteration.
   - **Peak TOPS Rule Mark**: Automatically highlights the maximum detected throughput with a callout banner.
   - **Interactive Scrubbing / Point Inspector**: Drag along the chart to inspect exact parameters ($C, H, W, K, L$), latency, and TOPS at any point.
   - **Data Table & CSV Export**: Built-in iOS Share Sheet support to export results as standard CSV.

5. **Live Execution Console**:
   - Monospaced execution log console showing warmup status, compilation time, iteration progress, and live TOPS calculations.

---

## Silicon Throughput Formula

The benchmark uses the canonical convolutional floating-point and integer operations formula:

$$\text{Total Operations} = 2 \times B \times H \times W \times C_{in} \times C_{out} \times K^2 \times L$$

$$\text{Throughput (TOPS)} = \frac{\text{Total Operations}}{\text{Average Execution Time (seconds)} \times 10^{12}}$$

$$\text{Latency} = \frac{\sum_{i=1}^{N} \text{time}_i}{N} \quad (\text{ms})$$

Where:
- $B$: Batch size
- $H, W$: Input feature map height and width
- $C_{in}$: Input channels
- $C_{out}$: Output channels
- $K$: Convolution kernel size ($K \times K$)
- $L$: Number of chained convolution layers
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
