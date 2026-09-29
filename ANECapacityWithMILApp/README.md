# ANECapacityWithMILApp

An iOS benchmarking application for evaluating Apple Neural Engine (ANE) and Apple Silicon compute capacity using Apple's **CoreML** and **Model Intermediate Language (MIL)**, with native support for **FP16**, **INT8 (W8A8 QDQ)**, and **FP8 (Float8E4M3 QDQ)** precision modes.

## Overview

`ANECapacityWithMILApp` provides an interactive, on-device benchmarking companion to `measure_conv_coreml.m` and mirrors the workflows and UI of `ANECapacityApp` (which benchmarks via `MetalPerformanceShadersGraph`).

Unlike standard CoreML applications that require offline Python model export (`coremltools`), `ANECapacityWithMILApp`:
- Evaluates models across configurable compute targets: **ANE** (`MLComputeUnits.cpuAndNeuralEngine`), **GPU** (`MLComputeUnits.cpuAndGPU`), **CPU** (`MLComputeUnits.cpuOnly`), and **All** (`MLComputeUnits.all`).
- Direct integration with private Apple Neural Engine runtime (`_ANEClient`) to extract hardware **PMU counters** (compute cycles, nominal cycles, output/input stalls, DMA read/write bytes, and ALU saturation percentage).
- Verifies output non-degeneracy (ensuring zero-skipping does not artificially inflate TOPS).
- Built with a clean Swift + Objective-C bridging architecture with zero external third-party package dependencies.

## Key Tabs & Features

1. **Benchmark Tab**:
   - Workload selector: Conv2D ($B \times C \times H \times W$, $K \times K$ kernel, $L$ chained layers) and MatMul / GEMM ($[B, M, K] \times [B, K, N]$ mapped to ANE 1x1 convolution).
   - Presets & Capacity Sweeps:
     - Channel Capacity Sweep ($C \in [32, 64, 128, 256, 384, 512]$)
     - ANE SRAM-Resident Sweep ($H=W=64$, $C \in [32..512]$)
     - Spatial Dimension Sweep ($H=W \in [64..768]$)
     - Chained Layer Depth Sweep ($L \in [1..100]$)
     - Pointwise Depth Sweep ($K=1, H=W=64, C=512, L \in [1..100]$)
     - Matrix Dimension / Depth / Rectangular GEMM Sweeps
   - Execution settings: Select Precision (FP16, INT8, FP8, Both, or All) and Compute Units.
   - Live execution console streaming latency, throughput, and PMU metrics.

2. **Figures Tab**:
   - Interactive Swift Charts displaying Throughput (TOPS), Latency (ms), Compute Cycles (Mcyc), Pipeline Stalls (Mcyc), DMA Volume (MB), and ALU Saturation (%).
   - Precision filtering (All, FP16, INT8, FP8) and draggable touch inspector.

3. **History Tab**:
   - Recorded run history with filter pills by device target and precision.
   - Comprehensive CSV export including full PMU performance counters and silicon metrics.

4. **Info Tab**:
   - Neural Engine & CoreML runtime information, target hardware architecture capabilities (H15..H19), and capacity calculation formulas.

## Architecture

- **`MILCapacityEngineBridge.{h,mm}`**: Objective-C++ bridge interfacing with `CoreML.framework` and `_ANEClient`. Evaluates CoreML models, times inferences, computes GOPs and TOPS, and collects hardware PMU counters via IOSurface.
- **`ANEClientBridge.{h,mm}`**: Direct hardware connection to private `AppleNeuralEngine.framework` runtime services.
- **`MILCapacityEngine.swift`**: Swift orchestration layer with `async/await` execution and simulation fallback for the iOS Simulator.
- **`BenchmarkViewModel.swift` & Views**: SwiftUI MVVM architecture.

## Building and Running

### Build via Command Line
```bash
xcodebuild -project ANECapacityWithMILApp/ANECapacityWithMILApp.xcodeproj \
           -scheme ANECapacityWithMILApp \
           -destination 'generic/platform=iOS' \
           CODE_SIGNING_ALLOWED=NO CODE_SIGN_IDENTITY="" CODE_SIGNING_REQUIRED=NO build
```

### Run on Physical Device
Open `ANECapacityWithMILApp/ANECapacityWithMILApp.xcodeproj` in Xcode, select your development team and iOS device (e.g. iPhone 16 Pro / iPhone 17 Pro / iPhone 18 Pro), and run.
