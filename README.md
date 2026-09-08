# MPSGraph Convolution Benchmark

This project benchmarks the performance of 2D convolutions on Apple Silicon using Metal Performance Shaders Graph (MPSGraph). It measures the compute capacity in TOPS (Trillions of Operations Per Second) for both the GPU and the Apple Neural Engine (ANE).

## Build Instructions

To compile the project, ensure you have clang and the necessary frameworks (Foundation, Metal, MetalPerformanceShadersGraph) installed (standard on macOS with Xcode Command Line Tools).

```bash
make
```

To build just the Swift version:
```bash
make measure_conv_swift
```

To clean the build artifacts:

```bash
make clean
```

### iOS Build
To build for iOS, you need to use the `xcrun` command to target the iPhone SDK and sign the binary.

```bash
# Compile for iOS (arm64)
xcrun -sdk iphoneos clang -fobjc-arc -O3 -framework Foundation -framework Metal -framework MetalPerformanceShadersGraph measure_conv_universal.m -o measure_conv_ios

# Sign the binary (replace 'Apple Development' with your identity)
codesign -s "Apple Development" measure_conv_ios
```

> **Note**: Running a standalone binary on a non-jailbroken iPhone is restricted. The easiest way to run this on a device is to wrap it in an iOS App.

#### Method 1: Create an Xcode Project (GUI)
1. Open Xcode and create a new **iOS App** (Objective-C).
2. **Delete** the following default files: `AppDelegate.h/m`, `SceneDelegate.h/m`, and `ViewController.h/m`.
3. **Replace** the contents of `main.m` with the code from `measure_conv_gui.m`.
4. **Info.plist** (Scene Manifest):
    - In the **Info** tab, find **Application Scene Manifest**.
    - **Delete** that entire row (to prevent the app from trying to use a SceneDelegate).
5. Add `Metal` and `MetalPerformanceShadersGraph` to the **Frameworks, Libraries, and Embedded Content**.
6. Run the app on your connected iPhone.

#### Method 2: Xcode Project (Console Only)
1. Follow the steps above but use `measure_conv_universal.m` instead.
2. Check the Xcode **Console** for the output.

## Running the Benchmark

### Standard Convolution Benchmarks
```bash
./measure_conv
```

### Advanced Silicon PMU Profiler & MPSGraphPackage Exporter (`measure_ane_pmu`)
`measure_ane_pmu` provides deep physical hardware profiling for Apple Neural Engine via `_ANEClient` and Apple PMU registers (`com.apple.ane.hardware-counters`), comparing FP16, INT8, QDQ, and GPU baselines while exporting self-contained `.mpsgraphpackage` bundles.

> **Note / Attribution**:
> `measure_ane_pmu` is based on and adapted from [**ane_pmu_profiler**](https://github.com/freedomtan/ane_pmu_profiler/), reusing its low-level Apple Neural Engine silicon PMU profiling, private `_ANEClient` telemetry interfaces, and hardware register mapping.
>
> For an in-depth microarchitectural analysis explaining how these counters behave under different tensor dimensions, see the [**Guide to Interpreting measure_ane_pmu Numbers**](How_to_Interpret_measure_ane_pmu_Numbers.md).

```bash
# Build and codesign with PMU entitlements
make measure_ane_pmu

# Run full benchmark across FP16, INT8, and QDQ with 20 chained layers
./measure_ane_pmu --layers 20 --iterations 10

# Export .mpsgraphpackage models to custom directory
./measure_ane_pmu --variant all --save-package ./packages

# Profile a specific variant without GPU
./measure_ane_pmu --variant int8 --layers 20 --iterations 20 --no-gpu
```

#### CLI Options
| Flag | Description | Default |
| :--- | :--- | :--- |
| `--variant <type>` | Precision variant (`all`, `fp16`, `int8`, `qdq`) | `all` |
| `--batch <B>` | Batch dimension | `1` |
| `--size <H>` | Spatial height and width ($H \times W$) | `256` |
| `--channels <C>` | Input and output channel depth ($C_i, C_o$) | `128` |
| `--layers <L>` | Number of chained convolution layers | `20` |
| `--iterations <N>` | Number of benchmark passes | `20` |
| `--save-package <dir>` | Serialize `.mpsgraphpackage` bundles | `./packages` |
| `--no-gpu` | Skip Metal GPU comparison | Disabled |
| `--no-pmu` | Skip physical silicon PMU hardware profiling | Disabled |

#### Self-Contained `.mpsgraphpackage` Export
When `--save-package` is enabled, `measure_ane_pmu` serializes each variant into a self-contained bundle containing:
- `manifest.plist` & compiled graph bytecode (`original_model_0.mpsgraph`, `specialized_model_1.mpsgraph`)
- `resources.bin` with constant weights
- `ane_bundle/`: Contains the low-level ANECIR bitcode (`*.bc.mlir`) and compiler options (`compiler_options_*.plist`) emitted by the MPSGraph compiler, allowing direct replay with `_ANEClient`.

## Benchmark Results

### Apple M4 Pro (H16g, 16 ANE Cores) — Physical Silicon PMU Telemetry
*Workload: 20 chained $3\times 3$ Conv layers on $[1, 128, 256, 256]$ tensor ($386.55\text{ GOPs} / 0.3865\text{ TOPs}$ per pass)*

| Variant | Device | Latency | Speed (TOPS) | Compute Cycles* | Output Stalls | Planar Cycles (L2PE) | DMA Traffic |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **GPU FP16** | Metal GPU | 39.13 ms | **9.88** | — | — | — | — |
| **ANE FP16** | Physical ANE | 20.61 ms | **18.76** | 157,200* | 503,317,431 | 10,494,464 | 34.99 MB |
| **ANE INT8** | Physical ANE | 10.78 ms | **35.87** | 105,045,873 | 233,509,350 | 5,247,232 | 18.35 MB |
| **ANE QDQ** | Physical ANE | 20.78 ms | **18.60** | 5,010,176 | 627,454,283 | 5,249,536 | 35.27 MB |

*\*Note on Compute Cycles: Under memory-bound workloads ($16\text{ MB}$ intermediate feature maps spilling on-chip L2 SRAM to DRAM), `kANE_NE_COMPUTE_CYCLES` is clock-gated OFF during the $503\text{M}$ output writeback stall cycles. In `conv_fp16`, back-to-back convolutions keep output writeback queues continuously saturated. In `conv_int8`, intermediate requantization casts on the Planar Engine create distinct computational phases, allowing the convolution engine to log $105\text{M}$ unstalled cycles. Total pipeline cycles (Compute + Stalls) account for 100% of runtime across all variants.*

#### Cache-Resident Benchmark ($H=64, W=64$, L=10 Layers)
*When intermediate feature maps ($\sim 1.0\text{ MB}$) fit entirely within on-chip L2 SRAM ($\sim 4-8\text{ MB}$), output stalls collapse by $>16\times$ and unstalled compute cycles become directly visible:*

| Variant ($64\times 64$, L=10) | Latency | Speed (TOPS) | Compute Cycles (`[13]`) | Output Stalls (`[15]`) | DMA Traffic (`[17]`) | ALU Efficiency |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **ANE FP16** | 1.15 ms | **10.50** | 463,416 | 15,587,386 | 1.80 MB | 13,033.2 MACs/cyc |
| **ANE INT8** | 0.77 ms | **15.72** | 484,910 | 7,547,107 | 1.23 MB | 12,455.5 MACs/cyc |

> **Key Architectural Observations from Silicon PMU:**
> 1. **Peak INT8 Realization**: Native INT8 delivers **35.87 TOPS** on Apple M4 Pro, hitting ~94.4% of Apple's advertised 38 TOPS hardware ceiling.
> 2. **Integer Scaling & DMA Reduction**: INT8 doubles throughput over FP16 (35.87 vs 18.76 TOPS) and cuts Unified Memory DMA traffic directly in half (18.35 MB vs 34.99 MB).
> 3. **Output Backpressure Stalls**: Because $128 \times 256 \times 256$ feature maps ($16\text{ MB}$) exceed on-chip L2 SRAM ($\sim 4 - 8\text{ MB}$), large spatial maps incur output backpressure to DRAM (`kANE_NE_OUTPUT_STALL_CYCLES`). Halving tensor size in INT8 cuts output stalls by $>2.1\times$ (503M $\to$ 233M cycles).
> 4. **L2 SRAM Fitting**: Reducing spatial dimensions to fit inside L2 SRAM ($H=64, W=64$) collapses output writeback stalls by $>16\times$ ($15.5\text{M}$ cycles) and unlocks peak ALU utilization ($>12,400\text{ MACs / cycle}$ across 16 cores, $>75\%$ theoretical saturation).
> 5. **QDQ Execution**: In QDQ (`dequantize -> conv -> quantize`), the internal convolution arithmetic executes in FP16 precision, matching FP16 throughput (~18.60 TOPS) and FP16 DMA footprint (~35.27 MB).
>
> *(For an exhaustive breakdown of each register, see [`How_to_Interpret_measure_ane_pmu_Numbers.md`](How_to_Interpret_measure_ane_pmu_Numbers.md).)*

### Historical Multi-Device Comparison Table

| Model | Device | Precision | Latency (Avg) | Speed (TOPS) |
| :--- | :--- | :--- | ---: | ---: |
| **Mac Mini M4 Pro** | **GPU** | FP16 | 39.05 ms | **9.90** |
| **Mac Mini M4 Pro** | **ANE** | FP16 | 20.90 ms | **18.50** |
| **Mac Mini M4 Pro** | **ANE** | INT8 | 10.76 ms | **35.91** |
| **Mac Mini M4 Pro** | **ANE** | DQ->FP16->Q | 20.65 ms | **18.72** |
| **MacBook Pro M1** | **GPU** | FP16 | 129.10 ms | **2.99** |
| **MacBook Pro M1** | **ANE** | FP16 | 35.70 ms | **10.83** |
| **MacBook Pro M1** | **ANE** | INT8 | 34.02 ms | **11.36** |
| **MacBook Pro M1** | **ANE** | DQ->FP16->Q | 33.99 ms | **11.37** |
| **iPhone 16 Pro** | **GPU** | FP16 | 149.35 ms | **2.59** |
| **iPhone 16 Pro** | **ANE** | FP16 | 12.81 ms | **30.18** |
| **iPhone 16 Pro** | **ANE** | INT8 | 7.98 ms | **48.46** |
| **iPhone 16 Pro** | **ANE** | DQ->FP16->Q | 6.41 ms | **60.29** |
| **iPhone 17 Pro** | **GPU** | FP16 | 57.02 ms | **6.78** |
| **iPhone 17 Pro** | **ANE** | FP16 | 8.70 ms | **44.41** |
| **iPhone 17 Pro** | **ANE** | INT8 | 7.85 ms | **49.27** |
| **iPhone 17 Pro** | **ANE** | DQ->FP16->Q | 6.12 ms | **63.20** |

*Note: Results may vary slightly depending on system load and thermal state.*
*Note: GPU INT8 convolution is not supported by MPSGraph on this device/configuration.*
