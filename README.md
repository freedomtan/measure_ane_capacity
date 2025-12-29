# MPSGraph Convolution Benchmark

This project benchmarks the performance of 2D convolutions on Apple Silicon using Metal Performance Shaders Graph (MPSGraph). It measures the compute capacity in TOPS (Trillions of Operations Per Second) for both the GPU and the Apple Neural Engine (ANE).

## Build Instructions

To compile the project, ensure you have clang and the necessary frameworks (Foundation, Metal, MetalPerformanceShadersGraph) installed (standard on macOS with Xcode Command Line Tools).

Run the following command in the terminal:

```bash
make
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

## Running the Benchmark

Execute the compiled binary:

```bash
```bash
./measure_conv
```

## Benchmark Results

### Apple M4 Pro

| Device | Precision | Latency (Avg) | Speed (TOPS) |
| :--- | :--- | ---: | ---: |
| **GPU** | FP16 | 39.05 ms | **9.90** |
| **ANE** | FP16 | 20.90 ms | **18.50** |
| **ANE** | INT8 | 10.76 ms | **35.91** |

### Apple M1

| Device | Precision | Latency (Avg) | Speed (TOPS) |
| :--- | :--- | ---: | ---: |
| **GPU** | FP16 | 129.10 ms | **2.99** |
| **ANE** | FP16 | 35.70 ms | **10.83** |
| **ANE** | INT8 | 34.02 ms | **11.36** |

*Note: Results may vary slightly depending on system load and thermal state.*
*Note: GPU INT8 convolution is not supported by MPSGraph on this device/configuration.*
