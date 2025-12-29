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

## Running the Benchmark

Execute the compiled binary:

```bash
./measure_conv_fp16
```

## Benchmark Results

### Apple M4 Pro

| Device | Precision | Latency (Avg) | Speed (TOPS) |
| :--- | :--- | :--- | :--- |
| **GPU** | FP16 | 39.25 ms | **9.85** |
| **ANE** | FP16 | 20.92 ms | **18.48** |

*Note: Results may vary slightly depending on system load and thermal state.*
