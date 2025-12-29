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
