import Foundation
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

// MARK: - Benchmark Errors
enum BenchmarkError: LocalizedError {
    case metalNotSupported
    case aneNotSupported
    case graphCompilationFailed(String)
    case bufferAllocationFailed
    case commandQueueCreationFailed
    case executionCancelled
    case executionFailed(String)
    
    var errorDescription: String? {
        switch self {
        case .metalNotSupported:
            return "Metal is not supported on this device."
        case .aneNotSupported:
            return "Apple Neural Engine (ANE) and MetalPerformanceShadersGraph require a physical iOS device (not supported in the iOS Simulator)."
        case .graphCompilationFailed(let reason):
            return "MPSGraph compilation failed: \(reason)"
        case .bufferAllocationFailed:
            return "Failed to allocate Metal buffer for input tensor."
        case .commandQueueCreationFailed:
            return "Failed to create Metal command queue."
        case .executionCancelled:
            return "Benchmark execution was cancelled by user."
        case .executionFailed(let reason):
            return "Execution failed: \(reason)"
        }
    }
}

// MARK: - Benchmark Engine
private func fillNonZeroData(
    buffer: UnsafeMutableRawPointer,
    byteCount: Int,
    dataType: MPSDataType,
    magnitude: Float = 1.0,
    seed: UInt64 = 0x5EED5EED5EED5EED
) {
    guard byteCount > 0 else { return }
    var state = seed
    if dataType.rawValue == 0x10430008 { // Float8e4m3
        // Multi-tier physical magnitudes [0.25, 0.5, 0.75, 1.0] to break symmetric binomial zero cancellation
        let posTiers: [UInt8] = [0x28, 0x30, 0x38, 0x40]
        let negTiers: [UInt8] = [0xA8, 0xB0, 0xB8, 0xC0]
        let ptr = buffer.bindMemory(to: UInt8.self, capacity: byteCount)
        for i in 0..<byteCount {
            state ^= state << 13
            state ^= state >> 7
            state ^= state << 17
            let tier = Int((state >> 1) & 3)
            ptr[i] = (state & 1) != 0 ? negTiers[tier] : posTiers[tier]
        }
    } else if dataType == .float16 {
        // Multi-tier non-zero dithering breaks binomial zero-cancellation at L=1
        // and maintains bounded, non-overflowing variance across deep chains (L=100).
        let ptr = buffer.bindMemory(to: UInt16.self, capacity: byteCount / 2)
        let count = byteCount / 2
        for i in 0..<count {
            state ^= state << 13
            state ^= state >> 7
            state ^= state << 17
            let tier = Float((state >> 1) & 3)
            let mag = magnitude * (0.85 + tier * 0.10)
            let val = Float16(mag)
            let posBits = val.bitPattern
            let negBits = posBits | 0x8000
            ptr[i] = (state & 1) != 0 ? negBits : posBits
        }
    } else {
        // INT8: Multi-tier non-zero pseudo-random magnitudes {±1, ±2, ±3, ±4} to break symmetric zero cancellation
        let ptr = buffer.bindMemory(to: Int8.self, capacity: byteCount)
        for i in 0..<byteCount {
            state ^= state << 13
            state ^= state >> 7
            state ^= state << 17
            let tier: Int8 = Int8((state >> 1) & 3) + 1
            ptr[i] = (state & 1) != 0 ? -tier : tier
        }
    }
}

private func countZeroElements(buffer: UnsafeMutableRawPointer, elementCount: Int, dataType: MPSDataType) -> Int {
    guard elementCount > 0 else { return 0 }
    if dataType == .float16 {
        let ptr = buffer.bindMemory(to: UInt16.self, capacity: elementCount)
        var zeros = 0
        for i in 0..<elementCount {
            // FP16 zero is 0x0000 (+0) or 0x8000 (-0).
            if ptr[i] == 0x0000 || ptr[i] == 0x8000 { zeros += 1 }
        }
        return zeros
    } else if dataType.rawValue == 0x10430008 { // Float8e4m3
        // E4M3 zero is 0x00 (+0) or 0x80 (-0).
        let ptr = buffer.bindMemory(to: UInt8.self, capacity: elementCount)
        var zeros = 0
        for i in 0..<elementCount {
            if ptr[i] == 0x00 || ptr[i] == 0x80 { zeros += 1 }
        }
        return zeros
    } else {
        // Int8 zero is 0x00.
        let ptr = buffer.bindMemory(to: UInt8.self, capacity: elementCount)
        var zeros = 0
        for i in 0..<elementCount {
            if ptr[i] == 0 { zeros += 1 }
        }
        return zeros
    }
}

private func createNonZeroData(byteCount: Int, dataType: MPSDataType, magnitude: Float = 1.0) -> Data {
    var data = Data(count: byteCount)
    data.withUnsafeMutableBytes { rawBuf in
        if let baseAddress = rawBuf.baseAddress {
            fillNonZeroData(buffer: baseAddress, byteCount: byteCount, dataType: dataType, magnitude: magnitude, seed: 0x5EED5EED5EED5EED)
        }
    }
    return data
}

// Same spirit as MILSpecBuilder.m's per-channel conv bias ("Zero-overhead
// fused hardware bias: alternating +/-0.05 to maintain signal vitality
// without extra cycles") -- the MPSGraph GPU INT8 QDQ path has no bias at
// all, which makes 0 an absorbing state once any activation underflows the
// quantization scale. +/-0.05 (MIL's value) only got MPSGraph's zero rate
// from 19.6% to 3.3% here; +/-0.3 was needed to reach a MIL-comparable
// <1% floor -- see docs/GPU_MXU_And_Convolution_Dispatch_Investigation.md Finding 4e for
// the full scale+bias tuning trail and the remaining open question of why
// the magnitude needed differs between the two backends.
private func createAlternatingBiasData(channels: Int) -> Data {
    var data = Data(count: channels * 2)
    let posB16 = Float16(0.3).bitPattern
    let negB16 = Float16(-0.3).bitPattern
    data.withUnsafeMutableBytes { rawBuf in
        guard let ptr = rawBuf.baseAddress?.assumingMemoryBound(to: UInt16.self) else { return }
        for c in 0..<channels {
            ptr[c] = (c & 1) != 0 ? negB16 : posB16
        }
    }
    return data
}

final class ANECapacityEngine {
    static let shared = ANECapacityEngine()
    
    private let mtlDevice: MTLDevice?
    
    init() {
        self.mtlDevice = MTLCreateSystemDefaultDevice()
    }
    
    var metalDeviceName: String {
        return mtlDevice?.name ?? "Unavailable"
    }
    
    var hasANE: Bool {
        #if targetEnvironment(simulator)
        return false
        #else
        guard mtlDevice != nil else { return false }
        let cd = MPSGraphCompilationDescriptor()
        return cd.responds(to: Selector(("setPreferredDevice:")))
        #endif
    }
    
    /**
     * Runs a single benchmark configuration matching measure_conv_universal.m and measure_ane_pmu.m
     */
    func runSingle(
        dimensions dims: ConvDimensions,
        precision: PrecisionMode,
        target: DeviceTarget,
        iterations: Int = 20,
        sweepType: SweepType = .none,
        sweepValue: Double = 0.0,
        sweepLabel: String = "",
        logHandler: @escaping (String) -> Void
    ) async throws -> BenchmarkResult {
        #if targetEnvironment(simulator)
        throw BenchmarkError.aneNotSupported
        #endif
        
        guard let device = mtlDevice else {
            throw BenchmarkError.metalNotSupported
        }
        
        if Task.isCancelled { throw BenchmarkError.executionCancelled }

        // GPU GEMM: bypass MPSGraph's public matrixMultiplication (which only
        // ever runs plain FP16 math underneath a quantize/dequantize wrapper --
        // Finding 4e) and dispatch the real native MPSNDArrayMatrixMultiplication/
        // MPSNDArrayQuantizedMatrixMultiplication kernels directly. This is the
        // only path that actually reaches A20 Pro's native FP8 ALUs; see
        // docs/GPU_MXU_And_Convolution_Dispatch_Investigation.md Finding 4l/4q.
        //
        // Exception: native FP8 GEMM hard-aborts on Apple9-family GPUs (A17
        // Pro/A18/A18 Pro -- see ANE comment below and the Apple10 guard in
        // runNativeGPUMatmul). On that hardware, fall through to the MPSGraph
        // dequant/quantize-wrapped path instead (the same one used for ANE):
        // it runs without crashing, just without any FP8 acceleration (Finding
        // 4e already established this path is plain FP16 math underneath).
        if dims.opType == .matmul && target == .gpu {
            if !(precision.isFP8 && !device.supportsFamily(.apple10)) {
                return try await runNativeGPUMatmul(
                    dims: dims,
                    precision: precision,
                    iterations: iterations,
                    sweepType: sweepType,
                    sweepValue: sweepValue,
                    sweepLabel: sweepLabel,
                    logHandler: logHandler,
                    device: device
                )
            }
            logHandler("[GPU \(precision.rawValue)] Native FP8 GEMM unsupported on this GPU (requires Apple10 GPU family or newer) -- falling back to MPSGraph dequant/quantize path (not hardware-accelerated, see Finding 4e).")
        }

        let mpsType = precision.mpsDataType
        let elementSize = precision.elementSize

        logHandler("[\(target.shortName) \(precision.rawValue)] Building graph: \(dims.shortDescription)...")
        
        // 1. Device and Compilation Descriptor Setup
        // On iOS/macOS MPSGraph, the device is always backed by the MTLDevice,
        // and the compilation descriptor specifies preferredDevice = 2 for ANE.
        let mDev = MPSGraphDevice(mtlDevice: device)
        let cd = MPSGraphCompilationDescriptor()
        
        if target == .ane {
            cd.optimizationLevel = .level1
            if cd.responds(to: Selector(("setPreferredDevice:"))) {
                cd.setValue(2, forKey: "preferredDevice") // 2 == MPSGraphDeviceTypeANE
            }
        } else {
            cd.optimizationLevel = .level0
        }
        
        // 2. Build MPSGraph
        let graph = MPSGraph()
        
        let inShape: [NSNumber]
        let input: MPSGraphTensor
        var cur: MPSGraphTensor
        let inputBufferLen: Int
        let outputBufferLen: Int
        let outShape: [NSNumber]
        
        let spatialH: Int
        let spatialW: Int
        
        if dims.opType == .matmul {
            // MatMul (GEMM: [B, M, K] x [B, K, N] -> [B, M, N]) is mapped as a 1x1 2D Convolution on ANE:
            //   Input:   [B, K, H, W] where H * W = M
            //   Weights: [N, K, 1, 1]
            //   Output:  [B, N, H, W]
            //
            // Rationale:
            // 1. Physical ANE MAC array aligns to 64-wide channels and executes reduction across Cin (K),
            //    not spatial dimensions.
            // 2. Spatial tile (H, W) divides M into power-of-2 2D spatial coordinates (e.g. 1024 -> 32x32),
            //    completely avoiding the degenerate C=1 layout failure that causes 100% zero outputs on device.
            // 3. MPSGraph recognizes [N, K, 1, 1] as standard OIHW weights layout, enabling full native
            //    W8A8 FP8 QDQ compilation to ANE binary_0.hwx on H18 and H19 with ZERO GPU fallback.
            let sW = 1 << ((Int(floor(log2(Double(dims.m))))) / 2)
            let sH = dims.m / sW
            spatialH = sH
            spatialW = sW
            
            inShape = [
                NSNumber(value: dims.batch),
                NSNumber(value: dims.k),
                NSNumber(value: spatialH),
                NSNumber(value: spatialW)
            ]
            let wShape: [NSNumber] = [
                NSNumber(value: dims.n),
                NSNumber(value: dims.k),
                1,
                1
            ]
            
            input = graph.placeholder(shape: inShape, dataType: mpsType, name: "in")
            cur = input
            
            guard let d = MPSGraphConvolution2DOpDescriptor(
                strideInX: 1,
                strideInY: 1,
                dilationRateInX: 1,
                dilationRateInY: 1,
                groups: 1,
                paddingStyle: .TF_SAME,
                dataLayout: .NCHW,
                weightsLayout: .OIHW
            ) else {
                throw BenchmarkError.graphCompilationFailed("Invalid convolution descriptor for 1x1 GEMM")
            }
            
            let weightMag = Float(0.96 / sqrt(Double(max(dims.k, 1))))
            
            if precision.isFP8 {
                guard #available(iOS 27.0, macOS 27.0, *) else {
                    throw BenchmarkError.executionFailed("FP8 requires iOS 27.0 / macOS 27.0 or newer.")
                }
                let wLength = dims.n * dims.k * 1 * 1 * 1
                let wData = createNonZeroData(byteCount: wLength, dataType: mpsType)
                let wFP8 = graph.constant(wData, shape: wShape, dataType: mpsType)
                
                // Full W8A8 QDQ: Unit scale 1.0 prevents ANE dynamic placeholder dequantizer underflow
                let fp8Scale: Double = 1.0
                let w = graph.dequantize(wFP8, scale: fp8Scale, zeroPoint: 0.0, dataType: .float16, name: "w_dequant")
                for _ in 0..<dims.layers {
                    let lhs = graph.dequantize(cur, scale: fp8Scale, zeroPoint: 0.0, dataType: .float16, name: "act_dequant")
                    let outFP16 = graph.convolution2D(lhs, weights: w, descriptor: d, name: nil)
                    cur = graph.quantize(outFP16, scale: fp8Scale, zeroPoint: 0.0, dataType: mpsType, name: "act_quant")
                }
            } else if mpsType == .int8 && target != .ane {
                let wLength = dims.n * dims.k * 1 * 1 * 2
                let wData = createNonZeroData(byteCount: wLength, dataType: .float16, magnitude: weightMag)
                let w = graph.constant(wData, shape: wShape, dataType: .float16)
                // Fix for all-zero output: plain cast() has no scale at all (int8
                // value N <-> float16 value N literally), and even quantize/
                // dequantize at scale=1.0 rounds any activation magnitude < 0.5
                // to exactly 0 -- with no bias anywhere in this chain, zero is an
                // absorbing state and the signal fully decays by L=20. Use a
                // small fixed scale (activations here run O(1-3) by design, same
                // as weightMag's unit-variance-per-layer convention) so typical
                // magnitudes map to a well-resolved, non-degenerate int8 range
                // instead of rounding to zero. See
                // docs/GPU_MXU_And_Convolution_Dispatch_Investigation.md Finding 4e. A small
                // scale alone (tried 1/8 and 1/32) only got from 100% zero
                // down to a scale-invariant 19.6% zero -- matching MIL's own
                // GPU INT8 path (which has always been clean, 0% zero) needed
                // its other difference too: a per-channel bias, breaking the
                // "zero is absorbing" cascade entirely rather than just
                // shrinking its basin.
                let int8Scale: Double = 1.0 / 32.0
                let biasData = createAlternatingBiasData(channels: dims.n)
                let bias = graph.constant(biasData, shape: [1, NSNumber(value: dims.n), 1, 1], dataType: .float16)
                for _ in 0..<dims.layers {
                    let lhs = graph.dequantize(cur, scale: int8Scale, zeroPoint: 0.0, dataType: .float16, name: "act_dequant")
                    let convOut = graph.convolution2D(lhs, weights: w, descriptor: d, name: nil)
                    let outFP16 = graph.addition(convOut, bias, name: "add_bias")
                    cur = graph.quantize(outFP16, scale: int8Scale, zeroPoint: 0.0, dataType: mpsType, name: "act_quant")
                }
            } else {
                let wLength = dims.n * dims.k * 1 * 1 * elementSize
                let wData = createNonZeroData(byteCount: wLength, dataType: mpsType, magnitude: weightMag)
                let w = graph.constant(wData, shape: wShape, dataType: mpsType)
                
                for _ in 0..<dims.layers {
                    cur = graph.convolution2D(cur, weights: w, descriptor: d, name: nil)
                    if mpsType == .int8 {
                        let fp = graph.cast(cur, to: .float16, name: "dequant")
                        cur = graph.cast(fp, to: .int8, name: "requant")
                    }
                }
            }
            inputBufferLen = dims.batch * dims.m * dims.k * elementSize
            outputBufferLen = dims.batch * dims.m * dims.n * elementSize
            outShape = [
                NSNumber(value: dims.batch),
                NSNumber(value: dims.n),
                NSNumber(value: spatialH),
                NSNumber(value: spatialW)
            ]
        } else {
            spatialH = dims.height
            spatialW = dims.width
            inShape = [
                NSNumber(value: dims.batch),
                NSNumber(value: dims.inChannels),
                NSNumber(value: dims.height),
                NSNumber(value: dims.width)
            ]
            let wShape: [NSNumber] = [
                NSNumber(value: dims.outChannels),
                NSNumber(value: dims.inChannels),
                NSNumber(value: dims.kernelSize),
                NSNumber(value: dims.kernelSize)
            ]
            
            input = graph.placeholder(shape: inShape, dataType: mpsType, name: "in")
            cur = input
            
            guard let d = MPSGraphConvolution2DOpDescriptor(
                strideInX: 1,
                strideInY: 1,
                dilationRateInX: 1,
                dilationRateInY: 1,
                groups: 1,
                paddingStyle: .TF_SAME,
                dataLayout: .NCHW,
                weightsLayout: .OIHW
            ) else {
                throw BenchmarkError.graphCompilationFailed("Invalid convolution descriptor")
            }
            
            let convReductionLen = dims.inChannels * dims.kernelSize * dims.kernelSize
            let weightMag = Float(0.96 / sqrt(Double(max(convReductionLen, 1))))
            
            if precision.isFP8 {
                guard #available(iOS 27.0, macOS 27.0, *) else {
                    throw BenchmarkError.executionFailed("FP8 requires iOS 27.0 / macOS 27.0 or newer.")
                }
                let wLength = dims.outChannels * dims.inChannels * dims.kernelSize * dims.kernelSize * 1
                let wData = createNonZeroData(byteCount: wLength, dataType: mpsType)
                let wFP8 = graph.constant(wData, shape: wShape, dataType: mpsType)
                
                // Full W8A8 QDQ: Unit scale 1.0 prevents ANE dynamic placeholder dequantizer underflow
                let fp8Scale: Double = 1.0
                let w = graph.dequantize(wFP8, scale: fp8Scale, zeroPoint: 0.0, dataType: .float16, name: "w_dequant")
                for _ in 0..<dims.layers {
                    let actFP16 = graph.dequantize(cur, scale: fp8Scale, zeroPoint: 0.0, dataType: .float16, name: "act_dequant")
                    let outFP16 = graph.convolution2D(actFP16, weights: w, descriptor: d, name: nil)
                    cur = graph.quantize(outFP16, scale: fp8Scale, zeroPoint: 0.0, dataType: mpsType, name: "act_quant")
                }
            } else if mpsType == .int8 && target != .ane {
                // MPS's GPU convolution kernel only supports FP32/FP16
                // operands -- unlike ANE, which has a native INT8 conv path.
                // Passing raw INT8 tensors to convolution2D here does not
                // throw a catchable NSException; it hits a hard assertion
                // inside MPSNDArrayConvolutionPreG13.mm ("Only FP32 or FP16
                // convolution supported") and calls abort(), taking the whole
                // app down. Route through the same QDQ pattern matmul's INT8
                // path already uses below: dequantize activations to FP16,
                // convolve in FP16 with FP16 weights, requantize the output.
                //
                // Fix for all-zero output: see the matching comment in the
                // matmul-as-1x1-conv branch above -- plain cast() has no scale,
                // and scale=1.0 quantize/dequantize rounds sub-0.5-magnitude
                // activations to exactly 0, which (with no bias anywhere in
                // this chain) is an absorbing state that fully decays the
                // signal by L=20. docs/GPU_MXU_And_Convolution_Dispatch_Investigation.md
                // Finding 4e.
                let wLength = dims.outChannels * dims.inChannels * dims.kernelSize * dims.kernelSize * 2
                let wData = createNonZeroData(byteCount: wLength, dataType: .float16, magnitude: weightMag)
                let w = graph.constant(wData, shape: wShape, dataType: .float16)

                // Also needs the bias fix (see matmul-as-1x1-conv branch
                // above): scale alone only got to a scale-invariant 19.6%
                // zero, matching MIL's clean 0% needed the bias too.
                let int8Scale: Double = 1.0 / 32.0
                let biasData = createAlternatingBiasData(channels: dims.outChannels)
                let bias = graph.constant(biasData, shape: [1, NSNumber(value: dims.outChannels), 1, 1], dataType: .float16)
                for _ in 0..<dims.layers {
                    let lhs = graph.dequantize(cur, scale: int8Scale, zeroPoint: 0.0, dataType: .float16, name: "act_dequant")
                    let convOut = graph.convolution2D(lhs, weights: w, descriptor: d, name: nil)
                    let outFP16 = graph.addition(convOut, bias, name: "add_bias")
                    cur = graph.quantize(outFP16, scale: int8Scale, zeroPoint: 0.0, dataType: mpsType, name: "act_quant")
                }
            } else {
                // Constant weights
                let wLength = dims.outChannels * dims.inChannels * dims.kernelSize * dims.kernelSize * elementSize
                let wData = createNonZeroData(byteCount: wLength, dataType: mpsType, magnitude: weightMag)
                let w = graph.constant(wData, shape: wShape, dataType: mpsType)

                // Chain L layers
                for _ in 0..<dims.layers {
                    cur = graph.convolution2D(cur, weights: w, descriptor: d, name: nil)

                    // Int8 simulated quantized flow: Int8 -> Conv -> FP16 dequant -> Int8 requant
                    if mpsType == .int8 {
                        let fp = graph.cast(cur, to: .float16, name: "dequant")
                        cur = graph.cast(fp, to: .int8, name: "requant")
                    }
                }
            }
            inputBufferLen = dims.batch * dims.height * dims.width * dims.inChannels * elementSize
            outputBufferLen = dims.batch * dims.height * dims.width * dims.outChannels * elementSize
            outShape = [
                NSNumber(value: dims.batch),
                NSNumber(value: dims.outChannels),
                NSNumber(value: dims.height),
                NSNumber(value: dims.width)
            ]
        }
        
        if Task.isCancelled { throw BenchmarkError.executionCancelled }
        
        // 3. Compile Executable with Exception Protection
        var beforeDirs: Set<String> = []
        if target == .ane {
            beforeDirs = ANEClientBridge.existingANETempDirectories()
        }
        
        logHandler("[\(target.shortName) \(precision.rawValue)] Compiling graph on \(target.shortName)...")
        let feeds = [input: MPSGraphShapedType(shape: inShape, dataType: mpsType)]
        
        var compiledExe: MPSGraphExecutable? = nil
        do {
            try ANEClientBridge.catchException {
                compiledExe = graph.compile(
                    with: mDev,
                    feeds: feeds,
                    targetTensors: [cur],
                    targetOperations: nil,
                    compilationDescriptor: cd
                )
            }
        } catch {
            throw BenchmarkError.graphCompilationFailed(error.localizedDescription)
        }
        
        guard let exe = compiledExe else {
            throw BenchmarkError.graphCompilationFailed("Compiler returned nil executable")
        }
        
        // 4. Allocate I/O buffers
        guard let iBuf = device.makeBuffer(length: inputBufferLen, options: []) else {
            throw BenchmarkError.bufferAllocationFailed
        }
        fillNonZeroData(buffer: iBuf.contents(), byteCount: inputBufferLen, dataType: mpsType, seed: 0x9E3779B97F4A7C15)
        let iData = MPSGraphTensorData(iBuf, shape: inShape, dataType: mpsType)

        // Captured (not results: nil) so the result can be checked for
        // degenerate all-zero output below -- a plausible-looking TOPS
        // number computed from wall-clock latency alone cannot distinguish
        // real work from H17+ zero-skip inflating a degenerate result, which
        // is exactly what FP8 W8A8 QDQ hits on some ANE generations (see
        // measure_conv_fp8.m's file-header note on H18).
        guard let oBuf = device.makeBuffer(length: outputBufferLen, options: []) else {
            throw BenchmarkError.bufferAllocationFailed
        }
        let oData = MPSGraphTensorData(oBuf, shape: outShape, dataType: mpsType)

        guard let queue = device.makeCommandQueue() else {
            throw BenchmarkError.commandQueueCreationFailed
        }

        let ed = MPSGraphExecutableExecutionDescriptor()
        ed.waitUntilCompleted = true

        // Optional GPU frame capture, gated by env var so default runs are
        // unaffected -- set ANE_CAPACITY_GPU_CAPTURE=1 (plus MetalCaptureEnabled
        // in Info.plist, already set) to dump a .gputrace per GPU benchmark
        // run into the app's Documents directory. Written to investigate
        // docs/GPU_MXU_And_Convolution_Dispatch_Investigation.md Finding 4e: whether FP8 and
        // INT8 convolution dispatch to different compiled GPU kernels despite
        // both nominally running the same FP16-only convolution2D op.
        var gpuCapture: MTLCaptureManager? = nil
        if target != .ane, ProcessInfo.processInfo.environment["ANE_CAPACITY_GPU_CAPTURE"] == "1" {
            let captureManager = MTLCaptureManager.shared()
            if captureManager.supportsDestination(.gpuTraceDocument) {
                let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
                let fileName = "capture_\(target.shortName)_\(precision.rawValue)_\(Int(Date().timeIntervalSince1970)).gputrace"
                let traceURL = docs.appendingPathComponent(fileName)
                let descriptor = MTLCaptureDescriptor()
                descriptor.captureObject = device
                descriptor.destination = .gpuTraceDocument
                descriptor.outputURL = traceURL
                do {
                    try captureManager.startCapture(with: descriptor)
                    gpuCapture = captureManager
                    logHandler("[\(target.shortName) \(precision.rawValue)] GPU capture -> \(traceURL.lastPathComponent)")
                } catch {
                    logHandler("[\(target.shortName) \(precision.rawValue)] GPU capture failed to start: \(error.localizedDescription)")
                }
            } else {
                logHandler("[\(target.shortName) \(precision.rawValue)] GPU capture requested but .gpuTraceDocument destination unsupported on this device/OS.")
            }
        }

        // 5. Warmup with Exception Protection
        logHandler("[\(target.shortName) \(precision.rawValue)] Warming up pipeline...")
        do {
            try ANEClientBridge.catchException {
                exe.run(with: queue, inputs: [iData], results: [oData], executionDescriptor: ed)
            }
        } catch {
            gpuCapture?.stopCapture()
            throw BenchmarkError.executionFailed("Warmup failed: \(error.localizedDescription)")
        }

        if let gpuCapture {
            // One compiled-executable dispatch is enough to see which actual
            // GPU kernel name got selected; stop immediately to keep the
            // trace small and avoid skewing the timed loop below.
            gpuCapture.stopCapture()
        }

        if Task.isCancelled { throw BenchmarkError.executionCancelled }

        // 6. Timed Benchmarking loop
        logHandler("[\(target.shortName) \(precision.rawValue)] Benchmarking \(iterations) iterations...")
        let startNanos = clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW)

        for i in 0..<iterations {
            if Task.isCancelled { throw BenchmarkError.executionCancelled }

            do {
                try ANEClientBridge.catchException {
                    exe.run(with: queue, inputs: [iData], results: [oData], executionDescriptor: ed)
                }
            } catch {
                throw BenchmarkError.executionFailed("Iteration \(i) failed: \(error.localizedDescription)")
            }

            // Yield every 5 iterations to allow UI updates
            if (i + 1) % 5 == 0 {
                await Task.yield()
            }
        }
        
        let endNanos = clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW)
        
        let durationSec = Double(endNanos - startNanos) / 1e9
        let avgSec = durationSec / Double(iterations)
        let avgMs = avgSec * 1000.0
        
        // Calculate TOPS: totalOps / (avgSec * 1e12)
        let tops = dims.totalOperations / (avgSec * 1e12)

        // A degenerate all-zero result reads as a plausible, even fast,
        // number here -- wall-clock latency alone can't tell real work from
        // H17+ zero-skip inflating throughput on garbage output. Confirmed on
        // iPhone 18 Pro: FP8 W8A8 QDQ's runtime activation quantize/
        // dequantize zeroes 100% of output on the ANE regardless of scale or
        // magnitude (see measure_conv_fp8.m). Surface it here rather than
        // silently reporting an inflated TOPS the UI can't distinguish from
        // a real one.
        let outputElementCount = outputBufferLen / elementSize
        let zeroCount = countZeroElements(buffer: oBuf.contents(), elementCount: outputElementCount, dataType: mpsType)
        if zeroCount == outputElementCount {
            logHandler("[\(target.shortName) \(precision.rawValue)] WARNING: 100% zero output "
                + "(\(outputElementCount)/\(outputElementCount) elements) -- this is not a real "
                + "measurement. The TOPS below is almost certainly hardware zero-skip inflating "
                + "a degenerate result, not real throughput.")
        }

        // 7. PMU Telemetry Harvesting
        var pmuCounters: [String: UInt64] = [:]
        var computeCycles: UInt64 = 0
        var nominalCycles: UInt64 = 0
        var outputStallCycles: UInt64 = 0
        var inputStallCycles: UInt64 = 0
        var dmaRwBytes: UInt64 = 0
        var dpeEnergy: UInt64 = 0
        var hwExecutionTimeNs: UInt64 = 0
        var macsPerCoreCycle: Double = 0.0
        var chipMacsPerCycle: Double = 0.0
        var aluSaturation: Double = 0.0
        var effectiveClockGhz: Double = 0.0
        
        if target == .ane {
            logHandler("   [PMU] Locating compiled ANE microcode bundle...")
            if let tempDir = ANEClientBridge.findNewANETempDirectory(since: beforeDirs) {
                let bundleURL = URL(fileURLWithPath: tempDir)
                logHandler("   [PMU] Found ANE microcode bundle: \(bundleURL.lastPathComponent)")
                
                let pmuRes = (dims.opType == .matmul)
                    ? ANEClientBridge.profileANECIRBundle(
                        at: bundleURL,
                        batch: UInt(dims.batch),
                        height: UInt(spatialH),
                        width: UInt(spatialW),
                        inChannels: UInt(dims.k),
                        outChannels: UInt(dims.n),
                        dataType: mpsType,
                        iterations: UInt(iterations),
                        totalMacs: dims.totalOperations / 2.0
                    )
                    : ANEClientBridge.profileANECIRBundle(
                        at: bundleURL,
                        batch: UInt(dims.batch),
                        height: UInt(dims.height),
                        width: UInt(dims.width),
                        inChannels: UInt(dims.inChannels),
                        outChannels: UInt(dims.outChannels),
                        dataType: mpsType,
                        iterations: UInt(iterations),
                        totalMacs: dims.totalOperations / 2.0
                    )
                
                if pmuRes.success && !pmuRes.performanceCounterDeltasPerIter.isEmpty {
                    for (k, v) in pmuRes.performanceCounterDeltasPerIter {
                        pmuCounters[k] = v.uint64Value
                    }
                    computeCycles = pmuCounters["kANE_NE_COMPUTE_CYCLES"] ?? 0
                    nominalCycles = pmuCounters["kANE_NE_NOMINAL_CYCLES"] ?? 0
                    outputStallCycles = pmuCounters["kANE_NE_OUTPUT_STALL_CYCLES"] ?? 0
                    inputStallCycles = pmuCounters["kANE_NE_INPUT_STALL_CYCLES"] ?? 0
                    dmaRwBytes = (pmuCounters["kANE_DMA_READWRITE_BYTES"] ?? 0) + (pmuCounters["kANE_DMA_READ_BYTES"] ?? 0)
                    dpeEnergy = pmuCounters["kANE_DPE_ENERGY"] ?? 0
                    hwExecutionTimeNs = pmuRes.hwExecutionTimeNs
                    macsPerCoreCycle = pmuRes.macsPerCoreCycle
                    chipMacsPerCycle = pmuRes.chipMacsPerCycle
                    aluSaturation = (precision == .int8) ? pmuRes.aluSaturationInt8 : pmuRes.aluSaturationFp16
                    effectiveClockGhz = pmuRes.effectiveCoreClockGhz
                    // Note: do not overwrite `tops` with `pmuRes.topsRealized`.
                    // `tops` is measured from real end-to-end timed iterations of the graph executable,
                    // whereas `pmuRes.topsRealized` evaluates an isolated microcode bundle with synthetic timing.
                    
                    let pmuSummary = String(
                        format: "   └─ PMU: Compute=+%@, Stalls=+%@, DMA=+%@, ALU Saturation=%.1f%%, Clock=%.2f GHz",
                        Self.formatCompact(computeCycles),
                        Self.formatCompact(outputStallCycles + inputStallCycles),
                        Self.formatBytes(dmaRwBytes),
                        aluSaturation,
                        effectiveClockGhz
                    )
                    logHandler(pmuSummary)
                } else {
                    logHandler("   [PMU Warning] Profiling bundle failed: \(pmuRes.statusMessage)")
                }
            } else {
                logHandler("   [PMU Note] No temporary ANE bundle emitted by MPSGraph on this runtime/device.")
            }
        }
        
        let resultMsg = String(
            format: "[%@ %@] Avg: %.2f ms, Throughput: %.4f TOPS (%.1f GFLOPs/iter), Zeros: %ld/%ld (%.1f%%)",
            target.shortName, precision.rawValue, avgMs, tops, dims.gflops,
            zeroCount, outputElementCount,
            outputElementCount > 0 ? Double(zeroCount) * 100.0 / Double(outputElementCount) : 0.0
        )
        logHandler(resultMsg)
        
        return BenchmarkResult(
            dimensions: dims,
            precision: precision,
            target: target,
            avgDurationMs: avgMs,
            tops: tops,
            iterations: iterations,
            sweepType: sweepType,
            sweepValue: sweepValue,
            sweepLabel: sweepLabel,
            pmuCounters: pmuCounters,
            computeCycles: computeCycles,
            nominalCycles: nominalCycles,
            outputStallCycles: outputStallCycles,
            inputStallCycles: inputStallCycles,
            dmaRwBytes: dmaRwBytes,
            dpeEnergy: dpeEnergy,
            hwExecutionTimeNs: hwExecutionTimeNs,
            macsPerCoreCycle: macsPerCoreCycle,
            chipMacsPerCycle: chipMacsPerCycle,
            aluSaturation: aluSaturation,
            effectiveClockGhz: effectiveClockGhz,
            zeroOutputCount: zeroCount,
            outputElementCount: outputElementCount
        )
    }
    
    /**
     * GPU GEMM via the native MPSNDArrayMatrixMultiplication /
     * MPSNDArrayQuantizedMatrixMultiplication kernels, bypassing MPSGraph's
     * public matrixMultiplication op entirely. Finding 4e (docs/GPU_MXU_And_
     * Convolution_Dispatch_Investigation.md) showed MPSGraph's quantize/
     * dequantize-wrapped matmul never reaches native low-precision hardware --
     * it's plain FP16 math underneath regardless of the wrapper dtype. This is
     * the path that actually dispatches to A20 Pro's native FP8 ALUs
     * (Finding 4l/4q), confirmed ~2x over FP16 once the GEMM is large/deep
     * enough (L=20 chained) to saturate the GPU.
     *
     * FP16/FP8 chain L layers deep (homogeneous dtype in/out, ping-ponging two
     * destination buffers -- layer n's output feeds layer n+1). INT8-affine
     * stays at L=1 regardless of dims.layers: MPSNDArrayQuantizedMatrixMultiplication
     * always dequantizes on write (destination dtype float16 != input dtype
     * int8), so there's no way to chain it without a requantize step this
     * plain NDArray API doesn't expose.
     */
    private func runNativeGPUMatmul(
        dims: ConvDimensions,
        precision: PrecisionMode,
        iterations: Int,
        sweepType: SweepType,
        sweepValue: Double,
        sweepLabel: String,
        logHandler: @escaping (String) -> Void,
        device: MTLDevice
    ) async throws -> BenchmarkResult {
        let mpsType = precision.mpsDataType
        let elementSize = precision.elementSize
        let B = dims.batch, M = dims.m, K = dims.k, N = dims.n
        let L = max(dims.layers, 1)

        logHandler("[GPU \(precision.rawValue)] Building native MPSNDArray GEMM (bypassing MPSGraph): \(dims.shortDescription)...")

        guard let queue = device.makeCommandQueue() else {
            throw BenchmarkError.commandQueueCreationFailed
        }

        func ndArray(_ dataType: MPSDataType, _ shape: [Int]) -> MPSNDArray {
            let desc = MPSNDArrayDescriptor(dataType: dataType, shape: shape.map { NSNumber(value: $0) })
            return MPSNDArray(device: device, descriptor: desc)
        }
        func fillND(_ arr: MPSNDArray, byteCount: Int, dataType: MPSDataType, magnitude: Float = 1.0) {
            var data = createNonZeroData(byteCount: byteCount, dataType: dataType, magnitude: magnitude)
            data.withUnsafeMutableBytes { raw in
                if let base = raw.baseAddress { arr.writeBytes(base, strideBytes: nil) }
            }
        }

        if Task.isCancelled { throw BenchmarkError.executionCancelled }

        var effectiveLayers = L
        var outputCount = 0
        var finalDest: MPSNDArray! = nil
        var dispatchBody: ((MTLCommandBuffer) -> Void)!

        if precision.isFP8 {
            guard #available(iOS 27.0, macOS 27.0, *) else {
                throw BenchmarkError.executionFailed("FP8 requires iOS 27.0 / macOS 27.0 or newer.")
            }
            // MPSNDArrayMatrixMultiplication hard-aborts (uncatchable assertion
            // failure, not a Swift error) on FP8 input/output on Apple9-family
            // GPUs (A17 Pro/A18/A18 Pro). FP8 GEMM is only safe from Apple10
            // (A19-gen) onward, so this must be checked before encode(), not
            // just the OS version.
            guard device.supportsFamily(.apple10) else {
                throw BenchmarkError.executionFailed("Native FP8 GEMM is not supported on this GPU (requires Apple10 GPU family or newer).")
            }
            let lhs = ndArray(mpsType, [B, 1, M, K])
            let rhs = ndArray(mpsType, [B, 1, K, N])
            fillND(lhs, byteCount: B * M * K * elementSize, dataType: mpsType)
            fillND(rhs, byteCount: B * K * N * elementSize, dataType: mpsType)
            let destA = ndArray(mpsType, [B, 1, M, N])
            let destB = ndArray(mpsType, [B, 1, M, N])
            let mm = MPSNDArrayMatrixMultiplication(device: device, sourceCount: 2)
            finalDest = (L % 2 == 1) ? destA : destB
            outputCount = B * M * N
            dispatchBody = { cb in
                var cur = lhs
                let pingPong = [destA, destB]
                for layer in 0..<L {
                    let dst = pingPong[layer % 2]
                    mm.encode(to: cb, sourceArrays: [cur, rhs], destinationArray: dst)
                    cur = dst
                }
            }
        } else if mpsType == .int8 {
            guard #available(iOS 18.0, macOS 15.0, *) else {
                throw BenchmarkError.executionFailed("MPSNDArrayQuantizedMatrixMultiplication requires iOS 18.0 / macOS 15.0 or newer.")
            }
            let lhs = ndArray(.int8, [B, 1, M, K])
            let rhs = ndArray(.int8, [B, 1, K, N])
            fillND(lhs, byteCount: B * M * K, dataType: .int8)
            fillND(rhs, byteCount: B * K * N, dataType: .int8)

            let lhsScale = ndArray(.float16, [1, 1, 1, 1])
            let lhsZero = ndArray(.int8, [1, 1, 1, 1])
            let rhsScale = ndArray(.float16, [1, 1, 1, 1])
            let rhsZero = ndArray(.int8, [1, 1, 1, 1])
            var scaleBits = Float16(1.0 / 32.0).bitPattern
            withUnsafeBytes(of: &scaleBits) { raw in
                lhsScale.writeBytes(UnsafeMutableRawPointer(mutating: raw.baseAddress!), strideBytes: nil)
                rhsScale.writeBytes(UnsafeMutableRawPointer(mutating: raw.baseAddress!), strideBytes: nil)
            }
            var zeroByte: Int8 = 0
            withUnsafeBytes(of: &zeroByte) { raw in
                lhsZero.writeBytes(UnsafeMutableRawPointer(mutating: raw.baseAddress!), strideBytes: nil)
                rhsZero.writeBytes(UnsafeMutableRawPointer(mutating: raw.baseAddress!), strideBytes: nil)
            }

            let leftQuantDesc = MPSNDArrayAffineQuantizationDescriptor(dataType: .int8, hasZeroPoint: true, hasMinValue: false)
            let rightQuantDesc = MPSNDArrayAffineQuantizationDescriptor(dataType: .int8, hasZeroPoint: true, hasMinValue: false)
            let qmm = MPSNDArrayQuantizedMatrixMultiplication(
                device: device,
                leftQuantizationDescriptor: leftQuantDesc,
                rightQuantizationDescriptor: rightQuantDesc
            )
            let dest = ndArray(.float16, [B, 1, M, N])
            let sourceArrays: [MPSNDArray] = [lhs, rhs, lhsScale, lhsZero, rhsScale, rhsZero]
            finalDest = dest
            outputCount = B * M * N
            effectiveLayers = 1
            dispatchBody = { cb in
                qmm.encode(to: cb, sourceArrays: sourceArrays, destinationArray: dest)
            }
            if L != 1 {
                logHandler("[GPU \(precision.rawValue)] Note: native INT8-affine kernel cannot chain layers (dequantizes to float16 on every call) -- running L=1, not the configured L=\(L).")
            }
        } else {
            let weightMag = Float(0.96 / sqrt(Double(max(K, 1))))
            let lhs = ndArray(mpsType, [B, 1, M, K])
            let rhs = ndArray(mpsType, [B, 1, K, N])
            fillND(lhs, byteCount: B * M * K * elementSize, dataType: mpsType)
            fillND(rhs, byteCount: B * K * N * elementSize, dataType: mpsType, magnitude: weightMag)
            let destA = ndArray(mpsType, [B, 1, M, N])
            let destB = ndArray(mpsType, [B, 1, M, N])
            let mm = MPSNDArrayMatrixMultiplication(device: device, sourceCount: 2)
            finalDest = (L % 2 == 1) ? destA : destB
            outputCount = B * M * N
            dispatchBody = { cb in
                var cur = lhs
                let pingPong = [destA, destB]
                for layer in 0..<L {
                    let dst = pingPong[layer % 2]
                    mm.encode(to: cb, sourceArrays: [cur, rhs], destinationArray: dst)
                    cur = dst
                }
            }
        }

        logHandler("[GPU \(precision.rawValue)] Warming up pipeline...")
        guard let warmupCb = queue.makeCommandBuffer() else {
            throw BenchmarkError.commandQueueCreationFailed
        }
        dispatchBody(warmupCb)
        warmupCb.commit()
        warmupCb.waitUntilCompleted()

        if Task.isCancelled { throw BenchmarkError.executionCancelled }

        logHandler("[GPU \(precision.rawValue)] Benchmarking \(iterations) iterations...")
        let startNanos = clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW)
        var lastCb: MTLCommandBuffer? = nil
        for i in 0..<iterations {
            if Task.isCancelled { throw BenchmarkError.executionCancelled }
            guard let cb = queue.makeCommandBuffer() else {
                throw BenchmarkError.commandQueueCreationFailed
            }
            dispatchBody(cb)
            cb.commit()
            lastCb = cb
            if (i + 1) % 5 == 0 { await Task.yield() }
        }
        lastCb?.waitUntilCompleted()
        let endNanos = clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW)

        let durationSec = Double(endNanos - startNanos) / 1e9
        let avgSec = durationSec / Double(iterations)
        let avgMs = avgSec * 1000.0

        let opsPerIter = 2.0 * Double(B) * Double(M) * Double(K) * Double(N) * Double(effectiveLayers)
        let tops = opsPerIter / (avgSec * 1e12)

        // INT8-affine's destination is always float16 (it dequantizes on write,
        // 2 bytes/element), never the nominal 1-byte int8 `elementSize` --
        // sizing the readback buffer off `elementSize` here previously
        // under-allocated by half for this path and overflowed on readBytes.
        let readbackType: MPSDataType = (mpsType == .int8) ? .float16 : mpsType
        let readbackElementSize = (mpsType == .int8) ? 2 : elementSize
        let outputByteCount = outputCount * readbackElementSize
        let outBuf = UnsafeMutableRawPointer.allocate(byteCount: max(outputByteCount, 1), alignment: 16)
        defer { outBuf.deallocate() }
        finalDest.readBytes(outBuf, strideBytes: nil)
        let zeroCount = countZeroElements(buffer: outBuf, elementCount: outputCount, dataType: readbackType)

        if zeroCount == outputCount {
            logHandler("[GPU \(precision.rawValue)] WARNING: 100% zero output (\(outputCount)/\(outputCount) elements) -- this is not a real measurement.")
        }

        let resultMsg = String(
            format: "[GPU %@] (native) Avg: %.2f ms, Throughput: %.4f TOPS (L=%d), Zeros: %ld/%ld (%.1f%%)",
            precision.rawValue, avgMs, tops, effectiveLayers,
            zeroCount, outputCount,
            outputCount > 0 ? Double(zeroCount) * 100.0 / Double(outputCount) : 0.0
        )
        logHandler(resultMsg)

        return BenchmarkResult(
            dimensions: dims,
            precision: precision,
            target: .gpu,
            avgDurationMs: avgMs,
            tops: tops,
            iterations: iterations,
            sweepType: sweepType,
            sweepValue: sweepValue,
            sweepLabel: sweepLabel,
            zeroOutputCount: zeroCount,
            outputElementCount: outputCount
        )
    }

    static func formatCompact(_ val: UInt64) -> String {
        if val >= 1_000_000 {
            return String(format: "%.2fM", Double(val) / 1_000_000.0)
        } else if val >= 1_000 {
            return String(format: "%.1fK", Double(val) / 1_000.0)
        }
        return "\(val)"
    }
    
    static func formatBytes(_ bytes: UInt64) -> String {
        if bytes >= 1024 * 1024 {
            return String(format: "%.1f MB", Double(bytes) / (1024.0 * 1024.0))
        } else if bytes >= 1024 {
            return String(format: "%.1f KB", Double(bytes) / 1024.0)
        }
        return "\(bytes) B"
    }
}
