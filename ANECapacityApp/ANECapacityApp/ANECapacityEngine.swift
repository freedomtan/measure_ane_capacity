import Foundation
import Metal
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
                for _ in 0..<dims.layers {
                    let lhs = graph.cast(cur, to: .float16, name: "dequant")
                    let outFP16 = graph.convolution2D(lhs, weights: w, descriptor: d, name: nil)
                    cur = graph.cast(outFP16, to: .int8, name: "requant")
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
                let wLength = dims.outChannels * dims.inChannels * dims.kernelSize * dims.kernelSize * 2
                let wData = createNonZeroData(byteCount: wLength, dataType: .float16, magnitude: weightMag)
                let w = graph.constant(wData, shape: wShape, dataType: .float16)

                for _ in 0..<dims.layers {
                    let lhs = graph.cast(cur, to: .float16, name: "dequant")
                    let outFP16 = graph.convolution2D(lhs, weights: w, descriptor: d, name: nil)
                    cur = graph.cast(outFP16, to: .int8, name: "requant")
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
            beforeDirs = (ANEClientBridge.existingANETempDirectories() as? Set<String>) ?? []
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

        // 5. Warmup with Exception Protection
        logHandler("[\(target.shortName) \(precision.rawValue)] Warming up pipeline...")
        do {
            try ANEClientBridge.catchException {
                exe.run(with: queue, inputs: [iData], results: [oData], executionDescriptor: ed)
            }
        } catch {
            throw BenchmarkError.executionFailed("Warmup failed: \(error.localizedDescription)")
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
        var tops = dims.totalOperations / (avgSec * 1e12)

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
                        formatCompact(computeCycles),
                        formatCompact(outputStallCycles + inputStallCycles),
                        formatBytes(dmaRwBytes),
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
            format: "[%@ %@] Avg: %.2f ms, Throughput: %.4f TOPS (%.1f GFLOPs/iter)",
            target.shortName, precision.rawValue, avgMs, tops, dims.gflops
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
    
    private func formatCompact(_ val: UInt64) -> String {
        if val >= 1_000_000 {
            return String(format: "%.2fM", Double(val) / 1_000_000.0)
        } else if val >= 1_000 {
            return String(format: "%.1fK", Double(val) / 1_000.0)
        }
        return "\(val)"
    }
    
    private func formatBytes(_ bytes: UInt64) -> String {
        if bytes >= 1024 * 1024 {
            return String(format: "%.1f MB", Double(bytes) / (1024.0 * 1024.0))
        } else if bytes >= 1024 {
            return String(format: "%.1f KB", Double(bytes) / 1024.0)
        }
        return "\(bytes) B"
    }
}
