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
private func fillNonZeroData(buffer: UnsafeMutableRawPointer, byteCount: Int, dataType: MPSDataType) {
    guard byteCount > 0 else { return }
    if dataType == .float16 {
        let ptr = buffer.bindMemory(to: UInt16.self, capacity: byteCount / 2)
        let count = byteCount / 2
        let patterns: [UInt16] = [0x2C00, 0xAC00, 0x2800, 0xA800]
        for i in 0..<count {
            ptr[i] = patterns[i & 3]
        }
    } else {
        let ptr = buffer.bindMemory(to: Int8.self, capacity: byteCount)
        let patterns: [Int8] = [1, -1, 2, -2]
        for i in 0..<byteCount {
            ptr[i] = patterns[i & 3]
        }
    }
}

private func createNonZeroData(byteCount: Int, dataType: MPSDataType) -> Data {
    var data = Data(count: byteCount)
    data.withUnsafeMutableBytes { rawBuf in
        if let baseAddress = rawBuf.baseAddress {
            fillNonZeroData(buffer: baseAddress, byteCount: byteCount, dataType: dataType)
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
        
        let inShape: [NSNumber] = [
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
        
        let input = graph.placeholder(shape: inShape, dataType: mpsType, name: "in")
        var cur = input
        
        // Constant weights
        let wLength = dims.outChannels * dims.inChannels * dims.kernelSize * dims.kernelSize * elementSize
        let wData = createNonZeroData(byteCount: wLength, dataType: mpsType)
        let w = graph.constant(wData, shape: wShape, dataType: mpsType)
        
        // Chain L layers
        for _ in 0..<dims.layers {
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
            
            cur = graph.convolution2D(cur, weights: w, descriptor: d, name: nil)
            
            // Int8 simulated quantized flow: Int8 -> Conv -> FP16 dequant -> Int8 requant
            if mpsType == .int8 {
                let fp = graph.cast(cur, to: .float16, name: "dequant")
                cur = graph.cast(fp, to: .int8, name: "requant")
            }
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
        let inputBufferLen = dims.batch * dims.height * dims.width * dims.inChannels * elementSize
        guard let iBuf = device.makeBuffer(length: inputBufferLen, options: []) else {
            throw BenchmarkError.bufferAllocationFailed
        }
        fillNonZeroData(buffer: iBuf.contents(), byteCount: inputBufferLen, dataType: mpsType)
        let iData = MPSGraphTensorData(iBuf, shape: inShape, dataType: mpsType)
        
        guard let queue = device.makeCommandQueue() else {
            throw BenchmarkError.commandQueueCreationFailed
        }
        
        let ed = MPSGraphExecutableExecutionDescriptor()
        ed.waitUntilCompleted = true
        
        // 5. Warmup with Exception Protection
        logHandler("[\(target.shortName) \(precision.rawValue)] Warming up pipeline...")
        do {
            try ANEClientBridge.catchException {
                exe.run(with: queue, inputs: [iData], results: nil, executionDescriptor: ed)
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
                    exe.run(with: queue, inputs: [iData], results: nil, executionDescriptor: ed)
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
                
                let pmuRes = ANEClientBridge.profileANECIRBundle(
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
                    if pmuRes.topsRealized > 0 {
                        tops = pmuRes.topsRealized
                    }
                    
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
            effectiveClockGhz: effectiveClockGhz
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
