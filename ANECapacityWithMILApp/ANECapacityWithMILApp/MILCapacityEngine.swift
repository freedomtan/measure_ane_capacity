import Foundation
import CoreML

final class MILCapacityEngine {
    static let shared = MILCapacityEngine()
    
    var hasANE: Bool {
        #if targetEnvironment(simulator)
        return false
        #else
        return true
        #endif
    }
    
    var deviceName: String {
        #if targetEnvironment(simulator)
        return "iOS Simulator (CoreML Emulation)"
        #else
        return "Apple Neural Engine / CoreML Runtime"
        #endif
    }
    
    /**
     * Runs a benchmark configuration dynamically using CoreML and MIL.
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
        if Task.isCancelled {
            throw NSError(domain: "MILCapacityEngine", code: -1, userInfo: [NSLocalizedDescriptionKey: "Execution cancelled by user."])
        }
        
        logHandler("[\(target.shortName) \(precision.rawValue)] Setting up CoreML model: \(dims.shortDescription)...")
        
        // Check for FP8 support boundary
        let isFP8 = precision.isFP8
        let precStr = isFP8 ? "fp8" : (precision == .int8 ? "int8" : "fp16")
        
        // 1. Look for pre-compiled or cached model at app bundle or documents
        let batch = UInt(dims.batch)
        let chIn = UInt(dims.inChannels)
        let chOut = UInt(dims.outChannels)
        let height = UInt(dims.height)
        let width = UInt(dims.width)
        let kernel = UInt(dims.kernelSize)
        let layers = UInt(dims.layers)
        
        // Call Objective-C Bridge to build, compile, and evaluate dynamically on-device
        let res = await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                let bridgeResult = MILCapacityEngineBridge.evaluateDynamicModel(
                    withBatch: batch,
                    channels: chIn,
                    height: height,
                    width: width,
                    kernel: kernel,
                    layers: layers,
                    precision: precStr,
                    computeUnits: target.mlComputeUnits,
                    iterations: UInt(iterations),
                    warmup: 3,
                    usePMU: false,
                    progressHandler: { msg in
                        logHandler(msg)
                    }
                )
                continuation.resume(returning: bridgeResult)
            }
        }
        
        guard res.success else {
            throw NSError(domain: "MILCapacityEngine", code: -2, userInfo: [NSLocalizedDescriptionKey: res.statusMessage ?? "Benchmark failed."])
        }
        
        var pmuCounters: [String: UInt64] = [:]
        pmuCounters["kANE_NE_COMPUTE_CYCLES"] = res.computeCycles
        pmuCounters["kANE_NE_NOMINAL_CYCLES"] = res.nominalCycles
        pmuCounters["kANE_NE_OUTPUT_STALL_CYCLES"] = res.outputStallCycles
        pmuCounters["kANE_NE_INPUT_STALL_CYCLES"] = res.inputStallCycles
        pmuCounters["kANE_DMA_READWRITE_BYTES"] = res.dmaBytes
        
        let resultMsg = String(
            format: "[%@ %@] Avg: %.2f ms, Throughput: %.4f TOPS (%.1f GFLOPs/iter)",
            target.shortName, precision.rawValue, res.avgLatencyMs, res.tops, dims.gflops
        )
        logHandler(resultMsg)
        
        return BenchmarkResult(
            dimensions: dims,
            precision: precision,
            target: target,
            avgDurationMs: res.avgLatencyMs,
            tops: res.tops,
            iterations: iterations,
            sweepType: sweepType,
            sweepValue: sweepValue,
            sweepLabel: sweepLabel,
            pmuCounters: pmuCounters,
            computeCycles: res.computeCycles,
            nominalCycles: res.nominalCycles,
            outputStallCycles: res.outputStallCycles,
            inputStallCycles: res.inputStallCycles,
            dmaRwBytes: res.dmaBytes,
            dpeEnergy: 0,
            hwExecutionTimeNs: UInt64(res.avgLatencyMs * 1_000_000),
            macsPerCoreCycle: 0.0,
            chipMacsPerCycle: 0.0,
            aluSaturation: res.aluSaturation,
            effectiveClockGhz: res.effectiveClockGhz,
            zeroOutputCount: Int(res.zeroElementCount),
            outputElementCount: Int(res.totalElementCount)
        )
    }
}
