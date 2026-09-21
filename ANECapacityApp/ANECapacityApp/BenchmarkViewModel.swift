import Foundation
import SwiftUI
import Combine

@MainActor
final class BenchmarkViewModel: ObservableObject {
    // Current operation type
    @Published var selectedOperation: OperationType = .conv2d {
        didSet {
            dimensions.opType = selectedOperation
            if selectedOperation == .matmul {
                selectedSweep = .matmulDimensions
            } else {
                selectedSweep = .channels
            }
        }
    }
    
    // Current custom dimensions
    @Published var dimensions: ConvDimensions = ConvDimensions()
    @Published var selectedPrecision: PrecisionMode = .both
    @Published var selectedTarget: DeviceTarget = .ane
    @Published var iterations: Int = 20
    
    // Sweep options
    @Published var selectedSweep: SweepType = .channels
    
    // Custom sweep ranges
    @Published var customChannelSteps: [Int] = [32, 64, 128, 256, 512, 1024]
    @Published var customSpatialSteps: [Int] = [64, 128, 256, 384, 512, 768]
    @Published var customDepthSteps: [Int] = [1, 5, 10, 20, 30, 40]
    @Published var customMatMulSteps: [Int] = [128, 256, 512, 1024, 2048]
    
    // Benchmark execution state
    @Published var isRunning: Bool = false
    @Published var progress: Double = 0.0
    @Published var statusMessage: String = "Ready to benchmark."
    @Published var consoleLogs: [String] = []
    
    // Results history
    @Published var results: [BenchmarkResult] = []
    
    // Currently active task for cancellation
    private var benchmarkTask: Task<Void, Never>? = nil
    
    // Hardware info
    var metalDeviceName: String {
        ANECapacityEngine.shared.metalDeviceName
    }
    
    var hasANE: Bool {
        ANECapacityEngine.shared.hasANE
    }
    
    init() {
        if !ANECapacityEngine.shared.hasANE {
            self.selectedTarget = .gpu
        }
        #if targetEnvironment(simulator)
        loadSampleResults()
        #endif
    }
    
    #if targetEnvironment(simulator)
    private func loadSampleResults() {
        let channelSteps = [32, 64, 128, 256, 512, 1024]
        let fp16Tops = [3.12, 6.84, 11.45, 14.82, 15.61, 15.84]
        let int8Tops = [6.24, 13.52, 22.81, 29.64, 31.22, 31.85]
        
        for (i, c) in channelSteps.enumerated() {
            let dims = ConvDimensions(batch: 1, height: 256, width: 256, inChannels: c, outChannels: c, kernelSize: 1, layers: 20)
            let ops = dims.totalOperations
            
            let fpTops = fp16Tops[i]
            let fpDurSec = ops / (fpTops * 1e12)
            results.append(BenchmarkResult(
                dimensions: dims,
                precision: .fp16,
                target: .ane,
                avgDurationMs: fpDurSec * 1000.0,
                tops: fpTops,
                iterations: 20,
                sweepType: .channels,
                sweepValue: Double(c),
                sweepLabel: "\(c)c",
                computeCycles: UInt64(Double(c) * 4200 + 120_000),
                nominalCycles: UInt64(Double(c) * 4500 + 130_000),
                outputStallCycles: UInt64(15_000 + i * 8_000),
                inputStallCycles: UInt64(10_000 + i * 5_000),
                dmaRwBytes: UInt64(dims.weightsBytes(precision: .fp16) + dims.inputBytes(precision: .fp16)),
                aluSaturation: min(94.5, 30.0 + Double(i) * 12.5),
                effectiveClockGhz: 1.80
            ))
            
            let inTops = int8Tops[i]
            let inDurSec = ops / (inTops * 1e12)
            results.append(BenchmarkResult(
                dimensions: dims,
                precision: .int8,
                target: .ane,
                avgDurationMs: inDurSec * 1000.0,
                tops: inTops,
                iterations: 20,
                sweepType: .channels,
                sweepValue: Double(c),
                sweepLabel: "\(c)c",
                computeCycles: UInt64(Double(c) * 3800 + 100_000),
                nominalCycles: UInt64(Double(c) * 4100 + 110_000),
                outputStallCycles: UInt64(12_000 + i * 6_000),
                inputStallCycles: UInt64(8_000 + i * 4_000),
                dmaRwBytes: UInt64(dims.weightsBytes(precision: .int8) + dims.inputBytes(precision: .int8)),
                aluSaturation: min(95.2, 32.0 + Double(i) * 12.2),
                effectiveClockGhz: 1.82
            ))
        }

        let matmulSteps = [128, 256, 512, 1024, 2048]
        let matmulFp16Tops = [2.85, 6.40, 12.10, 15.30, 15.90]
        let matmulInt8Tops = [5.60, 12.80, 24.10, 30.50, 31.80]
        
        for (i, sz) in matmulSteps.enumerated() {
            let dims = ConvDimensions(opType: .matmul, batch: 1, layers: 20, m: sz, k: sz, n: sz)
            let ops = dims.totalOperations
            
            let fpTops = matmulFp16Tops[i]
            let fpDurSec = ops / (fpTops * 1e12)
            results.append(BenchmarkResult(
                dimensions: dims,
                precision: .fp16,
                target: .ane,
                avgDurationMs: fpDurSec * 1000.0,
                tops: fpTops,
                iterations: 20,
                sweepType: .matmulDimensions,
                sweepValue: Double(sz),
                sweepLabel: "\(sz)",
                computeCycles: UInt64(Double(sz) * 3500 + 80_000),
                nominalCycles: UInt64(Double(sz) * 3800 + 90_000),
                outputStallCycles: UInt64(10_000 + i * 5_000),
                inputStallCycles: UInt64(8_000 + i * 4_000),
                dmaRwBytes: UInt64(dims.weightsBytes(precision: .fp16) + dims.inputBytes(precision: .fp16)),
                aluSaturation: min(95.0, 28.0 + Double(i) * 13.0),
                effectiveClockGhz: 1.80
            ))
            
            let inTops = matmulInt8Tops[i]
            let inDurSec = ops / (inTops * 1e12)
            results.append(BenchmarkResult(
                dimensions: dims,
                precision: .int8,
                target: .ane,
                avgDurationMs: inDurSec * 1000.0,
                tops: inTops,
                iterations: 20,
                sweepType: .matmulDimensions,
                sweepValue: Double(sz),
                sweepLabel: "\(sz)",
                computeCycles: UInt64(Double(sz) * 3200 + 70_000),
                nominalCycles: UInt64(Double(sz) * 3500 + 80_000),
                outputStallCycles: UInt64(8_000 + i * 4_000),
                inputStallCycles: UInt64(6_000 + i * 3_000),
                dmaRwBytes: UInt64(dims.weightsBytes(precision: .int8) + dims.inputBytes(precision: .int8)),
                aluSaturation: min(96.0, 30.0 + Double(i) * 13.0),
                effectiveClockGhz: 1.82
            ))
        }
    }
    #endif
    
    // Peak metrics
    var peakFP16TOPS: Double {
        results.filter { $0.precision == .fp16 && $0.target == .ane }.map(\.tops).max() ?? 0.0
    }
    
    var peakINT8TOPS: Double {
        results.filter { $0.precision == .int8 && $0.target == .ane }.map(\.tops).max() ?? 0.0
    }
    
    var peakFP8TOPS: Double {
        results.filter { $0.precision == .fp8 }.map(\.tops).max() ?? 0.0
    }
    
    var speedupRatio: Double? {
        guard peakFP16TOPS > 0, peakINT8TOPS > 0 else { return nil }
        return peakINT8TOPS / peakFP16TOPS
    }
    
    // Log appender
    func log(_ msg: String) {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm:ss"
        let timestamp = formatter.string(from: Date())
        let formatted = "[\(timestamp)] \(msg)"
        consoleLogs.append(formatted)
        if consoleLogs.count > 500 {
            consoleLogs.removeFirst(100)
        }
    }
    
    func clearLogs() {
        consoleLogs.removeAll()
    }
    
    func clearResults() {
        results.removeAll()
        log("Results cleared.")
    }
    
    // Start benchmark execution
    func startBenchmark() {
        guard !isRunning else { return }
        isRunning = true
        progress = 0.0
        
        benchmarkTask = Task {
            if selectedSweep == .none {
                await runSingleBenchmarkFlow()
            } else {
                await runSweepBenchmarkFlow()
            }
            
            self.isRunning = false
            self.statusMessage = "Benchmark completed."
            self.log("=== Benchmark Run Finished ===")
        }
    }
    
    func cancelBenchmark() {
        benchmarkTask?.cancel()
        benchmarkTask = nil
        isRunning = false
        statusMessage = "Benchmark cancelled."
        log("⚠️ Execution cancelled by user.")
    }
    
    private var precisionsToRun: [PrecisionMode] {
        switch selectedPrecision {
        case .both:
            return [.fp16, .int8]
        case .all:
            return [.fp16, .int8, .fp8]
        case .fp16, .int8, .fp8:
            return [selectedPrecision]
        }
    }
    
    // MARK: - Single Run Flow
    private func runSingleBenchmarkFlow() async {
        let precisions = precisionsToRun
        let totalSteps = Double(precisions.count)
        var currentStep = 0.0
        
        log("=== Starting Single Benchmark: \(dimensions.detailedDescription) ===")
        
        for prec in precisions {
            if Task.isCancelled { break }
            statusMessage = "Running \(selectedTarget.shortName) \(prec.rawValue)..."
            
            do {
                let res = try await ANECapacityEngine.shared.runSingle(
                    dimensions: dimensions,
                    precision: prec,
                    target: selectedTarget,
                    iterations: iterations,
                    sweepType: .none,
                    sweepValue: Double(dimensions.inChannels),
                    sweepLabel: "Channels",
                    logHandler: { [weak self] msg in
                        Task { @MainActor in self?.log(msg) }
                    }
                )
                results.append(res)
            } catch {
                log("❌ Error: \(error.localizedDescription)")
            }
            
            currentStep += 1.0
            progress = currentStep / totalSteps
        }
    }
    
    // MARK: - Sweep Benchmark Flow
    private func runSweepBenchmarkFlow() async {
        let precisions = precisionsToRun
        
        // Define sweep points
        struct SweepPoint {
            let dims: ConvDimensions
            let value: Double
            let label: String
        }
        
        var points: [SweepPoint] = []
        
        switch selectedSweep {
        case .none:
            break
        case .channels:
            for ch in customChannelSteps {
                var d = dimensions
                d.inChannels = ch
                d.outChannels = ch
                points.append(SweepPoint(dims: d, value: Double(ch), label: "\(ch)"))
            }
        case .spatial:
            for sz in customSpatialSteps {
                var d = dimensions
                d.height = sz
                d.width = sz
                points.append(SweepPoint(dims: d, value: Double(sz), label: "\(sz)x\(sz)"))
            }
        case .depth:
            for layers in customDepthSteps {
                var d = dimensions
                d.layers = layers
                points.append(SweepPoint(dims: d, value: Double(layers), label: "L=\(layers)"))
            }
        case .kernels:
            for k in [1, 3, 5] {
                var d = dimensions
                d.kernelSize = k
                points.append(SweepPoint(dims: d, value: Double(k), label: "\(k)x\(k)"))
            }
        case .fullCapacity:
            // Comprehensive sweep across channels with fixed H=256, W=256, K=3, L=20
            for ch in [32, 64, 128, 256, 512, 1024] {
                var d = ConvDimensions(opType: .conv2d, batch: 1, height: 256, width: 256, inChannels: ch, outChannels: ch, kernelSize: 3, layers: 20)
                points.append(SweepPoint(dims: d, value: Double(ch), label: "\(ch)c"))
            }
        case .matmulDimensions:
            for sz in customMatMulSteps {
                var d = dimensions
                d.opType = .matmul
                d.m = sz
                d.k = sz
                d.n = sz
                points.append(SweepPoint(dims: d, value: Double(sz), label: "\(sz)"))
            }
        case .matmulDepth:
            for layers in customDepthSteps {
                var d = dimensions
                d.opType = .matmul
                d.layers = layers
                points.append(SweepPoint(dims: d, value: Double(layers), label: "L=\(layers)"))
            }
        }
        
        let totalSteps = Double(points.count * precisionsToRun.count)
        var currentStep = 0.0
        
        log("=== Starting Sweep: \(selectedSweep.rawValue) (\(points.count) steps x \(precisionsToRun.count) types) ===")
        
        for pt in points {
            for prec in precisionsToRun {
                if Task.isCancelled { break }
                
                statusMessage = "[\(currentStep + 1)/\(Int(totalSteps))] \(selectedTarget.shortName) \(prec.rawValue) | \(pt.label)..."
                
                do {
                    let res = try await ANECapacityEngine.shared.runSingle(
                        dimensions: pt.dims,
                        precision: prec,
                        target: selectedTarget,
                        iterations: iterations,
                        sweepType: selectedSweep,
                        sweepValue: pt.value,
                        sweepLabel: pt.label,
                        logHandler: { [weak self] msg in
                            Task { @MainActor in self?.log(msg) }
                        }
                    )
                    results.append(res)
                } catch {
                    log("❌ Error at \(pt.label): \(error.localizedDescription)")
                }
                
                currentStep += 1.0
                progress = currentStep / totalSteps
                
                // Small pause between configurations to let hardware thermals / driver settle
                try? await Task.sleep(nanoseconds: 100_000_000) // 100ms
            }
        }
    }
    
    // Generate CSV string of results
    func exportCSV() -> String {
        var headers = [
            "Timestamp", "Device", "Precision", "Operation", "Batch", "Height", "Width",
            "InChannels", "OutChannels", "KernelSize", "M", "K", "N", "Layers", "TotalGFLOPs",
            "AvgDurationMs", "TOPS", "ZeroOutputCount", "OutputElementCount", "ZeroOutputPct",
            "HWExecutionTimeNs", "MACsPerCoreCycle",
            "TotalChipMACs", "ALUSaturationPct", "EffectiveClockGHz",
            "SweepType", "SweepValue", "SweepLabel", "Iterations",
            "NominalCycles", "ComputeCycles", "OutputStallCycles",
            "InputStallCycles", "DmaRwBytes", "DpeEnergy"
        ]
        
        let pmuKeys = BenchmarkResult.allPMUCounterKeys
        headers.append(contentsOf: pmuKeys)
        
        var csv = headers.joined(separator: ",") + "\n"
        let df = ISO8601DateFormatter()
        
        for r in results {
            var row: [String] = [
                df.string(from: r.timestamp),
                r.target.shortName,
                r.precision.rawValue,
                r.dimensions.opType.rawValue,
                "\(r.dimensions.batch)",
                "\(r.dimensions.height)",
                "\(r.dimensions.width)",
                "\(r.dimensions.inChannels)",
                "\(r.dimensions.outChannels)",
                "\(r.dimensions.kernelSize)",
                "\(r.dimensions.m)",
                "\(r.dimensions.k)",
                "\(r.dimensions.n)",
                "\(r.dimensions.layers)",
                String(format: "%.2f", r.dimensions.gflops),
                String(format: "%.3f", r.avgDurationMs),
                String(format: "%.4f", r.tops),
                "\(r.zeroOutputCount)",
                "\(r.outputElementCount)",
                String(format: "%.2f", r.zeroOutputPct),
                "\(r.hwExecutionTimeNs)",
                String(format: "%.2f", r.macsPerCoreCycle),
                String(format: "%.2f", r.chipMacsPerCycle),
                String(format: "%.2f", r.aluSaturation),
                String(format: "%.3f", r.effectiveClockGhz),
                r.sweepType.rawValue,
                "\(r.sweepValue)",
                "\"\(r.sweepLabel)\"",
                "\(r.iterations)",
                "\(r.nominalCycles)",
                "\(r.computeCycles)",
                "\(r.outputStallCycles)",
                "\(r.inputStallCycles)",
                "\(r.dmaRwBytes)",
                "\(r.dpeEnergy)"
            ]
            
            for key in pmuKeys {
                row.append("\(r.counterValue(for: key))")
            }
            
            csv += row.joined(separator: ",") + "\n"
        }
        return csv
    }
}
