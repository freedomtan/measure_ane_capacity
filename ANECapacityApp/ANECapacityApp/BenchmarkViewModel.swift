import Foundation
import SwiftUI
import Combine

@MainActor
final class BenchmarkViewModel: ObservableObject {
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
    }
    
    // Peak metrics
    var peakFP16TOPS: Double {
        results.filter { $0.precision == .fp16 && $0.target == .ane }.map(\.tops).max() ?? 0.0
    }
    
    var peakINT8TOPS: Double {
        results.filter { $0.precision == .int8 && $0.target == .ane }.map(\.tops).max() ?? 0.0
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
    
    // MARK: - Single Run Flow
    private func runSingleBenchmarkFlow() async {
        let precisionsToRun: [PrecisionMode] = (selectedPrecision == .both) ? [.fp16, .int8] : [selectedPrecision]
        let totalSteps = Double(precisionsToRun.count)
        var currentStep = 0.0
        
        log("=== Starting Single Benchmark: \(dimensions.detailedDescription) ===")
        
        for prec in precisionsToRun {
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
        let precisionsToRun: [PrecisionMode] = (selectedPrecision == .both) ? [.fp16, .int8] : [selectedPrecision]
        
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
                var d = ConvDimensions(batch: 1, height: 256, width: 256, inChannels: ch, outChannels: ch, kernelSize: 3, layers: 20)
                points.append(SweepPoint(dims: d, value: Double(ch), label: "\(ch)c"))
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
            "Timestamp", "Device", "Precision", "Batch", "Height", "Width",
            "InChannels", "OutChannels", "KernelSize", "Layers", "TotalGFLOPs",
            "AvgDurationMs", "TOPS", "HWExecutionTimeNs", "MACsPerCoreCycle",
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
                "\(r.dimensions.batch)",
                "\(r.dimensions.height)",
                "\(r.dimensions.width)",
                "\(r.dimensions.inChannels)",
                "\(r.dimensions.outChannels)",
                "\(r.dimensions.kernelSize)",
                "\(r.dimensions.layers)",
                String(format: "%.2f", r.dimensions.gflops),
                String(format: "%.3f", r.avgDurationMs),
                String(format: "%.4f", r.tops),
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
