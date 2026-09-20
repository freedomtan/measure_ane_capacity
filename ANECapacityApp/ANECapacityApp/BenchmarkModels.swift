import SwiftUI
import MetalPerformanceShadersGraph

// MARK: - Precision Mode
enum PrecisionMode: String, CaseIterable, Identifiable, Codable {
    case fp16 = "FP16"
    case int8 = "INT8"
    case fp8 = "FP8 (E4M3)"
    case both = "Both (FP16 & INT8)"
    case all = "All Precisions"
    
    var id: String { rawValue }
    
    var elementSize: Int {
        switch self {
        case .fp16: return 2
        case .int8, .fp8: return 1
        case .both, .all: return 2
        }
    }
    
    var isFP8: Bool {
        return self == .fp8
    }
    
    var mpsDataType: MPSDataType {
        switch self {
        case .fp16, .both, .all: return .float16
        case .int8: return .int8
        case .fp8:
            if #available(iOS 27.0, macOS 27.0, *) {
                return .float8e4m3
            } else {
                return MPSDataType(rawValue: 0x10430008) ?? .float16
            }
        }
    }
    
    var themeColor: Color {
        switch self {
        case .fp16: return .blue
        case .int8: return .orange
        case .fp8: return .mint
        case .both, .all: return .purple
        }
    }
}

// MARK: - Target Device
enum DeviceTarget: String, CaseIterable, Identifiable, Codable {
    case ane = "ANE (Neural Engine)"
    case gpu = "GPU (Metal)"
    
    var id: String { rawValue }
    
    var shortName: String {
        switch self {
        case .ane: return "ANE"
        case .gpu: return "GPU"
        }
    }
}

// MARK: - Operation Type
enum OperationType: String, CaseIterable, Identifiable, Codable {
    case conv2d = "Conv2D"
    case matmul = "MatMul (GEMM)"
    
    var id: String { rawValue }
    
    var icon: String {
        switch self {
        case .conv2d: return "square.grid.3x3.fill"
        case .matmul: return "rectangle.split.2x2.fill"
        }
    }
}

// MARK: - Sweep Type
enum SweepType: String, CaseIterable, Identifiable, Codable {
    case none = "Single Run"
    case channels = "Channel Capacity Sweep"
    case spatial = "Spatial Dimension (H=W) Sweep"
    case depth = "Chained Layer Depth Sweep"
    case kernels = "Kernel Size Sweep (1x1 vs 3x3)"
    case fullCapacity = "Full Capacity Comparison"
    case matmulDimensions = "Matrix Dimension (M=K=N) Sweep"
    case matmulDepth = "GEMM Chained Depth Sweep"
    
    var id: String { rawValue }
    
    var axisLabel: String {
        switch self {
        case .none: return "Run"
        case .channels: return "Channels (Ci = Co)"
        case .spatial: return "Spatial Resolution (H = W)"
        case .depth: return "Chained Layers (L)"
        case .kernels: return "Kernel Size (KxK)"
        case .fullCapacity: return "Channels (Ci = Co)"
        case .matmulDimensions: return "Matrix Size (M=K=N)"
        case .matmulDepth: return "Chained Layers (L)"
        }
    }
}

// MARK: - Convolution & GEMM Dimensions / Hyperparameters
struct ConvDimensions: Codable, Equatable {
    var opType: OperationType = .conv2d
    var batch: Int = 1
    
    // Conv2D dimensions
    var height: Int = 256
    var width: Int = 256
    var inChannels: Int = 128
    var outChannels: Int = 128
    var kernelSize: Int = 3
    var layers: Int = 20
    
    // MatMul (GEMM) dimensions: [B, M, K] x [B, K, N]
    var m: Int = 1024
    var k: Int = 1024
    var n: Int = 1024
    
    // Theoretical operations per single iteration
    var totalOperations: Double {
        if opType == .matmul {
            return 2.0 * Double(batch) * Double(m) * Double(k) * Double(n) * Double(layers)
        } else {
            return 2.0 * Double(batch) * Double(height) * Double(width) *
                   Double(inChannels) * Double(outChannels) *
                   Double(kernelSize * kernelSize) * Double(layers)
        }
    }
    
    var gflops: Double {
        return totalOperations / 1e9
    }
    
    func weightsBytes(precision: PrecisionMode) -> Int {
        if opType == .matmul {
            if precision.isFP8 {
                return batch * k * n * 1
            }
            return batch * k * n * 2
        }
        return outChannels * inChannels * kernelSize * kernelSize * precision.elementSize
    }
    
    func inputBytes(precision: PrecisionMode) -> Int {
        let elem = precision.elementSize
        if opType == .matmul {
            return batch * m * k * elem
        }
        return batch * height * width * inChannels * elem
    }
    
    var shortDescription: String {
        if opType == .matmul {
            return "GEMM [\(m)x\(k)x\(n)] L\(layers)"
        }
        return "\(inChannels)c \(height)x\(width) k\(kernelSize) L\(layers)"
    }
    
    var detailedDescription: String {
        if opType == .matmul {
            return "MatMul B:\(batch) | M:\(m) K:\(k) N:\(n) | L:\(layers)"
        }
        return "B:\(batch) | H:\(height) W:\(width) | Cin:\(inChannels) Cout:\(outChannels) | K:\(kernelSize)x\(kernelSize) | L:\(layers)"
    }
}

// MARK: - Benchmark Result Entry
struct BenchmarkResult: Identifiable, Codable {
    var id: UUID = UUID()
    var timestamp: Date = Date()
    var dimensions: ConvDimensions
    var precision: PrecisionMode // fp16 or int8
    var target: DeviceTarget
    var avgDurationMs: Double
    var tops: Double
    var iterations: Int
    var sweepType: SweepType
    var sweepValue: Double
    var sweepLabel: String
    
    // PMU Telemetry Counters & Hardware Metrics
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
    
    static let allPMUCounterKeys: [String] = [
        "kANE_AF_TO_L2_DATA",
        "kANE_AF_TO_KM_DATA",
        "kANE_L2_TO_AF_DATA",
        "kANE_L2_TO_NE_DATA",
        "kANE_NE_TO_L2_DATA",
        "kANE_INT8_CYCLES",
        "kANE_FP16_CYCLES",
        "kANE_L2_READ_STALL_CYCLES",
        "kANE_L2_WRITE_STALL_CYCLES",
        "kANE_KM_STALL_CYCLES",
        "kANE_NE_NOMINAL_CYCLES",
        "kANE_NE_THROTTLE_CYCLES",
        "kANE_L2_THROTTLE_CYCLES",
        "kANE_NE_COMPUTE_CYCLES",
        "kANE_NE_INPUT_STALL_CYCLES",
        "kANE_NE_OUTPUT_STALL_CYCLES",
        "kANE_NE_KERNEL_STALL_CYCLES",
        "kANE_DMA_READWRITE_BYTES",
        "kANE_DMA_READ_BYTES",
        "kANE_DPE_ENERGY",
        "kANE_L2_NOMINAL_CYCLES",
        "kANE_L2PE_COMPUTE_CYCLES",
        "kANE_L2PE_INPUT_STALL_CYCLES",
        "kANE_L2PE_OUTPUT_STALL_CYCLES"
    ]
    
    func counterValue(for key: String) -> UInt64 {
        return pmuCounters[key] ?? pmuCounters["\(key):"] ?? 0
    }
    
    init(
        id: UUID = UUID(),
        timestamp: Date = Date(),
        dimensions: ConvDimensions,
        precision: PrecisionMode,
        target: DeviceTarget,
        avgDurationMs: Double,
        tops: Double,
        iterations: Int,
        sweepType: SweepType,
        sweepValue: Double,
        sweepLabel: String,
        pmuCounters: [String: UInt64] = [:],
        computeCycles: UInt64 = 0,
        nominalCycles: UInt64 = 0,
        outputStallCycles: UInt64 = 0,
        inputStallCycles: UInt64 = 0,
        dmaRwBytes: UInt64 = 0,
        dpeEnergy: UInt64 = 0,
        hwExecutionTimeNs: UInt64 = 0,
        macsPerCoreCycle: Double = 0.0,
        chipMacsPerCycle: Double = 0.0,
        aluSaturation: Double = 0.0,
        effectiveClockGhz: Double = 0.0
    ) {
        self.id = id
        self.timestamp = timestamp
        self.dimensions = dimensions
        self.precision = precision
        self.target = target
        self.avgDurationMs = avgDurationMs
        self.tops = tops
        self.iterations = iterations
        self.sweepType = sweepType
        self.sweepValue = sweepValue
        self.sweepLabel = sweepLabel
        self.pmuCounters = pmuCounters
        self.computeCycles = computeCycles
        self.nominalCycles = nominalCycles
        self.outputStallCycles = outputStallCycles
        self.inputStallCycles = inputStallCycles
        self.dmaRwBytes = dmaRwBytes
        self.dpeEnergy = dpeEnergy
        self.hwExecutionTimeNs = hwExecutionTimeNs
        self.macsPerCoreCycle = macsPerCoreCycle
        self.chipMacsPerCycle = chipMacsPerCycle
        self.aluSaturation = aluSaturation
        self.effectiveClockGhz = effectiveClockGhz
    }
    
    var seriesName: String {
        return "\(target.shortName) \(precision.rawValue)"
    }
    
    var formattedDuration: String {
        return String(format: "%.2f ms", avgDurationMs)
    }
    
    var formattedTOPS: String {
        return String(format: "%.3f TOPS", tops)
    }
    
    var formattedGFLOPs: String {
        return String(format: "%.1f GFLOPs", dimensions.gflops)
    }
    
    var formattedComputeCycles: String {
        formatCompact(computeCycles)
    }
    
    var formattedStalls: String {
        formatCompact(outputStallCycles + inputStallCycles)
    }
    
    var formattedDMA: String {
        if dmaRwBytes >= 1024 * 1024 {
            return String(format: "%.1f MB", Double(dmaRwBytes) / (1024.0 * 1024.0))
        } else if dmaRwBytes >= 1024 {
            return String(format: "%.1f KB", Double(dmaRwBytes) / 1024.0)
        }
        return "\(dmaRwBytes) B"
    }
    
    var formattedSaturation: String {
        String(format: "%.1f%%", aluSaturation)
    }
    
    private func formatCompact(_ val: UInt64) -> String {
        if val >= 1_000_000 {
            return String(format: "%.2fM", Double(val) / 1_000_000.0)
        } else if val >= 1_000 {
            return String(format: "%.1fK", Double(val) / 1_000.0)
        }
        return "\(val)"
    }
}
