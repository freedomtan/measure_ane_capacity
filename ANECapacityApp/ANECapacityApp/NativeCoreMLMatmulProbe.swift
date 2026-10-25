import Foundation
import CoreML

/// On-device port of measure_matmul_coreml.m (docs/CPU_Matrix_Convolution_Capability.md
/// Finding 4), using the newly-ported MILMatmulChainConfig/MILBuildMatmulChainSpec
/// support in this app's own (pure-Objective-C, zero-dependency) MILSpecBuilder.m --
/// a separate implementation from the repo-root MILSpecBuilder.mm, which links real
/// protobuf and can't go into an iOS app bundle. Triggered by --native-coreml-matmul
/// (see ANECapacityApp.swift). Benchmarks all three CoreML compute units in one run,
/// since (unlike MPSGraphDevice.CPUDevice) CoreML doesn't crash on any of them.
enum NativeCoreMLMatmulProbe {
    static func run(size: Int? = nil) {
        let actualSize = resolveSize(size)
        for precision in [MILPrecision.FP16, .INT8, .FP8] {
            runOne(precision: precision, size: actualSize)
        }
    }

    /// Isolates a single precision's benchmark to one or more compute units, for
    /// controlled lldb/sample comparison of the exact BNNS call path per precision
    /// (see docs/CPU_Matrix_Convolution_Capability.md Finding 6's follow-up on whether
    /// FP8 takes a different SME2 code path than FP16 on A20 Pro, which supports the
    /// FEAT_SME_F8F16 ARM extension).
    static func runSingle(precision: MILPrecision, unitsFilter: String? = nil, size: Int? = nil) {
        let actualSize = resolveSize(size)
        runOne(precision: precision, unitsFilter: unitsFilter, size: actualSize)
    }

    private static func resolveSize(_ explicitSize: Int?) -> Int {
        if let explicitSize { return explicitSize }
        if let sIdx = ProcessInfo.processInfo.arguments.firstIndex(of: "--size"),
           sIdx + 1 < ProcessInfo.processInfo.arguments.count,
           let s = Int(ProcessInfo.processInfo.arguments[sIdx + 1]) {
            return s
        }
        return 2048 // Default to 2048x2048 to reach high operational intensity (>680 FLOPs/byte)
    }

    private static func runOne(precision: MILPrecision, unitsFilter: String? = nil, size: Int = 2048) {
        let uSize = UInt(size)
        let config = MILMatmulChainConfig(batch: 1, M: uSize, K: uSize, N: uSize, layers: 20, weightMode: .dense, precision: precision)
        let precisionName = MILPrecisionName(precision)
        print("[CoreMLMatmulProbe] === precision: \(precisionName) ===")

        var buildError: NSError?
        guard let specData = MILBuildMatmulChainSpec(config, &buildError) else {
            print("[CoreMLMatmulProbe] Failed to build spec: \(buildError?.localizedDescription ?? "unknown")")
            return
        }
        print("[CoreMLMatmulProbe] Spec built: \(specData.count) bytes")

        let tempPath = NSTemporaryDirectory() + "matmul_spec_\(precisionName)_\(ProcessInfo.processInfo.processIdentifier).mlmodel"
        do {
            try specData.write(to: URL(fileURLWithPath: tempPath))
        } catch {
            print("[CoreMLMatmulProbe] Failed to write temp spec: \(error)")
            return
        }
        defer { try? FileManager.default.removeItem(atPath: tempPath) }

        let sema = DispatchSemaphore(value: 0)
        var compiledURL: URL?
        var compileError: Error?
        MLModel.compileModel(at: URL(fileURLWithPath: tempPath)) { result in
            switch result {
            case .success(let url):
                let dest = NSTemporaryDirectory() + "compiled_matmul_\(precisionName)_\(ProcessInfo.processInfo.processIdentifier).mlmodelc"
                let destURL = URL(fileURLWithPath: dest)
                try? FileManager.default.removeItem(at: destURL)
                do {
                    try FileManager.default.copyItem(at: url, to: destURL)
                    compiledURL = destURL
                } catch {
                    compileError = error
                }
            case .failure(let error):
                compileError = error
            }
            sema.signal()
        }
        sema.wait()

        guard let compiledURL else {
            print("[CoreMLMatmulProbe] Compile failed: \(compileError?.localizedDescription ?? "unknown")")
            return
        }
        defer { try? FileManager.default.removeItem(at: compiledURL) }

        var units: [(String, MLComputeUnits)] = [
            ("ANE", .cpuAndNeuralEngine),
            ("GPU", .cpuAndGPU),
            ("CPU", .cpuOnly),
        ]
        if let unitsFilter {
            units = units.filter { $0.0 == unitsFilter }
        }

        let inputName = MILMatmulChainInputName()
        let outputName = MILMatmulChainOutputName(config)
        let iterations = 20
        let warmup = 3
        let batchD = Double(config.batch)
        let mD = Double(config.M)
        let kD = Double(config.K)
        let nD = Double(config.N)
        let layersD = Double(config.layers)
        let totalOps = 2.0 * batchD * mD * kD * nD * layersD

        for (label, computeUnits) in units {
            let mlConfig = MLModelConfiguration()
            mlConfig.computeUnits = computeUnits

            let loadStart = Date()
            guard let model = try? MLModel(contentsOf: compiledURL, configuration: mlConfig) else {
                print("[CoreMLMatmulProbe] [\(label)] Failed to load model")
                continue
            }
            let loadMs = Date().timeIntervalSince(loadStart) * 1000.0

            let shape: [NSNumber] = [NSNumber(value: config.batch), 1, NSNumber(value: config.M), NSNumber(value: config.K)]
            guard let input = try? MLMultiArray(shape: shape, dataType: .float16) else {
                print("[CoreMLMatmulProbe] [\(label)] Failed to allocate input")
                continue
            }
            let ptr = input.dataPointer.bindMemory(to: UInt16.self, capacity: input.count)
            var state: UInt64 = 0x9E3779B97F4A7C15
            for i in 0..<input.count {
                state ^= state << 13; state ^= state >> 7; state ^= state << 17
                ptr[i] = (state & 1 != 0) ? 0xAC00 : 0x2C00
            }

            guard let features = try? MLDictionaryFeatureProvider(dictionary: [inputName: input]) else {
                print("[CoreMLMatmulProbe] [\(label)] Failed to build feature provider")
                continue
            }

            var ok = true
            for _ in 0..<warmup {
                if (try? model.prediction(from: features)) == nil { ok = false; break }
            }
            guard ok else {
                print("[CoreMLMatmulProbe] [\(label)] Warmup prediction failed")
                continue
            }

            var result: MLFeatureProvider?
            let start = Date()
            for _ in 0..<iterations {
                guard let r = try? model.prediction(from: features) else { ok = false; break }
                result = r
            }
            guard ok else {
                print("[CoreMLMatmulProbe] [\(label)] Prediction failed")
                continue
            }
            let avg = Date().timeIntervalSince(start) / Double(iterations)
            let tops = (totalOps / 1e12) / avg

            print(String(format: "[CoreML %@ %@] Load: %.2f ms, Avg: %.2f ms, Speed: %.4f TOPS (%.1f FPS)",
                         label, precisionName, loadMs, avg * 1000.0, tops, 1.0 / avg))

            if let out = result?.featureValue(for: outputName)?.multiArrayValue {
                let outPtr = out.dataPointer.bindMemory(to: UInt16.self, capacity: out.count)
                var zeros = 0, nonFinite = 0
                var minAbs = Float.infinity, maxAbs: Float = 0
                for i in 0..<out.count {
                    let bits = outPtr[i]
                    let v = Float(Float16(bitPattern: bits))
                    if !v.isFinite { nonFinite += 1; continue }
                    if v == 0 { zeros += 1 }
                    let a = abs(v)
                    if a > 0 && a < minAbs { minAbs = a }
                    if a > maxAbs { maxAbs = a }
                }
                print("   Output check: \(out.count) elements, |v| range [\(minAbs.isFinite ? minAbs : 0), \(maxAbs)], zeros: \(zeros), non-finite: \(nonFinite)")
            }
        }
    }
}
