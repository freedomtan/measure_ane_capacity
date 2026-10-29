import SwiftUI

@main
struct ANECapacityApp: App {
    init() {
        if ProcessInfo.processInfo.arguments.contains("--native-int8-gemm") {
            let args = ProcessInfo.processInfo.arguments
            var size = 2048
            var layers = 1
            if let sIdx = args.firstIndex(of: "--size"), sIdx + 1 < args.count, let s = Int(args[sIdx + 1]) {
                size = s
            }
            if let lIdx = args.firstIndex(of: "--layers"), lIdx + 1 < args.count, let l = Int(args[lIdx + 1]) {
                layers = l
            }
            DispatchQueue.global().async {
                NativeInt8GEMMProbe.run(size: size, layers: layers)
                exit(0)
            }
        }
        if ProcessInfo.processInfo.arguments.contains("--native-int8-conv") {
            DispatchQueue.global().async {
                NativeInt8ConvProbe.run()
                exit(0)
            }
        }
        if ProcessInfo.processInfo.arguments.contains("--supports-mxu") {
            DispatchQueue.global().async {
                SupportsMXUProbe.run()
                exit(0)
            }
        }
        if ProcessInfo.processInfo.arguments.contains("--native-cpu-matmul") {
            DispatchQueue.global().async {
                NativeCPUMatmulConvProbe.runMatmul()
                exit(0)
            }
        }
        if ProcessInfo.processInfo.arguments.contains("--native-cpu-conv") {
            DispatchQueue.global().async {
                NativeCPUMatmulConvProbe.runConv()
                exit(0)
            }
        }
        if ProcessInfo.processInfo.arguments.contains("--sysctl-features") {
            DispatchQueue.global().async {
                SysctlFeatureProbe.run()
                exit(0)
            }
        }
        if ProcessInfo.processInfo.arguments.contains("--native-coreml-matmul") {
            DispatchQueue.global().async {
                NativeCoreMLMatmulProbe.run()
                exit(0)
            }
        }
        if ProcessInfo.processInfo.arguments.contains("--native-bnns-graph-matmul") {
            DispatchQueue.global().async {
                BNNSGraphMatmulProbeRun()
                exit(0)
            }
        }
        if ProcessInfo.processInfo.arguments.contains("--sme-microbench") {
            DispatchQueue.global().async {
                // Apple Silicon has no baseline (non-streaming) SVE -- only streaming SVE
                // via SME's `smstart`. check_sve_support() calls bare SVE outside streaming
                // mode and traps (SIGILL); skip it and go straight to what's supported.
                check_streaming_sve_support()
                check_sme_support()
                check_sme2_support()
                check_sme_f16f16_support()
                check_sme_b16b16_support()
                check_sme_f8f16_support()
                check_sme_f8f32_support()
                run_micro_benchmark(1, 1)
                run_gemm(1, 1)
                exit(0)
            }
        }
        if ProcessInfo.processInfo.arguments.contains("--sme2-fp8-sketch") {
            DispatchQueue.global().async {
                run_sme2_fp8_sketch()
                exit(0)
            }
        }
        if ProcessInfo.processInfo.arguments.contains("--sme2-fp8-peak") {
            DispatchQueue.global().async {
                run_sme2_fp8_peak()
                exit(0)
            }
        }
        if let idx = ProcessInfo.processInfo.arguments.firstIndex(of: "--native-coreml-matmul-single"),
           idx + 1 < ProcessInfo.processInfo.arguments.count {
            let precisionArg = ProcessInfo.processInfo.arguments[idx + 1]
            let precision: MILPrecision = precisionArg == "fp8" ? .FP8 : (precisionArg == "int8" ? .INT8 : .FP16)
            let args = ProcessInfo.processInfo.arguments
            var unitsFilter: String? = nil
            if let uidx = args.firstIndex(of: "--units"), uidx + 1 < args.count {
                unitsFilter = args[uidx + 1]
            }
            DispatchQueue.global().async {
                NativeCoreMLMatmulProbe.runSingle(precision: precision, unitsFilter: unitsFilter)
                exit(0)
            }
        }
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
