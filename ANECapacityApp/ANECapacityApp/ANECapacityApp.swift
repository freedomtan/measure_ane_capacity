import SwiftUI

@main
struct ANECapacityApp: App {
    init() {
        if ProcessInfo.processInfo.arguments.contains("--native-int8-gemm") {
            DispatchQueue.global().async {
                NativeInt8GEMMProbe.run()
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
