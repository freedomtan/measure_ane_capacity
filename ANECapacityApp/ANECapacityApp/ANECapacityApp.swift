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
        if ProcessInfo.processInfo.arguments.contains("--native-coreml-matmul") {
            DispatchQueue.global().async {
                NativeCoreMLMatmulProbe.run()
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
