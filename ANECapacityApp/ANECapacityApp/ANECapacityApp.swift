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
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
