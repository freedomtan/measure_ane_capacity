import SwiftUI
import Metal

struct DeviceInfoView: View {
    @ObservedObject var viewModel: BenchmarkViewModel
    
    var body: some View {
        NavigationView {
            List {
                Section(header: Text("Neural Engine Hardware")) {
                    infoRow(title: "ANE Device Available", value: viewModel.hasANE ? "Yes (Apple Neural Engine)" : "No (Simulator / Fallback)")
                    infoRow(title: "Framework", value: "MetalPerformanceShadersGraph")
                    infoRow(title: "Compilation Optimization", value: "Level 1 (ANE Target)")
                    infoRow(title: "Supported Precision", value: "FP16 (Half) & INT8 (Simulated QDQ)")
                }
                
                Section(header: Text("Metal Hardware Device")) {
                    infoRow(title: "Metal Device Name", value: viewModel.metalDeviceName)
                    if let dev = MTLCreateSystemDefaultDevice() {
                        infoRow(title: "Unified Memory", value: dev.hasUnifiedMemory ? "Yes" : "No")
                        infoRow(title: "Max Buffer Length", value: formatBytes(dev.maxBufferLength))
                        infoRow(title: "Max Threads/Group", value: "\(dev.maxThreadsPerThreadgroup.width)x\(dev.maxThreadsPerThreadgroup.height)x\(dev.maxThreadsPerThreadgroup.depth)")
                    }
                }
                
                Section(header: Text("How ANE Capacity is Measured")) {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Silicon Throughput Formula:")
                            .font(.caption)
                            .fontWeight(.bold)
                        Text("Throughput (TOPS) = (2 * B * H * W * Ci * Co * K² * L) / (avg_seconds * 10¹²)")
                            .font(.system(size: 11, design: .monospaced))
                            .padding(8)
                            .background(Color(.tertiarySystemBackground))
                            .cornerRadius(6)
                        
                        Text("• Ci, Co: Input and output channel dimensions (ANE matrix tiles typically 64x64 or 128x128).\n• H, W: Spatial feature map resolution.\n• K: 2D Convolution kernel size (K=3 for spatial conv, K=1 for GEMM).\n• L: Number of chained layers in graph (amortizes driver & command dispatch overhead to measure true silicon saturation).\n• INT8 Flow: Uses simulated dequantize-conv-requantize matching measure_conv_universal.m.")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    .padding(.vertical, 4)
                }
            }
            .listStyle(.insetGrouped)
            .navigationTitle("Device & Info")
        }
    }
    
    private func infoRow(title: String, value: String) -> some View {
        HStack {
            Text(title)
                .font(.subheadline)
            Spacer()
            Text(value)
                .font(.subheadline)
                .foregroundColor(.secondary)
        }
    }
    
    private func formatBytes(_ bytes: Int) -> String {
        let mb = Double(bytes) / (1024 * 1024)
        if mb >= 1024 {
            return String(format: "%.1f GB", mb / 1024.0)
        }
        return String(format: "%.0f MB", mb)
    }
}
