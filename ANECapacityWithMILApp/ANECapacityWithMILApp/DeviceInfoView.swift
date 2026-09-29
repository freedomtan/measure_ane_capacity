import SwiftUI
import CoreML

struct DeviceInfoView: View {
    @ObservedObject var viewModel: BenchmarkViewModel
    
    var body: some View {
        NavigationView {
            List {
                Section(header: Text("CoreML & Neural Engine Runtime")) {
                    infoRow(title: "Hardware Available", value: viewModel.hasANE ? "Yes (Apple Neural Engine)" : "No (Simulator / Emulation)")
                    infoRow(title: "Execution Framework", value: "CoreML.framework (MIL)")
                    infoRow(title: "Compute Target Units", value: "ANE, GPU, CPU, All")
                    infoRow(title: "Supported Precision", value: "FP16, INT8 (W8A8 QDQ), FP8 (Float8E4M3 QDQ)")
                }
                
                Section(header: Text("Target Hardware Architecture")) {
                    infoRow(title: "Device Placement", value: viewModel.hardwareDeviceName)
                    infoRow(title: "FP8 Target Architecture", value: "H18 (iPhone 17 Pro) / H19 (iPhone 18 Pro)")
                    infoRow(title: "FP16 & INT8 Target", value: "H15..H19 (A15..A19 / M2..M5)")
                    infoRow(title: "Hardware Telemetry", value: "_ANEClient PMU Counters")
                }
                
                Section(header: Text("How CoreML / MIL Capacity is Measured")) {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Silicon Throughput Formulas:")
                            .font(.caption)
                            .fontWeight(.bold)
                        Text("Conv2D: (2 * B * H * W * Ci * Co * K² * L) / (t * 10¹²)\nMatMul: (2 * B * M * K * N * L) / (t * 10¹²)")
                            .font(.system(size: 11, design: .monospaced))
                            .padding(8)
                            .background(Color(.tertiarySystemBackground))
                            .cornerRadius(6)
                        
                        Text("• Ci, Co / M, K, N: Channel or matrix dimensions (ANE tiles align to 64x64 or 128x128).\n• H, W: Spatial feature map resolution.\n• K: 2D Convolution kernel size.\n• L: Number of chained layers in graph (amortizes driver & command dispatch overhead to measure true silicon saturation).\n• FP8 (Float8E4M3): Uses MIL quantize/dequantize ops with specificationversion 10 (CoreML9 opset) for native hardware FP8 tensor cores.\n• INT8 (W8A8): Uses constexpr_blockwise_shift_scale and quantize/dequantize ops matching measure_conv_coreml.m.")
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
}
