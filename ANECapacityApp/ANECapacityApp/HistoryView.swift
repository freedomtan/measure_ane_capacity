import SwiftUI

struct HistoryView: View {
    @ObservedObject var viewModel: BenchmarkViewModel
    
    @State private var filterDevice: DeviceTarget? = nil
    @State private var filterPrecision: PrecisionMode? = nil
    @State private var showClearConfirmation: Bool = false
    
    var filteredResults: [BenchmarkResult] {
        viewModel.results.filter { r in
            if let dev = filterDevice, r.target != dev { return false }
            if let prec = filterPrecision, r.precision != prec { return false }
            return true
        }
    }
    
    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                if viewModel.results.isEmpty {
                    VStack(spacing: 16) {
                        Image(systemName: "tray")
                            .font(.system(size: 64))
                            .foregroundColor(.secondary)
                        Text("No Runs Recorded")
                            .font(.title3)
                            .fontWeight(.bold)
                        Text("Completed benchmark runs and sweeps will appear here.")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    // Filters Header
                    filtersHeader
                    
                    // List
                    List {
                        ForEach(filteredResults) { item in
                            resultRow(item)
                        }
                    }
                    .listStyle(.insetGrouped)
                }
            }
            .navigationTitle("History (\(viewModel.results.count))")
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    if !viewModel.results.isEmpty {
                        Button(role: .destructive, action: { showClearConfirmation = true }) {
                            Text("Clear")
                                .foregroundColor(.red)
                        }
                    }
                }
                
                ToolbarItem(placement: .navigationBarTrailing) {
                    if !viewModel.results.isEmpty {
                        ShareLink(
                            item: viewModel.exportCSV(),
                            preview: SharePreview("ANE_Capacity_Results.csv", image: Image(systemName: "doc.text"))
                        ) {
                            Label("Export CSV", systemImage: "square.and.arrow.up")
                        }
                    }
                }
            }
            .confirmationDialog("Clear All Results?", isPresented: $showClearConfirmation, titleVisibility: .visible) {
                Button("Delete All History", role: .destructive) {
                    viewModel.clearResults()
                }
            }
        }
    }
    
    private var filtersHeader: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                // Device filters
                filterPill(title: "All Devices", isSelected: filterDevice == nil) {
                    filterDevice = nil
                }
                ForEach(DeviceTarget.allCases) { dev in
                    filterPill(title: dev.shortName, isSelected: filterDevice == dev) {
                        filterDevice = dev
                    }
                }
                
                Divider().frame(height: 20)
                
                // Precision filters
                filterPill(title: "All Types", isSelected: filterPrecision == nil) {
                    filterPrecision = nil
                }
                filterPill(title: "FP16", isSelected: filterPrecision == .fp16) {
                    filterPrecision = .fp16
                }
                filterPill(title: "INT8", isSelected: filterPrecision == .int8) {
                    filterPrecision = .int8
                }
            }
            .padding(.horizontal)
            .padding(.vertical, 8)
        }
        .background(Color(.secondarySystemBackground))
    }
    
    private func filterPill(title: String, isSelected: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text(title)
                .font(.caption2)
                .fontWeight(.medium)
                .padding(.horizontal, 10)
                .padding(.vertical, 5)
                .background(isSelected ? Color.accentColor : Color(.tertiarySystemBackground))
                .foregroundColor(isSelected ? .white : .primary)
                .cornerRadius(12)
        }
    }
    
    private func resultRow(_ r: BenchmarkResult) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(r.dimensions.opType.rawValue)
                    .font(.system(size: 10, weight: .bold))
                    .padding(.horizontal, 5)
                    .padding(.vertical, 2)
                    .background(Color(.systemGray4))
                    .cornerRadius(5)
                
                Text("\(r.target.shortName) \(r.precision.rawValue)")
                    .font(.caption)
                    .fontWeight(.bold)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(r.precision == .fp16 ? Color.blue.opacity(0.15) : Color.orange.opacity(0.15))
                    .foregroundColor(r.precision == .fp16 ? .blue : .orange)
                    .cornerRadius(6)
                
                Text(r.sweepType == .none ? "Custom" : r.sweepLabel)
                    .font(.caption2)
                    .foregroundColor(.secondary)
                
                Spacer()
                
                Text(r.formattedTOPS)
                    .font(.subheadline)
                    .fontWeight(.bold)
                    .foregroundColor(.primary)
            }
            
            HStack {
                Text(r.dimensions.detailedDescription)
                    .font(.caption2)
                    .foregroundColor(.secondary)
                
                Spacer()
                
                Text(r.formattedDuration)
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
            
            if r.computeCycles > 0 || r.outputStallCycles > 0 || r.dmaRwBytes > 0 {
                HStack(spacing: 8) {
                    Label(r.formattedComputeCycles, systemImage: "cpu")
                        .foregroundColor(.blue)
                    Label(r.formattedStalls, systemImage: "exclamationmark.triangle")
                        .foregroundColor(r.outputStallCycles > 0 ? .orange : .secondary)
                    Label(r.formattedDMA, systemImage: "arrow.left.arrow.right")
                        .foregroundColor(.green)
                    if r.aluSaturation > 0 {
                        Text("ALU: \(r.formattedSaturation)")
                            .foregroundColor(.indigo)
                    }
                }
                .font(.system(size: 10, weight: .medium, design: .monospaced))
            }
        }
        .padding(.vertical, 4)
    }
}
