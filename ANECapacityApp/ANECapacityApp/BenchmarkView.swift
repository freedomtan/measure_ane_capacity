import SwiftUI

struct BenchmarkView: View {
    @ObservedObject var viewModel: BenchmarkViewModel
    
    @State private var isCustomMode: Bool = false
    @State private var showConsole: Bool = true
    
    var body: some View {
        NavigationView {
            ScrollViewReader { proxy in
                ScrollView {
                    VStack(spacing: 20) {
                        // Hardware status banner
                        hardwareStatusBanner
                        
                        // Mode Selector: Presets vs Custom
                        Picker("Run Mode", selection: $isCustomMode) {
                            Text("Capacity Sweeps").tag(false)
                            Text("Custom Size").tag(true)
                        }
                        .pickerStyle(.segmented)
                        
                        if isCustomMode {
                            customDimensionsCard
                        } else {
                            presetSweepsCard
                        }
                        
                        // Precision and Target Card
                        executionSettingsCard
                        
                        // Action Button & Progress
                        actionAndProgressCard
                        
                        // Console Log
                        if showConsole {
                            consoleCard(proxy: proxy)
                        }
                    }
                    .padding()
                }
            }
            .navigationTitle("ANE Capacity Benchmark")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: { showConsole.toggle() }) {
                        Image(systemName: showConsole ? "terminal.fill" : "terminal")
                    }
                }
            }
        }
    }
    
    // MARK: - Hardware Status Banner
    private var hardwareStatusBanner: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 12) {
                Image(systemName: viewModel.hasANE ? "cpu.fill" : "exclamationmark.triangle.fill")
                    .font(.title2)
                    .foregroundColor(viewModel.hasANE ? .green : .orange)
                
                VStack(alignment: .leading, spacing: 2) {
                    Text(viewModel.hasANE ? "Apple Neural Engine Active" : "iOS Simulator Detected")
                        .font(.subheadline)
                        .fontWeight(.bold)
                    Text("Metal Device: \(viewModel.metalDeviceName)")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
                
                Spacer()
                
                if viewModel.hasANE {
                    Text("ANE Ready")
                        .font(.caption2)
                        .fontWeight(.semibold)
                        .foregroundColor(.green)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(Color.green.opacity(0.15))
                        .cornerRadius(8)
                } else {
                    Text("Simulator")
                        .font(.caption2)
                        .fontWeight(.semibold)
                        .foregroundColor(.orange)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(Color.orange.opacity(0.15))
                        .cornerRadius(8)
                }
            }
            
            #if targetEnvironment(simulator)
            HStack(alignment: .top, spacing: 8) {
                Image(systemName: "info.circle.fill")
                    .foregroundColor(.secondary)
                    .font(.caption)
                Text("Physical ANE silicon and PMU counters require running on a physical iPhone or iPad. MPSGraph execution is disabled in the simulator.")
                    .font(.caption2)
                    .foregroundColor(.secondary)
                Spacer()
            }
            .padding(.top, 2)
            #endif
        }
        .padding()
        .background(Color(.secondarySystemBackground))
        .cornerRadius(14)
    }
    
    // MARK: - Preset Sweeps Card
    private var presetSweepsCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Select Capacity Sweep")
                .font(.headline)
            
            ForEach([SweepType.channels, SweepType.spatial, SweepType.depth, SweepType.kernels, SweepType.fullCapacity], id: \.self) { sweep in
                Button(action: { viewModel.selectedSweep = sweep }) {
                    HStack {
                        VStack(alignment: .leading, spacing: 3) {
                            Text(sweep.rawValue)
                                .font(.subheadline)
                                .fontWeight(.semibold)
                                .foregroundColor(.primary)
                            Text(sweepDescription(sweep))
                                .font(.caption2)
                                .foregroundColor(.secondary)
                        }
                        Spacer()
                        if viewModel.selectedSweep == sweep {
                            Image(systemName: "checkmark.circle.fill")
                                .foregroundColor(.accentColor)
                        } else {
                            Image(systemName: "circle")
                                .foregroundColor(.secondary)
                        }
                    }
                    .padding(12)
                    .background(viewModel.selectedSweep == sweep ? Color.accentColor.opacity(0.1) : Color(.tertiarySystemBackground))
                    .cornerRadius(10)
                }
                .buttonStyle(.plain)
            }
        }
        .padding()
        .background(Color(.secondarySystemBackground))
        .cornerRadius(16)
    }
    
    private func sweepDescription(_ s: SweepType) -> String {
        switch s {
        case .none: return "Run single test"
        case .channels: return "Sweeps C = 32, 64, 128, 256, 512, 1024 (Tests matrix array saturation)"
        case .spatial: return "Sweeps H=W = 64, 128, 256, 384, 512, 768 (Tests bandwidth scaling)"
        case .depth: return "Sweeps L = 1, 5, 10, 20, 30, 40 (Measures dispatch latency amortization)"
        case .kernels: return "Tests K=1x1 (GEMM) vs K=3x3 vs K=5x5 (2D Spatial Conv)"
        case .fullCapacity: return "Evaluates FP16 and INT8 across all major tensor footprints"
        }
    }
    
    // MARK: - Custom Dimensions Card
    private var customDimensionsCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Text("Custom Dimensions")
                    .font(.headline)
                Spacer()
                Text("\(String(format: "%.1f", viewModel.dimensions.gflops)) GFLOPs/iter")
                    .font(.caption)
                    .fontWeight(.semibold)
                    .foregroundColor(.accentColor)
            }
            
            // Channels
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("Input / Output Channels:")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Spacer()
                    Text("\(viewModel.dimensions.inChannels) c")
                        .font(.caption)
                        .fontWeight(.bold)
                }
                Picker("Channels", selection: $viewModel.dimensions.inChannels) {
                    ForEach([16, 32, 64, 128, 256, 512, 1024], id: \.self) { c in
                        Text("\(c)").tag(c)
                    }
                }
                .pickerStyle(.segmented)
                .onChange(of: viewModel.dimensions.inChannels) { newC in
                    viewModel.dimensions.outChannels = newC
                }
            }
            
            // Spatial Dimensions (H x W)
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("Spatial Resolution (H = W):")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Spacer()
                    Text("\(viewModel.dimensions.height) x \(viewModel.dimensions.width)")
                        .font(.caption)
                        .fontWeight(.bold)
                }
                Picker("Resolution", selection: $viewModel.dimensions.height) {
                    ForEach([64, 128, 256, 512, 768, 1024], id: \.self) { res in
                        Text("\(res)").tag(res)
                    }
                }
                .pickerStyle(.segmented)
                .onChange(of: viewModel.dimensions.height) { newH in
                    viewModel.dimensions.width = newH
                }
            }
            
            // Chained Layers (L)
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("Chained Layers (L):")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Spacer()
                    Text("\(viewModel.dimensions.layers) layers")
                        .font(.caption)
                        .fontWeight(.bold)
                }
                Picker("Layers", selection: $viewModel.dimensions.layers) {
                    ForEach([1, 5, 10, 20, 30, 40], id: \.self) { l in
                        Text("\(l)").tag(l)
                    }
                }
                .pickerStyle(.segmented)
            }
            
            // Kernel Size
            HStack {
                Text("Kernel Size (KxK):")
                    .font(.caption)
                    .foregroundColor(.secondary)
                Spacer()
                Picker("Kernel", selection: $viewModel.dimensions.kernelSize) {
                    Text("1x1").tag(1)
                    Text("3x3").tag(3)
                    Text("5x5").tag(5)
                }
                .pickerStyle(.segmented)
                .frame(width: 180)
            }
        }
        .padding()
        .background(Color(.secondarySystemBackground))
        .cornerRadius(16)
        .onAppear {
            viewModel.selectedSweep = .none
        }
    }
    
    // MARK: - Execution Settings Card
    private var executionSettingsCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Execution Settings")
                .font(.headline)
            
            // Precision
            VStack(alignment: .leading, spacing: 4) {
                Text("Data Type / Precision")
                    .font(.caption)
                    .foregroundColor(.secondary)
                Picker("Precision", selection: $viewModel.selectedPrecision) {
                    ForEach(PrecisionMode.allCases) { p in
                        Text(p.rawValue).tag(p)
                    }
                }
                .pickerStyle(.segmented)
            }
            
            // Target Device
            VStack(alignment: .leading, spacing: 4) {
                Text("Compute Target")
                    .font(.caption)
                    .foregroundColor(.secondary)
                Picker("Target", selection: $viewModel.selectedTarget) {
                    ForEach(DeviceTarget.allCases) { t in
                        Text(t.rawValue).tag(t)
                    }
                }
                .pickerStyle(.segmented)
            }
            
            // Iterations
            HStack {
                Text("Benchmark Iterations:")
                    .font(.caption)
                    .foregroundColor(.secondary)
                Spacer()
                Stepper("\(viewModel.iterations) runs", value: $viewModel.iterations, in: 5...100, step: 5)
                    .font(.subheadline)
            }
        }
        .padding()
        .background(Color(.secondarySystemBackground))
        .cornerRadius(16)
    }
    
    // MARK: - Action and Progress Card
    private var actionAndProgressCard: some View {
        VStack(spacing: 12) {
            if viewModel.isRunning {
                VStack(spacing: 8) {
                    HStack {
                        Text(viewModel.statusMessage)
                            .font(.caption)
                            .foregroundColor(.secondary)
                            .lineLimit(1)
                        Spacer()
                        Text("\(Int(viewModel.progress * 100))%")
                            .font(.caption)
                            .fontWeight(.bold)
                    }
                    
                    ProgressView(value: viewModel.progress)
                        .progressViewStyle(LinearProgressViewStyle())
                    
                    Button(role: .destructive, action: { viewModel.cancelBenchmark() }) {
                        Label("Stop Benchmark", systemImage: "stop.fill")
                            .font(.headline)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 14)
                            .background(Color.red.opacity(0.15))
                            .foregroundColor(.red)
                            .cornerRadius(12)
                    }
                }
            } else {
                Button(action: { viewModel.startBenchmark() }) {
                    Label(isCustomMode ? "Run Single Benchmark" : "Start Capacity Sweep", systemImage: "play.fill")
                        .font(.headline)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 14)
                        .background(Color.accentColor)
                        .foregroundColor(.white)
                        .cornerRadius(12)
                }
            }
        }
    }
    
    // MARK: - Console Log Card
    private func consoleCard(proxy: ScrollViewProxy) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Label("Live Execution Console", systemImage: "terminal")
                    .font(.caption)
                    .fontWeight(.bold)
                Spacer()
                Button(action: { viewModel.clearLogs() }) {
                    Text("Clear")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
            }
            
            ScrollView {
                VStack(alignment: .leading, spacing: 4) {
                    ForEach(Array(viewModel.consoleLogs.enumerated()), id: \.offset) { idx, log in
                        Text(log)
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundColor(.green)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .id(idx)
                    }
                }
            }
            .frame(height: 140)
            .padding(10)
            .background(Color.black.opacity(0.9))
            .cornerRadius(10)
            .onChange(of: viewModel.consoleLogs.count) { _ in
                if let last = viewModel.consoleLogs.indices.last {
                    proxy.scrollTo(last, anchor: .bottom)
                }
            }
        }
        .padding()
        .background(Color(.secondarySystemBackground))
        .cornerRadius(16)
    }
}
