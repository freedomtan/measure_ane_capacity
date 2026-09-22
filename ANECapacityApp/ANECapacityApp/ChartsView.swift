import SwiftUI
import Charts

enum MetricViewType: String, CaseIterable, Identifiable {
    case tops = "Throughput (TOPS)"
    case latency = "Latency (ms)"
    case computeCycles = "Compute (Mcyc)"
    case stalls = "Stalls (Mcyc)"
    case dma = "DMA (MB)"
    case saturation = "ALU Saturation (%)"
    
    var id: String { rawValue }
}

enum ChartPrecisionFilter: String, CaseIterable, Identifiable {
    case all = "All Precisions"
    case fp16 = "FP16"
    case int8 = "INT8"
    case fp8 = "FP8 (E4M3)"
    
    var id: String { rawValue }
}

struct ChartsView: View {
    @ObservedObject var viewModel: BenchmarkViewModel
    
    @State private var selectedMetric: MetricViewType = .tops
    @State private var selectedSweepFilter: SweepType = .channels
    @State private var selectedPrecisionFilter: ChartPrecisionFilter = .all
    @State private var selectedPoint: BenchmarkResult? = nil
    
    // Filtered results based on selected sweep type and precision
    var filteredResults: [BenchmarkResult] {
        viewModel.results.filter { r in
            guard r.sweepType == selectedSweepFilter else { return false }
            switch selectedPrecisionFilter {
            case .all: return true
            case .fp16: return r.precision == .fp16
            case .int8: return r.precision == .int8
            case .fp8: return r.precision == .fp8
            }
        }
    }
    
    // Whether current sweep has multiple precisions to filter
    var hasMultiplePrecisions: Bool {
        let precs = Set(viewModel.results.filter { $0.sweepType == selectedSweepFilter }.map(\.precision))
        return precs.count > 1
    }
    
    // Available sweeps that have data
    var availableSweeps: [SweepType] {
        let unique = Set(viewModel.results.map(\.sweepType))
        return SweepType.allCases.filter { unique.contains($0) }
    }
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    if viewModel.results.isEmpty {
                        emptyStateView
                    } else {
                        metricAndSweepPickers
                        
                        summaryCards
                        
                        chartCard
                        
                        if let selected = selectedPoint {
                            selectedPointCard(selected)
                        }
                        
                        comparisonTableCard
                    }
                }
                .padding()
            }
            .navigationTitle("Performance Figures")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    if !viewModel.results.isEmpty {
                        ShareLink(
                            item: viewModel.exportCSV(),
                            preview: SharePreview("ANE_Benchmark_Results.csv", image: Image(systemName: "chart.xyaxis.line"))
                        ) {
                            Label("Export Data", systemImage: "square.and.arrow.up")
                        }
                    }
                }
            }
            .onAppear {
                if let first = availableSweeps.first, !availableSweeps.contains(selectedSweepFilter) {
                    selectedSweepFilter = first
                }
            }
        }
    }
    
    // MARK: - Empty State
    private var emptyStateView: some View {
        VStack(spacing: 16) {
            Image(systemName: "chart.line.uptrend.xyaxis")
                .font(.system(size: 64))
                .foregroundColor(.secondary)
            Text("No Benchmark Data Yet")
                .font(.title2)
                .fontWeight(.bold)
            Text("Run a single benchmark or parameter sweep in the Benchmark tab to generate throughput and latency figures.")
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal)
        }
        .padding(.vertical, 80)
    }
    
    // MARK: - Pickers
    private var metricAndSweepPickers: some View {
        VStack(spacing: 12) {
            Picker("Metric", selection: $selectedMetric) {
                ForEach(MetricViewType.allCases) { m in
                    Text(m.rawValue).tag(m)
                }
            }
            .pickerStyle(.segmented)
            
            if hasMultiplePrecisions {
                Picker("Precision", selection: $selectedPrecisionFilter) {
                    ForEach(ChartPrecisionFilter.allCases) { p in
                        Text(p.rawValue).tag(p)
                    }
                }
                .pickerStyle(.segmented)
            }
            
            if availableSweeps.count > 1 {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(availableSweeps) { sw in
                            Button(action: {
                                selectedSweepFilter = sw
                                selectedPoint = nil
                            }) {
                                Text(sw.rawValue)
                                    .font(.caption)
                                    .fontWeight(.medium)
                                    .padding(.horizontal, 12)
                                    .padding(.vertical, 6)
                                    .background(selectedSweepFilter == sw ? Color.accentColor : Color(.secondarySystemBackground))
                                    .foregroundColor(selectedSweepFilter == sw ? .white : .primary)
                                    .cornerRadius(14)
                            }
                        }
                    }
                }
            }
        }
    }
    
    // MARK: - Summary Cards
    private var summaryCards: some View {
        HStack(spacing: 12) {
            statBadge(
                title: "Peak ANE FP16",
                value: String(format: "%.2f", viewModel.peakFP16TOPS),
                unit: "TOPS",
                color: .blue
            )
            
            statBadge(
                title: "Peak ANE INT8",
                value: String(format: "%.2f", viewModel.peakINT8TOPS),
                unit: "TOPS",
                color: .orange
            )
            
            if let ratio = viewModel.speedupRatio {
                statBadge(
                    title: "INT8 / FP16",
                    value: String(format: "%.2fx", ratio),
                    unit: "Speedup",
                    color: .green
                )
            }
        }
    }
    
    private func statBadge(title: String, value: String, unit: String, color: Color) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.caption2)
                .foregroundColor(.secondary)
            HStack(alignment: .firstTextBaseline, spacing: 2) {
                Text(value)
                    .font(.title3)
                    .fontWeight(.bold)
                    .foregroundColor(color)
                Text(unit)
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(Color(.secondarySystemBackground))
        .cornerRadius(12)
    }
    
    private func valueForMetric(_ item: BenchmarkResult) -> Double {
        switch selectedMetric {
        case .tops:
            return item.tops
        case .latency:
            return item.avgDurationMs
        case .computeCycles:
            return Double(item.computeCycles) / 1e6
        case .stalls:
            return Double(item.outputStallCycles + item.inputStallCycles) / 1e6
        case .dma:
            return Double(item.dmaRwBytes) / (1024.0 * 1024.0)
        case .saturation:
            return item.aluSaturation
        }
    }
    
    // MARK: - Main Chart Card
    private var chartCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(selectedMetric.rawValue)
                        .font(.headline)
                    Text("vs. \(selectedSweepFilter.axisLabel)")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                Spacer()
                
                // Interactive Legend Filters
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 6) {
                        legendFilterButton(title: "All", filter: .all, color: .secondary)
                        legendFilterButton(title: "FP16", filter: .fp16, color: .blue)
                        legendFilterButton(title: "INT8", filter: .int8, color: .orange)
                        legendFilterButton(title: "FP8", filter: .fp8, color: .mint)
                    }
                }
            }
            
            // Swift Chart
            Chart {
                ForEach(filteredResults) { item in
                    let yVal = valueForMetric(item)
                    
                    LineMark(
                        x: .value(selectedSweepFilter.axisLabel, item.sweepValue),
                        y: .value(selectedMetric.rawValue, yVal)
                    )
                    .foregroundStyle(by: .value("Series", item.seriesName))
                    .interpolationMethod(.monotone)
                    
                    PointMark(
                        x: .value(selectedSweepFilter.axisLabel, item.sweepValue),
                        y: .value(selectedMetric.rawValue, yVal)
                    )
                    .foregroundStyle(by: .value("Series", item.seriesName))
                    .symbol(by: .value("Series", item.seriesName))
                    .symbolSize(45)
                }
                
                // Selected point cursor and highlight ring
                if let sp = selectedPoint, filteredResults.contains(where: { $0.id == sp.id }) {
                    RuleMark(x: .value("Selected Step", sp.sweepValue))
                        .lineStyle(StrokeStyle(lineWidth: 1.5, dash: [4, 4]))
                        .foregroundStyle(Color.secondary.opacity(0.6))
                    
                    let spY = valueForMetric(sp)
                    PointMark(
                        x: .value(selectedSweepFilter.axisLabel, sp.sweepValue),
                        y: .value(selectedMetric.rawValue, spY)
                    )
                    .foregroundStyle(sp.precision.themeColor)
                    .symbolSize(140)
                }
                
                // Peak TOPS indicator line
                if selectedMetric == .tops, let maxTops = filteredResults.map(\.tops).max(), maxTops > 0 {
                    RuleMark(y: .value("Peak", maxTops))
                        .lineStyle(StrokeStyle(lineWidth: 1, dash: [4, 4]))
                        .foregroundStyle(Color.red.opacity(0.7))
                        .annotation(position: .top, alignment: .trailing) {
                            Text(String(format: "Max: %.2f TOPS", maxTops))
                                .font(.caption2)
                                .fontWeight(.bold)
                                .foregroundColor(.red)
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(Color(.systemBackground).opacity(0.8))
                                .cornerRadius(4)
                        }
                }
            }
            .chartForegroundStyleScale([
                "ANE FP16": Color.blue,
                "ANE INT8": Color.orange,
                "ANE FP8 (E4M3)": Color.mint,
                "GPU FP16": Color.purple,
                "GPU INT8": Color.indigo,
                "GPU FP8 (E4M3)": Color.cyan
            ])
            .chartXAxis {
                AxisMarks(values: .automatic) { value in
                    AxisGridLine()
                    AxisTick()
                    AxisValueLabel()
                }
            }
            .chartYAxis {
                AxisMarks(values: .automatic) { value in
                    AxisGridLine()
                    AxisTick()
                    AxisValueLabel()
                }
            }
            .frame(height: 280)
            .chartOverlay { proxy in
                GeometryReader { geo in
                    Rectangle().fill(.clear).contentShape(Rectangle())
                        .gesture(
                            DragGesture(minimumDistance: 0)
                                .onChanged { val in
                                    let plotOrigin = geo[proxy.plotAreaFrame].origin
                                    let xLoc = val.location.x - plotOrigin.x
                                    let yLoc = val.location.y - plotOrigin.y
                                    
                                    guard let xVal: Double = proxy.value(atX: xLoc) else { return }
                                    
                                    // Find closest sweep step
                                    guard let closestStep = filteredResults.min(by: { abs($0.sweepValue - xVal) < abs($1.sweepValue - xVal) })?.sweepValue else { return }
                                    
                                    let stepCandidates = filteredResults.filter { $0.sweepValue == closestStep }
                                    
                                    // If multiple points at this step (e.g. FP16 and INT8), pick closest along Y to user's touch
                                    if stepCandidates.count > 1, let touchY: Double = proxy.value(atY: yLoc) {
                                        selectedPoint = stepCandidates.min(by: {
                                            let y0 = valueForMetric($0)
                                            let y1 = valueForMetric($1)
                                            return abs(y0 - touchY) < abs(y1 - touchY)
                                        })
                                    } else {
                                        selectedPoint = stepCandidates.first
                                    }
                                }
                        )
                }
            }
        }
        .padding()
        .background(Color(.secondarySystemBackground))
        .cornerRadius(16)
    }
    
    private func legendFilterButton(title: String, filter: ChartPrecisionFilter, color: Color) -> some View {
        Button(action: {
            selectedPrecisionFilter = (selectedPrecisionFilter == filter && filter != .all) ? .all : filter
            selectedPoint = nil
        }) {
            HStack(spacing: 4) {
                Circle().fill(color).frame(width: 8, height: 8)
                Text(title)
                    .font(.caption2)
                    .fontWeight(selectedPrecisionFilter == filter ? .bold : .regular)
                    .foregroundColor(selectedPrecisionFilter == filter ? .primary : .secondary)
            }
            .padding(.horizontal, 6)
            .padding(.vertical, 3)
            .background(selectedPrecisionFilter == filter ? Color(.tertiarySystemBackground) : Color.clear)
            .cornerRadius(6)
        }
    }
    
    // MARK: - Selected Point Inspector Card
    private func selectedPointCard(_ r: BenchmarkResult) -> some View {
        // Find all benchmark results at this exact sweep step
        let stepResults = viewModel.results.filter {
            $0.sweepType == r.sweepType && $0.sweepValue == r.sweepValue
        }
        
        return VStack(alignment: .leading, spacing: 10) {
            HStack {
                Label(r.sweepLabel, systemImage: "slider.horizontal.3")
                    .font(.subheadline)
                    .fontWeight(.bold)
                
                Spacer()
                
                // Selectable Precision Pills in the Inspector Card
                if stepResults.count > 1 {
                    HStack(spacing: 6) {
                        ForEach(stepResults) { item in
                            Button(action: {
                                selectedPoint = item
                            }) {
                                Text("\(item.target.shortName) \(item.precision.rawValue)")
                                    .font(.caption2)
                                    .fontWeight(.bold)
                                    .padding(.horizontal, 8)
                                    .padding(.vertical, 4)
                                    .background(selectedPoint?.id == item.id ? item.precision.themeColor : Color(.tertiarySystemBackground))
                                    .foregroundColor(selectedPoint?.id == item.id ? .white : .primary)
                                    .cornerRadius(8)
                            }
                        }
                    }
                } else {
                    Text("\(r.target.shortName) \(r.precision.rawValue)")
                        .font(.caption)
                        .fontWeight(.bold)
                        .foregroundColor(r.precision.themeColor)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 3)
                        .background(Color(.tertiarySystemBackground))
                        .cornerRadius(6)
                }
            }
            
            Divider()
            
            HStack(spacing: 20) {
                VStack(alignment: .leading) {
                    Text("Throughput").font(.caption2).foregroundColor(.secondary)
                    Text(r.formattedTOPS)
                        .font(.subheadline)
                        .fontWeight(.bold)
                        .foregroundColor(r.precision.themeColor)
                }
                VStack(alignment: .leading) {
                    Text("Latency").font(.caption2).foregroundColor(.secondary)
                    Text(r.formattedDuration)
                        .font(.subheadline)
                        .fontWeight(.semibold)
                }
                VStack(alignment: .leading) {
                    Text(r.dimensions.opType == .matmul ? "Dimensions" : "Tensor Shape").font(.caption2).foregroundColor(.secondary)
                    Text(r.dimensions.opType == .matmul ? "[\(r.dimensions.m)x\(r.dimensions.k)x\(r.dimensions.n)]" : "\(r.dimensions.inChannels)c \(r.dimensions.height)x\(r.dimensions.width)")
                        .font(.subheadline)
                        .fontWeight(.semibold)
                }
                VStack(alignment: .leading) {
                    Text("Layers").font(.caption2).foregroundColor(.secondary)
                    Text("L=\(r.dimensions.layers)")
                        .font(.subheadline)
                        .fontWeight(.semibold)
                }
            }
            
            // Inline speedup indicator when both FP16 and INT8 exist at this step
            if let fp16 = stepResults.first(where: { $0.precision == .fp16 && $0.target == r.target }),
               let int8 = stepResults.first(where: { $0.precision == .int8 && $0.target == r.target }),
               fp16.tops > 0 {
                let speedup = int8.tops / fp16.tops
                HStack(spacing: 4) {
                    Image(systemName: "bolt.fill")
                        .foregroundColor(.green)
                        .font(.caption2)
                    Text(String(format: "INT8 Speedup at this step: %.2fx (FP16: %.1f TOPS → INT8: %.1f TOPS)", speedup, fp16.tops, int8.tops))
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
                .padding(.top, 2)
            }
            
            if r.computeCycles > 0 || r.nominalCycles > 0 || r.dmaRwBytes > 0 {
                Divider()
                HStack(spacing: 16) {
                    VStack(alignment: .leading) {
                        Text("Compute").font(.caption2).foregroundColor(.secondary)
                        Text(r.formattedComputeCycles).font(.subheadline).fontWeight(.bold).foregroundColor(.blue)
                    }
                    VStack(alignment: .leading) {
                        Text("Stalls").font(.caption2).foregroundColor(.secondary)
                        Text(r.formattedStalls).font(.subheadline).fontWeight(.bold).foregroundColor(.orange)
                    }
                    VStack(alignment: .leading) {
                        Text("DMA").font(.caption2).foregroundColor(.secondary)
                        Text(r.formattedDMA).font(.subheadline).fontWeight(.bold).foregroundColor(.green)
                    }
                    VStack(alignment: .leading) {
                        Text("ALU Sat").font(.caption2).foregroundColor(.secondary)
                        Text(r.formattedSaturation).font(.subheadline).fontWeight(.bold).foregroundColor(.indigo)
                    }
                    if r.effectiveClockGhz > 0 {
                        VStack(alignment: .leading) {
                            Text("Clock").font(.caption2).foregroundColor(.secondary)
                            Text(String(format: "%.2f GHz", r.effectiveClockGhz)).font(.subheadline).fontWeight(.bold)
                        }
                    }
                }
            }
        }
        .padding()
        .background(Color(.secondarySystemBackground))
        .cornerRadius(12)
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(r.precision.themeColor.opacity(0.5), lineWidth: 1.5)
        )
    }
    
    // MARK: - Comparison Table Card
    private var comparisonTableCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Data Table (\(filteredResults.count) points)")
                .font(.headline)
            
            VStack(spacing: 0) {
                // Table Header
                HStack {
                    Text("Sweep Step").font(.caption).fontWeight(.bold).frame(width: 90, alignment: .leading)
                    Text("Device").font(.caption).fontWeight(.bold).frame(width: 70, alignment: .leading)
                    Text("Latency").font(.caption).fontWeight(.bold).frame(maxWidth: .infinity, alignment: .trailing)
                    Text("TOPS").font(.caption).fontWeight(.bold).frame(maxWidth: .infinity, alignment: .trailing)
                }
                .padding(.vertical, 8)
                .background(Color(.tertiarySystemBackground))
                
                Divider()
                
                // Rows
                ForEach(filteredResults) { item in
                    HStack {
                        Text(item.sweepLabel)
                            .font(.caption2)
                            .frame(width: 90, alignment: .leading)
                        
                        Text("\(item.target.shortName) \(item.precision.rawValue)")
                            .font(.caption2)
                            .fontWeight(.medium)
                            .foregroundColor(item.precision.themeColor)
                            .frame(width: 70, alignment: .leading)
                        
                        Text(item.formattedDuration)
                            .font(.caption2)
                            .frame(maxWidth: .infinity, alignment: .trailing)
                        
                        Text(item.formattedTOPS)
                            .font(.caption2)
                            .fontWeight(.semibold)
                            .frame(maxWidth: .infinity, alignment: .trailing)
                    }
                    .padding(.vertical, 6)
                    Divider()
                }
            }
            .background(Color(.tertiarySystemBackground).opacity(0.5))
            .cornerRadius(8)
        }
        .padding()
        .background(Color(.secondarySystemBackground))
        .cornerRadius(16)
    }
}
