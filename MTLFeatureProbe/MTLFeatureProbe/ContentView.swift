import SwiftUI
import Metal
import UIKit

struct ContentView: View {
    private let device = MTLCreateSystemDefaultDevice()

    @State private var deviceSupports: [SelectorResult] = []
    @State private var familySupports: [SelectorResult] = []
    @State private var featureTable: [TableEntry] = []
    @State private var search = ""
    @State private var loaded = false

    var body: some View {
        NavigationStack {
            List {
                if let device {
                    Section("Device") {
                        LabeledContent("Name", value: device.name)
                    }
                }
                Section("-[MTLDevice supports*] (\(filtered(deviceSupports).count)/\(deviceSupports.count))") {
                    ForEach(filtered(deviceSupports)) { row in
                        rowView(row.name, row.value)
                    }
                }
                Section("-[MTLDeviceFeatureQueries familySupports*] (\(filtered(familySupports).count)/\(familySupports.count))") {
                    ForEach(filtered(familySupports)) { row in
                        rowView(row.name, row.value)
                    }
                }
                Section("_featureInfo[] diagnostic table (\(filteredTable.count)/\(featureTable.count))") {
                    ForEach(filteredTable) { row in
                        VStack(alignment: .leading, spacing: 2) {
                            rowView("[\(row.index)] \(row.name)", row.supported)
                            Text("requirement = \(row.requirement)")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
            }
            .searchable(text: $search, prompt: "Filter by name")
            .navigationTitle("MTL Feature Probe")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Copy Report", action: copyReport)
                }
            }
            .task {
                guard !loaded, let device else { return }
                loaded = true
                deviceSupports = MTLFeatureProbe.probeDeviceSupports(device)
                familySupports = MTLFeatureProbe.probeFamilySupports(device)
                featureTable = MTLFeatureProbe.probeFeatureInfoTable(device)

                if ProcessInfo.processInfo.arguments.contains("--dump") {
                    print("[MTLFeatureProbe] Device: \(device.name)")
                    for row in deviceSupports where row.name.localizedCaseInsensitiveContains("MXU") || row.name.localizedCaseInsensitiveContains("FP8") {
                        print("[MTLFeatureProbe] deviceSupports.\(row.name) = \(row.value)")
                    }
                    for row in familySupports where row.name.localizedCaseInsensitiveContains("MXU") || row.name.localizedCaseInsensitiveContains("FP8") {
                        print("[MTLFeatureProbe] familySupports.\(row.name) = \(row.value)")
                    }
                    for row in featureTable where row.name.localizedCaseInsensitiveContains("MXU") || row.name.localizedCaseInsensitiveContains("8-bit floating") {
                        print("[MTLFeatureProbe] featureTable[\(row.index)] \(row.name) req=\(row.requirement) supported=\(row.supported)")
                    }
                    exit(0)
                }
            }
        }
    }

    private func filtered(_ rows: [SelectorResult]) -> [SelectorResult] {
        guard !search.isEmpty else { return rows }
        return rows.filter { $0.name.localizedCaseInsensitiveContains(search) }
    }

    private var filteredTable: [TableEntry] {
        guard !search.isEmpty else { return featureTable }
        return featureTable.filter { $0.name.localizedCaseInsensitiveContains(search) }
    }

    @ViewBuilder
    private func rowView(_ name: String, _ value: Bool) -> some View {
        HStack {
            Text(name)
                .font(.system(.body, design: .monospaced))
            Spacer()
            Image(systemName: value ? "checkmark.circle.fill" : "xmark.circle")
                .foregroundStyle(value ? .green : .secondary)
        }
    }

    private func copyReport() {
        var lines: [String] = []
        if let device {
            lines.append("Device: \(device.name)")
            lines.append("")
        }
        lines.append("== MTLDevice supports* ==")
        lines += deviceSupports.map { "\($0.value ? "YES" : "NO ")  \($0.name)" }
        lines.append("")
        lines.append("== MTLDeviceFeatureQueries familySupports* ==")
        lines += familySupports.map { "\($0.value ? "YES" : "NO ")  \($0.name)" }
        lines.append("")
        lines.append("== _featureInfo[] ==")
        lines += featureTable.map { "[\($0.index)] \($0.name)  req=\($0.requirement) supported=\($0.supported ? 1 : 0)" }
        UIPasteboard.general.string = lines.joined(separator: "\n")
    }
}
