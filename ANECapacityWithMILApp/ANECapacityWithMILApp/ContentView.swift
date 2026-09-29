import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel = BenchmarkViewModel()
    @State private var selectedTab: Int = {
        if let idx = CommandLine.arguments.firstIndex(of: "-selectedTab"),
           idx + 1 < CommandLine.arguments.count,
           let tab = Int(CommandLine.arguments[idx + 1]) {
            return tab
        }
        return 0
    }()
    
    var body: some View {
        TabView(selection: $selectedTab) {
            BenchmarkView(viewModel: viewModel)
                .tabItem {
                    Label("Benchmark", systemImage: "gauge.with.needle")
                }
                .tag(0)
            
            ChartsView(viewModel: viewModel)
                .tabItem {
                    Label("Figures", systemImage: "chart.xyaxis.line")
                }
                .tag(1)
            
            HistoryView(viewModel: viewModel)
                .tabItem {
                    Label("History", systemImage: "list.bullet.rectangle")
                }
                .tag(2)
            
            DeviceInfoView(viewModel: viewModel)
                .tabItem {
                    Label("Info", systemImage: "cpu")
                }
                .tag(3)
        }
    }
}

#Preview {
    ContentView()
}
