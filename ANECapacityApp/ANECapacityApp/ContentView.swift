import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel = BenchmarkViewModel()
    
    var body: some View {
        TabView {
            BenchmarkView(viewModel: viewModel)
                .tabItem {
                    Label("Benchmark", systemImage: "gauge.with.needle")
                }
            
            ChartsView(viewModel: viewModel)
                .tabItem {
                    Label("Figures", systemImage: "chart.xyaxis.line")
                }
            
            HistoryView(viewModel: viewModel)
                .tabItem {
                    Label("History", systemImage: "list.bullet.rectangle")
                }
            
            DeviceInfoView(viewModel: viewModel)
                .tabItem {
                    Label("Info", systemImage: "cpu")
                }
        }
    }
}

#Preview {
    ContentView()
}
