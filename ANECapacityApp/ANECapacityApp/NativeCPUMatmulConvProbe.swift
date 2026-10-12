import Foundation
import Metal
import MetalPerformanceShadersGraph

/// On-device port of measure_matmul_universal.m / measure_conv_universal.m's
/// kTargetCPU path (docs/CPU_Matrix_Convolution_Capability.md Finding 1/3):
/// does +[MPSGraphDevice CPUDevice] crash the same way here, or is that a
/// macOS-27-MPSGraph-build-specific bug? Triggered by --native-cpu-matmul /
/// --native-cpu-conv (see ANECapacityApp.swift) -- run as two separate
/// launches, not one, since both the macOS CLI's compile-time crash and this
/// probe's execution-time crash abort the whole process; isolating matmul
/// from conv the same way the CLI isolates CPU from GPU/ANE.
///
/// Deliberately tiny shapes (64^3 matmul, 32x32x8ch conv, 1 layer): the
/// full-size shapes from the macOS CLI took minutes to even finish
/// *compiling* for CPUDevice on-device with no console output in between
/// (unbuffered stdout confirmed it wasn't a true hang, just a very slow
/// compile) -- these shapes compile in a few seconds so the crash (which
/// turned out to be at a different pipeline stage than on macOS -- see the
/// doc) surfaces quickly instead.
enum NativeCPUMatmulConvProbe {
    private static func cpuDevice() -> MPSGraphDevice? {
        let sel = Selector(("CPUDevice"))
        guard MPSGraphDevice.responds(to: sel) else {
            print("[CPUProbe] +[MPSGraphDevice CPUDevice] not available on this OS/SDK")
            return nil
        }
        typealias Fn = @convention(c) (AnyObject, Selector) -> AnyObject?
        let imp = unsafeBitCast(class_getMethodImplementation(object_getClass(MPSGraphDevice.self), sel), to: Fn.self)
        return imp(MPSGraphDevice.self, sel) as? MPSGraphDevice
    }

    static func runMatmul() {
        setvbuf(stdout, nil, _IONBF, 0)
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            print("[CPUProbe] No Metal device/queue"); return
        }
        guard let cpuDev = cpuDevice() else { return }
        print("[CPUProbe] Device: \(device.name), targeting MPSGraphDevice.CPUDevice for matmul")

        let graph = MPSGraph()
        let B = 1, M = 64, K = 64, N = 64, L = 1
        let inShape = [B, 1, M, K].map { NSNumber(value: $0) }
        let wShape = [B, 1, K, N].map { NSNumber(value: $0) }

        let input = graph.placeholder(shape: inShape, dataType: .float16, name: "in")
        var cur = input

        var wBuf = [UInt16](repeating: 0, count: B * K * N)
        for i in 0..<wBuf.count { wBuf[i] = (i & 1 != 0) ? 0xA800 : 0x2800 }
        let wData = Data(bytes: wBuf, count: wBuf.count * 2)
        let w = graph.constant(wData, shape: wShape, dataType: .float16)

        for i in 0..<L {
            cur = graph.matrixMultiplication(primary: cur, secondary: w, name: "matmul_\(i)")
        }

        let feeds = [input: MPSGraphShapedType(shape: inShape, dataType: .float16)]
        let cd = MPSGraphCompilationDescriptor()
        cd.optimizationLevel = .level0

        print("[CPUProbe] Compiling matmul graph for CPUDevice...")
        let exe = graph.compile(with: cpuDev, feeds: feeds, targetTensors: [cur], targetOperations: nil, compilationDescriptor: cd)
        print("[CPUProbe] Compile returned without crashing: \(exe)")

        let iBuf = device.makeBuffer(length: B * M * K * 2, options: [])!
        let iData = MPSGraphTensorData(iBuf, shape: inShape, dataType: .float16)
        let oBuf = device.makeBuffer(length: B * M * N * 2, options: [])!
        let oData = MPSGraphTensorData(oBuf, shape: inShape, dataType: .float16)

        let ed = MPSGraphExecutableExecutionDescriptor()
        ed.waitUntilCompleted = true
        exe.run(with: queue, inputs: [iData], results: [oData], executionDescriptor: ed)
        print("[CPUProbe] Matmul run succeeded, no crash.")
    }

    static func runConv() {
        setvbuf(stdout, nil, _IONBF, 0)
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            print("[CPUProbe] No Metal device/queue"); return
        }
        guard let cpuDev = cpuDevice() else { return }
        print("[CPUProbe] Device: \(device.name), targeting MPSGraphDevice.CPUDevice for conv")

        let graph = MPSGraph()
        let B = 1, H = 32, W = 32, Ci = 8, Co = 8, K = 3, L = 1
        let inShape = [B, Ci, H, W].map { NSNumber(value: $0) }
        let wShape = [Co, Ci, K, K].map { NSNumber(value: $0) }

        let input = graph.placeholder(shape: inShape, dataType: .float16, name: "in")
        var cur = input

        var wBuf = [UInt16](repeating: 0, count: Co * Ci * K * K)
        for i in 0..<wBuf.count { wBuf[i] = (i & 1 != 0) ? 0xA800 : 0x2800 }
        let wData = Data(bytes: wBuf, count: wBuf.count * 2)
        let w = graph.constant(wData, shape: wShape, dataType: .float16)

        let desc = MPSGraphConvolution2DOpDescriptor(
            strideInX: 1, strideInY: 1, dilationRateInX: 1, dilationRateInY: 1,
            groups: 1, paddingStyle: .TF_SAME, dataLayout: .NCHW, weightsLayout: .OIHW
        )!

        for i in 0..<L {
            cur = graph.convolution2D(cur, weights: w, descriptor: desc, name: "conv_\(i)")
        }

        let feeds = [input: MPSGraphShapedType(shape: inShape, dataType: .float16)]
        let cd = MPSGraphCompilationDescriptor()
        cd.optimizationLevel = .level0

        print("[CPUProbe] Compiling conv graph for CPUDevice...")
        let exe = graph.compile(with: cpuDev, feeds: feeds, targetTensors: [cur], targetOperations: nil, compilationDescriptor: cd)
        print("[CPUProbe] Compile returned without crashing: \(exe)")

        let iBuf = device.makeBuffer(length: B * H * W * Ci * 2, options: [])!
        let iData = MPSGraphTensorData(iBuf, shape: inShape, dataType: .float16)
        let oBuf = device.makeBuffer(length: B * H * W * Co * 2, options: [])!
        let oData = MPSGraphTensorData(oBuf, shape: inShape, dataType: .float16)

        let ed = MPSGraphExecutableExecutionDescriptor()
        ed.waitUntilCompleted = true
        exe.run(with: queue, inputs: [iData], results: [oData], executionDescriptor: ed)
        print("[CPUProbe] Conv run succeeded, no crash.")
    }
}
