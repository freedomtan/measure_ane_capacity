import Foundation
import Metal
import MetalPerformanceShadersGraph

// Runtime access to private API
extension MPSGraphDevice {
    static var aneDevice: MPSGraphDevice? {
        let selector = Selector("ANEDevice")
        if MPSGraphDevice.responds(to: selector) {
            return MPSGraphDevice.perform(selector).takeUnretainedValue() as? MPSGraphDevice
        }
        return nil
    }
}

func runBench(device: MTLDevice, useANE: Bool, dataType: MPSDataType, name: String) {
    let graph = MPSGraph()

    // Settings for high-throughput
    let B: NSNumber = 1
    let H: NSNumber = 256
    let W: NSNumber = 256
    let Ci: NSNumber = 128
    let Co: NSNumber = 128
    let K: NSNumber = 3
    let L: Int = 20

    let inShape = [B, Ci, H, W]
    let wShape = [Co, Ci, K, K]

    let input = graph.placeholder(shape: inShape, dataType: dataType, name: "in")
    var cur = input

    let elementSize = (dataType == .float16) ? 2 : 1
    
    // Weights allocation
    let wLength = Co.intValue * Ci.intValue * K.intValue * K.intValue * elementSize
    let wData = NSMutableData(length: wLength)!
    let w = graph.constant(wData as Data, shape: wShape, dataType: dataType)

    for _ in 0..<L {
         let d = MPSGraphConvolution2DOpDescriptor(
            strideInX: 1,
            strideInY: 1,
            dilationRateInX: 1,
            dilationRateInY: 1,
            groups: 1,
            paddingStyle: .TF_SAME,
            dataLayout: .NCHW,
            weightsLayout: .OIHW)!
            
        cur = graph.convolution2D(cur, weights: w, descriptor: d, name: nil)
    }
    
    var mDev: MPSGraphDevice?
    if useANE {
        guard let ane = MPSGraphDevice.aneDevice else {
             print("[\(name)] Skipped: ANE not supported.")
             return
        }
        mDev = ane
    } else {
        mDev = MPSGraphDevice(mtlDevice: device)
    }

    let feeds = [input: MPSGraphShapedType(shape: inShape, dataType: dataType)]
    
    let cd = MPSGraphCompilationDescriptor()
    cd.optimizationLevel = useANE ? .level1 : .level0
    
    // Fix non-optional return from compile
    let exe = graph.compile(with: mDev, feeds: feeds, targetTensors: [cur], targetOperations: nil, compilationDescriptor: cd)
    
    let bufferLength = B.intValue * H.intValue * W.intValue * Ci.intValue * elementSize
    let iBuf = device.makeBuffer(length: bufferLength, options: [])!
    
    // Fix argument label (likely just '(_:shape:dataType:)')
    let iData = MPSGraphTensorData(iBuf, shape: inShape, dataType: dataType)
    
    let q = device.makeCommandQueue()!
    let ed = MPSGraphExecutableExecutionDescriptor()
    ed.waitUntilCompleted = true
    
    // Warmup
    exe.run(with: q, inputs: [iData], results: nil, executionDescriptor: ed)
    
    let iterations = 20
    let start = clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW)
    
    for _ in 0..<iterations {
        exe.run(with: q, inputs: [iData], results: nil, executionDescriptor: ed)
    }
    
    let end = clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW)
    let duration = Double(end - start) / 1e9
    let avg = duration / Double(iterations)
    
    let tops = (2.0 * Double(B.intValue * H.intValue * W.intValue * Ci.intValue * Co.intValue * K.intValue * K.intValue * L)) / (avg * 1e12)
    
    print(String(format: "[%@] Avg: %.2f ms, Speed: %.4f TOPS", name, avg * 1000.0, tops))
}

// Top level code (remove @main struct)
if let device = MTLCreateSystemDefaultDevice() {
    runBench(device: device, useANE: false, dataType: .float16, name: "GPU FP16")
    runBench(device: device, useANE: true, dataType: .float16, name: "ANE FP16")
    
    // GPU INT8 not supported
    runBench(device: device, useANE: true, dataType: .int8, name: "ANE INT8")
} else {
    print("Error: Metal not supported")
}
