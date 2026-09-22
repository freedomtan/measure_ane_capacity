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

func fillNonZeroData(buffer: UnsafeMutableRawPointer, byteCount: Int, dataType: MPSDataType, seed: UInt64 = 0x5EED5EED5EED5EED) {
    guard byteCount > 0 else { return }
    var state = seed
    if dataType == .float16 {
        let ptr = buffer.bindMemory(to: UInt16.self, capacity: byteCount / 2)
        let count = byteCount / 2
        for i in 0..<count {
            state ^= state << 13
            state ^= state >> 7
            state ^= state << 17
            ptr[i] = (state & 1) != 0 ? 0xA800 : 0x2800 // -0.03125, +0.03125
        }
    } else {
        let ptr = buffer.bindMemory(to: Int8.self, capacity: byteCount)
        for i in 0..<byteCount {
            state ^= state << 13
            state ^= state >> 7
            state ^= state << 17
            ptr[i] = (state & 1) != 0 ? -1 : 1
        }
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
    fillNonZeroData(buffer: wData.mutableBytes, byteCount: wLength, dataType: dataType, seed: 0x5EED5EED5EED5EED)
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
            weightsLayout: .OIHW
        )!
            
        cur = graph.convolution2D(cur, weights: w, descriptor: d, name: nil)
        
        // Realistic Quantized flow: Int8 -> (Conv) -> Int32/FP16 -> (Select/Scale) -> Int8
        // We simulate this by casting to FP16 and back to Int8 to force the graph to handle dequant/quant
        if dataType == .int8 {
            let fp = graph.cast(cur, to: .float16, name: "dequant")
            cur = graph.cast(fp, to: .int8, name: "requant")
        }
    }
    
    let mDev = MPSGraphDevice(mtlDevice: device)
    let feeds = [input: MPSGraphShapedType(shape: inShape, dataType: dataType)]
    
    let cd = MPSGraphCompilationDescriptor()
    if useANE {
        cd.optimizationLevel = .level1
        if cd.responds(to: Selector(("setPreferredDevice:"))) {
            cd.setValue(2, forKey: "preferredDevice")
        }
    } else {
        cd.optimizationLevel = .level0
    }
    
    let exe = graph.compile(with: mDev, feeds: feeds, targetTensors: [cur], targetOperations: nil, compilationDescriptor: cd)
    
    let bufferLength = B.intValue * H.intValue * W.intValue * Ci.intValue * elementSize
    let iBuf = device.makeBuffer(length: bufferLength, options: [])!
    fillNonZeroData(buffer: iBuf.contents(), byteCount: bufferLength, dataType: dataType, seed: 0x9E3779B97F4A7C15)
    
    let iData = MPSGraphTensorData(iBuf, shape: inShape, dataType: dataType)

    let oBuf = device.makeBuffer(length: bufferLength, options: [])!
    let oData = MPSGraphTensorData(oBuf, shape: inShape, dataType: dataType)
    
    let q = device.makeCommandQueue()!
    let ed = MPSGraphExecutableExecutionDescriptor()
    ed.waitUntilCompleted = true
    
    // Warmup
    exe.run(with: q, inputs: [iData], results: [oData], executionDescriptor: ed)
    
    let iterations = 20
    let start = clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW)
    
    for _ in 0..<iterations {
        exe.run(with: q, inputs: [iData], results: [oData], executionDescriptor: ed)
    }
    
    let end = clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW)
    let duration = Double(end - start) / 1e9
    let avg = duration / Double(iterations)
    
    let tops = (2.0 * Double(B.intValue * H.intValue * W.intValue * Ci.intValue * Co.intValue * K.intValue * K.intValue * L)) / (avg * 1e12)
    
    let totalElem = B.intValue * H.intValue * W.intValue * Co.intValue
    var zeroCount = 0
    if dataType == .float16 {
        let ptr = oBuf.contents().bindMemory(to: UInt16.self, capacity: totalElem)
        for k in 0..<totalElem {
            if ptr[k] == 0x0000 || ptr[k] == 0x8000 { zeroCount += 1 }
        }
        print(String(format: "[%@] Avg: %.2f ms, Speed: %.4f TOPS | Check: [0x%04x, 0x%04x, 0x%04x, 0x%04x] (zeros: %d/%d)",
                     name, avg * 1000.0, tops, ptr[0], ptr[1], ptr[2], ptr[3], zeroCount, totalElem))
    } else {
        let ptr = oBuf.contents().bindMemory(to: Int8.self, capacity: totalElem)
        for k in 0..<totalElem {
            if ptr[k] == 0 { zeroCount += 1 }
        }
        print(String(format: "[%@] Avg: %.2f ms, Speed: %.4f TOPS | Check: [%d, %d, %d, %d] (zeros: %d/%d)",
                     name, avg * 1000.0, tops, ptr[0], ptr[1], ptr[2], ptr[3], zeroCount, totalElem))
    }
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
