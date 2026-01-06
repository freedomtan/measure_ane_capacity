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

func checkType() {
    let graph = MPSGraph()
    
    let inShape = [1, 32, 64, 64] as [NSNumber]
    let wShape = [32, 32, 3, 3] as [NSNumber]
    
    // Create Int8 placeholders/constants
    let input = graph.placeholder(shape: inShape, dataType: .int8, name: "in")
    
    // Weights
    let wLength = 32 * 32 * 3 * 3
    let wData = NSMutableData(length: wLength)!
    let w = graph.constant(wData as Data, shape: wShape, dataType: .int8)
    
    // Conv
    let d = MPSGraphConvolution2DOpDescriptor(
        strideInX: 1, strideInY: 1,
        dilationRateInX: 1, dilationRateInY: 1,
        groups: 1, paddingStyle: .TF_SAME,
        dataLayout: .NCHW, weightsLayout: .OIHW)!
    
    var cur = graph.convolution2D(input, weights: w, descriptor: d, name: nil)
    
    // Attempt to cast to FP16 to verify if this promotes the chain to use Dequantize hardware
    cur = graph.cast(cur, to: .float16, name: "cast")
    
    print("Input DataType: \(input.dataType.rawValue)")
    print("Output DataType: \(cur.dataType.rawValue)")
    
    if cur.dataType == .float16 {
        print("Result is Float16 (Correct for Mixed Precision Quantization)")
    } else if cur.dataType == .float32 {
        print("Result is Float32 (Correct for Mixed Precision Quantization)")
    } else if cur.dataType == .int32 {
        print("Result is Int32 (Correct for Integer Quantization)")
    } else if cur.dataType == .int8 {
        print("Result is Int8 (Potentially wrapping/overflowing - check if this is desired)")
    } else {
        print("Result is Unknown: \(cur.dataType.rawValue)")
    }
}

checkType()
