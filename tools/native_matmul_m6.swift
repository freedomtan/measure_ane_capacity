// Standalone macOS CLI port of ANECapacityApp/NativeInt8GEMMProbe.swift, for
// running the native MPSNDArrayMatrixMultiplication / MPSNDArrayQuantizedMatrixMultiplication
// FP16 vs INT8-affine vs FP8-E4M3 GEMM comparison directly on a Mac (no app/Xcode project
// needed -- the main ANECapacityApp project is iphoneos-only).
//
// L=20-layer chained variant: for FP16/FP8 (same dtype in/out, square MxNxK so the
// output of one layer is a valid input to the next), encodes 20 dependent GEMMs per
// command buffer, ping-ponging between two destination buffers, matching the
// 20-layer chain methodology used by the MPSGraph matmulDimensions sweep elsewhere
// in this repo. INT8-affine's output is float16 (dequantized), not int8, so a true
// chain would need a requantize step between layers that doesn't exist as a plain
// MPSNDArray op outside MPSGraph -- kept at L=1 for that path, reported as such.
import Foundation
import Metal
import MetalPerformanceShaders

guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
    print("[Int8GEMMProbe] No Metal device/queue"); exit(1)
}
print("[Int8GEMMProbe] Device: \(device.name)")

let M = 8192, N = 8192, K = 8192
let L = 20

func ndArray(dataType: MPSDataType, shape: [Int]) -> MPSNDArray {
    let desc = MPSNDArrayDescriptor(dataType: dataType, shape: shape.map { NSNumber(value: $0) })
    return MPSNDArray(device: device, descriptor: desc)
}

func fillInt8(_ arr: MPSNDArray, count: Int, seed: UInt64) {
    var buf = [Int8](repeating: 0, count: count)
    var state = seed
    for i in 0..<count {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        let v = Int8(truncatingIfNeeded: Int((state >> 56) & 0x07) - 4)
        buf[i] = v == 0 ? 1 : v
    }
    buf.withUnsafeMutableBytes { raw in arr.writeBytes(raw.baseAddress!, strideBytes: nil) }
}
func fillHalfConstant(_ arr: MPSNDArray, count: Int, value: Float16) {
    var buf = [Float16](repeating: value, count: count)
    buf.withUnsafeMutableBytes { raw in arr.writeBytes(raw.baseAddress!, strideBytes: nil) }
}
func fillInt8Constant(_ arr: MPSNDArray, count: Int, value: Int8) {
    var buf = [Int8](repeating: value, count: count)
    buf.withUnsafeMutableBytes { raw in arr.writeBytes(raw.baseAddress!, strideBytes: nil) }
}

let lhs = ndArray(dataType: .int8, shape: [1, 1, M, K])
let rhs = ndArray(dataType: .int8, shape: [1, 1, K, N])
fillInt8(lhs, count: M * K, seed: 0x9E3779B97F4A7C15)
fillInt8(rhs, count: K * N, seed: 0xC2B2AE3D27D4EB4F)

let lhsScale = ndArray(dataType: .float16, shape: [1, 1, 1, 1])
let lhsZero = ndArray(dataType: .int8, shape: [1, 1, 1, 1])
let rhsScale = ndArray(dataType: .float16, shape: [1, 1, 1, 1])
let rhsZero = ndArray(dataType: .int8, shape: [1, 1, 1, 1])
fillHalfConstant(lhsScale, count: 1, value: Float16(1.0 / 32.0))
fillInt8Constant(lhsZero, count: 1, value: 0)
fillHalfConstant(rhsScale, count: 1, value: Float16(1.0 / 32.0))
fillInt8Constant(rhsZero, count: 1, value: 0)

guard #available(macOS 15.0, *) else {
    print("[Int8GEMMProbe] MPSNDArrayQuantizedMatrixMultiplication requires macOS 15+")
    exit(1)
}

let leftQuantDesc = MPSNDArrayAffineQuantizationDescriptor(dataType: .int8, hasZeroPoint: true, hasMinValue: false)
let rightQuantDesc = MPSNDArrayAffineQuantizationDescriptor(dataType: .int8, hasZeroPoint: true, hasMinValue: false)
let qmm = MPSNDArrayQuantizedMatrixMultiplication(
    device: device,
    leftQuantizationDescriptor: leftQuantDesc,
    rightQuantizationDescriptor: rightQuantDesc
)
let sourceArrays: [MPSNDArray] = [lhs, rhs, lhsScale, lhsZero, rhsScale, rhsZero]

func runTimed(label: String, layers: Int, dest: MPSNDArray, destIsFloat32: Bool = false, destIsRawByte: Bool = false, body: (MTLCommandBuffer) -> Void) {
    do {
        let cb = queue.makeCommandBuffer()!
        body(cb); cb.commit(); cb.waitUntilCompleted()
    }
    Thread.sleep(forTimeInterval: 2.0)

    let timedIterations = 10
    let start = clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW)
    var lastCb: MTLCommandBuffer? = nil
    for _ in 0..<timedIterations {
        guard let cb = queue.makeCommandBuffer() else { break }
        body(cb)
        cb.commit()
        lastCb = cb
    }
    lastCb?.waitUntilCompleted()
    let end = clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW)
    let avgMs = Double(end - start) / 1e6 / Double(timedIterations)
    let flops = 2.0 * Double(M) * Double(N) * Double(K) * Double(layers)
    let tops = flops / (avgMs / 1000.0) / 1e12
    print("[Int8GEMMProbe][\(label)] L=\(layers) Avg: \(avgMs) ms, \(tops) TOPS")

    if destIsRawByte {
        var out = [UInt8](repeating: 0, count: M * N)
        out.withUnsafeMutableBytes { raw in dest.readBytes(raw.baseAddress!, strideBytes: nil) }
        let nonZero = out.filter { $0 != 0 }.count
        print("[Int8GEMMProbe][\(label)] nonZero=\(nonZero)/\(out.count) sampleBytes=\(out.prefix(8))")
    } else if destIsFloat32 {
        var out = [Float32](repeating: 0, count: M * N)
        out.withUnsafeMutableBytes { raw in dest.readBytes(raw.baseAddress!, strideBytes: nil) }
        let nonZero = out.filter { $0 != 0 }.count
        print("[Int8GEMMProbe][\(label)] nonZero=\(nonZero)/\(out.count) sample=\(out.prefix(8))")
    } else {
        var out = [Float16](repeating: 0, count: M * N)
        out.withUnsafeMutableBytes { raw in dest.readBytes(raw.baseAddress!, strideBytes: nil) }
        let nonZero = out.filter { $0 != 0 }.count
        print("[Int8GEMMProbe][\(label)] nonZero=\(nonZero)/\(out.count) sample=\(out.prefix(8).map { Float($0) })")
    }
}

// INT8-affine: kept single-shot (L=1) -- output dtype (float16) != input dtype
// (int8), so a true chain needs a requantize step this plain-NDArray API doesn't
// provide.
let int8GemmDest = ndArray(dataType: .float16, shape: [1, 1, M, N])
runTimed(label: "GPU Native INT8-Affine GEMM (MPSNDArrayQuantizedMatrixMultiplication)", layers: 1, dest: int8GemmDest) { cb in
    qmm.encode(to: cb, sourceArrays: sourceArrays, destinationArray: int8GemmDest)
}

// FP16: homogeneous dtype in/out, square MxNxK -> chainable. Ping-pong between two
// dest buffers, feeding each layer's output as the next layer's left operand.
let lhsF = ndArray(dataType: .float16, shape: [1, 1, M, K])
let rhsF = ndArray(dataType: .float16, shape: [1, 1, K, N])
var lhsFBuf = [Float16](repeating: 0, count: M * K)
var rhsFBuf = [Float16](repeating: 0, count: K * N)
for i in 0..<lhsFBuf.count { lhsFBuf[i] = Float16(((i % 7) - 3)) * 0.1 + 0.05 }
for i in 0..<rhsFBuf.count { rhsFBuf[i] = Float16(((i % 5) - 2)) * 0.1 + 0.05 }
lhsFBuf.withUnsafeMutableBytes { raw in lhsF.writeBytes(raw.baseAddress!, strideBytes: nil) }
rhsFBuf.withUnsafeMutableBytes { raw in rhsF.writeBytes(raw.baseAddress!, strideBytes: nil) }

let mm = MPSNDArrayMatrixMultiplication(device: device, sourceCount: 2)
let fp16DestA = ndArray(dataType: .float16, shape: [1, 1, M, N])
let fp16DestB = ndArray(dataType: .float16, shape: [1, 1, M, N])
runTimed(label: "GPU FP16 GEMM (MPSNDArrayMatrixMultiplication, baseline)", layers: L, dest: fp16DestB) { cb in
    var cur = lhsF
    let pingPong = [fp16DestA, fp16DestB]
    for layer in 0..<L {
        let dst = pingPong[layer % 2]
        mm.encode(to: cb, sourceArrays: [cur, rhsF], destinationArray: dst)
        cur = dst
    }
}

func fillFP8E4M3(_ arr: MPSNDArray, count: Int) {
    let patterns: [UInt8] = [0x38, 0x30, 0x3C, 0x28, 0x3A, 0x34, 0x3E, 0x20]
    var buf = [UInt8](repeating: 0, count: count)
    for i in 0..<count { buf[i] = patterns[i % patterns.count] }
    buf.withUnsafeMutableBytes { raw in arr.writeBytes(raw.baseAddress!, strideBytes: nil) }
}

if #available(macOS 26.0, *) {
    let lhs8 = ndArray(dataType: .float8e4m3, shape: [1, 1, M, K])
    let rhs8 = ndArray(dataType: .float8e4m3, shape: [1, 1, K, N])
    fillFP8E4M3(lhs8, count: M * K)
    fillFP8E4M3(rhs8, count: K * N)
    let fp8DestA = ndArray(dataType: .float8e4m3, shape: [1, 1, M, N])
    let fp8DestB = ndArray(dataType: .float8e4m3, shape: [1, 1, M, N])
    let mm8 = MPSNDArrayMatrixMultiplication(device: device, sourceCount: 2)
    runTimed(label: "GPU Native FP8-E4M3 GEMM (MPSNDArrayMatrixMultiplication)", layers: L, dest: fp8DestB, destIsRawByte: true) { cb in
        var cur = lhs8
        let pingPong = [fp8DestA, fp8DestB]
        for layer in 0..<L {
            let dst = pingPong[layer % 2]
            mm8.encode(to: cb, sourceArrays: [cur, rhs8], destinationArray: dst)
            cur = dst
        }
    }
} else {
    print("[Int8GEMMProbe] FP8 requires macOS 26+")
}

print("[Int8GEMMProbe] Done.")
