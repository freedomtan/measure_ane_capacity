import Foundation
import Metal
import MetalPerformanceShaders

/// Standalone probe for whether the public, documented
/// `MPSNDArrayQuantizedMatrixMultiplication` int8-affine GEMM path is faster
/// or slower than a plain FP16 GEMM on this device's actual GPU. Triggered by
/// `--native-int8-gemm` on launch (see ANECapacityApp.swift). Not part of the
/// normal ConvDimensions/BenchmarkResult pipeline -- this is a standalone
/// sanity check for docs/MPSGraph_MXU_SDPA_Investigation.md Finding 4h/4i.
enum NativeInt8GEMMProbe {
    static func run() {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            print("[Int8GEMMProbe] No Metal device/queue"); return
        }
        print("[Int8GEMMProbe] Device: \(device.name)")

        let M = 2048, N = 2048, K = 2048
        let iterations = 20

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

        guard #available(iOS 18.0, *) else {
            print("[Int8GEMMProbe] MPSNDArrayQuantizedMatrixMultiplication requires iOS 18+")
            return
        }

        let leftQuantDesc = MPSNDArrayAffineQuantizationDescriptor(dataType: .int8, hasZeroPoint: true, hasMinValue: false)
        let rightQuantDesc = MPSNDArrayAffineQuantizationDescriptor(dataType: .int8, hasZeroPoint: true, hasMinValue: false)
        let qmm = MPSNDArrayQuantizedMatrixMultiplication(
            device: device,
            leftQuantizationDescriptor: leftQuantDesc,
            rightQuantizationDescriptor: rightQuantDesc
        )
        let sourceArrays: [MPSNDArray] = [lhs, rhs, lhsScale, lhsZero, rhsScale, rhsZero]

        func runTimed(label: String, dest: MPSNDArray, body: (MTLCommandBuffer) -> Void) {
            do {
                let cb = queue.makeCommandBuffer()!
                body(cb); cb.commit(); cb.waitUntilCompleted()
            }
            let start = clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW)
            for _ in 0..<iterations {
                let cb = queue.makeCommandBuffer()!
                body(cb); cb.commit(); cb.waitUntilCompleted()
            }
            let end = clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW)
            let avgMs = Double(end - start) / 1e6 / Double(iterations)
            let flops = 2.0 * Double(M) * Double(N) * Double(K)
            let tops = flops / (avgMs / 1000.0) / 1e12
            print("[Int8GEMMProbe][\(label)] Avg: \(avgMs) ms, \(tops) TOPS")

            var out = [Float16](repeating: 0, count: M * N)
            out.withUnsafeMutableBytes { raw in dest.readBytes(raw.baseAddress!, strideBytes: nil) }
            let nonZero = out.filter { $0 != 0 }.count
            print("[Int8GEMMProbe][\(label)] nonZero=\(nonZero)/\(out.count) sample=\(out.prefix(8).map { Float($0) })")
        }

        let int8GemmDest = ndArray(dataType: .float16, shape: [1, 1, M, N])
        runTimed(label: "GPU Native INT8-Affine GEMM (MPSNDArrayQuantizedMatrixMultiplication)", dest: int8GemmDest) { cb in
            qmm.encode(to: cb, sourceArrays: sourceArrays, destinationArray: int8GemmDest)
        }

        let lhsF = ndArray(dataType: .float16, shape: [1, 1, M, K])
        let rhsF = ndArray(dataType: .float16, shape: [1, 1, K, N])
        var lhsFBuf = [Float16](repeating: 0, count: M * K)
        var rhsFBuf = [Float16](repeating: 0, count: K * N)
        for i in 0..<lhsFBuf.count { lhsFBuf[i] = Float16(((i % 7) - 3)) * 0.1 + 0.05 }
        for i in 0..<rhsFBuf.count { rhsFBuf[i] = Float16(((i % 5) - 2)) * 0.1 + 0.05 }
        lhsFBuf.withUnsafeMutableBytes { raw in lhsF.writeBytes(raw.baseAddress!, strideBytes: nil) }
        rhsFBuf.withUnsafeMutableBytes { raw in rhsF.writeBytes(raw.baseAddress!, strideBytes: nil) }

        let mm = MPSNDArrayMatrixMultiplication(device: device, sourceCount: 2)
        let fp16Dest = ndArray(dataType: .float16, shape: [1, 1, M, N])
        runTimed(label: "GPU FP16 GEMM (MPSNDArrayMatrixMultiplication, baseline)", dest: fp16Dest) { cb in
            mm.encode(to: cb, sourceArrays: [lhsF, rhsF], destinationArray: fp16Dest)
        }

        print("[Int8GEMMProbe] Done.")
    }
}
