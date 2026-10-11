import Foundation
import Metal
import MetalPerformanceShaders

/// Open-thread item #1 from docs/MPSGraph_MXU_SDPA_Investigation.md Finding 4m:
/// does native FP8 convolution dispatch through the plain MPSNDArrayConvolution2D
/// class, where int8/plain-FP16 did not reach a non-float-only device-behavior
/// class cleanly? Motivating hypothesis: Finding 4i's isMXUDTypeSupported
/// disassembly showed the A18-equivalent device-behavior class unconditionally
/// accepts float8-vs-float8 operand pairs, while the older PreG13 class (Finding
/// 4k) hardcodes "Only FP32 or FP16 convolution supported." Finding 4m found
/// A18 Pro routes this exact Cin=Cout=16/KH=KW=16 config through PreG13, while
/// A20 Pro routes the identical call through the A18-equivalent class instead --
/// so FP8 is expected to hard-crash on A18 Pro (PreG13 gate) but may dispatch
/// cleanly on A20 Pro (A18-equivalent class's own eligibility logic).
enum NativeInt8ConvProbe {
    static func run() {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            print("[ConvProbe] No Metal device/queue"); return
        }
        print("[ConvProbe] Device: \(device.name)")

        // The one channel/kernel-size configuration confirmed to dispatch
        // without crashing in Finding 4m (Cin=Cout=16 exactly, matching the
        // MXU's native 16x16x16 tile size).
        let Cin = 16, Cout = 16, H = 17, W = 17, KH = 16, KW = 16
        let outH = H - KH + 1, outW = W - KW + 1
        print("[ConvProbe] Cin=\(Cin) Cout=\(Cout) H=\(H) W=\(W) KH=\(KH) KW=\(KW) outH=\(outH) outW=\(outW)")

        // Raw 1-byte E4M3 encodings (sign:1 exp:4 bias7 mantissa:3) for a small
        // cycling set of exactly-representable, all-positive values -- same
        // pattern used in NativeInt8GEMMProbe's FP8 test, avoiding both
        // overflow and the exact-zero-sum cancellation bug found there.
        let fp8Patterns: [(byte: UInt8, value: Float)] = [
            (0x38, 1.0), (0x30, 0.5), (0x3C, 1.5), (0x28, 0.25),
            (0x3A, 1.25), (0x34, 0.75), (0x3E, 1.75), (0x20, 0.125),
        ]
        func fillFP8(count: Int) -> (bytes: [UInt8], values: [Float]) {
            var bytes = [UInt8](repeating: 0, count: count)
            var values = [Float](repeating: 0, count: count)
            for i in 0..<count {
                let p = fp8Patterns[i % fp8Patterns.count]
                bytes[i] = p.byte
                values[i] = p.value
            }
            return (bytes, values)
        }
        let (inputBytes, inputData) = fillFP8(count: Cin * H * W)
        let (weightBytes, weightData) = fillFP8(count: Cout * Cin * KH * KW)

        guard let descClass = NSClassFromString("MPSNDArrayConvolution2DDescriptor") as? NSObject.Type else {
            print("[ConvProbe] descriptor class not found"); return
        }
        let desc = descClass.init()
        desc.setValue(KW, forKey: "kernelWidth")
        desc.setValue(KH, forKey: "kernelHeight")
        desc.setValue(Cin, forKey: "inputFeatureChannels")
        desc.setValue(Cout, forKey: "outputFeatureChannels")
        desc.setValue(1, forKey: "strideInPixelsX")
        desc.setValue(1, forKey: "strideInPixelsY")
        desc.setValue(1, forKey: "groups")
        desc.setValue(1, forKey: "dilationRateInX")
        desc.setValue(1, forKey: "dilationRateInY")
        desc.setValue(0, forKey: "dataFormat")
        desc.setValue(0, forKey: "weightsFormat")
        desc.setValue(1, forKey: "channelMultiplier")

        guard let convClass = NSClassFromString("MPSNDArrayConvolution2D") as? NSObject.Type else {
            print("[ConvProbe] class not found"); return
        }
        let sel = Selector(("initWithDevice:ndArrayConvolution2DDescriptor:"))
        typealias AllocFn = @convention(c) (AnyObject, Selector) -> AnyObject
        let allocSel = Selector(("alloc"))
        let allocImp = unsafeBitCast(class_getMethodImplementation(object_getClass(convClass), allocSel), to: AllocFn.self)
        let allocated = allocImp(convClass, allocSel)
        typealias InitFn = @convention(c) (AnyObject, Selector, AnyObject, AnyObject) -> AnyObject?
        let imp = unsafeBitCast(class_getMethodImplementation(convClass, sel), to: InitFn.self)
        guard let convInstance = imp(allocated, sel, device as AnyObject, desc) else {
            print("[ConvProbe] Failed to construct MPSNDArrayConvolution2D"); return
        }
        let kernel = unsafeDowncast(convInstance, to: MPSNDArrayMultiaryKernel.self)
        print("[ConvProbe] MPSNDArrayConvolution2D (plain, FP8-E4M3) instance created: \(kernel)")

        func ndArray(dataType: MPSDataType, shape: [Int]) -> MPSNDArray {
            let d = MPSNDArrayDescriptor(dataType: dataType, shape: shape.map { NSNumber(value: $0) })
            return MPSNDArray(device: device, descriptor: d)
        }
        func writeBytes(_ arr: MPSNDArray, _ bytes: [UInt8]) {
            var b = bytes
            b.withUnsafeMutableBytes { raw in arr.writeBytes(raw.baseAddress!, strideBytes: nil) }
        }

        guard #available(iOS 27.0, *) else {
            print("[ConvProbe] FP8 requires iOS 27+"); return
        }
        let inputArr = ndArray(dataType: .float8e4m3, shape: [1, Cin, H, W])
        let weightArr = ndArray(dataType: .float8e4m3, shape: [Cout, Cin, KH, KW])
        writeBytes(inputArr, inputBytes)
        writeBytes(weightArr, weightBytes)
        // Destination also FP8: NativeInt8GEMMProbe's FP8 GEMM test found this
        // was the one combination that dispatched on A20 Pro (FP16/FP32
        // destinations crashed identically on A18 Pro regardless, so this
        // choice shouldn't matter for the A18-Pro-crashes-either-way side of
        // the comparison, and matches the one confirmed-working side).
        let destArr = ndArray(dataType: .float8e4m3, shape: [1, Cout, outH, outW])

        let cb = queue.makeCommandBuffer()!
        kernel.encode(to: cb, sourceArrays: [inputArr, weightArr], destinationArray: destArr)
        cb.commit()
        cb.waitUntilCompleted()

        var outBytes = [UInt8](repeating: 0, count: Cout * outH * outW)
        outBytes.withUnsafeMutableBytes { raw in destArr.readBytes(raw.baseAddress!, strideBytes: nil) }
        let nonZero = outBytes.filter { $0 != 0 }.count
        print("[ConvProbe] Dispatch succeeded, no crash. GPU result: nonZero=\(nonZero)/\(outBytes.count) sampleBytes=\(outBytes.prefix(16))")

        // CPU reference, NCHW/OIHW (the convention confirmed structurally
        // correct, if not numerically matching, in Finding 4m) -- printed for
        // comparison even though Finding 4m's full layout sweep found no exact
        // match; this is a coarse sanity check (right order of magnitude?) not
        // a byte-for-byte correctness claim.
        func inputOffset(_ c: Int, _ h: Int, _ w: Int) -> Int { c*H*W + h*W + w }
        func weightOffset(_ o: Int, _ c: Int, _ kh: Int, _ kw: Int) -> Int { o*Cin*KH*KW + c*KH*KW + kh*KW + kw }
        var expected = [Float](repeating: 0, count: Cout * outH * outW)
        for o in 0..<Cout { for oh in 0..<outH { for ow in 0..<outW {
            var acc: Float = 0
            for c in 0..<Cin { for kh in 0..<KH { for kw in 0..<KW {
                acc += inputData[inputOffset(c, oh+kh, ow+kw)] * weightData[weightOffset(o, c, kh, kw)]
            }}}
            expected[o*outH*outW + oh*outW + ow] = acc
        }}}
        print("[ConvProbe] CPU reference (NCHW/OIHW), first 8: \(expected.prefix(8))")
    }
}
