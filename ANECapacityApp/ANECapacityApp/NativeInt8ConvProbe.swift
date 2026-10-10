import Foundation
import Metal
import MetalPerformanceShaders

/// Probe for the private, undocumented `MPSNDArrayQuantizedConvolution`
/// class (i.e. reaching `conv2d_a18/a20_int32_int8_int8` in the metallib,
/// per docs/MPSGraph_MXU_SDPA_Investigation.md Finding 4h/4i). Plain
/// `MPSNDArrayConvolution2D` (tried first, see git history on this file)
/// never worked for int8 regardless of any shape/order permutation tried --
/// turned out it isn't the int8 entry point at all. By analogy with
/// `MPSNDArrayMatrixMultiplication` vs. the dedicated
/// `MPSNDArrayQuantizedMatrixMultiplication` subclass (Finding 4j), there is
/// a parallel dedicated subclass for convolution:
/// `-[MPSNDArrayQuantizedConvolution initWithDevice:
/// ndArrayConvolution2DDescriptor:dataQuantizationDescriptor:
/// weightsQuantizationDescriptor:]`, confirmed via `nm -C` on the extracted
/// framework binary. Driven here via KVC + raw objc_msgSend, same technique
/// as the GEMM probe, then force-downcast to the public
/// `MPSNDArrayMultiaryKernel` base class to call the normal, public, typed
/// `encodeToCommandBuffer:sourceArrays:destinationArray:`.
enum NativeInt8ConvProbe {
    static func makeQuantizedConvInstance(
        device: MTLDevice, convDesc: NSObject, dataQuantDesc: NSObject, weightsQuantDesc: NSObject
    ) -> AnyObject? {
        guard let cls = NSClassFromString("MPSNDArrayQuantizedConvolution") as? NSObject.Type else { return nil }
        let sel = Selector(("initWithDevice:ndArrayConvolution2DDescriptor:dataQuantizationDescriptor:weightsQuantizationDescriptor:"))
        guard cls.instancesRespond(to: sel) else { return nil }
        typealias AllocFn = @convention(c) (AnyObject, Selector) -> AnyObject
        let allocSel = Selector(("alloc"))
        let allocImp = unsafeBitCast(class_getMethodImplementation(object_getClass(cls), allocSel), to: AllocFn.self)
        let allocated = allocImp(cls, allocSel)
        typealias InitFn = @convention(c) (AnyObject, Selector, AnyObject, AnyObject, AnyObject, AnyObject) -> AnyObject?
        let imp = unsafeBitCast(class_getMethodImplementation(cls, sel), to: InitFn.self)
        return imp(allocated, sel, device as AnyObject, convDesc, dataQuantDesc, weightsQuantDesc)
    }

    static func run() {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            print("[ConvProbe] No Metal device/queue"); return
        }
        print("[ConvProbe] Device: \(device.name)")

        guard let descClass = NSClassFromString("MPSNDArrayConvolution2DDescriptor") as? NSObject.Type else {
            print("[ConvProbe] MPSNDArrayConvolution2DDescriptor class not found"); return
        }
        let desc = descClass.init()

        // Cin=1 (product Cin*KH*KW=256) brought the "not enough taps" error
        // back even with KH=KW=16 -- so the real gate is the *combined*
        // Cin*KH*KW reduction length, not KH*KW alone (4096, Cin=16, worked;
        // 256, Cin=1, didn't). Back to the proven-working Cin=Cout=16, with
        // H/W shrunk to the minimum for a small, hand-printable case.
        let Cin = 16, Cout = 16, H = 17, W = 17, KH = 16, KW = 16
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

        guard #available(iOS 18.0, *) else {
            print("[ConvProbe] MPSNDArrayAffineQuantizationDescriptor requires iOS 18+"); return
        }
        let dataQuantDesc = MPSNDArrayAffineQuantizationDescriptor(dataType: .int8, hasZeroPoint: false, hasMinValue: false)
        let weightsQuantDesc = MPSNDArrayAffineQuantizationDescriptor(dataType: .int8, hasZeroPoint: false, hasMinValue: false)

        guard let convInstance = makeQuantizedConvInstance(
            device: device, convDesc: desc, dataQuantDesc: dataQuantDesc, weightsQuantDesc: weightsQuantDesc
        ) else {
            print("[ConvProbe] Failed to construct MPSNDArrayQuantizedConvolution"); return
        }
        let kernel = unsafeDowncast(convInstance, to: MPSNDArrayMultiaryKernel.self)
        print("[ConvProbe] MPSNDArrayQuantizedConvolution instance created: \(kernel)")

        func ndArray(dataType: MPSDataType, shape: [Int]) -> MPSNDArray {
            let d = MPSNDArrayDescriptor(dataType: dataType, shape: shape.map { NSNumber(value: $0) })
            return MPSNDArray(device: device, descriptor: d)
        }
        func writeInt8(_ arr: MPSNDArray, _ data: [Int8]) {
            var d = data
            d.withUnsafeMutableBytes { raw in arr.writeBytes(raw.baseAddress!, strideBytes: nil) }
        }
        func writeHalf(_ arr: MPSNDArray, _ value: Float16) {
            var d = [value]
            d.withUnsafeMutableBytes { raw in arr.writeBytes(raw.baseAddress!, strideBytes: nil) }
        }
        func writeInt8Scalar(_ arr: MPSNDArray, _ value: Int8) {
            var d = [value]
            d.withUnsafeMutableBytes { raw in arr.writeBytes(raw.baseAddress!, strideBytes: nil) }
        }
        func readHalf(_ arr: MPSNDArray, count: Int) -> [Float16] {
            var out = [Float16](repeating: 0, count: count)
            out.withUnsafeMutableBytes { raw in arr.readBytes(raw.baseAddress!, strideBytes: nil) }
            return out
        }

        let scale: Float = 1.0 / 4.0
        func fillBoundedRandom(count: Int, seed: UInt64) -> [Int8] {
            var buf = [Int8](repeating: 0, count: count)
            var state = seed
            for i in 0..<count {
                state = state &* 6364136223846793005 &+ 1442695040888963407
                let v = Int8(truncatingIfNeeded: Int((state >> 56) & 0x07) - 4)
                buf[i] = v == 0 ? 1 : v
            }
            return buf
        }
        let inputData = fillBoundedRandom(count: Cin * H * W, seed: 0x9E3779B97F4A7C15)
        let weightData = fillBoundedRandom(count: Cout * Cin * KH * KW, seed: 0xC2B2AE3D27D4EB4F)

        let outH = H - KH + 1, outW = W - KW + 1

        // Comprehensive layout sweep instead of one-guess-at-a-time: the
        // SAME flat inputData/weightData buffers reinterpreted under every
        // plausible axis convention (does dataFormat=0 mean NCHW or NHWC?
        // does weightsFormat=0 mean OIHW, IOHW, OHWI, or HWIO? is the real
        // kernel spatially flipped? is the output NCHW or NHWC?), each
        // compared against the actual GPU result below. Reports only the
        // best match per combination (not 32 full dumps) to keep this a
        // single device round-trip instead of many.
        func inputOffset(_ layout: String, _ c: Int, _ h: Int, _ w: Int) -> Int {
            layout == "nchw" ? c*H*W + h*W + w : h*W*Cin + w*Cin + c
        }
        func weightOffset(_ layout: String, _ o: Int, _ c: Int, _ kh: Int, _ kw: Int) -> Int {
            switch layout {
            case "oihw": return o*Cin*KH*KW + c*KH*KW + kh*KW + kw
            case "iohw": return c*Cout*KH*KW + o*KH*KW + kh*KW + kw
            case "ohwi": return o*KH*KW*Cin + kh*KW*Cin + kw*Cin + c
            default /* hwio */: return kh*KW*Cin*Cout + kw*Cin*Cout + c*Cout + o
            }
        }
        func outputOffset(_ layout: String, _ o: Int, _ oh: Int, _ ow: Int) -> Int {
            layout == "nchw" ? o*outH*outW + oh*outW + ow : oh*outW*Cout + ow*Cout + o
        }
        func computeExpected(inLayout: String, wLayout: String, outLayout: String, flip: Bool) -> [Float] {
            var result = [Float](repeating: 0, count: Cout * outH * outW)
            for o in 0..<Cout { for oh in 0..<outH { for ow in 0..<outW {
                var acc: Float = 0
                for c in 0..<Cin { for kh in 0..<KH { for kw in 0..<KW {
                    let (ikh, ikw) = flip ? (KH-1-kh, KW-1-kw) : (kh, kw)
                    let inVal = inputData[inputOffset(inLayout, c, oh+ikh, ow+ikw)]
                    let wVal = weightData[weightOffset(wLayout, o, c, kh, kw)]
                    acc += Float(inVal) * Float(wVal)
                }}}
                result[outputOffset(outLayout, o, oh, ow)] = acc * scale * scale
            }}}
            return result
        }

        var candidates: [(String, [Float])] = []
        for inLayout in ["nchw", "nhwc"] {
            for wLayout in ["oihw", "iohw", "ohwi", "hwio"] {
                for outLayout in ["nchw", "nhwc"] {
                    for flip in [false, true] {
                        let label = "in=\(inLayout) w=\(wLayout) out=\(outLayout) flip=\(flip)"
                        candidates.append((label, computeExpected(inLayout: inLayout, wLayout: wLayout, outLayout: outLayout, flip: flip)))
                    }
                }
            }
        }

        let inputArr = ndArray(dataType: .int8, shape: [1, Cin, H, W])
        let weightArr = ndArray(dataType: .int8, shape: [Cout, Cin, KH, KW])
        writeInt8(inputArr, inputData)
        writeInt8(weightArr, weightData)

        // Scalar [1,1,1,1] per-tensor scale worked fine for the GEMM class
        // (Finding 4j) but the 32-way layout sweep above found no match at
        // all (best diff no better than random) -- testing whether conv
        // instead needs per-channel-shaped scale arrays structurally, even
        // though the actual value is uniform across channels either way.
        func writeHalfBroadcast(_ arr: MPSNDArray, count: Int, _ value: Float16) {
            var d = [Float16](repeating: value, count: count)
            d.withUnsafeMutableBytes { raw in arr.writeBytes(raw.baseAddress!, strideBytes: nil) }
        }
        // Per-channel shapes triggered a NEW, worse error ("Unsupported
        // quantization scheme") -- confirms [1,1,1,1] per-tensor scalar is
        // structurally correct; the mismatch must be elsewhere.
        let dataScale = ndArray(dataType: .float16, shape: [1, 1, 1, 1])
        let weightsScale = ndArray(dataType: .float16, shape: [1, 1, 1, 1])
        writeHalfBroadcast(dataScale, count: 1, Float16(scale))
        writeHalfBroadcast(weightsScale, count: 1, Float16(scale))

        let destArr = ndArray(dataType: .float16, shape: [1, Cout, outH, outW])

        let cb = queue.makeCommandBuffer()!
        // Order by analogy with MPSNDArrayQuantizedMatrixMultiplication's
        // documented [LHS, RHS, <LHS quant inputs>, <RHS quant inputs>]
        // (Finding 4j): [data, weights, dataScale, dataZero, weightsScale, weightsZero].
        kernel.encode(to: cb, sourceArrays: [inputArr, weightArr, dataScale, weightsScale], destinationArray: destArr)
        cb.commit()
        cb.waitUntilCompleted()

        let out = readHalf(destArr, count: Cout * outH * outW)
        print("[ConvProbe] GPU result: \(out.map { Float($0) })")

        func closeEnough(_ a: Float16, _ b: Float) -> Bool {
            abs(Float(a) - b) <= max(1.0, abs(b) * 0.02)
        }
        func maxAbsDiff(_ candidate: [Float]) -> Float {
            zip(out, candidate).map { abs(Float($0) - $1) }.max() ?? -1
        }
        var ranked = candidates.map { (label: $0.0, diff: maxAbsDiff($0.1), exact: zip(out, $0.1).allSatisfy { closeEnough($0, $1) }) }
        ranked.sort { $0.diff < $1.diff }
        print("[ConvProbe] Layout sweep, best 5 by max-abs-diff (0 or near-0 = found it):")
        for r in ranked.prefix(5) {
            print("[ConvProbe]   diff=\(r.diff) exact=\(r.exact)  \(r.label)")
        }
        if let exact = ranked.first(where: { $0.exact }) {
            print("[ConvProbe] EXACT MATCH FOUND: \(exact.label)")
        } else {
            print("[ConvProbe] No exact match among \(candidates.count) layout candidates.")
        }
    }
}
