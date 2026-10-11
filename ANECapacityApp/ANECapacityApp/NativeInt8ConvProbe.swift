import Foundation
import Metal
import MetalPerformanceShaders

/// Sanity check: verify against the plain, trusted, non-quantized
/// MPSNDArrayConvolution2D class (FP16) using fully asymmetric dimensions
/// (Cin != Cout, H != W, KH != KW) so no accidental symmetry can mask an
/// axis-swap bug the way earlier square tests might have. Runs the same
/// 32-way layout sweep (input NCHW/NHWC x weight OIHW/IOHW/OHWI/HWIO x
/// output NCHW/NHWC x kernel flip) used on the quantized path, which found
/// no match there either -- isolating whether the issue is in the
/// quantization path specifically or in this low-level class's conventions
/// generally (see git history on this file for the quantized-path attempts).
///
/// GPU dispatch runs BEFORE the CPU-side layout sweep is computed, since the
/// sweep is O(32 * Cout*outH*outW*Cin*KH*KW) and dwarfs dispatch cost at
/// large channel counts -- no point paying it if the kernel is about to
/// hard-abort on a validation gate anyway.
enum NativeInt8ConvProbe {
    static func run() {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            print("[ConvProbe] No Metal device/queue"); return
        }
        print("[ConvProbe] Device: \(device.name)")

        let Cin = 512, Cout = 512, H = 34, W = 34, KH = 3, KW = 3
        let outH = H - KH + 1, outW = W - KW + 1
        print("[ConvProbe] Cin=\(Cin) Cout=\(Cout) H=\(H) W=\(W) KH=\(KH) KW=\(KW) outH=\(outH) outW=\(outW)")

        func fillBoundedRandom(count: Int, seed: UInt64) -> [Float] {
            var buf = [Float](repeating: 0, count: count)
            var state = seed
            for i in 0..<count {
                state = state &* 6364136223846793005 &+ 1442695040888963407
                let v = Float(Int((state >> 56) & 0x07)) - 4
                buf[i] = v == 0 ? 1 : v
            }
            return buf
        }
        let inputData = fillBoundedRandom(count: Cin * H * W, seed: 0x9E3779B97F4A7C15)
        let weightData = fillBoundedRandom(count: Cout * Cin * KH * KW, seed: 0xC2B2AE3D27D4EB4F)

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
        print("[ConvProbe] MPSNDArrayConvolution2D (plain, FP16) instance created: \(kernel)")

        func ndArray(dataType: MPSDataType, shape: [Int]) -> MPSNDArray {
            let d = MPSNDArrayDescriptor(dataType: dataType, shape: shape.map { NSNumber(value: $0) })
            return MPSNDArray(device: device, descriptor: d)
        }
        func writeHalf(_ arr: MPSNDArray, _ data: [Float]) {
            var d = data.map { Float16($0) }
            d.withUnsafeMutableBytes { raw in arr.writeBytes(raw.baseAddress!, strideBytes: nil) }
        }
        func readHalf(_ arr: MPSNDArray, count: Int) -> [Float16] {
            var out = [Float16](repeating: 0, count: count)
            out.withUnsafeMutableBytes { raw in arr.readBytes(raw.baseAddress!, strideBytes: nil) }
            return out
        }

        let inputArr = ndArray(dataType: .float16, shape: [1, Cin, H, W])
        let weightArr = ndArray(dataType: .float16, shape: [Cout, Cin, KH, KW])
        writeHalf(inputArr, inputData)
        writeHalf(weightArr, weightData)
        let destArr = ndArray(dataType: .float16, shape: [1, Cout, outH, outW])

        let cb = queue.makeCommandBuffer()!
        kernel.encode(to: cb, sourceArrays: [inputArr, weightArr], destinationArray: destArr)
        cb.commit()
        cb.waitUntilCompleted()

        let out = readHalf(destArr, count: Cout * outH * outW)
        print("[ConvProbe] GPU result (first 16): \(out.prefix(16).map { Float($0) })")
        print("[ConvProbe] Dispatch succeeded, no crash -- computing layout sweep now")

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
                    guard oh+ikh < H, ow+ikw < W else { continue }
                    let inVal = inputData[inputOffset(inLayout, c, oh+ikh, ow+ikw)]
                    let wVal = weightData[weightOffset(wLayout, o, c, kh, kw)]
                    acc += inVal * wVal
                }}}
                result[outputOffset(outLayout, o, oh, ow)] = acc
            }}}
            return result
        }

        func maxAbsDiff(_ candidate: [Float]) -> Float {
            zip(out, candidate).map { abs(Float($0) - $1) }.max() ?? -1
        }

        var ranked: [(label: String, diff: Float)] = []
        for inLayout in ["nchw", "nhwc"] {
            for wLayout in ["oihw", "iohw", "ohwi", "hwio"] {
                for outLayout in ["nchw", "nhwc"] {
                    for flip in [false, true] {
                        let label = "in=\(inLayout) w=\(wLayout) out=\(outLayout) flip=\(flip)"
                        let diff = maxAbsDiff(computeExpected(inLayout: inLayout, wLayout: wLayout, outLayout: outLayout, flip: flip))
                        ranked.append((label, diff))
                    }
                }
            }
        }
        ranked.sort { $0.diff < $1.diff }
        print("[ConvProbe] Layout sweep (asymmetric dims), best 5 by max-abs-diff (near-0 = found it):")
        for r in ranked.prefix(5) {
            print("[ConvProbe]   diff=\(r.diff)  \(r.label)")
        }
    }
}
